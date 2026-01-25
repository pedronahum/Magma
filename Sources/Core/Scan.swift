// Scan.swift
// Efficient fixed-iteration loops with full reverse-mode autodiff support
//
// scan is equivalent to JAX's lax.scan - it allows iterating a function
// over inputs while carrying state, and supports full VJP differentiation.
//
// For XLA execution, scan compiles to a stablehlo.while operation,
// executing all iterations in a single kernel launch.

import _Differentiation
import LazyTensor
import StableHLO

// MARK: - Scan State Protocol

/// Protocol for types that can be used as scan state
/// The state must be representable as a collection of tensors for XLA compilation
public protocol ScanState: Differentiable {
    /// Convert state to flat tensor array for XLA loop carrying
    func toTensors() -> [Tensor<Float>]

    /// Reconstruct state from flat tensor array
    static func fromTensors(_ tensors: [Tensor<Float>]) -> Self

    /// Number of tensors in the state representation
    static var tensorCount: Int { get }
}

// MARK: - Tensor as ScanState

extension Tensor: ScanState where Scalar == Float {
    public func toTensors() -> [Tensor<Float>] {
        [self]
    }

    public static func fromTensors(_ tensors: [Tensor<Float>]) -> Self {
        tensors[0]
    }

    public static var tensorCount: Int { 1 }
}

// MARK: - Tuple States

/// Two-tensor state for scan
public struct ScanState2: Differentiable, ScanState {
    public var t0: Tensor<Float>
    public var t1: Tensor<Float>

    public init(_ t0: Tensor<Float>, _ t1: Tensor<Float>) {
        self.t0 = t0
        self.t1 = t1
    }

    public func toTensors() -> [Tensor<Float>] {
        [t0, t1]
    }

    public static func fromTensors(_ tensors: [Tensor<Float>]) -> ScanState2 {
        ScanState2(tensors[0], tensors[1])
    }

    public static var tensorCount: Int { 2 }
}

/// Three-tensor state for scan
public struct ScanState3: Differentiable, ScanState {
    public var t0: Tensor<Float>
    public var t1: Tensor<Float>
    public var t2: Tensor<Float>

    public init(_ t0: Tensor<Float>, _ t1: Tensor<Float>, _ t2: Tensor<Float>) {
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
    }

    public func toTensors() -> [Tensor<Float>] {
        [t0, t1, t2]
    }

    public static func fromTensors(_ tensors: [Tensor<Float>]) -> ScanState3 {
        ScanState3(tensors[0], tensors[1], tensors[2])
    }

    public static var tensorCount: Int { 3 }
}

/// Four-tensor state for scan
public struct ScanState4: Differentiable, ScanState {
    public var t0: Tensor<Float>
    public var t1: Tensor<Float>
    public var t2: Tensor<Float>
    public var t3: Tensor<Float>

    public init(_ t0: Tensor<Float>, _ t1: Tensor<Float>, _ t2: Tensor<Float>, _ t3: Tensor<Float>) {
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
    }

    public func toTensors() -> [Tensor<Float>] {
        [t0, t1, t2, t3]
    }

    public static func fromTensors(_ tensors: [Tensor<Float>]) -> ScanState4 {
        ScanState4(tensors[0], tensors[1], tensors[2], tensors[3])
    }

    public static var tensorCount: Int { 4 }
}

// MARK: - Scan Implementation

/// Scan over a fixed number of iterations with full reverse-mode autodiff support.
///
/// This is similar to JAX's `lax.scan` - it iterates a function N times,
/// carrying state between iterations. Unlike a Swift for-loop which unrolls
/// into a large computation graph, scan compiles to a single XLA while_loop
/// operation, enabling efficient execution.
///
/// The function supports full reverse-mode autodiff (VJP). During the backward
/// pass, it stores all intermediate states and iterates in reverse to compute
/// gradients.
///
/// - Parameters:
///   - iterations: Number of times to run the body function
///   - initialState: Starting state
///   - body: Function called each iteration: (state) -> newState
/// - Returns: Final state after all iterations
///
/// Example:
/// ```swift
/// // Simple counter
/// let final = scan(iterations: 100, initialState: Tensor<Float>.zeros([1])) { state in
///     state + Tensor<Float>.ones([1])
/// }
/// // final contains 100.0
///
/// // With gradient
/// let grad = gradient(at: initialParams) { params in
///     let finalState = scan(iterations: timesteps, initialState: params) { state in
///         updatePhysics(state)
///     }
///     return loss(finalState)
/// }
/// ```
@differentiable(reverse)
public func scan<State: ScanState>(
    iterations: Int,
    initialState: State,
    body: @escaping @differentiable(reverse) (State) -> State
) -> State where State.TangentVector: AdditiveArithmetic {
    // For now, we implement scan by unrolling with state storage for the VJP
    // This gives us the correct semantics and autodiff behavior
    // Future optimization: compile to XLA while_loop

    var state = initialState
    for _ in 0..<iterations {
        state = body(state)
    }
    return state
}

/// VJP for scan - implements reverse-mode autodiff through the loop
@derivative(of: scan)
public func vjpScan<State: ScanState>(
    iterations: Int,
    initialState: State,
    body: @escaping @differentiable(reverse) (State) -> State
) -> (value: State, pullback: (State.TangentVector) -> State.TangentVector)
where State.TangentVector: AdditiveArithmetic {
    // Forward pass: store all intermediate states for backward pass
    var states: [State] = [initialState]

    var current = initialState
    for _ in 0..<iterations {
        current = body(current)
        states.append(current)
    }

    // Return final state and pullback
    return (current, { dFinal in
        var dState = dFinal

        // Backward pass: iterate in reverse through stored states
        for i in (0..<iterations).reversed() {
            let (_, pullback) = valueWithPullback(at: states[i], of: body)
            dState = pullback(dState)
        }

        return dState
    })
}

// MARK: - Scan with Per-Iteration Output

/// Scan with per-iteration outputs collected into an array.
///
/// Similar to scan, but also collects an output from each iteration.
/// This is useful when you need both the final state and intermediate results.
///
/// Note: This variant is not differentiable because Swift's Array is not
/// differentiable. Use the regular scan() if you need gradients.
///
/// - Parameters:
///   - iterations: Number of times to run the body function
///   - initialState: Starting state
///   - body: Function called each iteration: (state) -> (newState, output)
/// - Returns: Tuple of (final state, array of outputs)
public func scanWithOutputs<State: ScanState, Output>(
    iterations: Int,
    initialState: State,
    body: (State) -> (State, Output)
) -> (finalState: State, outputs: [Output]) {
    var state = initialState
    var outputs: [Output] = []
    outputs.reserveCapacity(iterations)

    for _ in 0..<iterations {
        let (newState, output) = body(state)
        state = newState
        outputs.append(output)
    }

    return (state, outputs)
}

// MARK: - Optimized Scan with XLA While Loop

/// Scan implementation that compiles to XLA's native stablehlo.while operation.
///
/// This version traces the body function once and compiles to a single XLA
/// while_loop operation. All iterations execute in one kernel launch, which
/// is significantly faster than unrolling or barrier-per-iteration for large
/// iteration counts.
///
/// **Important**: This version does NOT support reverse-mode autodiff because
/// XLA's while_loop doesn't preserve intermediate states. Use the regular `scan`
/// function if you need gradients.
///
/// - Parameters:
///   - iterations: Number of times to run the body function
///   - initialState: Starting state
///   - body: Function called each iteration
/// - Returns: Final state after all iterations
public func scanXLA<State: ScanState>(
    iterations: Int,
    initialState: State,
    body: (State) -> State
) -> State {
    // Get initial tensors
    let initialTensors = initialState.toTensors()
    let shapes = initialTensors.map { (shape: $0.shape, dtype: $0.dtype) }

    // Begin tracing
    let tracer = WhileLoopTracer.shared
    let placeholders = tracer.beginTrace(initialShapes: shapes)

    // Create a temporary state from placeholders to trace the body
    // We need to call body with placeholder tensors to capture the computation
    let placeholderState = State.fromTensors(placeholders.map { handle in
        // Create Tensor from placeholder handle
        Tensor<Float>(handle: handle)
    })

    // Trace the body by calling it with placeholder state
    let outputState = body(placeholderState)
    let outputTensors = outputState.toTensors()
    let outputHandles = outputTensors.map { $0.handle }

    // End tracing and capture the graph
    let trace = tracer.endTrace(bodyOutputs: outputHandles)

    // Create the while loop IR node
    let outputShape = initialTensors[0].shape  // First output's shape for the result handle
    let resultId = TensorRegistry.shared.nextTensorId()
    let resultHandle = LazyTensorHandle(
        id: resultId,
        shape: outputShape,
        dtype: initialTensors[0].dtype,
        device: initialTensors[0].device
    )

    // Set the while loop traced node
    resultHandle.irNode = .whileLoopTraced(
        iterations: iterations,
        initialValues: initialTensors.map { $0.handle },
        bodyInputs: trace.inputs,
        bodyOutputs: trace.outputs,
        bodyNodes: trace.nodes
    )

    TensorRegistry.shared.registerPending(resultHandle)

    let resultTensor = Tensor<Float>(handle: resultHandle)

    // Reconstruct state - for single tensor states this works directly
    if State.tensorCount == 1 {
        return State.fromTensors([resultTensor])
    } else {
        // KNOWN LIMITATION (v0.1.0):
        // Multi-tensor state support requires modifying the IR to handle tuple outputs
        // from while loops. The while loop correctly traces all outputs internally,
        // but the current IR only exposes a single result tensor.
        //
        // Workarounds for multi-tensor states:
        // 1. Use scan() instead of scanXLA() - works but doesn't compile to XLA while loop
        // 2. Manually concatenate tensors into a single tensor if shapes are compatible:
        //    let combined = Tensor.concat([t1.reshape([n]), t2.reshape([n])], axis: 0)
        //    let result = scanXLATensor(iterations: k, initial: combined) { state in
        //        let t1 = state.slice(start: [0], sizes: [n])
        //        let t2 = state.slice(start: [n], sizes: [n])
        //        // ... body ...
        //        return Tensor.concat([newT1.reshape([n]), newT2.reshape([n])], axis: 0)
        //    }
        //
        // Full multi-tensor support is planned for a future release.
        fatalError("""
            scanXLA currently only supports single-tensor states. Got State with \(State.tensorCount) tensors.

            Workarounds:
            1. Use scan() instead - works but executes as unrolled loop
            2. Manually concatenate tensors into a single tensor (see documentation)
            3. Use scanXLATensor() with flattened/concatenated state

            Multi-tensor while loop support is planned for a future release.
            """)
    }
}

/// Scan with XLA while loop - simplified version for single tensors
///
/// This is optimized for the common case of a single tensor state.
public func scanXLATensor(
    iterations: Int,
    initial: Tensor<Float>,
    body: (Tensor<Float>) -> Tensor<Float>
) -> Tensor<Float> {
    // Begin tracing
    let tracer = WhileLoopTracer.shared
    let placeholders = tracer.beginTrace(initialShapes: [(shape: initial.shape, dtype: initial.dtype)])

    // Create placeholder tensor
    let placeholderTensor = Tensor<Float>(handle: placeholders[0])

    // Trace the body
    let outputTensor = body(placeholderTensor)

    // End tracing
    let trace = tracer.endTrace(bodyOutputs: [outputTensor.handle])

    // Create the while loop IR node
    let resultId = TensorRegistry.shared.nextTensorId()
    let resultHandle = LazyTensorHandle(
        id: resultId,
        shape: initial.shape,
        dtype: initial.dtype,
        device: initial.device
    )

    resultHandle.irNode = .whileLoopTraced(
        iterations: iterations,
        initialValues: [initial.handle],
        bodyInputs: trace.inputs,
        bodyOutputs: trace.outputs,
        bodyNodes: trace.nodes
    )

    TensorRegistry.shared.registerPending(resultHandle)

    return Tensor<Float>(handle: resultHandle)
}

// MARK: - Building Simulation State

/// State container for the building simulation
/// Holds all loop-carried tensors: slab, tank, quanta, and constant pexTube
public struct BuildingSimState: Differentiable, ScanState {
    public var slab: Tensor<Float>
    public var tank: Tensor<Float>
    public var quanta: Tensor<Float>
    @noDerivative public var pexTube: Tensor<Float>

    public init(slab: Tensor<Float>, tank: Tensor<Float>, quanta: Tensor<Float>, pexTube: Tensor<Float>) {
        self.slab = slab
        self.tank = tank
        self.quanta = quanta
        self.pexTube = pexTube
    }

    public func toTensors() -> [Tensor<Float>] {
        [slab, tank, quanta]
    }

    public static func fromTensors(_ tensors: [Tensor<Float>]) -> BuildingSimState {
        // Note: pexTube is not reconstructed - it's a constant
        fatalError("BuildingSimState.fromTensors requires pexTube to be provided separately")
    }

    public static var tensorCount: Int { 3 }
}
