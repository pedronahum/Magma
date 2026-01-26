// Magma - LazyTensor
// x10-style lazy execution with graph tracing and compilation caching
//
// This module provides:
// - LazyTensorHandle: Reference to a node in the computation graph
// - IRNode: Operation representation in the graph
// - IRGraph: Full computation graph with topological ordering
// - LazyTensorBarrier: Triggers compilation and execution

import Foundation
import StableHLO
import XLARuntime

// MARK: - Lazy Tensor Handle

/// Handle to a lazy tensor value in the computation graph
///
/// LazyTensorHandle represents a tensor that may or may not be materialized.
/// Operations on lazy tensors build up a computation graph that is
/// compiled and executed when `LazyTensorBarrier()` is called.
public final class LazyTensorHandle: @unchecked Sendable {

    /// Unique identifier for this handle
    public let id: UInt64

    /// Shape of the tensor
    public let shape: [Int]

    /// Data type of tensor elements
    public let dtype: DType

    /// Device where this tensor resides
    public let device: Device

    /// The IR node that produces this tensor (nil if materialized)
    public var irNode: IRNode?

    /// Materialized buffer (nil if not yet computed)
    public var materializedBuffer: PJRTBuffer?

    /// Whether this tensor has been materialized
    public var isMaterialized: Bool {
        materializedBuffer != nil
    }

    /// Whether this tensor is "live" - explicitly requested by the user for materialization.
    /// Live tensors are outputs that must be computed. Non-live tensors are intermediates
    /// that may be garbage collected if no longer needed.
    /// Inspired by TensorFlow Swift's lazy tensor design.
    public var isLive: Bool = false

    /// Create a new lazy tensor handle
    public init(id: UInt64, shape: [Int], dtype: DType, device: Device) {
        self.id = id
        self.shape = shape
        self.dtype = dtype
        self.device = device
    }
}

// MARK: - IR Node

/// Represents an operation in the computation graph
public indirect enum IRNode: @unchecked Sendable {
    /// Data from a materialized buffer
    case data(PJRTBuffer)

    /// Constant value
    case constant(values: [Float], shape: [Int])

    /// Operation with inputs
    case operation(op: OpKind, inputs: [LazyTensorHandle], attributes: [String: Any])

    /// While loop with traced body
    /// - iterations: Number of loop iterations
    /// - initialValues: Input handles for loop-carried state
    /// - bodyInputs: Placeholder handles used during body tracing (map to block args)
    /// - bodyOutputs: Output handles from body tracing (become loop outputs)
    /// - bodyNodes: All nodes created during body tracing (the body's computation graph)
    case whileLoopTraced(
        iterations: Int,
        initialValues: [LazyTensorHandle],
        bodyInputs: [LazyTensorHandle],
        bodyOutputs: [LazyTensorHandle],
        bodyNodes: [LazyTensorHandle]
    )
}

// MARK: - Operation Kinds

/// All supported operation types
public enum OpKind: String, Sendable {
    // Elementwise binary
    case add, subtract, multiply, divide
    case maximum, minimum, power

    // Elementwise unary
    case negate, abs, exponential, log
    case sqrt, rsqrt, sine, cosine, tanh
    case floor, ceil

    // Type conversion
    case convert

    // Matrix operations
    case matmul, batchedMatmul, transpose, reshape, broadcast

    // Reductions
    case reduceSum, reduceMax, reduceMin, reduceMean

    // Activations
    case relu, sigmoid, softmax, gelu, leakyRelu, elu, silu

    // Clamp/Clip
    case clamp

    // Convolutions
    case conv1d, conv2d, convTranspose2d, maxPool2d, avgPool2d

    // Normalization
    case batchNorm, layerNorm

    // Comparison
    case equal, less, greater, select

    // Indexing/Slicing
    case slice, pad, gather, scatter, concatenate

    // Control Flow
    case whileLoop, cond

    // Random Number Generation
    case rngUniform, rngNormal

    // Fused Operations (for optimization)
    // These are detected by fusion passes and emitted as custom_call ops

    /// Fused scaled dot-product attention: softmax(Q @ K^T / scale) @ V
    case fusedScaledDotProductAttention

    /// Fused layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    case fusedLayerNorm

    /// Fused RMS normalization: x / sqrt(mean(x^2) + eps) * weight
    case fusedRMSNorm

    /// Fused matrix multiply with bias and activation: activation(x @ w + b)
    case fusedMatMulBiasActivation

    /// Fused rotary position embedding (RoPE)
    case fusedRoPE

    /// Fused softmax (numerically stable)
    case fusedSoftmax

    /// Fused GELU activation (exact or approximate)
    case fusedGelu
}

// MARK: - IR Graph

/// Complete computation graph ready for compilation
public final class IRGraph: @unchecked Sendable {

    /// Nodes in topological order
    public var nodes: [LazyTensorHandle] = []

    /// Output handles
    public var outputs: [LazyTensorHandle] = []

    /// Create an empty graph
    public init() {}

    /// Add an output to the graph
    public func addOutput(_ handle: LazyTensorHandle) {
        outputs.append(handle)
    }

    /// Emit StableHLO MLIR for this graph
    public func emitStableHLO(name: String) -> String {
        let emitter = StableHLOEmitter(graph: self)
        return emitter.emit(name: name)
    }
}

// MARK: - Execution Context
//
// Inspired by TensorFlow Swift's LazyTensorContext, this provides thread-local
// state for lazy tensor execution. This enables safe concurrent execution of
// independent tensor computations without cross-thread interference.

/// Thread-local execution context for lazy tensor operations
///
/// Each thread maintains its own context with:
/// - Local pending tensors (not shared across threads)
/// - Configuration flags for the current execution
/// - Statistics for this thread's executions
///
/// This design is inspired by TensorFlow Swift's LazyTensorContext pattern.
public final class ExecutionContext: @unchecked Sendable {

    /// Thread-local storage key
    private static let threadLocalKey = "Magma.ExecutionContext"

    /// Get the current thread's execution context
    public static var current: ExecutionContext {
        #if os(Linux)
        // On Linux, use a simple global for now (proper TLS requires more setup)
        return _globalContext
        #else
        if let context = Thread.current.threadDictionary[threadLocalKey] as? ExecutionContext {
            return context
        }
        let context = ExecutionContext()
        Thread.current.threadDictionary[threadLocalKey] = context
        return context
        #endif
    }

    /// Whether shape tracking is enabled (for debugging)
    public var isShapeTrackingEnabled: Bool = false

    /// Whether to promote constants to inputs (for cache optimization)
    public var shouldPromoteConstants: Bool = true

    /// Local tensor ID counter (thread-local to avoid contention)
    private var localNextId: UInt64 = 0

    /// Local pending tensors (per-thread, per-device)
    private var localPending: [String: [LazyTensorHandle]] = [:]
    private var localPendingSet: [String: Set<UInt64>] = [:]

    /// Statistics for this context
    public var executionCount: Int = 0
    public var totalNodesExecuted: Int = 0

    /// Private initializer - use ExecutionContext.current instead
    fileprivate init() {}

    /// Generate a locally-unique tensor ID
    /// Combined with thread ID for global uniqueness
    public func nextLocalId() -> UInt64 {
        let id = localNextId
        localNextId += 1
        return id
    }

    /// Mark a tensor for local materialization
    public func markForMaterialization(_ handle: LazyTensorHandle) {
        let key = handle.device.description
        if localPendingSet[key] == nil {
            localPendingSet[key] = Set()
            localPending[key] = []
        }
        if !localPendingSet[key]!.contains(handle.id) {
            localPendingSet[key]!.insert(handle.id)
            localPending[key]!.append(handle)
        }
    }

    /// Take pending tensors for a device (thread-local)
    public func takeLocalPending(for device: Device) -> [LazyTensorHandle] {
        let key = device.description
        let result = localPending[key] ?? []
        localPending[key] = []
        localPendingSet[key] = Set()
        return result
    }

    /// Clear all local state
    public func reset() {
        localPending.removeAll()
        localPendingSet.removeAll()
        executionCount = 0
        totalNodesExecuted = 0
    }
}

#if os(Linux)
/// Global context fallback for Linux (until proper TLS is implemented)
nonisolated(unsafe) private var _globalContext = ExecutionContext()
#endif

// MARK: - Tensor Registry

/// Global registry for tracking live lazy tensors
public final class TensorRegistry: @unchecked Sendable {

    /// Shared instance
    public static let shared = TensorRegistry()

    /// Next available tensor ID
    private var nextId: UInt64 = 0

    /// Pending (unmaterialized) tensors by device - these are tensors that need to be computed
    /// Only tensors explicitly marked for materialization are added here
    private var pending: [String: [LazyTensorHandle]] = [:]

    /// Tensors that have been marked for materialization (to avoid duplicates)
    private var pendingSet: [String: Set<UInt64>] = [:]

    /// Lock for thread safety
    private let lock = NSLock()

    private init() {}

    /// Generate a new unique tensor ID
    public func nextTensorId() -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let id = nextId
        nextId += 1
        return id
    }

    /// Register a pending tensor (for internal tracking - doesn't mark for output)
    public func registerPending(_ handle: LazyTensorHandle) {
        // Record to tracer if tracing is active (for while loop body capture)
        WhileLoopTracer.shared.recordNode(handle)

        // This is otherwise a no-op for intermediate tensors
        // Only tensors marked via markForMaterialization will become outputs
    }

    /// Mark a tensor for materialization (this tensor will be computed as an output)
    /// Call this when the user explicitly requests tensor values (e.g., scalars())
    public func markForMaterialization(_ handle: LazyTensorHandle) {
        lock.lock()
        defer { lock.unlock() }

        // Skip if already materialized
        if handle.isMaterialized {
            return
        }

        let key = handle.device.description
        if pendingSet[key] == nil {
            pendingSet[key] = Set()
            pending[key] = []
        }

        // Only add if not already pending
        if !pendingSet[key]!.contains(handle.id) {
            pendingSet[key]!.insert(handle.id)
            pending[key]!.append(handle)
        }
    }

    /// Get and clear pending tensors for a device
    public func takePending(for device: Device) -> [LazyTensorHandle] {
        lock.lock()
        defer { lock.unlock() }
        let key = device.description
        let result = pending[key] ?? []
        pending[key] = []
        pendingSet[key] = Set()
        return result
    }

    /// Current pending (unmaterialized) tensor count
    public var pendingCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return pending.values.reduce(0) { $0 + $1.count }
    }

    /// Clear all pending tensors
    public func clearAll() {
        lock.lock()
        defer { lock.unlock() }
        pending.removeAll()
        pendingSet.removeAll()
    }
}

// MARK: - Constant Promotion
//
// Inspired by TensorFlow Swift's LazyTensorTraceCache, constant promotion converts
// small scalar/vector constants into function inputs. This enables cache reuse
// across computations that differ only in constant values (e.g., learning rates,
// epsilon values, different batch constants).

/// Represents a constant that has been promoted to a function input
public struct PromotedConstant: Sendable {
    /// Original node ID in the graph
    public let originalNodeId: UInt64

    /// Shape of the constant
    public let shape: [Int]

    /// Data type
    public let dtype: DType

    /// The actual values (extracted from the graph)
    public let values: [Float]

    /// Index of this constant in the promoted inputs list
    public let inputIndex: Int
}

/// Result of constant promotion analysis
public struct ConstantPromotionResult: Sendable {
    /// Hash computed without constant values (structure-only)
    public let structuralHash: String

    /// Constants that were promoted to inputs
    public let promotedConstants: [PromotedConstant]

    /// Whether promotion was applied
    public var wasPromoted: Bool { !promotedConstants.isEmpty }
}

// MARK: - Compilation Cache

/// Cache for compiled executables with constant promotion support
///
/// The cache uses structural hashing (graph structure without constant values)
/// to enable cache reuse across computations that differ only in scalar constants.
/// This is particularly beneficial for training loops where hyperparameters
/// like learning rate or epsilon vary between iterations.
public final class CompilationCache: @unchecked Sendable {

    /// Shared cache instance
    public static let shared = CompilationCache()

    /// Cache entries keyed by structural hash
    private var cache: [String: PJRTExecutable] = [:]

    /// Lock for thread safety
    private let lock = NSLock()

    /// Maximum number of elements for a constant to be promotable
    /// Small scalars and vectors benefit most from promotion
    public static let promotionThreshold: Int = 16

    private init() {}

    /// Look up a cached executable
    public func get(hash: String) -> PJRTExecutable? {
        lock.lock()
        defer { lock.unlock() }
        return cache[hash]
    }

    /// Store an executable in the cache
    public func put(hash: String, executable: PJRTExecutable) {
        lock.lock()
        defer { lock.unlock() }
        cache[hash] = executable
    }

    /// Clear all cached entries
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
        hitCount = 0
        missCount = 0
        promotionHitCount = 0
    }

    /// Cache statistics
    public var hitCount: Int = 0
    public var missCount: Int = 0
    public var promotionHitCount: Int = 0  // Hits due to constant promotion

    /// Cache hit rate
    public var hitRate: Double {
        let total = hitCount + missCount
        return total > 0 ? Double(hitCount) / Double(total) : 0
    }

    /// Percentage of hits that were due to constant promotion
    public var promotionBenefit: Double {
        guard hitCount > 0 else { return 0 }
        return Double(promotionHitCount) / Double(hitCount)
    }
}

// MARK: - Metal Compilation Cache

#if os(macOS) && canImport(MetalHLO)

/// Cache for compiled Metal executables
public final class MetalCompilationCache: @unchecked Sendable {

    /// Shared cache instance
    public static let shared = MetalCompilationCache()

    /// Cache entries keyed by structural hash
    private var cache: [String: MetalHLOExecutable] = [:]

    /// Lock for thread safety
    private let lock = NSLock()

    private init() {}

    /// Look up a cached executable
    public func get(hash: String) -> MetalHLOExecutable? {
        lock.lock()
        defer { lock.unlock() }
        return cache[hash]
    }

    /// Store an executable in the cache
    public func put(hash: String, executable: MetalHLOExecutable) {
        lock.lock()
        defer { lock.unlock() }
        cache[hash] = executable
    }

    /// Clear all cached entries
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
        hitCount = 0
        missCount = 0
    }

    /// Cache statistics
    public var hitCount: Int = 0
    public var missCount: Int = 0

    /// Cache hit rate
    public var hitRate: Double {
        let total = hitCount + missCount
        return total > 0 ? Double(hitCount) / Double(total) : 0
    }
}

#endif

// MARK: - Lazy Tensor Barrier

/// Global XLA client (lazily initialized)
/// Access is protected by _clientLock
nonisolated(unsafe) private var _globalClient: PJRTClient?
private let _clientLock = NSLock()

/// Get or create the global XLA client
public func getGlobalClient(backend: Backend = .cpu) throws -> PJRTClient {
    _clientLock.lock()
    defer { _clientLock.unlock() }

    if let client = _globalClient {
        return client
    }

    let client = try PJRTClient.create(backend: backend)
    _globalClient = client
    return client
}

// MARK: - Metal Client

#if os(macOS) && canImport(MetalHLO)

/// Global MetalHLO client (lazily initialized)
nonisolated(unsafe) private var _globalMetalClient: MetalHLOClient?
private let _metalClientLock = NSLock()

/// Get or create the global MetalHLO client
public func getMetalHLOClient() throws -> MetalHLOClient {
    _metalClientLock.lock()
    defer { _metalClientLock.unlock() }

    if let client = _globalMetalClient {
        return client
    }

    let client = try MetalHLOClient.create()
    _globalMetalClient = client
    return client
}

#endif

/// Trigger compilation and execution of all pending operations
///
/// When called, all lazy tensors on the specified device are:
/// 1. Collected into a graph
/// 2. Validated for shape/type errors
/// 3. Analyzed for constant promotion (cache optimization)
/// 4. Converted to StableHLO MLIR
/// 5. Compiled (or fetched from cache)
/// 6. Executed
/// 7. Results stored back in tensor handles
///
/// Example:
/// ```swift
/// let y = x.matmul(w).relu()  // Lazy - no computation yet
/// LazyTensorBarrier()          // Compile and execute everything
/// print(y.scalars())           // Now we can read the results
/// ```
public func LazyTensorBarrier(on device: Device = .default) {
    // Route to Metal-specific barrier if using Metal backend
    #if os(macOS) && canImport(MetalHLO)
    if device.backend == .metal {
        MetalLazyTensorBarrier(on: device)
        return
    }
    #endif

    // 1. Collect all pending tensors for this device
    let pending = TensorRegistry.shared.takePending(for: device)
    if pending.isEmpty {
        return // Nothing to execute
    }

    // Find all outputs (tensors that need to be materialized)
    let outputs = pending.filter { !$0.isMaterialized }
    if outputs.isEmpty {
        return
    }

    // Mark outputs as live (for future memory optimization)
    for output in outputs {
        output.isLive = true
    }

    // 2. Build IR graph
    let graph = IRGraph()
    for output in outputs {
        graph.addOutput(output)
    }
    graph.buildTopologicalOrder()

    // 3. Validate graph (catch errors early with better messages)
    do {
        try graph.validate()
    } catch {
        print("Magma: Graph validation failed: \(error)")
        return
    }

    // 4. Analyze for constant promotion
    let promotionResult = graph.analyzeForConstantPromotion()
    let structuralHash = promotionResult.structuralHash

    // 5. Check compilation cache using structural hash
    let cache = CompilationCache.shared
    var executable: PJRTExecutable
    if let cached = cache.get(hash: structuralHash) {
        cache.hitCount += 1
        if promotionResult.wasPromoted {
            cache.promotionHitCount += 1
        }
        executable = cached
    } else {
        cache.missCount += 1

        // 6. Emit StableHLO MLIR with constant promotion
        let emitter = StableHLOEmitter(graph: graph)
        let mlir = emitter.emit(
            name: "lazy_graph_\(structuralHash.prefix(8))",
            promotedConstants: promotionResult.promotedConstants
        )

        // 7. Compile
        do {
            let client = try getGlobalClient(backend: device.backend)
            executable = try client.compile(mlir)
            cache.put(hash: structuralHash, executable: executable)
        } catch {
            // Always write failing MLIR to debug file for first unique failure
            let debugPath = "/tmp/magma_debug_\(structuralHash.prefix(8)).mlir"
            // Check if we haven't already written this hash
            if !FileManager.default.fileExists(atPath: debugPath) {
                print("Magma: MLIR that failed to compile (hash=\(structuralHash.prefix(8))):")
                print(mlir.prefix(3000))
                print("... (truncated)")
                if let data = mlir.data(using: .utf8) {
                    let url = URL(fileURLWithPath: debugPath)
                    try? data.write(to: url)
                    print("Magma: Full MLIR written to \(debugPath)")
                }
            }
            print("Magma: Compilation failed: \(error)")
            return
        }
    }

    // 8. Collect input buffers
    // First: data nodes (pre-materialized large tensors)
    var inputBuffers: [PJRTBuffer] = []
    for node in graph.nodes {
        if case .data(let buffer) = node.irNode {
            inputBuffers.append(buffer)
        }
    }

    // Second: promoted constants (need to create buffers for their values)
    if promotionResult.wasPromoted {
        do {
            let client = try getGlobalClient(backend: device.backend)
            for promoted in promotionResult.promotedConstants.sorted(by: { $0.inputIndex < $1.inputIndex }) {
                let buffer = try client.createBuffer(
                    promoted.values,
                    shape: promoted.shape,
                    elementType: .float32,
                    device: nil
                )
                inputBuffers.append(buffer)
            }
        } catch {
            print("Magma: Failed to create buffers for promoted constants: \(error)")
            return
        }
    }

    // 9. Execute
    do {
        let outputBuffers = try executable.execute(inputBuffers)

        // 10. Update tensor handles with results
        for (i, output) in outputs.enumerated() {
            if i < outputBuffers.count {
                let buffer = outputBuffers[i]
                output.materializedBuffer = buffer
                // Set irNode to .data so the tensor can be used as input in future computations
                // This replaces the computation graph with a direct reference to the materialized buffer
                output.irNode = .data(buffer)
            }
        }
    } catch {
        print("Magma: Execution failed: \(error)")
    }
}

// MARK: - Metal Lazy Tensor Barrier

#if os(macOS) && canImport(MetalHLO)

/// Metal-specific lazy tensor barrier
///
/// This function handles execution on Metal GPUs via MetalHLO.
/// It follows the same pattern as the PJRT-based barrier but uses MetalHLO for compilation and execution.
private func MetalLazyTensorBarrier(on device: Device) {
    // 1. Collect all pending tensors for this device
    let pending = TensorRegistry.shared.takePending(for: device)
    if pending.isEmpty {
        return // Nothing to execute
    }

    // Find all outputs (tensors that need to be materialized)
    let outputs = pending.filter { !$0.isMaterialized }
    if outputs.isEmpty {
        return
    }

    // Mark outputs as live (for future memory optimization)
    for output in outputs {
        output.isLive = true
    }

    // 2. Build IR graph
    let graph = IRGraph()
    for output in outputs {
        graph.addOutput(output)
    }
    graph.buildTopologicalOrder()

    // 3. Validate graph (catch errors early with better messages)
    do {
        try graph.validate()
    } catch {
        print("Magma [Metal]: Graph validation failed: \(error)")
        return
    }

    // 3.5. Run optimization passes
    let optimizedGraph: IRGraph
    if ProcessInfo.processInfo.environment["MAGMA_NO_OPT"] != "1" {
        let passManager = PassManager.shared
        optimizedGraph = passManager.run(on: graph)

        if ProcessInfo.processInfo.environment["MAGMA_DEBUG"] == "1" {
            let originalCount = graph.nodes.count
            let optimizedCount = optimizedGraph.nodes.count
            if originalCount != optimizedCount {
                print("Magma [Metal]: Optimization reduced \(originalCount) -> \(optimizedCount) nodes")
            }
        }
    } else {
        optimizedGraph = graph
    }

    // Use optimized graph from here on

    // 4. Analyze for constant promotion
    let promotionResult = optimizedGraph.analyzeForConstantPromotion()
    let structuralHash = promotionResult.structuralHash

    // 5. Check Metal compilation cache using structural hash
    let cache = MetalCompilationCache.shared
    var executable: MetalHLOExecutable
    if let cached = cache.get(hash: structuralHash) {
        cache.hitCount += 1
        executable = cached
    } else {
        cache.missCount += 1

        // 6. Emit StableHLO MLIR with constant promotion
        let emitter = StableHLOEmitter(graph: optimizedGraph)
        let mlir = emitter.emit(
            name: "metal_graph_\(structuralHash.prefix(8))",
            promotedConstants: promotionResult.promotedConstants
        )

        // Debug: Print the MLIR being generated (opt-in via MAGMA_DEBUG=1)
        if ProcessInfo.processInfo.environment["MAGMA_DEBUG"] == "1" {
            print("Magma [Metal]: Generated MLIR:")
            print(mlir)
            print("Magma [Metal]: Promoted constants count: \(promotionResult.promotedConstants.count)")
            print("Magma [Metal]: Was promoted: \(promotionResult.wasPromoted)")
        }

        // 7. Compile via MetalHLO
        do {
            let client = try getMetalHLOClient()
            executable = try client.compile(mlir)

            if ProcessInfo.processInfo.environment["MAGMA_DEBUG"] == "1" {
                print("Magma [Metal]: Compiled executable - inputs: \(executable.inputCount), outputs: \(executable.outputCount)")
            }

            cache.put(hash: structuralHash, executable: executable)
        } catch {
            // Always write failing MLIR to debug file for first unique failure
            let debugPath = "/tmp/magma_metal_debug_\(structuralHash.prefix(8)).mlir"
            if !FileManager.default.fileExists(atPath: debugPath) {
                print("Magma [Metal]: MLIR that failed to compile (hash=\(structuralHash.prefix(8))):")
                print(mlir.prefix(3000))
                print("... (truncated)")
                if let data = mlir.data(using: .utf8) {
                    let url = URL(fileURLWithPath: debugPath)
                    try? data.write(to: url)
                    print("Magma [Metal]: Full MLIR written to \(debugPath)")
                }
            }
            print("Magma [Metal]: Compilation failed: \(error)")
            return
        }
    }

    // 8. Collect input buffers
    // For Metal, we need to handle both data nodes and promoted constants
    // Note: Data nodes from PJRT buffers need to be converted to Metal buffers

    var inputBuffers: [MetalHLOBuffer] = []

    // First: Check for any data nodes - these would come from pre-existing PJRT buffers
    // which is not typical for Metal-first execution, but we should handle it
    for node in optimizedGraph.nodes {
        if case .data(let pjrtBuffer) = node.irNode {
            // Transfer PJRT buffer data to Metal
            do {
                let client = try getMetalHLOClient()
                let hostData = try pjrtBuffer.toFloatArray()
                // Use optimized non-throwing createBuffer for Float arrays
                let metalBuffer = client.createBuffer(
                    hostData,
                    shape: node.shape
                )
                inputBuffers.append(metalBuffer)
            } catch {
                print("Magma [Metal]: Failed to convert PJRT buffer to Metal: \(error)")
                return
            }
        }
    }

    // Second: promoted constants (need to create buffers for their values)
    if promotionResult.wasPromoted {
        do {
            let client = try getMetalHLOClient()
            for promoted in promotionResult.promotedConstants.sorted(by: { $0.inputIndex < $1.inputIndex }) {
                // Use optimized non-throwing createBuffer for Float arrays
                let buffer = client.createBuffer(
                    promoted.values,
                    shape: promoted.shape
                )
                inputBuffers.append(buffer)
            }
        } catch {
            print("Magma [Metal]: Failed to create buffers for promoted constants: \(error)")
            return
        }
    }

    let debugEnabled = ProcessInfo.processInfo.environment["MAGMA_DEBUG"] == "1"

    if debugEnabled {
        print("Magma [Metal]: Input buffers count: \(inputBuffers.count)")
        print("Magma [Metal]: Outputs to materialize: \(outputs.count)")
        for (i, output) in outputs.enumerated() {
            print("Magma [Metal]: Output \(i) handle ID: \(output.id), shape: \(output.shape)")
        }
    }

    // 9. Execute
    do {
        let outputBuffers = try executable.execute(inputBuffers)

        if debugEnabled {
            print("Magma [Metal]: Output buffers received: \(outputBuffers.count)")
        }

        // 10. Update tensor handles with results
        // For Metal, we store the MetalHLO buffer data back to host and create a "virtual" PJRT buffer
        // This allows seamless interop with the rest of the Magma system
        for (i, output) in outputs.enumerated() {
            if i < outputBuffers.count {
                let metalBuffer = outputBuffers[i]
                // Create a materialized result that can be read
                output.materializedBuffer = nil  // No PJRT buffer for Metal
                // Store a marker that this is a Metal-materialized tensor
                // For now, we store the data in a special way
                // In the future, we might want a proper dual-buffer system
                let hostData = try metalBuffer.toFloatArray()

                if debugEnabled {
                    print("Magma [Metal]: Output \(i) - shape: \(output.shape), data count: \(hostData.count), first few: \(Array(hostData.prefix(5)))")
                }

                // Re-materialize via PJRT for compatibility (if needed)
                // For now, mark as materialized without a PJRT buffer
                output.irNode = .constant(values: hostData, shape: output.shape)
            }
        }
    } catch {
        print("Magma [Metal]: Execution failed: \(error)")
    }
}

#endif // os(macOS) && canImport(MetalHLO)

/// Execute a graph synchronously and return output buffers
///
/// This is a lower-level API that executes a pre-built graph.
/// For most use cases, prefer LazyTensorBarrier().
public func executeGraph(_ graph: IRGraph, on device: Device = .default) throws -> [PJRTBuffer] {
    graph.buildTopologicalOrder()

    // Run optimization passes
    let optimizedGraph: IRGraph
    if ProcessInfo.processInfo.environment["MAGMA_NO_OPT"] != "1" {
        let passManager = PassManager.shared
        optimizedGraph = passManager.run(on: graph)
    } else {
        optimizedGraph = graph
    }

    let emitter = StableHLOEmitter(graph: optimizedGraph)
    let graphHash = optimizedGraph.computeHash()
    let mlir = emitter.emit(name: "graph_\(graphHash.prefix(8))")

    // Check cache
    let cache = CompilationCache.shared
    var executable: PJRTExecutable

    if let cached = cache.get(hash: graphHash) {
        cache.hitCount += 1
        executable = cached
    } else {
        cache.missCount += 1
        let client = try getGlobalClient(backend: device.backend)
        executable = try client.compile(mlir)
        cache.put(hash: graphHash, executable: executable)
    }

    // Collect input buffers
    var inputBuffers: [PJRTBuffer] = []
    for node in graph.nodes {
        if case .data(let buffer) = node.irNode {
            inputBuffers.append(buffer)
        }
    }

    return try executable.execute(inputBuffers)
}

// MARK: - While Loop Tracing

/// Context for tracing a function body for while loop emission
///
/// During tracing, we create placeholder inputs that represent loop-carried variables.
/// All operations performed using these placeholders are recorded, and the final
/// outputs are captured. This traced graph can then be emitted as a stablehlo.while.
public final class WhileLoopTracer: @unchecked Sendable {

    /// Whether tracing is currently active
    public private(set) var isTracing: Bool = false

    /// Placeholder inputs created for this trace (represent block arguments)
    public private(set) var bodyInputs: [LazyTensorHandle] = []

    /// Nodes created during tracing (the body computation graph)
    public private(set) var tracedNodes: [LazyTensorHandle] = []

    /// Set of traced node IDs for O(1) lookup
    private var tracedNodeIds: Set<UInt64> = []

    /// Starting tensor ID for this trace (to identify traced nodes)
    private var startingId: UInt64 = 0

    /// Shared tracer instance
    public static let shared = WhileLoopTracer()

    private init() {}

    /// Begin tracing a while loop body
    ///
    /// Creates placeholder inputs for each initial value. Operations performed
    /// on these placeholders will be recorded as the body graph.
    ///
    /// - Parameter initialShapes: Shapes and dtypes of loop-carried variables
    /// - Returns: Placeholder handles to use as body inputs
    public func beginTrace(initialShapes: [(shape: [Int], dtype: DType)]) -> [LazyTensorHandle] {
        precondition(!isTracing, "Cannot nest while loop traces")
        isTracing = true
        bodyInputs = []
        tracedNodes = []
        tracedNodeIds = []
        startingId = TensorRegistry.shared.nextTensorId() + 1000000  // Use high IDs for placeholders

        // Create placeholder inputs (these represent block arguments in MLIR)
        for (i, spec) in initialShapes.enumerated() {
            let id = startingId + UInt64(i)
            let handle = LazyTensorHandle(id: id, shape: spec.shape, dtype: spec.dtype, device: .default)
            // Mark as a "placeholder" - no irNode means it's an external input
            handle.irNode = nil
            bodyInputs.append(handle)
        }

        return bodyInputs
    }

    /// Record a node created during tracing
    ///
    /// Call this from operation creation to track nodes belonging to the trace.
    public func recordNode(_ handle: LazyTensorHandle) {
        guard isTracing else { return }
        if !tracedNodeIds.contains(handle.id) {
            tracedNodeIds.insert(handle.id)
            tracedNodes.append(handle)
        }
    }

    /// Check if a handle is a placeholder input for current trace
    public func isPlaceholder(_ handle: LazyTensorHandle) -> Bool {
        guard isTracing else { return false }
        return bodyInputs.contains { $0.id == handle.id }
    }

    /// End tracing and return the captured graph
    ///
    /// - Parameter bodyOutputs: The output handles from the body function
    /// - Returns: Tuple of (bodyInputs, bodyOutputs, tracedNodes)
    public func endTrace(bodyOutputs: [LazyTensorHandle]) -> (
        inputs: [LazyTensorHandle],
        outputs: [LazyTensorHandle],
        nodes: [LazyTensorHandle]
    ) {
        precondition(isTracing, "Not currently tracing")
        isTracing = false

        let result = (inputs: bodyInputs, outputs: bodyOutputs, nodes: tracedNodes)

        // Clear state
        bodyInputs = []
        tracedNodes = []
        tracedNodeIds = []

        return result
    }

    /// Abort tracing (for error recovery)
    public func abortTrace() {
        isTracing = false
        bodyInputs = []
        tracedNodes = []
        tracedNodeIds = []
    }
}

// MARK: - Metrics

/// Print compilation and execution metrics
public func PrintMetrics() {
    let cache = CompilationCache.shared
    let hitRatePercent = Int(cache.hitRate * 100)
    let promotionBenefitPercent = Int(cache.promotionBenefit * 100)
    print("=== Magma Metrics ===")
    print("Cache hits: \(cache.hitCount)")
    print("Cache misses: \(cache.missCount)")
    print("Hit rate: \(hitRatePercent)%")
    print("Promotion hits: \(cache.promotionHitCount) (\(promotionBenefitPercent)% of hits from constant promotion)")
    print("Pending tensors: \(TensorRegistry.shared.pendingCount)")
}

// MARK: - NSLock (for cross-platform)

#if os(Linux)
import Glibc

final class NSLock: @unchecked Sendable {
    private var mutex = pthread_mutex_t()

    init() {
        pthread_mutex_init(&mutex, nil)
    }

    deinit {
        pthread_mutex_destroy(&mutex)
    }

    func lock() {
        pthread_mutex_lock(&mutex)
    }

    func unlock() {
        pthread_mutex_unlock(&mutex)
    }
}
#else
import Foundation
#endif
