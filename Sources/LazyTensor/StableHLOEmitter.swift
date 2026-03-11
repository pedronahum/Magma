// Magma - StableHLOEmitter
// Converts IRGraph to StableHLO MLIR

import Foundation
import StableHLO

/// Converts an IRGraph to StableHLO MLIR text
public final class StableHLOEmitter {

    /// The IR graph to emit
    private let graph: IRGraph

    /// Map from LazyTensorHandle ID to MLIR Value
    private var valueMap: [UInt64: Value] = [:]

    /// The MLIR builder
    private let builder: MLIRBuilder

    /// Set of node IDs that should be treated as inputs (for constant promotion)
    private var promotedConstantIds: Set<UInt64> = []

    /// Create an emitter for the given graph
    public init(graph: IRGraph) {
        self.graph = graph
        self.builder = MLIRBuilder()
    }

    /// Emit StableHLO MLIR for the graph
    public func emit(name: String) -> String {
        emit(name: name, promotedConstants: [])
    }

    /// Emit StableHLO MLIR for the graph with constant promotion
    ///
    /// When constants are promoted, they become function inputs instead of
    /// being embedded in the MLIR. This enables cache reuse across graphs
    /// that differ only in constant values.
    ///
    /// - Parameters:
    ///   - name: Module/function name
    ///   - promotedConstants: Constants to treat as inputs
    /// - Returns: MLIR text
    public func emit(name: String, promotedConstants: [PromotedConstant]) -> String {
        // Track which constants are promoted
        promotedConstantIds = Set(promotedConstants.map { $0.originalNodeId })

        // First, collect all input handles (data nodes and promoted constants)
        var inputHandles: [LazyTensorHandle] = []

        // Walk the graph in topological order
        let sortedNodes = topologicalSort()

        // Collect data inputs first (they come before promoted constants)
        // Both .data (PJRT) and .metalData (Metal) are on-device inputs
        for handle in sortedNodes {
            var isDataInput = false
            if case .data = handle.irNode { isDataInput = true }
            #if os(macOS) && canImport(MetalHLO)
            if case .metalData = handle.irNode { isDataInput = true }
            #endif
            if isDataInput {
                inputHandles.append(handle)
            }
        }

        // Create function arguments for data inputs
        for handle in inputHandles {
            let tensorType = TensorType(shape: handle.shape, dtype: handle.dtype)
            let arg = builder.argument(tensorType)
            valueMap[handle.id] = arg
        }

        // Create function arguments for promoted constants (sorted by inputIndex)
        let sortedPromoted = promotedConstants.sorted { $0.inputIndex < $1.inputIndex }
        for promoted in sortedPromoted {
            // Find the original handle
            if let handle = sortedNodes.first(where: { $0.id == promoted.originalNodeId }) {
                let tensorType = TensorType(shape: promoted.shape, dtype: promoted.dtype)
                let arg = builder.argument(tensorType)
                valueMap[handle.id] = arg
            }
        }

        // Emit operations
        for handle in sortedNodes {
            if valueMap[handle.id] != nil {
                continue // Already emitted (argument, promoted constant, or previously emitted)
            }

            guard let node = handle.irNode else {
                continue
            }

            let value = emitNode(handle: handle, node: node)
            valueMap[handle.id] = value
        }

        // Collect outputs
        let outputs = graph.outputs.compactMap { valueMap[$0.id] }

        return builder.build(name: name, outputs: outputs)
    }

    /// Emit a single IR node
    private func emitNode(handle: LazyTensorHandle, node: IRNode) -> Value {
        switch node {
        case .constant(let values, let shape):
            return builder.constant(values.map { Double($0) }, shape: shape, dtype: handle.dtype)

        case .data:
            // Should have been handled as an argument
            fatalError("Data node should have been created as argument")

        #if os(macOS) && canImport(MetalHLO)
        case .metalData:
            // Should have been handled as an argument (same as .data)
            fatalError("MetalData node should have been created as argument")
        #endif

        case .operation(let op, let inputs, let attributes):
            return emitOperation(op: op, inputs: inputs, output: handle, attributes: attributes)

        case .whileLoopTraced(let iterations, let initialValues, let bodyInputs, let bodyOutputs, let bodyNodes):
            return emitWhileLoop(
                iterations: iterations,
                initialValues: initialValues,
                bodyInputs: bodyInputs,
                bodyOutputs: bodyOutputs,
                bodyNodes: bodyNodes,
                outputHandle: handle
            )
        }
    }

    /// Emit a while loop with traced body
    ///
    /// This emits a stablehlo.while operation using the traced body graph.
    /// The loop runs for a fixed number of iterations using a counter.
    private func emitWhileLoop(
        iterations: Int,
        initialValues: [LazyTensorHandle],
        bodyInputs: [LazyTensorHandle],
        bodyOutputs: [LazyTensorHandle],
        bodyNodes: [LazyTensorHandle],
        outputHandle: LazyTensorHandle
    ) -> Value {
        // Get MLIR values for initial loop-carried state
        var loopInitialValues: [Value] = []
        for input in initialValues {
            guard let value = valueMap[input.id] else {
                fatalError("While loop initial value not found: id=\(input.id)")
            }
            loopInitialValues.append(value)
        }

        // Add iteration counter and max as first two loop-carried variables
        let iterInit = builder.constant([0.0], shape: [], dtype: .float32)
        let iterMax = builder.constant([Double(iterations)], shape: [], dtype: .float32)

        // Full initial values: [iter, iterMax, state0, state1, ...]
        let fullInitialValues = [iterInit, iterMax] + loopInitialValues

        // Build types for all loop-carried values
        let allTypes: [TensorType] = fullInitialValues.map { $0.type }
        let allTypesStr = allTypes.map { $0.mlirType }.joined(separator: ", ")

        // Create result values for the while loop
        let resultValues = builder.allocateValues(types: allTypes)
        let resultsStr = resultValues.map { $0.name }.joined(separator: ", ")

        // Create input string
        let inputsStr = fullInitialValues.map { $0.displayName }.joined(separator: ", ")

        // Create block arguments for condition region (use unique ids based on builder state)
        var condBlockArgs: [Value] = []
        var condArgDefs: [String] = []
        var nextCondId = 100  // Start at 100 to avoid conflicts with other values
        for type in allTypes {
            let argValue = Value(id: nextCondId, type: type, isBlockArg: true, blockArgPrefix: "cond")
            condBlockArgs.append(argValue)
            condArgDefs.append("\(argValue.displayName): \(type.mlirType)")
            nextCondId += 1
        }

        // Build condition region with its own builder
        let condBuilder = MLIRBuilder()
        condBuilder.setNextValueId(200)  // Start after block args
        // Condition: iter < itermax (condBlockArgs[0] < condBlockArgs[1])
        let condResult = condBuilder.compare(condBlockArgs[0], condBlockArgs[1], direction: .lt)
        let condOps = condBuilder.getOperations().map { "    " + $0 }.joined(separator: "\n")

        // Create block arguments for body region
        var bodyBlockArgs: [Value] = []
        var bodyArgDefs: [String] = []
        var nextBodyId = 300  // Start at 300 to avoid conflicts
        for type in allTypes {
            let argValue = Value(id: nextBodyId, type: type, isBlockArg: true, blockArgPrefix: "body")
            bodyBlockArgs.append(argValue)
            bodyArgDefs.append("\(argValue.displayName): \(type.mlirType)")
            nextBodyId += 1
        }

        // Build body region with its own builder
        let bodyBuilder = MLIRBuilder()
        bodyBuilder.setNextValueId(400)  // Start after body block args

        // Map body placeholder handles to body block arg values
        var localValueMap: [UInt64: Value] = [:]
        for (i, input) in bodyInputs.enumerated() {
            localValueMap[input.id] = bodyBlockArgs[i + 2]  // +2 for iter, itermax
        }

        // Emit body nodes using the body builder
        for node in bodyNodes {
            guard let irNode = node.irNode else { continue }

            switch irNode {
            case .constant(let values, let shape):
                localValueMap[node.id] = bodyBuilder.constant(values.map { Double($0) }, shape: shape, dtype: node.dtype)

            case .operation(let op, let inputs, let attributes):
                let inputValues = inputs.map { input -> Value in
                    if let v = localValueMap[input.id] {
                        return v
                    } else if let v = self.valueMap[input.id] {
                        // Reference to value outside the loop - need to pass through
                        // For now, this is a limitation - we'd need to add these as loop-carried vars
                        fatalError("While body: external reference \(input.id) not yet supported")
                    } else {
                        fatalError("While body: missing input \(input.id) for \(op)")
                    }
                }
                let result = emitOperationWithBuilder(bodyBuilder, op: op, inputs: inputValues, output: node, attributes: attributes)
                localValueMap[node.id] = result

            case .data:
                fatalError("Unexpected node type in while body")
            #if os(macOS) && canImport(MetalHLO)
            case .metalData:
                fatalError("Unexpected node type in while body")
            #endif
            case .whileLoopTraced:
                fatalError("Unexpected node type in while body")
            }
        }

        // Get output values
        var outputValues: [Value] = []
        for output in bodyOutputs {
            guard let v = localValueMap[output.id] else {
                fatalError("While body output not found: id=\(output.id)")
            }
            outputValues.append(v)
        }

        // Increment iteration counter
        let one = bodyBuilder.constant([1.0], shape: [], dtype: .float32)
        let newIter = bodyBuilder.add(bodyBlockArgs[0], one)

        // Body returns: [newIter, itermax, newState0, newState1, ...]
        let bodyReturnValues = [newIter, bodyBlockArgs[1]] + outputValues
        let bodyReturnStr = bodyReturnValues.map { $0.displayName }.joined(separator: ", ")
        let bodyOps = bodyBuilder.getOperations().map { "    " + $0 }.joined(separator: "\n")

        // Build the while operation string using the generic op format
        // from stablehlo spec: "stablehlo.while"(%operands) ({ cond }, { body })
        let whileOp = """
            \(resultsStr) = "stablehlo.while"(\(inputsStr)) ({
            ^bb0(\(condArgDefs.joined(separator: ", "))):
        \(condOps)
              "stablehlo.return"(\(condResult.displayName)) : (\(condResult.type.mlirType)) -> ()
            }, {
            ^bb0(\(bodyArgDefs.joined(separator: ", "))):
        \(bodyOps)
              "stablehlo.return"(\(bodyReturnStr)) : (\(allTypesStr)) -> ()
            }) : (\(allTypesStr)) -> (\(allTypesStr))
        """
        builder.addRawOperation(whileOp)

        // Return the first state result (index 2, after iter and iterMax)
        if resultValues.count > 2 {
            return resultValues[2]
        } else {
            return resultValues[0]
        }
    }

    /// Emit an operation using a specific builder (for while loop bodies)
    private func emitOperationWithBuilder(
        _ targetBuilder: MLIRBuilder,
        op: OpKind,
        inputs: [Value],
        output: LazyTensorHandle,
        attributes: [String: Any]
    ) -> Value {
        switch op {
        // Binary elementwise
        case .add:
            return targetBuilder.add(inputs[0], inputs[1])
        case .subtract:
            return targetBuilder.subtract(inputs[0], inputs[1])
        case .multiply:
            return targetBuilder.multiply(inputs[0], inputs[1])
        case .divide:
            return targetBuilder.divide(inputs[0], inputs[1])
        case .maximum:
            return targetBuilder.maximum(inputs[0], inputs[1])
        case .minimum:
            return targetBuilder.minimum(inputs[0], inputs[1])
        case .power:
            return targetBuilder.power(inputs[0], inputs[1])

        // Unary elementwise
        case .negate:
            return targetBuilder.negate(inputs[0])
        case .abs:
            return targetBuilder.abs(inputs[0])
        case .exponential:
            return targetBuilder.exponential(inputs[0])
        case .log:
            return targetBuilder.log(inputs[0])
        case .sqrt:
            return targetBuilder.sqrt(inputs[0])
        case .rsqrt:
            return targetBuilder.rsqrt(inputs[0])
        case .sine:
            return targetBuilder.sine(inputs[0])
        case .cosine:
            return targetBuilder.cosine(inputs[0])
        case .tanh:
            return targetBuilder.tanh(inputs[0])
        case .floor:
            return targetBuilder.floor(inputs[0])
        case .ceil:
            return targetBuilder.ceil(inputs[0])

        // Type conversion
        case .convert:
            let targetDtype = attributes["targetDtype"] as! DType
            return targetBuilder.convert(inputs[0], to: targetDtype)

        // Matrix operations
        case .matmul:
            return targetBuilder.dot(inputs[0], inputs[1])
        case .batchedMatmul:
            return targetBuilder.batchedDot(inputs[0], inputs[1])
        case .transpose:
            let permutation = attributes["permutation"] as? [Int]
            return targetBuilder.transpose(inputs[0], permutation: permutation)
        case .reshape:
            let newShape = attributes["shape"] as! [Int]
            return targetBuilder.reshape(inputs[0], to: newShape)
        case .broadcast:
            let targetShape = attributes["shape"] as! [Int]
            return targetBuilder.broadcast(inputs[0], to: targetShape)

        // Activations
        case .relu:
            return targetBuilder.relu(inputs[0])
        case .sigmoid:
            return targetBuilder.sigmoid(inputs[0])

        // Reductions
        case .reduceSum:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].type.rank)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            return targetBuilder.reduceSum(inputs[0], axes: axes, keepDims: keepDims)
        case .reduceMax:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].type.rank)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            return targetBuilder.reduceMax(inputs[0], axes: axes, keepDims: keepDims)
        case .reduceMin:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].type.rank)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            return targetBuilder.reduceMin(inputs[0], axes: axes, keepDims: keepDims)
        case .reduceMean:
            // Mean = sum / count
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].type.rank)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            let sum = targetBuilder.reduceSum(inputs[0], axes: axes, keepDims: keepDims)
            let count = axes.reduce(1) { $0 * inputs[0].type.shape[$1] }
            let countValue = targetBuilder.constant(Double(count), type: sum.type)
            return targetBuilder.divide(sum, countValue)

        // Comparisons
        case .equal:
            return targetBuilder.equal(inputs[0], inputs[1])
        case .less:
            return targetBuilder.less(inputs[0], inputs[1])
        case .greater:
            return targetBuilder.greater(inputs[0], inputs[1])
        case .select:
            return targetBuilder.select(inputs[0], inputs[1], inputs[2])

        // Slice and pad
        case .slice:
            let starts = attributes["starts"] as! [Int]
            let limits = attributes["limits"] as! [Int]
            let strides = attributes["strides"] as! [Int]
            return targetBuilder.slice(inputs[0], starts: starts, limits: limits, strides: strides)
        case .pad:
            let low = attributes["low"] as! [Int]
            let high = attributes["high"] as! [Int]
            return targetBuilder.pad(inputs[0], low: low, high: high)
        case .concatenate:
            let dim = attributes["dimension"] as! Int
            return targetBuilder.concatenate(inputs, dimension: dim)

        // Gather and scatter
        case .gather:
            let axis = attributes["axis"] as! Int
            return targetBuilder.gather(inputs[0], indices: inputs[1], axis: axis)
        case .scatter:
            let axis = attributes["axis"] as! Int
            return targetBuilder.scatter(inputs[0], indices: inputs[1], updates: inputs[2], axis: axis)

        // Control flow - not expected in while body
        case .whileLoop, .cond:
            fatalError("Nested control flow not supported in while body")

        // Operations not yet supported in while loop body
        case .softmax, .gelu, .leakyRelu, .elu, .silu, .clamp,
             .conv1d, .conv2d, .convTranspose2d, .maxPool2d, .avgPool2d, .batchNorm, .layerNorm,
             .rngUniform, .rngNormal:
            fatalError("Operation \(op) not yet supported in while loop body")
        }
    }

    /// Emit an operation given pre-resolved input values
    ///
    /// This is used for emitting operations inside while loop bodies where
    /// we have Value inputs directly rather than going through the valueMap.
    private func emitOperationWithValues(
        op: OpKind,
        inputs: [Value],
        output: LazyTensorHandle,
        attributes: [String: Any]
    ) -> Value {
        switch op {
        // Binary elementwise
        case .add:
            return builder.add(inputs[0], inputs[1])
        case .subtract:
            return builder.subtract(inputs[0], inputs[1])
        case .multiply:
            return builder.multiply(inputs[0], inputs[1])
        case .divide:
            return builder.divide(inputs[0], inputs[1])
        case .maximum:
            return builder.maximum(inputs[0], inputs[1])
        case .minimum:
            return builder.minimum(inputs[0], inputs[1])
        case .power:
            return builder.power(inputs[0], inputs[1])

        // Unary elementwise
        case .negate:
            return builder.negate(inputs[0])
        case .abs:
            return builder.abs(inputs[0])
        case .exponential:
            return builder.exponential(inputs[0])
        case .log:
            return builder.log(inputs[0])
        case .sqrt:
            return builder.sqrt(inputs[0])
        case .rsqrt:
            return builder.rsqrt(inputs[0])
        case .sine:
            return builder.sine(inputs[0])
        case .cosine:
            return builder.cosine(inputs[0])
        case .tanh:
            return builder.tanh(inputs[0])
        case .floor:
            return builder.floor(inputs[0])
        case .ceil:
            return builder.ceil(inputs[0])

        // Type conversion
        case .convert:
            let targetDtype = attributes["targetDtype"] as! DType
            return builder.convert(inputs[0], to: targetDtype)

        // Matrix operations
        case .matmul:
            return builder.dot(inputs[0], inputs[1])
        case .batchedMatmul:
            return builder.batchedDot(inputs[0], inputs[1])
        case .transpose:
            let permutation = attributes["permutation"] as? [Int]
            return builder.transpose(inputs[0], permutation: permutation)
        case .reshape:
            let newShape = attributes["shape"] as! [Int]
            return builder.reshape(inputs[0], to: newShape)
        case .broadcast:
            let targetShape = attributes["shape"] as! [Int]
            return builder.broadcast(inputs[0], to: targetShape)

        // Reductions
        case .reduceSum:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].type.shape.count)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            return builder.reduceSum(inputs[0], axes: axes, keepDims: keepDims)
        case .reduceMax:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].type.shape.count)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            return builder.reduceMax(inputs[0], axes: axes, keepDims: keepDims)
        case .reduceMin:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].type.shape.count)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            return builder.reduceMin(inputs[0], axes: axes, keepDims: keepDims)
        case .reduceMean:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].type.shape.count)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            let sum = builder.reduceSum(inputs[0], axes: axes, keepDims: keepDims)
            let count = axes.reduce(1) { $0 * inputs[0].type.shape[$1] }
            let countVal = builder.constant(Double(count), type: sum.type)
            return builder.divide(sum, countVal)

        // Activations
        case .relu:
            return builder.relu(inputs[0])
        case .sigmoid:
            return builder.sigmoid(inputs[0])
        case .softmax:
            let axis = (attributes["axis"] as? Int) ?? -1
            let actualAxis = axis < 0 ? inputs[0].type.shape.count + axis : axis
            let maxVal = builder.reduceMax(inputs[0], axes: [actualAxis], keepDims: true)
            let shifted = builder.subtract(inputs[0], maxVal)
            let expVal = builder.exponential(shifted)
            let sumExp = builder.reduceSum(expVal, axes: [actualAxis], keepDims: true)
            return builder.divide(expVal, sumExp)
        case .gelu:
            let x = inputs[0]
            let half = builder.constant(0.5, type: x.type)
            let one = builder.constant(1.0, type: x.type)
            let sqrtTwoPi = builder.constant(0.7978845608, type: x.type)
            let coeff = builder.constant(0.044715, type: x.type)
            let x3 = builder.multiply(x, builder.multiply(x, x))
            let inner = builder.add(x, builder.multiply(coeff, x3))
            let tanhArg = builder.multiply(sqrtTwoPi, inner)
            let tanhVal = builder.tanh(tanhArg)
            let onePlusTanh = builder.add(one, tanhVal)
            return builder.multiply(half, builder.multiply(x, onePlusTanh))
        case .leakyRelu:
            let x = inputs[0]
            let alpha = (attributes["negativeSlope"] as? Float) ?? 0.01
            let alphaTensor = builder.constant(Double(alpha), type: x.type)
            let alphaX = builder.multiply(alphaTensor, x)
            return builder.maximum(x, alphaX)
        case .elu:
            let x = inputs[0]
            let alpha = (attributes["alpha"] as? Float) ?? 1.0
            let zero = builder.constant(0.0, type: x.type)
            let one = builder.constant(1.0, type: x.type)
            let alphaTensor = builder.constant(Double(alpha), type: x.type)
            let expX = builder.exponential(x)
            let expMinus1 = builder.subtract(expX, one)
            let negPart = builder.multiply(alphaTensor, expMinus1)
            let cond = builder.greater(x, zero)
            return builder.select(cond, x, negPart)
        case .silu:
            let x = inputs[0]
            let sigX = builder.sigmoid(x)
            return builder.multiply(x, sigX)

        // Comparison
        case .less:
            return builder.less(inputs[0], inputs[1])
        case .greater:
            return builder.greater(inputs[0], inputs[1])
        case .equal:
            return builder.equal(inputs[0], inputs[1])
        case .select:
            return builder.select(inputs[0], inputs[1], inputs[2])

        // Slicing
        case .slice:
            let starts = attributes["start"] as? [Int] ?? []
            let limits = attributes["limit"] as? [Int] ?? []
            let strides = attributes["strides"] as? [Int] ?? Array(repeating: 1, count: starts.count)
            return builder.slice(inputs[0], starts: starts, limits: limits, strides: strides)
        case .pad:
            let lowPad = attributes["lowPad"] as? [Int] ?? []
            let highPad = attributes["highPad"] as? [Int] ?? []
            return builder.pad(inputs[0], low: lowPad, high: highPad)
        case .concatenate:
            let dimension = attributes["dimension"] as? Int ?? 0
            return builder.concatenate(inputs, dimension: dimension)
        case .clamp:
            let x = inputs[0]
            let minVal = (attributes["min"] as? Float) ?? -Float.infinity
            let maxVal = (attributes["max"] as? Float) ?? Float.infinity
            let minTensor = builder.constant(Double(minVal), type: x.type)
            let maxTensor = builder.constant(Double(maxVal), type: x.type)
            return builder.minimum(builder.maximum(x, minTensor), maxTensor)
        case .gather:
            let axis = (attributes["axis"] as? Int) ?? 0
            return builder.gather(inputs[0], indices: inputs[1], axis: axis)
        case .scatter:
            let axis = (attributes["axis"] as? Int) ?? 0
            return builder.scatter(inputs[0], indices: inputs[1], updates: inputs[2], axis: axis)

        case .whileLoop, .cond:
            fatalError("Nested control flow not supported in while loop body")

        case .rngUniform, .rngNormal:
            fatalError("RNG operations not supported in while loop body (use deterministic ops)")

        case .conv1d, .conv2d, .convTranspose2d, .maxPool2d, .avgPool2d, .batchNorm, .layerNorm:
            fatalError("Operation \(op) not yet implemented in while loop body")
        }
    }

    /// Emit an operation
    private func emitOperation(
        op: OpKind,
        inputs: [LazyTensorHandle],
        output: LazyTensorHandle,
        attributes: [String: Any]
    ) -> Value {
        // Get input values
        let inputValues = inputs.compactMap { valueMap[$0.id] }
        guard inputValues.count == inputs.count else {
            // Debug output for error diagnosis
            var debugMsg = "StableHLOEmitter: Missing inputs for \(op)\n"
            debugMsg += "  Expected \(inputs.count) inputs, got \(inputValues.count)\n"
            for (i, input) in inputs.enumerated() {
                let found = valueMap[input.id] != nil
                debugMsg += "  Input \(i): id=\(input.id), shape=\(input.shape), found=\(found)\n"
                if let node = input.irNode {
                    switch node {
                    case .constant(_, let shape):
                        debugMsg += "    -> constant with shape \(shape)\n"
                    case .data:
                        debugMsg += "    -> data node\n"
                    #if os(macOS) && canImport(MetalHLO)
                    case .metalData:
                        debugMsg += "    -> metalData node\n"
                    #endif
                    case .operation(let innerOp, _, _):
                        debugMsg += "    -> operation \(innerOp)\n"
                    case .whileLoopTraced(let iters, _, _, _, _):
                        debugMsg += "    -> while loop (\(iters) iterations)\n"
                    }
                } else {
                    debugMsg += "    -> no irNode\n"
                }
            }
            fatalError(debugMsg)
        }

        switch op {
        // Binary elementwise
        case .add:
            return builder.add(inputValues[0], inputValues[1])
        case .subtract:
            return builder.subtract(inputValues[0], inputValues[1])
        case .multiply:
            return builder.multiply(inputValues[0], inputValues[1])
        case .divide:
            return builder.divide(inputValues[0], inputValues[1])
        case .maximum:
            return builder.maximum(inputValues[0], inputValues[1])
        case .minimum:
            return builder.minimum(inputValues[0], inputValues[1])
        case .power:
            return builder.power(inputValues[0], inputValues[1])

        // Unary elementwise
        case .negate:
            return builder.negate(inputValues[0])
        case .abs:
            return builder.abs(inputValues[0])
        case .exponential:
            return builder.exponential(inputValues[0])
        case .log:
            return builder.log(inputValues[0])
        case .sqrt:
            return builder.sqrt(inputValues[0])
        case .rsqrt:
            return builder.rsqrt(inputValues[0])
        case .sine:
            return builder.sine(inputValues[0])
        case .cosine:
            return builder.cosine(inputValues[0])
        case .tanh:
            return builder.tanh(inputValues[0])
        case .floor:
            return builder.floor(inputValues[0])
        case .ceil:
            return builder.ceil(inputValues[0])

        // Type conversion
        case .convert:
            let targetDtype = attributes["targetDtype"] as! DType
            return builder.convert(inputValues[0], to: targetDtype)

        // Matrix operations
        case .matmul:
            return builder.dot(inputValues[0], inputValues[1])
        case .batchedMatmul:
            // Batched matrix multiplication using dot_general
            return builder.batchedDot(inputValues[0], inputValues[1])
        case .transpose:
            let permutation = attributes["permutation"] as? [Int]
            return builder.transpose(inputValues[0], permutation: permutation)
        case .reshape:
            let newShape = attributes["shape"] as? [Int] ?? output.shape
            return builder.reshape(inputValues[0], to: newShape)
        case .broadcast:
            let targetShape = attributes["shape"] as? [Int] ?? output.shape
            return builder.broadcast(inputValues[0], to: targetShape)

        // Reductions
        case .reduceSum:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].shape.count)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            return builder.reduceSum(inputValues[0], axes: axes, keepDims: keepDims)
        case .reduceMax:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].shape.count)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            return builder.reduceMax(inputValues[0], axes: axes, keepDims: keepDims)
        case .reduceMin:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].shape.count)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            return builder.reduceMin(inputValues[0], axes: axes, keepDims: keepDims)
        case .reduceMean:
            // Mean = sum / count
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].shape.count)
            let keepDims = attributes["keepDims"] as? Bool ?? false
            let sum = builder.reduceSum(inputValues[0], axes: axes, keepDims: keepDims)
            let count = axes.reduce(1) { $0 * inputs[0].shape[$1] }
            let countVal = builder.constant(Double(count), type: sum.type)
            return builder.divide(sum, countVal)

        // Activations
        case .relu:
            return builder.relu(inputValues[0])
        case .sigmoid:
            return builder.sigmoid(inputValues[0])
        case .softmax:
            // softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
            let axis = (attributes["axis"] as? Int) ?? -1
            let actualAxis = axis < 0 ? inputs[0].shape.count + axis : axis
            let maxVal = builder.reduceMax(inputValues[0], axes: [actualAxis], keepDims: true)
            let shifted = builder.subtract(inputValues[0], maxVal)
            let expVal = builder.exponential(shifted)
            let sumExp = builder.reduceSum(expVal, axes: [actualAxis], keepDims: true)
            return builder.divide(expVal, sumExp)
        case .gelu:
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let x = inputValues[0]
            let half = builder.constant(0.5, type: x.type)
            let one = builder.constant(1.0, type: x.type)
            let sqrtTwoPi = builder.constant(0.7978845608, type: x.type) // sqrt(2/pi)
            let coeff = builder.constant(0.044715, type: x.type)
            let x3 = builder.multiply(x, builder.multiply(x, x))
            let inner = builder.add(x, builder.multiply(coeff, x3))
            let tanhArg = builder.multiply(sqrtTwoPi, inner)
            let tanhVal = builder.tanh(tanhArg)
            let onePlusTanh = builder.add(one, tanhVal)
            return builder.multiply(half, builder.multiply(x, onePlusTanh))

        // Comparison
        case .equal:
            return builder.equal(inputValues[0], inputValues[1])
        case .less:
            return builder.less(inputValues[0], inputValues[1])
        case .greater:
            return builder.greater(inputValues[0], inputValues[1])
        case .select:
            return builder.select(inputValues[0], inputValues[1], inputValues[2])

        // Slice operation
        case .slice:
            let starts = attributes["start"] as? [Int] ?? []
            let limits = attributes["limit"] as? [Int] ?? []
            let strides = attributes["strides"] as? [Int] ?? Array(repeating: 1, count: starts.count)
            return builder.slice(inputValues[0], starts: starts, limits: limits, strides: strides)

        // Pad operation
        case .pad:
            let lowPad = attributes["lowPad"] as? [Int] ?? []
            let highPad = attributes["highPad"] as? [Int] ?? []
            return builder.pad(inputValues[0], low: lowPad, high: highPad)

        // Concatenate operation
        case .concatenate:
            let dimension = attributes["dimension"] as? Int ?? 0
            return builder.concatenate(inputValues, dimension: dimension)

        // Leaky ReLU: max(x, alpha * x)
        case .leakyRelu:
            let x = inputValues[0]
            let alpha = (attributes["negativeSlope"] as? Float) ?? 0.01
            let alphaTensor = builder.constant(Double(alpha), type: x.type)
            let alphaX = builder.multiply(alphaTensor, x)
            return builder.maximum(x, alphaX)

        // ELU: x if x > 0, else alpha * (exp(x) - 1)
        case .elu:
            let x = inputValues[0]
            let alpha = (attributes["alpha"] as? Float) ?? 1.0
            let zero = builder.constant(0.0, type: x.type)
            let one = builder.constant(1.0, type: x.type)
            let alphaTensor = builder.constant(Double(alpha), type: x.type)
            let expX = builder.exponential(x)
            let expMinus1 = builder.subtract(expX, one)
            let negPart = builder.multiply(alphaTensor, expMinus1)
            let cond = builder.greater(x, zero)
            return builder.select(cond, x, negPart)

        // SiLU: x * sigmoid(x)
        case .silu:
            let x = inputValues[0]
            let sigX = builder.sigmoid(x)
            return builder.multiply(x, sigX)

        // Clamp: max(min, min(x, max))
        case .clamp:
            let x = inputValues[0]
            let minVal = (attributes["min"] as? Float) ?? -1.0
            let maxVal = (attributes["max"] as? Float) ?? 1.0
            let minTensor = builder.constant(Double(minVal), type: x.type)
            let maxTensor = builder.constant(Double(maxVal), type: x.type)
            let clampedLow = builder.maximum(x, minTensor)
            return builder.minimum(clampedLow, maxTensor)

        // Gather operation
        case .gather:
            let axis = (attributes["axis"] as? Int) ?? 0
            return builder.gather(inputValues[0], indices: inputValues[1], axis: axis)

        // Scatter operation
        case .scatter:
            let axis = (attributes["axis"] as? Int) ?? 0
            return builder.scatter(inputValues[0], indices: inputValues[1], updates: inputValues[2], axis: axis)

        // Control flow operations
        case .whileLoop, .cond:
            // Control flow ops require special handling with closures
            // They cannot be emitted through the simple IR -> MLIR path
            // These need to be handled at a higher level with explicit region building
            fatalError("Control flow operations require explicit region building - use MLIRBuilder directly")

        // Random number generation
        case .rngUniform:
            // inputValues[0] = low, inputValues[1] = high
            return builder.rngUniform(inputValues[0], inputValues[1], shape: output.shape)
        case .rngNormal:
            // inputValues[0] = mean, inputValues[1] = stddev
            return builder.rngNormal(inputValues[0], inputValues[1], shape: output.shape)

        // Not yet implemented
        case .conv1d, .conv2d, .convTranspose2d, .maxPool2d, .avgPool2d, .batchNorm, .layerNorm:
            fatalError("Operation \(op) not yet implemented in StableHLOEmitter")
        }
    }

    /// Topologically sort the graph nodes
    private func topologicalSort() -> [LazyTensorHandle] {
        var result: [LazyTensorHandle] = []
        var visited = Set<UInt64>()
        var visiting = Set<UInt64>() // For cycle detection

        func visit(_ handle: LazyTensorHandle) {
            if visited.contains(handle.id) {
                return
            }
            if visiting.contains(handle.id) {
                fatalError("Cycle detected in computation graph")
            }

            visiting.insert(handle.id)

            // Visit dependencies first
            if let node = handle.irNode {
                switch node {
                case .operation(_, let inputs, _):
                    for input in inputs {
                        visit(input)
                    }
                case .whileLoopTraced(_, let initialValues, _, _, _):
                    // Visit initial values for the while loop
                    for input in initialValues {
                        visit(input)
                    }
                case .constant, .data:
                    break
                #if os(macOS) && canImport(MetalHLO)
                case .metalData:
                    break
                #endif
                }
            }

            visiting.remove(handle.id)
            visited.insert(handle.id)
            result.append(handle)
        }

        // Start from outputs
        for output in graph.outputs {
            visit(output)
        }

        return result
    }
}

// MARK: - Persistent Subgraph Cache
//
// Inspired by TensorFlow Swift's ObjectIdentifier-based operation dependency caching
// (LazyTensorTrace.swift). Caches the topological subtree for each handle by pointer
// identity so that repeated barriers reuse pre-computed subgraph orderings for
// persistent handles (weights, materialized intermediate tensors) rather than
// re-traversing them every barrier.
//
// Entries are invalidated in LazyTensorHandle.irNode.didSet when a handle is updated.

nonisolated(unsafe) var _subgraphCache: [ObjectIdentifier: [LazyTensorHandle]] = [:]
private let _subgraphCacheMaxEntries = 8000

/// Invalidate the cached subgraph entry for a specific handle.
/// Must be called when a handle's irNode changes.
internal func invalidateSubgraphCache(for handle: LazyTensorHandle) {
    _subgraphCache.removeValue(forKey: ObjectIdentifier(handle))
}

// MARK: - IRGraph Extension

extension IRGraph {

    /// Build the topological order from outputs.
    ///
    /// Uses pointer identity (ObjectIdentifier) for visited tracking and consults
    /// the module-level persistent subgraph cache for handles whose subtrees are
    /// already known. Persistent handles (weights as .metalData, pre-materialized
    /// tensors) benefit most: their cached subtree is reused without any recursion.
    public func buildTopologicalOrder() {
        var result: [LazyTensorHandle] = []
        var visited = Set<ObjectIdentifier>()

        func visit(_ handle: LazyTensorHandle) {
            let oid = ObjectIdentifier(handle)
            guard !visited.contains(oid) else { return }

            // Fast path: use cached subtree if available (built in a previous barrier).
            // The cache is invalidated when irNode changes, so this is always up-to-date.
            if let cached = _subgraphCache[oid] {
                for node in cached {
                    let noid = ObjectIdentifier(node)
                    if visited.insert(noid).inserted {
                        result.append(node)
                    }
                }
                return
            }

            visited.insert(oid)
            let subtreeStart = result.count

            if let node = handle.irNode {
                switch node {
                case .operation(_, let inputs, _):
                    for input in inputs {
                        visit(input)
                    }
                case .whileLoopTraced(_, let initialValues, _, _, _):
                    for input in initialValues {
                        visit(input)
                    }
                case .constant, .data:
                    break
                #if os(macOS) && canImport(MetalHLO)
                case .metalData:
                    break
                #endif
                }
            }

            result.append(handle)

            // Cache this handle's subtree for reuse in subsequent barriers.
            // Cap the cache size to bound memory growth.
            if _subgraphCache.count < _subgraphCacheMaxEntries {
                _subgraphCache[oid] = Array(result[subtreeStart...])
            }
        }

        for output in outputs {
            visit(output)
        }

        nodes = result
    }

    /// Compute a hash for this graph (for caching)
    /// This is the legacy hash that includes constant values
    public func computeHash() -> String {
        var components: [String] = []

        // Build a map from tensor ID to relative index in topological order
        var nodeIdToIndex: [UInt64: Int] = [:]
        for (index, node) in nodes.enumerated() {
            nodeIdToIndex[node.id] = index
        }

        for node in nodes {
            if let irNode = node.irNode {
                switch irNode {
                case .constant(let values, let shape):
                    components.append("const:\(shape):\(values.prefix(4))")
                case .data:
                    components.append("data:\(node.shape):\(node.dtype)")
                #if os(macOS) && canImport(MetalHLO)
                case .metalData:
                    components.append("data:\(node.shape):\(node.dtype)")
                #endif
                case .operation(let op, let inputs, _):
                    // Use RELATIVE indices not absolute IDs for cache compatibility
                    let inputIndices = inputs.map { input -> String in
                        if let idx = nodeIdToIndex[input.id] {
                            return String(idx)
                        } else {
                            return "ext:\(input.shape)"
                        }
                    }.joined(separator: ",")
                    components.append("\(op.rawValue):[\(inputIndices)]:\(node.shape)")
                case .whileLoopTraced(let iterations, let initialValues, _, _, _):
                    let inputIndices = initialValues.map { input -> String in
                        if let idx = nodeIdToIndex[input.id] {
                            return String(idx)
                        } else {
                            return "ext:\(input.shape)"
                        }
                    }.joined(separator: ",")
                    components.append("while:\(iterations):[\(inputIndices)]:\(node.shape)")
                }
            }
        }

        // Simple hash - in production you'd use a proper hash function
        // Use magnitude to ensure non-negative, as MLIR module names can't start with '-'
        let combined = components.joined(separator: "|")
        return String(combined.hashValue.magnitude)
    }

    /// Analyze the graph for constant promotion and compute structural hash
    ///
    /// This enables cache reuse across graphs that differ only in small constant values.
    /// Inspired by TensorFlow Swift's LazyTensorTraceCache constant promotion.
    ///
    /// - Returns: Structural hash and list of promoted constants
    public func analyzeForConstantPromotion() -> ConstantPromotionResult {
        var structuralComponents: [String] = []
        var promotedConstants: [PromotedConstant] = []
        var promotedInputIndex = 0

        // IMPORTANT: Only analyze nodes that are reachable from outputs.
        // After constant folding, some nodes become dead (e.g., original constant inputs
        // when their operations are folded). We must not promote dead nodes.
        let reachableNodes = computeReachableNodes()

        // First, count existing data inputs to offset promoted constant indices
        let dataInputCount = reachableNodes.filter {
            if case .data = $0.irNode { return true }
            #if os(macOS) && canImport(MetalHLO)
            if case .metalData = $0.irNode { return true }
            #endif
            return false
        }.count

        promotedInputIndex = dataInputCount

        // Build a map from tensor ID to relative index in topological order
        // This is crucial for cache hits - we need structural equivalence,
        // not identity equivalence. Two graphs with the same structure but
        // different tensor IDs should hash the same.
        var nodeIdToIndex: [UInt64: Int] = [:]
        for (index, node) in reachableNodes.enumerated() {
            nodeIdToIndex[node.id] = index
        }

        for node in reachableNodes {
            guard let irNode = node.irNode else { continue }

            switch irNode {
            case .constant(let values, let shape):
                let elementCount = shape.isEmpty ? 1 : shape.reduce(1, *)

                // Promote small constants (scalars and small vectors)
                if elementCount <= CompilationCache.promotionThreshold {
                    // Use structural description (shape + dtype) instead of values
                    structuralComponents.append("promoted_const:\(shape):\(node.dtype)")

                    promotedConstants.append(PromotedConstant(
                        originalNodeId: node.id,
                        shape: shape,
                        dtype: node.dtype,
                        values: values,
                        inputIndex: promotedInputIndex
                    ))
                    promotedInputIndex += 1
                } else {
                    // Large constants: include values in hash (not promoted)
                    structuralComponents.append("const:\(shape):\(values.prefix(4))")
                }

            case .data:
                structuralComponents.append("data:\(node.shape):\(node.dtype)")

            #if os(macOS) && canImport(MetalHLO)
            case .metalData:
                structuralComponents.append("data:\(node.shape):\(node.dtype)")
            #endif

            case .operation(let op, let inputs, _):
                // Use RELATIVE indices (position in topological order) not absolute IDs
                // This ensures structurally equivalent graphs hash the same
                let inputIndices = inputs.map { input -> String in
                    if let idx = nodeIdToIndex[input.id] {
                        return String(idx)
                    } else {
                        // Input not in this graph (external dependency) - use shape as identifier
                        return "ext:\(input.shape)"
                    }
                }.joined(separator: ",")
                structuralComponents.append("\(op.rawValue):[\(inputIndices)]:\(node.shape)")

            case .whileLoopTraced(let iterations, let initialValues, _, _, _):
                let inputIndices = initialValues.map { input -> String in
                    if let idx = nodeIdToIndex[input.id] {
                        return String(idx)
                    } else {
                        return "ext:\(input.shape)"
                    }
                }.joined(separator: ",")
                structuralComponents.append("while:\(iterations):[\(inputIndices)]:\(node.shape)")
            }
        }

        let combined = structuralComponents.joined(separator: "|")
        let structuralHash = String(combined.hashValue.magnitude)

        return ConstantPromotionResult(
            structuralHash: structuralHash,
            promotedConstants: promotedConstants
        )
    }

    /// Compute the set of nodes reachable from outputs by walking backwards through dependencies.
    /// This excludes dead nodes that may remain in the graph after optimization.
    private func computeReachableNodes() -> [LazyTensorHandle] {
        var result: [LazyTensorHandle] = []
        var visited = Set<UInt64>()

        func visit(_ handle: LazyTensorHandle) {
            if visited.contains(handle.id) {
                return
            }
            visited.insert(handle.id)

            // Visit dependencies first
            if let node = handle.irNode {
                switch node {
                case .operation(_, let inputs, _):
                    for input in inputs {
                        visit(input)
                    }
                case .whileLoopTraced(_, let initialValues, _, _, _):
                    for input in initialValues {
                        visit(input)
                    }
                case .constant, .data:
                    break
                #if os(macOS) && canImport(MetalHLO)
                case .metalData:
                    break
                #endif
                }
            }

            result.append(handle)
        }

        // Start from outputs
        for output in outputs {
            visit(output)
        }

        return result
    }

    // MARK: - Graph Validation

    /// Validation error types
    public enum ValidationError: Error, CustomStringConvertible {
        case shapeMismatch(operation: OpKind, expected: String, got: String, nodeId: UInt64)
        case typeMismatch(operation: OpKind, types: [DType], nodeId: UInt64)
        case missingInput(operation: OpKind, inputIndex: Int, nodeId: UInt64)
        case invalidAttribute(operation: OpKind, attribute: String, nodeId: UInt64)
        case cycleDetected(nodeIds: [UInt64])

        public var description: String {
            switch self {
            case .shapeMismatch(let op, let expected, let got, let nodeId):
                return "Shape mismatch in \(op) (node \(nodeId)): expected \(expected), got \(got)"
            case .typeMismatch(let op, let types, let nodeId):
                return "Type mismatch in \(op) (node \(nodeId)): incompatible types \(types)"
            case .missingInput(let op, let idx, let nodeId):
                return "Missing input \(idx) for \(op) (node \(nodeId))"
            case .invalidAttribute(let op, let attr, let nodeId):
                return "Invalid attribute '\(attr)' for \(op) (node \(nodeId))"
            case .cycleDetected(let nodeIds):
                return "Cycle detected in graph involving nodes: \(nodeIds)"
            }
        }
    }

    /// Validate the computation graph before MLIR emission
    ///
    /// Catches errors early with descriptive messages, inspired by TensorFlow Swift's
    /// shape inference validation. This runs before compilation to provide better
    /// error messages than MLIR/XLA compiler errors.
    ///
    /// - Throws: ValidationError if the graph is invalid
    public func validate() throws {
        // Check for cycles during topological sort
        var visited = Set<UInt64>()
        var visiting = Set<UInt64>()
        var cyclePath: [UInt64] = []

        func checkCycle(_ handle: LazyTensorHandle) throws {
            if visited.contains(handle.id) { return }

            if visiting.contains(handle.id) {
                cyclePath.append(handle.id)
                throw ValidationError.cycleDetected(nodeIds: cyclePath)
            }

            visiting.insert(handle.id)
            cyclePath.append(handle.id)

            if let node = handle.irNode {
                if case .operation(_, let inputs, _) = node {
                    for input in inputs {
                        try checkCycle(input)
                    }
                }
            }

            cyclePath.removeLast()
            visiting.remove(handle.id)
            visited.insert(handle.id)
        }

        for output in outputs {
            try checkCycle(output)
        }

        // Validate each node
        for node in nodes {
            try validateNode(node)
        }
    }

    /// Validate a single node's shape and type constraints
    private func validateNode(_ handle: LazyTensorHandle) throws {
        guard let irNode = handle.irNode else { return }

        switch irNode {
        case .constant(let values, let shape):
            // Validate constant shape matches value count
            let expectedCount = shape.isEmpty ? 1 : shape.reduce(1, *)
            if values.count != expectedCount {
                throw ValidationError.shapeMismatch(
                    operation: .add, // Using add as placeholder for constant
                    expected: "values count \(expectedCount) for shape \(shape)",
                    got: "values count \(values.count)",
                    nodeId: handle.id
                )
            }

        case .data:
            // Data nodes are always valid
            break

        #if os(macOS) && canImport(MetalHLO)
        case .metalData:
            // Metal data nodes are always valid
            break
        #endif

        case .operation(let op, let inputs, let attributes):
            try validateOperation(op: op, inputs: inputs, output: handle, attributes: attributes)

        case .whileLoopTraced:
            // While loop validation could be more thorough, but for now just accept it
            break
        }
    }

    /// Validate operation-specific constraints
    private func validateOperation(
        op: OpKind,
        inputs: [LazyTensorHandle],
        output: LazyTensorHandle,
        attributes: [String: Any]
    ) throws {
        switch op {
        // Binary operations: shapes must be broadcastable
        case .add, .subtract, .multiply, .divide, .maximum, .minimum, .power:
            guard inputs.count == 2 else {
                throw ValidationError.missingInput(operation: op, inputIndex: inputs.count, nodeId: output.id)
            }
            // Could add broadcasting validation here

        // Matrix multiplication
        case .matmul:
            guard inputs.count == 2 else {
                throw ValidationError.missingInput(operation: op, inputIndex: inputs.count, nodeId: output.id)
            }
            let lhs = inputs[0]
            let rhs = inputs[1]
            if lhs.shape.count == 2 && rhs.shape.count == 2 {
                if lhs.shape[1] != rhs.shape[0] {
                    throw ValidationError.shapeMismatch(
                        operation: op,
                        expected: "inner dimensions to match: [\(lhs.shape[0]), K] @ [K, \(rhs.shape[1])]",
                        got: "[\(lhs.shape[0]), \(lhs.shape[1])] @ [\(rhs.shape[0]), \(rhs.shape[1])]",
                        nodeId: output.id
                    )
                }
            }

        // Batched matmul
        case .batchedMatmul:
            guard inputs.count == 2 else {
                throw ValidationError.missingInput(operation: op, inputIndex: inputs.count, nodeId: output.id)
            }
            let lhs = inputs[0]
            let rhs = inputs[1]
            if lhs.shape.count >= 2 && rhs.shape.count >= 2 {
                let k1 = lhs.shape[lhs.shape.count - 1]
                let k2 = rhs.shape[rhs.shape.count - 2]
                if k1 != k2 {
                    throw ValidationError.shapeMismatch(
                        operation: op,
                        expected: "inner dimensions to match",
                        got: "k=\(k1) vs k=\(k2)",
                        nodeId: output.id
                    )
                }
            }

        // Reshape: element count must match
        case .reshape:
            guard inputs.count == 1 else {
                throw ValidationError.missingInput(operation: op, inputIndex: 0, nodeId: output.id)
            }
            let inputCount = inputs[0].shape.isEmpty ? 1 : inputs[0].shape.reduce(1, *)
            let outputCount = output.shape.isEmpty ? 1 : output.shape.reduce(1, *)
            if inputCount != outputCount {
                throw ValidationError.shapeMismatch(
                    operation: op,
                    expected: "\(inputCount) elements",
                    got: "\(outputCount) elements",
                    nodeId: output.id
                )
            }

        // Unary operations
        case .negate, .abs, .exponential, .log, .sqrt, .rsqrt, .sine, .cosine, .tanh, .floor, .ceil,
             .relu, .sigmoid, .gelu, .silu, .elu, .leakyRelu:
            guard inputs.count >= 1 else {
                throw ValidationError.missingInput(operation: op, inputIndex: 0, nodeId: output.id)
            }

        // Reduction operations
        case .reduceSum, .reduceMax, .reduceMin, .reduceMean:
            guard inputs.count >= 1 else {
                throw ValidationError.missingInput(operation: op, inputIndex: 0, nodeId: output.id)
            }
            if let axes = attributes["axes"] as? [Int] {
                for axis in axes {
                    let normalizedAxis = axis < 0 ? inputs[0].shape.count + axis : axis
                    if normalizedAxis < 0 || normalizedAxis >= inputs[0].shape.count {
                        throw ValidationError.invalidAttribute(
                            operation: op,
                            attribute: "axis \(axis) out of bounds for rank \(inputs[0].shape.count)",
                            nodeId: output.id
                        )
                    }
                }
            }

        // Comparison operations need 2 inputs
        case .equal, .less, .greater:
            guard inputs.count == 2 else {
                throw ValidationError.missingInput(operation: op, inputIndex: inputs.count, nodeId: output.id)
            }

        // Select needs 3 inputs
        case .select:
            guard inputs.count == 3 else {
                throw ValidationError.missingInput(operation: op, inputIndex: inputs.count, nodeId: output.id)
            }

        // Default: just check we have at least one input for most ops
        default:
            break
        }
    }
}
