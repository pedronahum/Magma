// Magma - StableHLO
// Pure Swift MLIR/StableHLO builder

/// Builds StableHLO MLIR modules as text
///
/// This is a pure Swift implementation with no C dependencies.
/// It generates valid MLIR text that can be compiled by XLA.
///
/// Example:
/// ```swift
/// let builder = MLIRBuilder()
/// let x = builder.argument(TensorType(shape: [32, 784], dtype: .float32))
/// let w = builder.argument(TensorType(shape: [784, 256], dtype: .float32))
/// let y = builder.dot(x, w)
/// let z = builder.relu(y)
/// let mlir = builder.build(name: "matmul_relu", outputs: [z])
/// ```
public final class MLIRBuilder: @unchecked Sendable {
    
    // MARK: - State
    
    /// Function arguments
    private var arguments: [Argument] = []
    
    /// Operations (in order)
    private var operations: [String] = []
    
    /// Next value ID
    private var nextValueId: Int = 0
    
    /// Constants defined in the module
    private var constants: [(Value, String)] = []
    
    // MARK: - Initialization
    
    public init() {}
    
    // MARK: - Arguments
    
    /// Add a function argument
    @discardableResult
    public func argument(_ type: TensorType) -> Value {
        let arg = Argument(index: arguments.count, type: type)
        arguments.append(arg)
        return arg.asValue
    }
    
    // MARK: - Value Creation

    private func nextValue(type: TensorType) -> Value {
        let value = Value(id: nextValueId, type: type)
        nextValueId += 1
        return value
    }

    /// Allocate multiple result values without emitting any operations
    /// Used for creating results of control flow operations
    public func allocateValues(types: [TensorType]) -> [Value] {
        return types.map { nextValue(type: $0) }
    }

    /// Create block arguments for a region (used in control flow)
    /// These have different naming (%arg0, %arg1, etc.) and are marked as block args
    /// The prefix is used to create unique names (e.g., "cond" -> %cond_0, %cond_1)
    public func createBlockArgs(types: [TensorType], prefix: String = "arg") -> [Value] {
        var args: [Value] = []
        for (i, type) in types.enumerated() {
            let arg = Value(id: i, type: type, isBlockArg: true, blockArgPrefix: prefix)
            args.append(arg)
        }
        return args
    }

    /// Get the operations list for inspection (used in control flow)
    public func getOperations() -> [String] {
        return operations
    }

    /// Add a raw operation string (used for complex control flow)
    public func addRawOperation(_ op: String) {
        operations.append(op)
    }

    /// Set the next value ID (used to avoid conflicts with block args)
    public func setNextValueId(_ id: Int) {
        nextValueId = id
    }
    
    // MARK: - Constants
    
    /// Create a constant tensor filled with a single value
    public func constant(_ value: Double, type: TensorType) -> Value {
        let result = nextValue(type: type)
        let valueStr = type.dtype.isFloatingPoint ? "\(value)" : "\(Int(value))"
        let op = "    \(result.name) = stablehlo.constant dense<\(valueStr)> : \(type.mlirType)"
        operations.append(op)
        return result
    }
    
    /// Create a constant tensor from an array of values
    public func constant(_ values: [Double], shape: [Int], dtype: DType) -> Value {
        let type = TensorType(shape: shape, dtype: dtype)
        let result = nextValue(type: type)

        let valuesStr: String
        if values.count == 1 {
            // Single value - splat constant
            valuesStr = dtype.isFloatingPoint ? "\(values[0])" : "\(Int(values[0]))"
        } else if values.count > 1 && isSplatConstant(values) {
            // All values are the same - use splat representation
            valuesStr = dtype.isFloatingPoint ? "\(values[0])" : "\(Int(values[0]))"
        } else {
            // Dense constant with shaped array format
            valuesStr = formatDenseArray(values: values, shape: shape, dtype: dtype)
        }

        let op = "    \(result.name) = stablehlo.constant dense<\(valuesStr)> : \(type.mlirType)"
        operations.append(op)
        return result
    }

    /// Check if all values in the array are identical (for splat constants)
    private func isSplatConstant(_ values: [Double]) -> Bool {
        guard let first = values.first else { return true }
        return values.allSatisfy { $0 == first }
    }

    /// Format values into a properly nested dense array string for MLIR
    /// e.g., shape [2,3] with values [1,2,3,4,5,6] becomes "[[1,2,3],[4,5,6]]"
    private func formatDenseArray(values: [Double], shape: [Int], dtype: DType) -> String {
        if shape.isEmpty {
            // Scalar
            return dtype.isFloatingPoint ? "\(values[0])" : "\(Int(values[0]))"
        }

        if shape.count == 1 {
            // 1D array
            let formatted = values.map { dtype.isFloatingPoint ? "\($0)" : "\(Int($0))" }
            return "[\(formatted.joined(separator: ", "))]"
        }

        // Multi-dimensional: recursively format
        func formatRecursive(values: ArraySlice<Double>, remainingShape: [Int]) -> String {
            if remainingShape.count == 1 {
                // Base case: format 1D slice
                let formatted = values.map { dtype.isFloatingPoint ? "\($0)" : "\(Int($0))" }
                return "[\(formatted.joined(separator: ", "))]"
            }

            // Recursive case: divide into sub-arrays
            let firstDim = remainingShape[0]
            let restShape = Array(remainingShape.dropFirst())
            let subArraySize = restShape.reduce(1, *)

            var parts: [String] = []
            for i in 0..<firstDim {
                let startIdx = values.startIndex + i * subArraySize
                let endIdx = startIdx + subArraySize
                let subSlice = values[startIdx..<endIdx]
                parts.append(formatRecursive(values: subSlice, remainingShape: restShape))
            }
            return "[\(parts.joined(separator: ", "))]"
        }

        return formatRecursive(values: values[...], remainingShape: shape)
    }
    
    // MARK: - Element-wise Binary Operations
    
    /// Element-wise addition
    public func add(_ lhs: Value, _ rhs: Value) -> Value {
        binaryOp("add", lhs, rhs)
    }
    
    /// Element-wise subtraction
    public func subtract(_ lhs: Value, _ rhs: Value) -> Value {
        binaryOp("subtract", lhs, rhs)
    }
    
    /// Element-wise multiplication
    public func multiply(_ lhs: Value, _ rhs: Value) -> Value {
        binaryOp("multiply", lhs, rhs)
    }
    
    /// Element-wise division
    public func divide(_ lhs: Value, _ rhs: Value) -> Value {
        binaryOp("divide", lhs, rhs)
    }
    
    /// Element-wise maximum
    public func maximum(_ lhs: Value, _ rhs: Value) -> Value {
        binaryOp("maximum", lhs, rhs)
    }
    
    /// Element-wise minimum
    public func minimum(_ lhs: Value, _ rhs: Value) -> Value {
        binaryOp("minimum", lhs, rhs)
    }
    
    /// Element-wise power
    public func power(_ lhs: Value, _ rhs: Value) -> Value {
        binaryOp("power", lhs, rhs)
    }
    
    private func binaryOp(_ name: String, _ lhs: Value, _ rhs: Value) -> Value {
        // Handle broadcasting: broadcast smaller tensor to match larger
        var lhsVal = lhs
        var rhsVal = rhs

        // Handle type conversion for mixed boolean/float operations
        // Convert boolean to the other operand's type if they differ
        if lhsVal.type.dtype == .bool && rhsVal.type.dtype != .bool {
            lhsVal = convert(lhsVal, to: rhsVal.type.dtype)
        } else if rhsVal.type.dtype == .bool && lhsVal.type.dtype != .bool {
            rhsVal = convert(rhsVal, to: lhsVal.type.dtype)
        }

        if lhsVal.type != rhsVal.type {
            // Need to broadcast one or both operands
            let targetShape = computeBroadcastShape(lhsVal.type.shape, rhsVal.type.shape)

            if lhsVal.type.shape != targetShape {
                lhsVal = broadcast(lhsVal, to: targetShape)
            }
            if rhsVal.type.shape != targetShape {
                rhsVal = broadcast(rhsVal, to: targetShape)
            }
        }

        let result = nextValue(type: lhsVal.type)
        let op = "    \(result.name) = stablehlo.\(name) \(lhsVal.displayName), \(rhsVal.displayName) : \(lhsVal.type.mlirType)"
        operations.append(op)
        return result
    }

    /// Compute the broadcast shape for two shapes following NumPy-style broadcasting
    private func computeBroadcastShape(_ lhs: [Int], _ rhs: [Int]) -> [Int] {
        let maxRank = max(lhs.count, rhs.count)

        // Pad shapes to same rank
        let lhsPadded = Array(repeating: 1, count: maxRank - lhs.count) + lhs
        let rhsPadded = Array(repeating: 1, count: maxRank - rhs.count) + rhs

        // Compute result shape
        var resultShape: [Int] = []
        for i in 0..<maxRank {
            if lhsPadded[i] == rhsPadded[i] {
                resultShape.append(lhsPadded[i])
            } else if lhsPadded[i] == 1 {
                resultShape.append(rhsPadded[i])
            } else if rhsPadded[i] == 1 {
                resultShape.append(lhsPadded[i])
            } else {
                fatalError("Shapes \(lhs) and \(rhs) cannot be broadcast together")
            }
        }
        return resultShape
    }
    
    // MARK: - Element-wise Unary Operations
    
    /// Element-wise negation
    public func negate(_ input: Value) -> Value {
        unaryOp("negate", input)
    }
    
    /// Element-wise absolute value
    public func abs(_ input: Value) -> Value {
        unaryOp("abs", input)
    }
    
    /// Element-wise exponential
    public func exponential(_ input: Value) -> Value {
        unaryOp("exponential", input)
    }
    
    /// Element-wise natural logarithm
    public func log(_ input: Value) -> Value {
        unaryOp("log", input)
    }
    
    /// Element-wise square root
    public func sqrt(_ input: Value) -> Value {
        unaryOp("sqrt", input)
    }
    
    /// Element-wise reciprocal square root
    public func rsqrt(_ input: Value) -> Value {
        unaryOp("rsqrt", input)
    }
    
    /// Element-wise sine
    public func sine(_ input: Value) -> Value {
        unaryOp("sine", input)
    }
    
    /// Element-wise cosine
    public func cosine(_ input: Value) -> Value {
        unaryOp("cosine", input)
    }
    
    /// Element-wise tanh
    public func tanh(_ input: Value) -> Value {
        unaryOp("tanh", input)
    }
    
    /// Element-wise floor
    public func floor(_ input: Value) -> Value {
        unaryOp("floor", input)
    }
    
    /// Element-wise ceiling
    public func ceil(_ input: Value) -> Value {
        unaryOp("ceil", input)
    }

    /// Convert tensor to a different dtype
    public func convert(_ input: Value, to dtype: DType) -> Value {
        let resultType = TensorType(shape: input.type.shape, dtype: dtype)
        let result = nextValue(type: resultType)
        let op = "    \(result.name) = stablehlo.convert \(input.displayName) : (\(input.type.mlirType)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }

    private func unaryOp(_ name: String, _ input: Value) -> Value {
        let result = nextValue(type: input.type)
        let op = "    \(result.name) = stablehlo.\(name) \(input.displayName) : \(input.type.mlirType)"
        operations.append(op)
        return result
    }
    
    // MARK: - Matrix Operations
    
    /// Matrix multiplication (dot product)
    public func dot(_ lhs: Value, _ rhs: Value) -> Value {
        let resultType = lhs.type.matmulResult(with: rhs.type)
        let result = nextValue(type: resultType)
        let op = "    \(result.name) = stablehlo.dot \(lhs.displayName), \(rhs.displayName) : (\(lhs.type.mlirType), \(rhs.type.mlirType)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }

    /// Batched matrix multiplication using dot_general.
    ///
    /// For tensors of shape [batch..., m, k] @ [batch..., k, n] -> [batch..., m, n]
    /// Uses stablehlo.dot_general with batch dimensions.
    public func batchedDot(_ lhs: Value, _ rhs: Value) -> Value {
        // Compute result type: batch dims + [m, n]
        let lhsShape = lhs.type.shape
        let rhsShape = rhs.type.shape
        let rank = lhsShape.count

        // Batch dimensions are all but last 2
        let batchDims = Array(0..<(rank - 2))
        let m = lhsShape[rank - 2]
        let n = rhsShape[rank - 1]

        let resultShape = Array(lhsShape.dropLast(2)) + [m, n]
        let resultType = TensorType(shape: resultShape, dtype: lhs.type.dtype)
        let result = nextValue(type: resultType)

        // Build dot_dimension_numbers
        // lhs_batching_dimensions = [0, 1, ...] (all batch dims)
        // rhs_batching_dimensions = [0, 1, ...] (all batch dims)
        // lhs_contracting_dimensions = [last dim] (k)
        // rhs_contracting_dimensions = [second to last dim] (k)
        let batchDimsStr = batchDims.map(String.init).joined(separator: ", ")
        let lhsContractDim = rank - 1  // k dimension in lhs
        let rhsContractDim = rank - 2  // k dimension in rhs

        // stablehlo.dot_general format:
        // dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [...], rhs_batching_dimensions = [...],
        //                                        lhs_contracting_dimensions = [...], rhs_contracting_dimensions = [...]>
        let dotNumbers = "#stablehlo.dot<lhs_batching_dimensions = [\(batchDimsStr)], rhs_batching_dimensions = [\(batchDimsStr)], lhs_contracting_dimensions = [\(lhsContractDim)], rhs_contracting_dimensions = [\(rhsContractDim)]>"

        let op = "    \(result.name) = stablehlo.dot_general \(lhs.displayName), \(rhs.displayName), \(dotNumbers) : (\(lhs.type.mlirType), \(rhs.type.mlirType)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }

    /// Transpose
    public func transpose(_ input: Value, permutation: [Int]? = nil) -> Value {
        let perm = permutation ?? Array((0..<input.type.rank).reversed())
        let resultType = input.type.permuted(perm)
        let result = nextValue(type: resultType)
        let permStr = perm.map(String.init).joined(separator: ", ")
        let op = "    \(result.name) = stablehlo.transpose \(input.displayName), dims = [\(permStr)] : (\(input.type.mlirType)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }
    
    /// Reshape
    public func reshape(_ input: Value, to newShape: [Int]) -> Value {
        let resultType = input.type.reshaped(to: newShape)
        let result = nextValue(type: resultType)
        let op = "    \(result.name) = stablehlo.reshape \(input.displayName) : (\(input.type.mlirType)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }
    
    /// Broadcast to target shape
    public func broadcast(_ input: Value, to shape: [Int]) -> Value {
        let resultType = TensorType(shape: shape, dtype: input.type.dtype)
        let result = nextValue(type: resultType)
        
        // Compute broadcast dimensions
        let inputRank = input.type.rank
        let outputRank = shape.count
        let broadcastDims = Array((outputRank - inputRank)..<outputRank)
        let dimsStr = broadcastDims.map(String.init).joined(separator: ", ")
        
        let op = "    \(result.name) = stablehlo.broadcast_in_dim \(input.displayName), dims = [\(dimsStr)] : (\(input.type.mlirType)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }
    
    // MARK: - Reductions
    
    /// Sum reduction along axes
    public func reduceSum(_ input: Value, axes: [Int], keepDims: Bool = false) -> Value {
        reduce("add", input, axes: axes, keepDims: keepDims, initValue: 0.0)
    }
    
    /// Max reduction along axes
    public func reduceMax(_ input: Value, axes: [Int], keepDims: Bool = false) -> Value {
        reduce("maximum", input, axes: axes, keepDims: keepDims, initValue: -.infinity)
    }
    
    /// Min reduction along axes
    public func reduceMin(_ input: Value, axes: [Int], keepDims: Bool = false) -> Value {
        reduce("minimum", input, axes: axes, keepDims: keepDims, initValue: .infinity)
    }
    
    private func reduce(_ op: String, _ input: Value, axes: [Int], keepDims: Bool, initValue: Double) -> Value {
        // StableHLO reduce always removes dimensions - compute the reduced shape
        var reducedShape = input.type.shape
        for axis in axes.sorted().reversed() {
            reducedShape.remove(at: axis)
        }

        let reducedType = TensorType(shape: reducedShape, dtype: input.type.dtype)
        let reducedResult = nextValue(type: reducedType)

        // Create init value
        let scalarType = TensorType(shape: [], dtype: input.type.dtype)
        let initValueStr: String
        if input.type.dtype.isFloatingPoint {
            if initValue.isInfinite {
                // Use hex representation for infinity values
                // float32: -inf = 0xFF800000, +inf = 0x7F800000
                if initValue < 0 {
                    initValueStr = "0xFF800000"
                } else {
                    initValueStr = "0x7F800000"
                }
            } else {
                initValueStr = "\(initValue)"
            }
        } else {
            initValueStr = "\(Int(initValue))"
        }
        let initVal = nextValue(type: scalarType)
        operations.append("    \(initVal.name) = stablehlo.constant dense<\(initValueStr)> : \(scalarType.mlirType)")

        let axesStr = axes.map(String.init).joined(separator: ", ")

        let reduceOp = """
            \(reducedResult.name) = stablehlo.reduce(\(input.displayName) init: \(initVal.name)) across dimensions = [\(axesStr)] : (\(input.type.mlirType), \(scalarType.mlirType)) -> \(reducedType.mlirType)
             reducer(%arg_a: \(scalarType.mlirType), %arg_b: \(scalarType.mlirType)) {
              %r = stablehlo.\(op) %arg_a, %arg_b : \(scalarType.mlirType)
              stablehlo.return %r : \(scalarType.mlirType)
            }
        """
        operations.append(reduceOp)

        // If keepDims is true, reshape to add back the removed dimensions as size 1
        if keepDims {
            var finalShape = input.type.shape
            for axis in axes.sorted() {
                finalShape[axis] = 1
            }
            let finalType = TensorType(shape: finalShape, dtype: input.type.dtype)
            let finalResult = nextValue(type: finalType)
            let reshapeOp = "    \(finalResult.name) = stablehlo.reshape \(reducedResult.displayName) : (\(reducedType.mlirType)) -> \(finalType.mlirType)"
            operations.append(reshapeOp)
            return finalResult
        }

        return reducedResult
    }
    
    // MARK: - Activations (Composite)
    
    /// ReLU activation: max(0, x)
    public func relu(_ input: Value) -> Value {
        let zero = constant(0.0, type: input.type)
        return maximum(input, zero)
    }
    
    /// Sigmoid activation: 1 / (1 + exp(-x))
    public func sigmoid(_ input: Value) -> Value {
        let negX = negate(input)
        let expNegX = exponential(negX)
        let one = constant(1.0, type: input.type)
        let onePlusExp = add(one, expNegX)
        return divide(one, onePlusExp)
    }
    
    // MARK: - Comparison Operations
    
    /// Element-wise equality comparison
    public func equal(_ lhs: Value, _ rhs: Value) -> Value {
        compareOp("EQ", lhs, rhs)
    }
    
    /// Element-wise less-than comparison
    public func less(_ lhs: Value, _ rhs: Value) -> Value {
        compareOp("LT", lhs, rhs)
    }
    
    /// Element-wise greater-than comparison
    public func greater(_ lhs: Value, _ rhs: Value) -> Value {
        compareOp("GT", lhs, rhs)
    }
    
    private func compareOp(_ comparison: String, _ lhs: Value, _ rhs: Value) -> Value {
        let resultType = TensorType(shape: lhs.type.shape, dtype: .bool)
        let result = nextValue(type: resultType)
        let compareType = lhs.type.dtype.isFloatingPoint ? "FLOAT" : "SIGNED"
        let op = "    \(result.name) = stablehlo.compare \(comparison), \(lhs.displayName), \(rhs.displayName), \(compareType) : (\(lhs.type.mlirType), \(rhs.type.mlirType)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }
    
    /// Select between two values based on condition
    public func select(_ condition: Value, _ onTrue: Value, _ onFalse: Value) -> Value {
        let result = nextValue(type: onTrue.type)
        let op = "    \(result.name) = stablehlo.select \(condition.displayName), \(onTrue.displayName), \(onFalse.displayName) : (\(condition.type.mlirType), \(onTrue.type.mlirType), \(onFalse.type.mlirType)) -> \(onTrue.type.mlirType)"
        operations.append(op)
        return result
    }

    // MARK: - Slice and Pad Operations

    /// Slice a tensor along specified dimensions
    public func slice(_ input: Value, starts: [Int], limits: [Int], strides: [Int]) -> Value {
        var newShape = input.type.shape
        for i in 0..<newShape.count {
            newShape[i] = (limits[i] - starts[i] + strides[i] - 1) / strides[i]
        }
        let resultType = TensorType(shape: newShape, dtype: input.type.dtype)
        let result = nextValue(type: resultType)

        // StableHLO slice format: [start0:limit0:stride0, start1:limit1:stride1, ...]
        var sliceParts: [String] = []
        for i in 0..<starts.count {
            sliceParts.append("\(starts[i]):\(limits[i]):\(strides[i])")
        }
        let sliceStr = sliceParts.joined(separator: ", ")

        let op = "    \(result.name) = stablehlo.slice \(input.displayName) [\(sliceStr)] : (\(input.type.mlirType)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }

    /// Pad a tensor with zeros
    public func pad(_ input: Value, low: [Int], high: [Int]) -> Value {
        var newShape = input.type.shape
        for i in 0..<newShape.count {
            newShape[i] = newShape[i] + low[i] + high[i]
        }
        let resultType = TensorType(shape: newShape, dtype: input.type.dtype)
        let result = nextValue(type: resultType)

        // Create zero constant for padding value
        let zeroName = "%pad_zero_\(nextValueId)"
        nextValueId += 1
        let zeroOp = "    \(zeroName) = stablehlo.constant dense<0.0> : tensor<\(input.type.dtype.mlirName)>"
        operations.append(zeroOp)

        let lowStr = low.map(String.init).joined(separator: ", ")
        let highStr = high.map(String.init).joined(separator: ", ")
        let interiorStr = Array(repeating: "0", count: low.count).joined(separator: ", ")

        let op = "    \(result.name) = stablehlo.pad \(input.displayName), \(zeroName), low = [\(lowStr)], high = [\(highStr)], interior = [\(interiorStr)] : (\(input.type.mlirType), tensor<\(input.type.dtype.mlirName)>) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }

    /// Concatenate tensors along a dimension
    public func concatenate(_ inputs: [Value], dimension: Int) -> Value {
        guard !inputs.isEmpty else {
            fatalError("Cannot concatenate empty array")
        }

        var newShape = inputs[0].type.shape
        var totalSize = 0
        for input in inputs {
            totalSize += input.type.shape[dimension]
        }
        newShape[dimension] = totalSize

        let resultType = TensorType(shape: newShape, dtype: inputs[0].type.dtype)
        let result = nextValue(type: resultType)

        let inputsStr = inputs.map { $0.displayName }.joined(separator: ", ")
        let inputTypesStr = inputs.map { $0.type.mlirType }.joined(separator: ", ")

        let op = "    \(result.name) = stablehlo.concatenate \(inputsStr), dim = \(dimension) : (\(inputTypesStr)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }

    // MARK: - Gather Operation

    /// Gather slices from input using indices
    ///
    /// This implements a simplified gather that extracts slices along a single dimension.
    /// - Parameters:
    ///   - input: Source tensor to gather from
    ///   - indices: Index tensor with gather indices
    ///   - axis: The axis along which to gather
    /// - Returns: Gathered tensor
    public func gather(_ input: Value, indices: Value, axis: Int) -> Value {
        // StableHLO gather requires integer indices - convert if needed
        let intIndices: Value
        if indices.type.dtype == .float32 || indices.type.dtype == .float16 || indices.type.dtype == .bfloat16 {
            // Convert float indices to int64
            let intType = TensorType(shape: indices.type.shape, dtype: .int64)
            intIndices = nextValue(type: intType)
            let convertOp = "    \(intIndices.name) = stablehlo.convert \(indices.displayName) : (\(indices.type.mlirType)) -> \(intType.mlirType)"
            operations.append(convertOp)
        } else {
            intIndices = indices
        }

        // Compute result shape: replace the gather axis with indices shape
        var resultShape = input.type.shape

        // For 1D indices gathering along axis: result replaces axis dim with indices shape
        if intIndices.type.shape.count == 1 {
            resultShape[axis] = intIndices.type.shape[0]
        } else {
            // Multi-dimensional indices
            resultShape.remove(at: axis)
            resultShape.insert(contentsOf: intIndices.type.shape, at: axis)
        }

        let resultType = TensorType(shape: resultShape, dtype: input.type.dtype)
        let result = nextValue(type: resultType)

        // Build dimension numbers for stablehlo.gather
        // offset_dims: dimensions in result that correspond to slices from input
        // collapsed_slice_dims: dimensions that are collapsed (size 1 slices)
        // start_index_map: which dimension of input each index in indices refers to
        // index_vector_dim: which dimension of indices contains the index vectors

        let rank = input.type.shape.count
        let offsetDims = (0..<rank).filter { $0 != axis }.map { String($0) }.joined(separator: ", ")
        let collapsedSliceDims = "\(axis)"
        let startIndexMap = "\(axis)"

        // slice_sizes: size 1 along gather axis, full size elsewhere
        var sliceSizes = input.type.shape
        sliceSizes[axis] = 1
        let sliceSizesStr = sliceSizes.map { String($0) }.joined(separator: ", ")

        let indexVectorDim = intIndices.type.shape.count  // indices treated as scalar indices

        let op = """
            \(result.name) = "stablehlo.gather"(\(input.displayName), \(intIndices.displayName)) {
              dimension_numbers = #stablehlo.gather<
                offset_dims = [\(offsetDims)],
                collapsed_slice_dims = [\(collapsedSliceDims)],
                start_index_map = [\(startIndexMap)],
                index_vector_dim = \(indexVectorDim)
              >,
              slice_sizes = array<i64: \(sliceSizesStr)>,
              indices_are_sorted = false
            } : (\(input.type.mlirType), \(intIndices.type.mlirType)) -> \(resultType.mlirType)
        """
        operations.append(op)
        return result
    }

    // MARK: - Scatter Operation

    /// Scatter updates into input at specified indices
    ///
    /// - Parameters:
    ///   - input: Base tensor to scatter into
    ///   - indices: Index tensor specifying where to scatter
    ///   - updates: Values to scatter
    ///   - axis: The axis along which to scatter
    /// - Returns: Result tensor with scattered values
    public func scatter(_ input: Value, indices: Value, updates: Value, axis: Int) -> Value {
        let resultType = input.type  // Result has same shape as input
        let result = nextValue(type: resultType)

        let rank = input.type.shape.count
        let updateWindowDims = (0..<rank).filter { $0 != axis }.map { String($0) }.joined(separator: ", ")
        let insertedWindowDims = "\(axis)"
        let scatterDimsToOperandDims = "\(axis)"
        let indexVectorDim = indices.type.shape.count

        let scalarType = TensorType(shape: [], dtype: input.type.dtype)

        let op = """
            \(result.name) = "stablehlo.scatter"(\(input.displayName), \(indices.displayName), \(updates.displayName)) ({
            ^bb0(%arg0: \(scalarType.mlirType), %arg1: \(scalarType.mlirType)):
              stablehlo.return %arg1 : \(scalarType.mlirType)
            }) {
              scatter_dimension_numbers = #stablehlo.scatter<
                update_window_dims = [\(updateWindowDims)],
                inserted_window_dims = [\(insertedWindowDims)],
                scatter_dims_to_operand_dims = [\(scatterDimsToOperandDims)],
                index_vector_dim = \(indexVectorDim)
              >,
              indices_are_sorted = false,
              unique_indices = false
            } : (\(input.type.mlirType), \(indices.type.mlirType), \(updates.type.mlirType)) -> \(resultType.mlirType)
        """
        operations.append(op)
        return result
    }

    // MARK: - Iota Operation

    /// Create a tensor with values counting along an axis
    ///
    /// - Parameters:
    ///   - shape: Output tensor shape
    ///   - axis: Axis along which to count
    ///   - dtype: Output data type
    /// - Returns: Iota tensor
    public func iota(shape: [Int], axis: Int, dtype: DType) -> Value {
        let resultType = TensorType(shape: shape, dtype: dtype)
        let result = nextValue(type: resultType)

        let op = "    \(result.name) = stablehlo.iota dim = \(axis) : \(resultType.mlirType)"
        operations.append(op)
        return result
    }

    // MARK: - Control Flow Operations

    /// While loop operation
    ///
    /// Executes a loop where the condition and body are closures that build MLIR.
    /// The loop continues while the condition returns true.
    ///
    /// - Parameters:
    ///   - initialValues: Initial values for loop-carried variables
    ///   - conditionBuilder: Closure that builds the condition, takes loop vars and returns a boolean tensor
    ///   - bodyBuilder: Closure that builds the body, takes loop vars and returns updated values
    /// - Returns: Final values of loop-carried variables
    public func whileLoop(
        initialValues: [Value],
        conditionBuilder: ([Value]) -> Value,
        bodyBuilder: ([Value]) -> [Value]
    ) -> [Value] {
        guard !initialValues.isEmpty else {
            fatalError("while_loop requires at least one loop-carried variable")
        }

        // Create result types (same as initial values)
        let resultTypes = initialValues.map { $0.type }
        let results = resultTypes.map { nextValue(type: $0) }

        // Build the while operation with condition and body regions
        let inputsStr = initialValues.map { $0.displayName }.joined(separator: ", ")
        let inputTypesStr = initialValues.map { $0.type.mlirType }.joined(separator: ", ")
        let resultTypesStr = resultTypes.map { $0.mlirType }.joined(separator: ", ")
        let resultsStr = results.map { $0.name }.joined(separator: ", ")

        // Create block arguments for condition region
        var condBlockArgs: [Value] = []
        var condArgDefs: [String] = []
        for (i, type) in resultTypes.enumerated() {
            let argValue = Value(id: nextValueId + i, type: type, isBlockArg: true)
            condBlockArgs.append(argValue)
            condArgDefs.append("\(argValue.name): \(type.mlirType)")
        }
        nextValueId += resultTypes.count

        // Build condition region using a nested builder
        let condBuilder = MLIRBuilder()
        condBuilder.nextValueId = nextValueId
        let condResult = conditionBuilder(condBlockArgs)
        nextValueId = condBuilder.nextValueId + 1  // Account for the result

        let condOps = condBuilder.operations.map { "  " + $0 }.joined(separator: "\n")

        // Create block arguments for body region
        var bodyBlockArgs: [Value] = []
        var bodyArgDefs: [String] = []
        for (i, type) in resultTypes.enumerated() {
            let argValue = Value(id: nextValueId + i, type: type, isBlockArg: true)
            bodyBlockArgs.append(argValue)
            bodyArgDefs.append("\(argValue.name): \(type.mlirType)")
        }
        nextValueId += resultTypes.count

        // Build body region using a nested builder
        let bodyBuilderInst = MLIRBuilder()
        bodyBuilderInst.nextValueId = nextValueId
        let bodyResults = bodyBuilder(bodyBlockArgs)
        nextValueId = bodyBuilderInst.nextValueId

        let bodyOps = bodyBuilderInst.operations.map { "  " + $0 }.joined(separator: "\n")
        let bodyReturnVals = bodyResults.map { $0.displayName }.joined(separator: ", ")

        let op = """
            \(resultsStr) = stablehlo.while(\(inputsStr)) : (\(inputTypesStr)) -> (\(resultTypesStr))
             cond {
            ^bb0(\(condArgDefs.joined(separator: ", "))):
        \(condOps)
              stablehlo.return \(condResult.displayName) : \(condResult.type.mlirType)
            } do {
            ^bb0(\(bodyArgDefs.joined(separator: ", "))):
        \(bodyOps)
              stablehlo.return \(bodyReturnVals) : \(resultTypesStr)
            }
        """
        operations.append(op)

        return results
    }

    /// Conditional (if-then-else) operation
    ///
    /// Executes one of two branches based on a boolean predicate.
    ///
    /// - Parameters:
    ///   - predicate: Boolean scalar tensor (0-d tensor<i1>)
    ///   - resultTypes: Types of the results
    ///   - trueBranch: Closure that builds the true branch
    ///   - falseBranch: Closure that builds the false branch
    /// - Returns: Results from the selected branch
    public func cond(
        predicate: Value,
        resultTypes: [TensorType],
        trueBranch: () -> [Value],
        falseBranch: () -> [Value]
    ) -> [Value] {
        let results = resultTypes.map { nextValue(type: $0) }
        let resultsStr = results.map { $0.name }.joined(separator: ", ")
        let resultTypesStr = resultTypes.map { $0.mlirType }.joined(separator: ", ")

        // Build true branch
        let trueBuilder = MLIRBuilder()
        trueBuilder.nextValueId = nextValueId
        let trueResults = trueBranch()
        nextValueId = trueBuilder.nextValueId

        let trueOps = trueBuilder.operations.isEmpty ? "" :
            trueBuilder.operations.map { "  " + $0 }.joined(separator: "\n") + "\n"
        let trueReturnVals = trueResults.map { $0.displayName }.joined(separator: ", ")

        // Build false branch
        let falseBuilder = MLIRBuilder()
        falseBuilder.nextValueId = nextValueId
        let falseResults = falseBranch()
        nextValueId = falseBuilder.nextValueId

        let falseOps = falseBuilder.operations.isEmpty ? "" :
            falseBuilder.operations.map { "  " + $0 }.joined(separator: "\n") + "\n"
        let falseReturnVals = falseResults.map { $0.displayName }.joined(separator: ", ")

        let op = """
            \(resultsStr) = stablehlo.if \(predicate.displayName) -> (\(resultTypesStr)) {
        \(trueOps)      stablehlo.return \(trueReturnVals) : \(resultTypesStr)
            } else {
        \(falseOps)      stablehlo.return \(falseReturnVals) : \(resultTypesStr)
            }
        """
        operations.append(op)

        return results
    }

    /// Create a boolean scalar constant
    public func boolConstant(_ value: Bool) -> Value {
        let type = TensorType(shape: [], dtype: .bool)
        let result = nextValue(type: type)
        let boolStr = value ? "true" : "false"
        let op = "    \(result.name) = stablehlo.constant dense<\(boolStr)> : tensor<i1>"
        operations.append(op)
        return result
    }

    /// Compare two values and return a boolean result
    public func compare(_ lhs: Value, _ rhs: Value, direction: CompareDirection) -> Value {
        let resultType = TensorType(shape: lhs.type.shape, dtype: .bool)
        let result = nextValue(type: resultType)
        let op = "    \(result.name) = stablehlo.compare \(direction.rawValue), \(lhs.displayName), \(rhs.displayName) : (\(lhs.type.mlirType), \(rhs.type.mlirType)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }

    // MARK: - Random Number Generation

    /// Generate random numbers with uniform distribution in [low, high)
    public func rngUniform(_ low: Value, _ high: Value, shape: [Int]) -> Value {
        let resultType = TensorType(shape: shape, dtype: low.type.dtype)
        let result = nextValue(type: resultType)
        let shapeStr = shape.map { String($0) }.joined(separator: ", ")

        let op = "    \(result.name) = stablehlo.rng \(low.displayName), \(high.displayName), shape = [\(shapeStr)], distribution = UNIFORM : (\(low.type.mlirType), \(high.type.mlirType)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }

    /// Generate random numbers with normal distribution (mean, stddev)
    public func rngNormal(_ mean: Value, _ stddev: Value, shape: [Int]) -> Value {
        let resultType = TensorType(shape: shape, dtype: mean.type.dtype)
        let result = nextValue(type: resultType)
        let shapeStr = shape.map { String($0) }.joined(separator: ", ")

        let op = "    \(result.name) = stablehlo.rng \(mean.displayName), \(stddev.displayName), shape = [\(shapeStr)], distribution = NORMAL : (\(mean.type.mlirType), \(stddev.type.mlirType)) -> \(resultType.mlirType)"
        operations.append(op)
        return result
    }

    // MARK: - Build Module

    /// Build the final MLIR module
    public func build(name: String, outputs: [Value]) -> String {
        let argDefs = arguments.map { "\($0.name): \($0.type.mlirType)" }.joined(separator: ", ")
        let outputTypes = outputs.map { $0.type.mlirType }.joined(separator: ", ")
        let returnValues = outputs.map { $0.displayName }.joined(separator: ", ")
        
        let body = operations.joined(separator: "\n")
        
        return """
        module @\(name) {
          func.func @main(\(argDefs)) -> (\(outputTypes)) {
        \(body)
            return \(returnValues) : \(outputTypes)
          }
        }
        """
    }
}
