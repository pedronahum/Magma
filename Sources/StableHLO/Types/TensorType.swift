// Magma - StableHLO
// Tensor type representation for MLIR

/// Represents a tensor type in StableHLO IR
public struct TensorType: Hashable, Sendable, CustomStringConvertible {
    /// Shape dimensions (e.g., [32, 784] for a batch of 32 vectors of 784 elements)
    public let shape: [Int]
    
    /// Element data type
    public let dtype: DType
    
    /// Create a tensor type
    public init(shape: [Int], dtype: DType) {
        self.shape = shape
        self.dtype = dtype
    }
    
    /// Number of dimensions (rank)
    public var rank: Int { shape.count }
    
    /// Total number of elements
    public var elementCount: Int {
        shape.isEmpty ? 1 : shape.reduce(1, *)
    }
    
    /// Total size in bytes
    public var byteSize: Int {
        elementCount * dtype.byteSize
    }
    
    /// Whether this is a scalar (rank 0)
    public var isScalar: Bool {
        shape.isEmpty
    }
    
    /// MLIR type string (e.g., "tensor<32x784xf32>")
    public var mlirType: String {
        if shape.isEmpty {
            return "tensor<\(dtype.mlirName)>"
        } else {
            let shapeStr = shape.map(String.init).joined(separator: "x")
            return "tensor<\(shapeStr)x\(dtype.mlirName)>"
        }
    }
    
    public var description: String { mlirType }
    
    // MARK: - Shape Operations
    
    /// Create a new type with a different shape but same dtype
    public func reshaped(to newShape: [Int]) -> TensorType {
        precondition(
            newShape.reduce(1, *) == elementCount,
            "New shape \(newShape) has different element count than \(shape)"
        )
        return TensorType(shape: newShape, dtype: dtype)
    }
    
    /// Create a new type with transposed shape
    public func transposed() -> TensorType {
        precondition(rank == 2, "Transpose requires rank 2, got \(rank)")
        return TensorType(shape: [shape[1], shape[0]], dtype: dtype)
    }
    
    /// Create a new type with permuted dimensions
    public func permuted(_ permutation: [Int]) -> TensorType {
        precondition(permutation.count == rank, "Permutation must have same length as rank")
        let newShape = permutation.map { shape[$0] }
        return TensorType(shape: newShape, dtype: dtype)
    }
    
    /// Result type after matrix multiplication with another tensor
    public func matmulResult(with other: TensorType) -> TensorType {
        precondition(rank == 2 && other.rank == 2, "Matmul requires rank 2 tensors")
        precondition(shape[1] == other.shape[0], "Inner dimensions must match")
        precondition(dtype == other.dtype, "DTypes must match")
        return TensorType(shape: [shape[0], other.shape[1]], dtype: dtype)
    }
    
    /// Result type after reduction along an axis
    public func reduced(along axis: Int, keepDims: Bool = false) -> TensorType {
        var newShape = shape
        if keepDims {
            newShape[axis] = 1
        } else {
            newShape.remove(at: axis)
        }
        return TensorType(shape: newShape, dtype: dtype)
    }
    
    /// Result type after broadcasting with another tensor
    public func broadcast(with other: TensorType) -> TensorType? {
        let resultShape = broadcastShapes(shape, other.shape)
        guard let resultShape = resultShape else { return nil }
        return TensorType(shape: resultShape, dtype: dtype)
    }
}

// MARK: - Shape Utilities

/// Compute broadcast shape for two shapes, or nil if incompatible
public func broadcastShapes(_ a: [Int], _ b: [Int]) -> [Int]? {
    let maxRank = max(a.count, b.count)
    let paddedA = Array(repeating: 1, count: maxRank - a.count) + a
    let paddedB = Array(repeating: 1, count: maxRank - b.count) + b
    
    var result: [Int] = []
    for (dimA, dimB) in zip(paddedA, paddedB) {
        if dimA == dimB {
            result.append(dimA)
        } else if dimA == 1 {
            result.append(dimB)
        } else if dimB == 1 {
            result.append(dimA)
        } else {
            return nil  // Incompatible
        }
    }
    return result
}
