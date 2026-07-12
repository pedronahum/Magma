// Magma - Autodiff Support
// Swift differentiable programming integration for Tensor

import _Differentiation
import LazyTensor
import StableHLO
import XLARuntime

// MARK: - Equatable Conformance

extension Tensor: Equatable where Scalar: TensorScalar & BinaryFloatingPoint {
    /// Tensors are equal if they have the same shape and handle ID
    /// Note: This is identity equality, not value equality
    public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
        lhs.handle.id == rhs.handle.id && lhs.shape == rhs.shape
    }
}

// MARK: - AdditiveArithmetic Conformance

extension Tensor: AdditiveArithmetic where Scalar: TensorScalar & BinaryFloatingPoint {
    /// Zero tensor (scalar zero, will broadcast as needed)
    public static var zero: Tensor {
        Tensor.zeros([])
    }

    /// Element-wise addition (already defined in Tensor.swift, just declare conformance)
    // public static func + (lhs: Tensor, rhs: Tensor) -> Tensor - defined in Tensor.swift

    /// Element-wise subtraction (already defined in Tensor.swift)
    // public static func - (lhs: Tensor, rhs: Tensor) -> Tensor - defined in Tensor.swift
}

// MARK: - Differentiable Conformance

extension Tensor: Differentiable where Scalar: TensorScalar & BinaryFloatingPoint {
    /// The tangent vector type is the same as the tensor itself.
    public typealias TangentVector = Tensor<Scalar>

    /// Move along a tangent vector.
    public mutating func move(by offset: TangentVector) {
        self = self + offset
    }
}

// MARK: - VJPs for Arithmetic Operations

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    // NOTE: these operators broadcast their operands, so each pullback must
    // reduce the incoming cotangent back to its operand's original shape
    // (summing over the broadcast axes) via `sumAlongBroadcastDims(to:)`.
    // Without this, e.g. `x[B,N] + bias[N]` yields a bias gradient of shape
    // [B,N] instead of [N]. For same-shape operands the helper is a no-op.

    /// VJP for addition: d(a+b)/da = 1, d(a+b)/db = 1
    @derivative(of: +)
    public static func vjpAdd(lhs: Tensor, rhs: Tensor) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        let lhsShape = lhs.shape, rhsShape = rhs.shape
        return (lhs + rhs, { v in
            (v.sumAlongBroadcastDims(to: lhsShape), v.sumAlongBroadcastDims(to: rhsShape))
        })
    }

    /// VJP for subtraction: d(a-b)/da = 1, d(a-b)/db = -1
    @derivative(of: -)
    public static func vjpSubtract(lhs: Tensor, rhs: Tensor) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        let lhsShape = lhs.shape, rhsShape = rhs.shape
        return (lhs - rhs, { v in
            (v.sumAlongBroadcastDims(to: lhsShape), v.negated().sumAlongBroadcastDims(to: rhsShape))
        })
    }

    /// VJP for multiplication: d(a*b)/da = b, d(a*b)/db = a
    @derivative(of: *)
    public static func vjpMultiply(lhs: Tensor, rhs: Tensor) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        let lhsShape = lhs.shape, rhsShape = rhs.shape
        return (lhs * rhs, { v in
            ((v * rhs).sumAlongBroadcastDims(to: lhsShape),
             (v * lhs).sumAlongBroadcastDims(to: rhsShape))
        })
    }

    /// VJP for division: d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    @derivative(of: /)
    public static func vjpDivide(lhs: Tensor, rhs: Tensor) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        let lhsShape = lhs.shape, rhsShape = rhs.shape
        let result = lhs / rhs
        return (result, { v in
            let dLhs = (v / rhs).sumAlongBroadcastDims(to: lhsShape)
            let dRhs = ((v.negated()) * result / rhs).sumAlongBroadcastDims(to: rhsShape)
            return (dLhs, dRhs)
        })
    }
}

// MARK: - VJPs for Unary Operations

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for negation
    @derivative(of: negated)
    public func vjpNegated() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        return (self.negated(), { v in v.negated() })
    }

    /// VJP for unary minus prefix operator
    @derivative(of: -)
    public static func vjpNegate(_ tensor: Tensor) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        return (-tensor, { v in -v })
    }
}

// MARK: - VJPs for Activations

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for ReLU: d(relu(x))/dx = 1 if x > 0, else 0
    @derivative(of: relu)
    public func vjpRelu() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.relu()
        return (result, { v in
            // Gradient is v where self > 0, else 0
            // Approximate by checking if result > 0
            v * result.reluGradMask()
        })
    }

    /// Helper for ReLU gradient mask
    internal func reluGradMask() -> Tensor {
        // Returns 1 where self > 0, 0 otherwise
        // Compare with a zero tensor
        let zeroTensor = Tensor<Scalar>.zeros(shape, on: device)
        return self.greaterThan(zeroTensor)
    }

    /// VJP for sigmoid: d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
    @derivative(of: sigmoid)
    public func vjpSigmoid() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let s = self.sigmoid()
        return (s, { v in
            let one = Tensor.ones(s.shape, on: s.device)
            return v * s * (one - s)
        })
    }

    /// VJP for tanh: d(tanh(x))/dx = 1 - tanh(x)^2
    @derivative(of: tanh)
    public func vjpTanh() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let t = self.tanh()
        return (t, { v in
            let one = Tensor.ones(t.shape, on: t.device)
            return v * (one - t * t)
        })
    }

    /// VJP for exp: d(exp(x))/dx = exp(x)
    @derivative(of: exp)
    public func vjpExp() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let e = self.exp()
        return (e, { v in v * e })
    }

    /// VJP for log: d(log(x))/dx = 1/x
    @derivative(of: log)
    public func vjpLog() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        return (self.log(), { v in v / self })
    }

}

// MARK: - VJPs for Matrix Operations

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for matmul: d(A@B)/dA = dOut@B^T, d(A@B)/dB = A^T@dOut
    @derivative(of: matmul)
    public func vjpMatmul(_ other: Tensor) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        let result = self.matmul(other)
        return (result, { v in
            let dSelf = v.matmul(other.transpose())
            let dOther = self.transpose().matmul(v)
            return (dSelf, dOther)
        })
    }

    /// VJP for transpose: gradient just transposes back
    @derivative(of: transpose)
    public func vjpTranspose() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        return (self.transpose(), { v in v.transpose() })
    }

    /// VJP for reshape: just reshape back to original shape
    @derivative(of: reshape)
    public func vjpReshape(_ newShape: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let originalShape = self.shape
        return (self.reshape(newShape), { v in v.reshape(originalShape) })
    }

    /// VJP for broadcast
    @derivative(of: broadcast)
    public func vjpBroadcast(to targetShape: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let originalShape = self.shape
        return (self.broadcast(to: targetShape), { v in
            // Sum along dimensions that were broadcasted
            if originalShape == targetShape {
                return v
            }
            // Simplified: just sum and reshape
            // Proper implementation should sum along broadcast dimensions
            if originalShape.isEmpty {
                // Broadcasting from scalar - sum everything
                return v.sum()
            }
            // For now, reshape (may need proper reduce for broadcasting)
            return v.sumAlongBroadcastDims(to: originalShape)
        })
    }

    /// Helper to sum along broadcast dimensions
    internal func sumAlongBroadcastDims(to targetShape: [Int]) -> Tensor {
        if targetShape.isEmpty {
            return self.sum()
        }
        // Compute which dimensions were broadcast (target has 1, self has > 1)
        // Left-pad target shape with 1s if ranks differ
        let selfRank = self.shape.count
        let targetRank = targetShape.count
        let paddedTarget = Array(repeating: 1, count: selfRank - targetRank) + targetShape

        var reduceDims: [Int] = []
        for i in 0..<selfRank {
            if paddedTarget[i] == 1 && self.shape[i] > 1 {
                reduceDims.append(i)
            }
        }

        if reduceDims.isEmpty {
            // No broadcast dims to reduce, just reshape if ranks differ
            if selfRank != targetRank {
                return self.reshape(targetShape)
            }
            return self
        }

        let summed = self.sum(dims: reduceDims, keepDims: true)
        if summed.shape == targetShape {
            return summed
        }
        return summed.reshape(targetShape)
    }
}

// MARK: - VJPs for Reductions

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for sum: gradient broadcasts to original shape
    @derivative(of: sum)
    public func vjpSum() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let originalShape = self.shape
        return (self.sum(), { v in
            v.broadcast(to: originalShape)
        })
    }

    /// VJP for mean: gradient is 1/n broadcast to original shape
    @derivative(of: mean)
    public func vjpMean() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let originalShape = self.shape
        let n = Float(self.elementCount)
        return (self.mean(), { v in
            let scaled = v / Tensor.full([], Scalar(n), on: v.device)
            return scaled.broadcast(to: originalShape)
        })
    }
}

// MARK: - VJPs for Additional Operations

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for abs: d|x|/dx = sign(x) = x / |x|
    @derivative(of: abs)
    public func vjpAbs() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.abs()
        return (result, { v in
            // Gradient of abs is sign(x): 1 if x > 0, -1 if x < 0, 0 if x == 0
            // Use self / abs(self) with protection against division by zero
            let eps = Tensor.full(self.shape, Scalar(1e-12), on: self.device)
            let sign = self / (result + eps)
            return v * sign
        })
    }
}

// MARK: - VJPs for Batched Matrix Operations

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for batchedMatmul: [..., M, K] @ [..., K, N] -> [..., M, N]
    /// dL/dA = dL/dC @ B^T, dL/dB = A^T @ dL/dC
    @derivative(of: batchedMatmul)
    public func vjpBatchedMatmul(_ other: Tensor) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        let result = self.batchedMatmul(other)
        return (result, { v in
            let dSelf = v.batchedMatmul(other.transpose(-1, -2))
            let dOther = self.transpose(-1, -2).batchedMatmul(v)
            return (dSelf, dOther)
        })
    }

    /// VJP for transpose with dimension arguments
    @derivative(of: transpose(_:_:))
    public func vjpTransposeDims(_ dim1: Int, _ dim2: Int) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        return (self.transpose(dim1, dim2), { v in v.transpose(dim1, dim2) })
    }
}

// MARK: - VJPs for Dimension-wise Reductions

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for sum along dimensions
    @derivative(of: sum(dims:keepDims:))
    public func vjpSumDims(dims: [Int], keepDims: Bool) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let originalShape = self.shape
        let result = self.sum(dims: dims, keepDims: keepDims)
        return (result, { v in
            var grad = v
            if !keepDims {
                // Re-insert the reduced dimensions as size 1
                let normalizedDims = dims.map { $0 < 0 ? originalShape.count + $0 : $0 }.sorted()
                var expandedShape = v.shape
                for dim in normalizedDims {
                    expandedShape.insert(1, at: dim)
                }
                grad = v.reshape(expandedShape)
            }
            return grad.broadcast(to: originalShape)
        })
    }

    /// VJP for mean along dimensions
    @derivative(of: mean(dims:keepDims:))
    public func vjpMeanDims(dims: [Int], keepDims: Bool) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let originalShape = self.shape
        let normalizedDims = dims.map { $0 < 0 ? originalShape.count + $0 : $0 }
        // Count of elements in the reduced dimensions
        let count = normalizedDims.reduce(1) { $0 * originalShape[$1] }
        let result = self.mean(dims: dims, keepDims: keepDims)
        return (result, { v in
            var grad = v / Tensor.full(v.shape, Scalar(count), on: v.device)
            if !keepDims {
                let sortedDims = normalizedDims.sorted()
                var expandedShape = v.shape
                for dim in sortedDims {
                    expandedShape.insert(1, at: dim)
                }
                grad = grad.reshape(expandedShape)
            }
            return grad.broadcast(to: originalShape)
        })
    }
}

// MARK: - VJPs for Softmax

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for softmax along a dimension
    /// Gradient: s * (v - sum(v * s, dim=dim))
    @derivative(of: softmax)
    public func vjpSoftmax(dim: Int) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let s = self.softmax(dim: dim)
        return (s, { v in
            let sumVS = (v * s).sum(dims: [dim], keepDims: true)
            return s * (v - sumVS.broadcast(to: s.shape))
        })
    }
}

// MARK: - VJPs for GELU (proper gradient)

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for gelu, matching the tanh approximation the forward op emits:
    ///   g(x) = 0.5·x·(1 + tanh(u)),  u = c·(x + a·x³),  c = √(2/π), a = 0.044715
    /// so the exact pullback is
    ///   g'(x) = 0.5·(1 + tanh u) + 0.5·x·(1 − tanh²u)·u',   u' = c·(1 + 3a·x²)
    /// The previous pullback used the derivative of the *sigmoid* approximation
    /// (x·sigmoid(1.702x)), which is not the derivative of the value the forward
    /// actually computes — gradients were systematically off in the transition
    /// region.
    @derivative(of: gelu)
    public func vjpGelu() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.gelu()
        return (result, { [x = self] v in
            let shape = x.shape, device = x.device
            let c = Tensor.full(shape, Scalar(0.7978845608), on: device)   // √(2/π)
            let a = Tensor.full(shape, Scalar(0.044715), on: device)
            let three = Tensor.full(shape, Scalar(3), on: device)
            let half = Tensor.full(shape, Scalar(0.5), on: device)
            let one = Tensor.ones(shape, on: device)
            let x2 = x * x
            let u = c * (x + a * (x2 * x))
            let t = u.tanh()
            let uPrime = c * (one + three * a * x2)
            let geluGrad = half * (one + t) + half * x * (one - t * t) * uPrime
            return v * geluGrad
        })
    }
}

// MARK: - Gradient Computation Functions

/// Compute the gradient of a scalar-valued function at a point.
///
/// Example:
/// ```swift
/// let x = Tensor<Float>.ones([2, 3])
/// let grad = gradient(at: x) { $0.sum() }
/// ```
public func gradient<T: Differentiable>(
    at x: T,
    of f: @differentiable(reverse) (T) -> Tensor<Float>
) -> T.TangentVector where T.TangentVector: AdditiveArithmetic {
    let (_, pullback) = valueWithPullback(at: x, of: f)
    return pullback(Tensor<Float>.ones([], on: .default))
}

/// Compute both the value and gradient of a scalar-valued function.
///
/// Example:
/// ```swift
/// let x = Tensor<Float>.ones([2, 3])
/// let (value, grad) = valueWithGradient(at: x) { $0.sum() }
/// ```
public func valueWithGradient<T: Differentiable>(
    at x: T,
    of f: @differentiable(reverse) (T) -> Tensor<Float>
) -> (value: Tensor<Float>, gradient: T.TangentVector) where T.TangentVector: AdditiveArithmetic {
    let (value, pullback) = valueWithPullback(at: x, of: f)
    let gradient = pullback(Tensor<Float>.ones([], on: .default))
    return (value, gradient)
}

/// Compute gradient with respect to multiple inputs
public func gradient<T: Differentiable, U: Differentiable>(
    at x: T, _ y: U,
    of f: @differentiable(reverse) (T, U) -> Tensor<Float>
) -> (T.TangentVector, U.TangentVector) where T.TangentVector: AdditiveArithmetic, U.TangentVector: AdditiveArithmetic {
    let (_, pullback) = valueWithPullback(at: x, y, of: f)
    return pullback(Tensor<Float>.ones([], on: .default))
}
