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
        // A lower-rank cotangent is broadcastable to the target: typically the
        // rank-0 `Tensor.zero` that autodiff passes for an output that does not
        // reach the result. Expand it rather than reducing.
        if selfRank < targetRank {
            return self.broadcast(to: targetShape)
        }
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

// MARK: - VJPs for Parameterized Activations

// These activations are single lazy ops (or have a numerically better
// gradient than differentiating their definition), so they get custom VJPs.
// A registered derivative is visible to other modules, which is what lets user
// code differentiate through them.

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// Broadcast a cotangent to `shape`. Autodiff passes the rank-0
    /// `Tensor.zero` for an output that does not reach the result.
    internal func broadcastCotangent(to shape: [Int]) -> Tensor {
        self.shape == shape ? self : self.broadcast(to: shape)
    }

    /// VJP for leakyRelu: 1 for x > 0, `negativeSlope` otherwise.
    @derivative(of: leakyRelu, wrt: self)
    public func vjpLeakyRelu(negativeSlope: Float) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let positive = self.greaterThan(Scalar(0))
        let slope = Tensor.full([], Scalar(negativeSlope), on: device)
        let one = Tensor.ones([], on: device)
        let grad = positive + (one - positive) * slope
        return (self.leakyRelu(negativeSlope: negativeSlope), { v in v * grad })
    }

    /// VJP for elu: 1 for x > 0, `alpha * exp(x)` otherwise.
    @derivative(of: elu, wrt: self)
    public func vjpElu(alpha: Float) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let grad = eluGradient(alpha: Scalar(alpha))
        return (self.elu(alpha: alpha), { v in v * grad })
    }

    /// VJP for selu: `scale` for x > 0, `scale * alpha * exp(x)` otherwise.
    @derivative(of: selu)
    public func vjpSelu() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let alpha = Scalar(1.6732632423543772848170429916717)
        let scale = Tensor.full([], Scalar(1.0507009873554804934193349852946), on: device)
        let grad = eluGradient(alpha: alpha) * scale
        return (self.selu(), { v in v * grad })
    }

    private func eluGradient(alpha: Scalar) -> Tensor {
        let positive = self.greaterThan(Scalar(0))
        let one = Tensor.ones([], on: device)
        // Clamp before exp so a large positive input cannot give inf, which the
        // mask below would turn into 0 * inf = NaN.
        let negativeBranch = self.clamp(min: -.infinity, max: 0).exp() * Tensor.full([], alpha, on: device)
        return positive + (one - positive) * negativeBranch
    }

    /// VJP for hardtanh (and `clamp`): 1 strictly inside `(minVal, maxVal)`,
    /// 0 outside, where the output is constant.
    @derivative(of: hardtanh, wrt: self)
    public func vjpHardtanh(minVal: Float, maxVal: Float) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let inside = self.greaterThan(Scalar(minVal)) * self.lessThan(Scalar(maxVal))
        return (self.hardtanh(minVal: minVal, maxVal: maxVal), { v in v * inside })
    }

    /// VJP for pow: `exponent * x^(exponent - 1)`.
    @derivative(of: pow, wrt: self)
    public func vjpPow(_ exponent: Float) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.pow(exponent)
        let grad: Tensor
        if exponent == 0 {
            grad = Tensor.zeros(shape, on: device)
        } else {
            grad = self.pow(exponent - 1) * Tensor.full([], Scalar(exponent), on: device)
        }
        return (result, { v in v * grad })
    }

    /// VJP for softplus: `sigmoid(x)`, which stays finite where `exp(x)`
    /// overflows.
    @derivative(of: softplus)
    public func vjpSoftplus() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let grad = self.sigmoid()
        return (self.softplus(), { v in v * grad })
    }

    /// VJP for logSoftmax: `v - softmax(x) * sum(v, dim)`.
    @derivative(of: logSoftmax)
    public func vjpLogSoftmax(dim: Int) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.logSoftmax(dim: dim)
        let axis = dim < 0 ? rank + dim : dim
        let shape = self.shape
        return (result, { v in
            let v = v.broadcastCotangent(to: shape)
            let sumV = v.sum(dims: [axis], keepDims: true).broadcast(to: shape)
            return v - result.exp() * sumV
        })
    }
}

// MARK: - VJPs for Max Reductions

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// Pullback shared by max/min reductions (a subgradient): the cotangent
    /// goes to the elements equal to the extremum, split evenly among ties.
    ///
    /// - Parameters:
    ///   - keptResult: the reduced value with the reduced axes kept as size 1.
    ///   - keptCotangent: the cotangent in the same kept shape.
    ///   - axes: the (non-negative) reduced axes.
    internal func extremumPullback(keptResult: Tensor, keptCotangent: Tensor, axes: [Int]) -> Tensor {
        let hits = self.equalTo(keptResult.broadcast(to: shape))
        let count = hits.sum(dims: axes, keepDims: true).broadcast(to: shape)
        return keptCotangent.broadcast(to: shape) * hits / count
    }

    /// Shape of a reduction over `axes` with the reduced axes kept as size 1.
    internal func keptReductionShape(_ axes: [Int]) -> [Int] {
        var kept = shape
        for axis in axes { kept[axis] = 1 }
        return kept
    }

    /// VJP for the full max reduction (subgradient; ties share the gradient).
    @derivative(of: max)
    public func vjpMax() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.max()
        let axes = Array(0..<rank)
        let kept = keptReductionShape(axes)
        return (result, { v in
            self.extremumPullback(
                keptResult: result.reshape(kept),
                keptCotangent: v.broadcastCotangent(to: []).reshape(kept),
                axes: axes)
        })
    }

    /// VJP for max along dimensions (subgradient; ties share the gradient).
    @derivative(of: max(dims:keepDims:))
    public func vjpMaxDims(dims: [Int], keepDims: Bool) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.max(dims: dims, keepDims: keepDims)
        let axes = dims.map { $0 < 0 ? rank + $0 : $0 }
        let kept = keptReductionShape(axes)
        let resultShape = result.shape
        return (result, { v in
            self.extremumPullback(
                keptResult: result.reshape(kept),
                keptCotangent: v.broadcastCotangent(to: resultShape).reshape(kept),
                axes: axes)
        })
    }
}

// MARK: - VJP for Variance

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for variance: `2 * (x - mean(x)) / d`, with `d = N - 1` when
    /// `unbiased` (and N > 1), else `N`, where N is the number of reduced
    /// elements. (A custom VJP: differentiating the body crashes the
    /// Differentiation pass on Swift 6.3.)
    @derivative(of: variance)
    public func vjpVariance(
        dims: [Int], keepDims: Bool, unbiased: Bool
    ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.variance(dims: dims, keepDims: keepDims, unbiased: unbiased)
        let axes = dims.map { $0 < 0 ? rank + $0 : $0 }
        let count = axes.reduce(1) { $0 * shape[$1] }
        let denominator = unbiased && count > 1 ? count - 1 : count
        let kept = keptReductionShape(axes)
        let inputShape = self.shape, resultShape = result.shape
        let centered = self - self.mean(dims: axes, keepDims: true).broadcast(to: inputShape)
        let scale = Tensor.full([], Scalar(2) / Scalar(denominator), on: device)
        return (result, { v in
            let vFull = v.broadcastCotangent(to: resultShape).reshape(kept).broadcast(to: inputShape)
            return vFull * centered * scale
        })
    }
}

// MARK: - VJPs for Masking

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for maskedFill: masked positions route the cotangent to `value`,
    /// the others to `self`. The mask is not differentiated.
    @derivative(of: maskedFill, wrt: (self, value))
    public func vjpMaskedFill(
        mask: Tensor, value: Tensor
    ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        let result = self.maskedFill(mask: mask, value: value)
        let selected = mask.notEqual(Scalar(0))
        let one = Tensor.ones([], on: device)
        let shape = self.shape, valueShape = value.shape
        return (result, { v in
            let v = v.broadcastCotangent(to: shape)
            return (v * (one - selected), (v * selected).sumAlongBroadcastDims(to: valueShape))
        })
    }
}

// MARK: - VJPs for Device and Precision Conversions

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for moving a tensor between devices: the cotangent moves back.
    @derivative(of: to(device:))
    public func vjpToDevice(_ targetDevice: Device) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let sourceDevice = device
        return (self.to(device: targetDevice), { v in v.to(device: sourceDevice) })
    }
}

extension Tensor where Scalar == Float {

    /// VJP for toReducedPrecision: the cotangent is converted back to the
    /// input's dtype.
    @derivative(of: toReducedPrecision)
    public func vjpToReducedPrecision() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let sourceDtype = dtype
        return (self.toReducedPrecision(), { v in v.to(sourceDtype) })
    }

    /// VJP for toFullPrecision: the cotangent is converted back to the
    /// input's dtype.
    @derivative(of: toFullPrecision)
    public func vjpToFullPrecision() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let sourceDtype = dtype
        return (self.toFullPrecision(), { v in v.to(sourceDtype) })
    }

    /// VJP for a dtype conversion: the cotangent is converted back to the
    /// input's dtype.
    @derivative(of: to(_:))
    public func vjpToDtype(_ targetDtype: DType) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let sourceDtype = dtype
        return (self.to(targetDtype), { v in v.to(sourceDtype) })
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
