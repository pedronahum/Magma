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

    /// VJP for addition: d(a+b)/da = 1, d(a+b)/db = 1
    @derivative(of: +)
    public static func vjpAdd(lhs: Tensor, rhs: Tensor) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        return (lhs + rhs, { v in (v, v) })
    }

    /// VJP for subtraction: d(a-b)/da = 1, d(a-b)/db = -1
    @derivative(of: -)
    public static func vjpSubtract(lhs: Tensor, rhs: Tensor) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        return (lhs - rhs, { v in (v, v.negated()) })
    }

    /// VJP for multiplication: d(a*b)/da = b, d(a*b)/db = a
    @derivative(of: *)
    public static func vjpMultiply(lhs: Tensor, rhs: Tensor) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        return (lhs * rhs, { v in (v * rhs, v * lhs) })
    }

    /// VJP for division: d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    @derivative(of: /)
    public static func vjpDivide(lhs: Tensor, rhs: Tensor) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        let result = lhs / rhs
        return (result, { v in
            let dLhs = v / rhs
            let dRhs = (v.negated()) * result / rhs
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

    /// VJP for gelu
    @derivative(of: gelu)
    public func vjpGelu() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.gelu()
        return (result, { v in
            // GELU gradient is complex, approximate for now
            // d/dx[x * Phi(x)] where Phi is the CDF of standard normal
            // Approximation: gradient ≈ sigmoid(1.702 * x) * (1 + 1.702 * x * (1 - sigmoid(1.702 * x)))
            // For simplicity, use numerical approximation
            v * self.geluGrad()
        })
    }

    /// Helper for GELU gradient (approximation)
    internal func geluGrad() -> Tensor {
        // Approximate GELU gradient
        // Using tanh approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        // Gradient is complex, so we use a simplified form
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        // For now, return ones as placeholder - proper implementation needed
        handle.irNode = .constant(values: Array(repeating: Float(1.0), count: elementCount), shape: shape)
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
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
        // Simplified: if target is smaller, sum and reshape
        if targetShape.isEmpty {
            return self.sum()
        }
        // For matching ranks, sum along dimensions where target is 1
        // For now, use a simplified approach
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: targetShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .reduceSum, inputs: [self.handle], attributes: ["targetShape": targetShape])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
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

// MARK: - VJPs for Softmax

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for softmax
    @derivative(of: softmax)
    public func vjpSoftmax(dim: Int) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let s = self.softmax(dim: dim)
        return (s, { v in
            // Softmax gradient: s * (v - sum(v * s))
            let sumVS = (v * s).sum()  // Simplified - should be along dim
            return s * (v - sumVS.broadcast(to: s.shape))
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
