// Magma - Generic tangent-walking optimizer (experimental / proof-of-concept)
//
// The "Design A" optimizer: it updates any Differentiable + KeyPathIterable model
// from its `TangentVector`, with no `Parameter` list and no per-type code. It
// walks the model's writable `Tensor` slots and the tangent's `Tensor` slots in
// lockstep (they share declaration order) — the traversal S4TF got from compiler-
// synthesized KeyPathIterable, here supplied by the reflection-backed protocol.
//
// This is the piece that was previously thought to require an unavailable compiler
// feature; it composes today. Momentum/Adam are a straightforward extension (keep
// per-slot state keyed by the same key paths).

import _Differentiation

/// One SGD step: `param -= learningRate * grad`, for every `Tensor<Float>` slot in
/// the model, driven by reflection over the model and its tangent.
public func sgdUpdate<M: Differentiable & KeyPathIterable>(
    _ model: inout M,
    gradient: M.TangentVector,
    learningRate: Float
) {
    let paramKPs = M.writableKeyPaths(to: Tensor<Float>.self)
    // The tangent is a synthesized type (can't add a conformance); reflect it
    // directly — the free helper needs no protocol conformance.
    let gradKPs = reflectKeyPaths(of: M.TangentVector.self, to: Tensor<Float>.self)
    precondition(
        paramKPs.count == gradKPs.count,
        "model/tangent Tensor-slot mismatch (\(paramKPs.count) vs \(gradKPs.count))")

    let lr = Tensor<Float>.full([], learningRate)
    for (pkp, gkp) in zip(paramKPs, gradKPs) {
        model[keyPath: pkp] = model[keyPath: pkp] - gradient[keyPath: gkp] * lr
    }
}

/// A stateful SGD-with-momentum optimizer over any nested Differentiable +
/// KeyPathIterable model. Per-tensor velocity is kept as a flat array keyed by the
/// (stable) recursive traversal order, proving stateful optimizers work over
/// `Model.TangentVector` — the case that most needed the missing key-path
/// traversal. Materializes params and velocity each step so the lazy graph stays
/// flat across a training loop.
public struct MomentumSGD<M: Differentiable & KeyPathIterable> {
    public var learningRate: Float
    public var momentum: Float
    private var velocity: [Tensor<Float>] = []

    public init(learningRate: Float, momentum: Float = 0.9) {
        precondition(momentum >= 0 && momentum < 1, "momentum in [0, 1)")
        self.learningRate = learningRate
        self.momentum = momentum
    }

    public mutating func update(_ model: inout M, gradient: M.TangentVector) {
        let paramKPs = model.recursivelyWritableTensorKeyPaths()
        let gradKPs = recursivelyTensorKeyPaths(of: gradient)
        precondition(
            paramKPs.count == gradKPs.count,
            "model/tangent Tensor-slot mismatch (\(paramKPs.count) vs \(gradKPs.count))")

        if velocity.isEmpty {
            velocity = paramKPs.map { Tensor<Float>.zeros(model[keyPath: $0].shape) }
        }
        let lr = Tensor<Float>.full([], learningRate)
        let mom = Tensor<Float>.full([], momentum)

        for i in paramKPs.indices {
            let v = velocity[i] * mom + gradient[keyPath: gradKPs[i]]   // v = μ·v + g
            let p = model[keyPath: paramKPs[i]] - v * lr                // w -= lr·v
            // Collapse the lazy graph so it doesn't grow across steps.
            velocity[i] = Tensor<Float>(v.scalars(), shape: v.shape)
            model[keyPath: paramKPs[i]] = Tensor<Float>(p.scalars(), shape: p.shape)
        }
    }
}
