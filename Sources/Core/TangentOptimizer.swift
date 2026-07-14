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
import Foundation

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

/// Adam over any Differentiable + KeyPathIterable model. Non-generic: its state
/// (first/second moments per tensor slot) is a flat `[Tensor]` keyed by the stable
/// recursive traversal order, so `update` is generic in the model and no verbose
/// `Adam<VeryNestedSequential>` annotation is needed. Materializes params and
/// moments each step to keep the lazy graph flat. Mirrors `optim.Adam`'s math.
public struct Adam {
    public var learningRate: Float
    public var beta1: Float
    public var beta2: Float
    public var epsilon: Float
    private var m: [Tensor<Float>] = []   // first moment
    private var v: [Tensor<Float>] = []   // second moment
    private var t: Int = 0

    public init(learningRate: Float = 0.001, beta1: Float = 0.9,
                beta2: Float = 0.999, epsilon: Float = 1e-8) {
        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    }

    public mutating func update<M: Differentiable & KeyPathIterable>(
        _ model: inout M, gradient: M.TangentVector
    ) {
        let paramKPs = model.recursivelyWritableTensorKeyPaths()
        let gradKPs = recursivelyTensorKeyPaths(of: gradient)
        precondition(
            paramKPs.count == gradKPs.count,
            "model/tangent Tensor-slot mismatch (\(paramKPs.count) vs \(gradKPs.count))")

        if m.isEmpty {
            m = paramKPs.map { Tensor<Float>.zeros(model[keyPath: $0].shape) }
            v = m
        }
        t += 1
        let b1 = Tensor<Float>.full([], beta1)
        let b2 = Tensor<Float>.full([], beta2)
        let oneMinusB1 = Tensor<Float>.full([], 1 - beta1)
        let oneMinusB2 = Tensor<Float>.full([], 1 - beta2)
        let biasCorr1 = Tensor<Float>.full([], 1 - pow(beta1, Float(t)))
        let biasCorr2 = Tensor<Float>.full([], 1 - pow(beta2, Float(t)))
        let lr = Tensor<Float>.full([], learningRate)
        let eps = Tensor<Float>.full([], epsilon)

        for i in paramKPs.indices {
            let g = gradient[keyPath: gradKPs[i]]
            let mi = m[i] * b1 + g * oneMinusB1                 // m = β1·m + (1-β1)·g
            let vi = v[i] * b2 + (g * g) * oneMinusB2           // v = β2·v + (1-β2)·g²
            let mHat = mi / biasCorr1
            let vHat = vi / biasCorr2
            let step = lr * (mHat / (vHat.sqrt() + eps))        // lr·m̂ / (√v̂ + ε)
            let p = model[keyPath: paramKPs[i]] - step
            // Collapse graphs so state/params don't grow across steps.
            m[i] = Tensor<Float>(mi.scalars(), shape: mi.shape)
            v[i] = Tensor<Float>(vi.scalars(), shape: vi.shape)
            model[keyPath: paramKPs[i]] = Tensor<Float>(p.scalars(), shape: p.shape)
        }
    }
}
