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
