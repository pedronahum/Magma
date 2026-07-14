// Magma - Gradient helpers for value-semantic layers held as `some Layer`
//
// Differentiating a value whose static type is an opaque return type (`some
// Layer`, as produced by `sequential { ... }` or any `-> some Layer` factory)
// crashes the compiler's Differentiation SIL pass — see
// Documentation/KNOWN_COMPILER_ISSUES.md. The crash is specifically about
// emitting a differentiability witness for a closure whose parameter is the
// opaque archetype.
//
// These helpers sidestep it: the differentiated closure lives inside a generic
// `<M: Layer>` function, so its parameter is an ordinary generic archetype (which
// differentiates fine), and the caller's loss is typed over concrete tensors
// (`pred`, `target`) rather than over the model. That lets models be held and
// trained as `some Layer` with the ergonomic spelling intact.

import _Differentiation

/// Reverse-mode gradient of `lossFn(model(input), target)` w.r.t. `model`.
/// `model` may be held as `some Layer`.
public func modelGradient<M: Layer>(
    of model: M,
    input: Tensor<Float>,
    target: Tensor<Float>,
    lossFn: @differentiable(reverse) (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
) -> M.TangentVector {
    gradient(at: model) { m in lossFn(m(input), target) }
}

/// Loss value and its gradient together (useful for logging the loss while
/// training). `model` may be held as `some Layer`.
public func modelValueWithGradient<M: Layer>(
    of model: M,
    input: Tensor<Float>,
    target: Tensor<Float>,
    lossFn: @differentiable(reverse) (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
) -> (value: Tensor<Float>, gradient: M.TangentVector) {
    valueWithGradient(at: model) { m in lossFn(m(input), target) }
}
