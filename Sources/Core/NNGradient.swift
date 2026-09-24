// Magma - Autodiff bridge for the reference-semantic nn.* API
//
// An `nn.Module` is not itself `Differentiable` (its weights are `Parameter`
// reference cells, read inside `forward`), so you cannot write
// `gradient(at: someModule)`. The optimizers instead consume gradients you supply.
// This helper closes that gap for the manual path: given a set of `Parameter`s and
// a differentiable loss written over their *values*, it returns the loss plus an
// identity-keyed gradient map ready for `optimizer.step(_:)`.
//
// This is a bridge for the reference-semantic API, not a substitute for real
// whole-model autodiff. For ergonomic "differentiate the entire model" training —
// no parameter list, no rewriting the forward over a values array — use the
// value-semantic `Layer` API (`sequential`, `modelGradient`; see ValueLayers.swift
// and the README Quick Example).

import _Differentiation

/// Loss value and per-parameter gradients for a differentiable loss expressed over
/// a set of parameters' values.
///
/// The `loss` closure receives the current values of `parameters` **in order** and
/// returns a scalar loss; its gradient w.r.t. each value is paired back with the
/// owning `Parameter` by identity, so the result can be fed straight to
/// `optimizer.step(_:)` (the `[Parameter: Tensor]` overload) without any
/// order-matching.
///
/// ```swift
/// let (loss, grads) = parameterGradients(of: fc.parameters()) { p in
///     let pred = x.matmul(p[0].transpose()) + p[1].broadcast(to: [batch, 1]) // p[0]=weight, p[1]=bias
///     let r = pred - y
///     return (r * r).sum() / n
/// }
/// optimizer.step(grads)
/// ```
///
/// Every parameter gets an entry: one the loss does not depend on gets a zero
/// gradient of its own shape. A parameter listed more than once (e.g. tied
/// weights, or concatenated `parameters()` lists that share a layer) gets the
/// sum of the gradients of all its occurrences.
public func parameterGradients(
    of parameters: [Parameter],
    loss: @differentiable(reverse) ([Tensor<Float>]) -> Tensor<Float>
) -> (value: Tensor<Float>, gradients: [Parameter: Tensor<Float>]) {
    let values = parameters.map { $0.value }
    let (value, gradView) = valueWithGradient(at: values, of: loss)
    // The gradient of `[Tensor]` is an Array.DifferentiableView; `.base` is the
    // underlying `[Tensor]` (Tensor is its own TangentVector). A value the loss
    // does not depend on gets `Tensor.zero`, a shape-[] scalar, and when the
    // loss depends on no value at all `.base` is empty. Replace both with
    // explicit zeros shaped like the parameter.
    let grads: [Tensor<Float>] = values.indices.map { i in
        let v = values[i]
        if i < gradView.base.count {
            let g = gradView.base[i]
            let isScalarZeroStandIn = g.shape.isEmpty && !v.shape.isEmpty
            if !isScalarZeroStandIn { return g }
        }
        return Tensor<Float>.zeros(v.shape, on: v.device)
    }
    // Accumulate, rather than trap on, parameters that appear more than once.
    return (value, Dictionary(zip(parameters, grads), uniquingKeysWith: +))
}
