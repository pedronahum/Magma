// Magma - Value-semantic differentiable layers (experimental, "Design A")
//
// The S4TF-style alternative to the reference-semantic `nn.Module`/`Parameter`
// API: a layer is a Differentiable value type whose weights are stored `Tensor`s,
// its forward is `@differentiable`, and it is `KeyPathIterable` so a generic
// optimizer can reach its tensors by reflection (see TangentOptimizer). Layers
// compose into a typed, differentiable `Sequential` via a result builder.
//
// This is the migration target de-risked earlier; it coexists with the existing
// `nn.*` layers rather than replacing them yet. Currently `Tensor -> Tensor`
// layers.

import _Differentiation

/// A value-semantic, differentiable `Tensor -> Tensor` layer.
public protocol Layer: Differentiable, KeyPathIterable {
    @differentiable(reverse)
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float>
}

/// Fully-connected layer: `x @ weight + bias`.
public struct Linear: Layer {
    public var weight: Tensor<Float>
    public var bias: Tensor<Float>

    public init(weight: Tensor<Float>, bias: Tensor<Float>) {
        self.weight = weight
        self.bias = bias
    }

    @differentiable(reverse)
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.matmul(weight) + bias
    }
}

/// ReLU activation (no parameters).
public struct ReLU: Layer {
    public init() {}
    @differentiable(reverse)
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.relu()
    }
}

/// Sigmoid activation (no parameters).
public struct Sigmoid: Layer {
    public init() {}
    @differentiable(reverse)
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        input.sigmoid()
    }
}

/// Typed composition of two layers: `layer2(layer1(x))`. Differentiable is
/// synthesized (both sub-layers are Differentiable); `KeyPathIterable` reaches
/// both sub-layers' tensors recursively.
public struct Sequential2<L1: Layer, L2: Layer>: Layer {
    public var layer1: L1
    public var layer2: L2

    public init(_ layer1: L1, _ layer2: L2) {
        self.layer1 = layer1
        self.layer2 = layer2
    }

    @differentiable(reverse)
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        layer2(layer1(input))
    }
}

/// Builds a chain of layers into a left-nested `Sequential2` tree, preserving
/// each layer's concrete type (so the whole model stays a Differentiable struct).
@resultBuilder
public enum LayerBuilder {
    public static func buildPartialBlock<L: Layer>(first: L) -> L { first }
    public static func buildPartialBlock<L0: Layer, L1: Layer>(
        accumulated: L0, next: L1
    ) -> Sequential2<L0, L1> {
        Sequential2(accumulated, next)
    }
}

/// Compose layers in declaration order, e.g.
/// `sequential { Linear(...); ReLU(); Linear(...) }`.
public func sequential<Body: Layer>(@LayerBuilder _ content: () -> Body) -> Body {
    content()
}
