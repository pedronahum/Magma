// Magma - Value-semantic layer proof (Design A de-risk)
// Prove the whole "Design A" mechanism composes on this toolchain: a
// value-semantic Differentiable layer (weights are stored Tensors, forward is
// @differentiable), a KeyPathIterable backed by reflection (no compiler
// synthesis), and a generic optimizer that walks Model.TangentVector — trained
// end to end with real autodiff. CPU-materialized.

import Testing
import _Differentiation
@testable import Magma

// A value-semantic dense layer. Differentiable is auto-synthesized (all stored
// props are Tensors); KeyPathIterable conformance is empty (reflection default).
private struct Dense: Differentiable, KeyPathIterable {
    var weight: Tensor<Float>
    var bias: Tensor<Float>

    @differentiable(reverse)
    func callAsFunction(_ x: Tensor<Float>) -> Tensor<Float> {
        x.matmul(weight) + bias
    }
}

@Suite("Value-Semantic Layer Proof", .serialized)
struct ValueLayerProofTests {

    @Test("KeyPathIterable enumerates a layer's (and its tangent's) Tensor slots")
    func keyPathEnumeration() {
        // Reflection finds the two Tensor stored properties, writable on the model.
        #expect(Dense.writableKeyPaths(to: Tensor<Float>.self).count == 2)
        // ...and on the synthesized tangent (via the conformance-free helper).
        #expect(reflectKeyPaths(of: Dense.TangentVector.self, to: Tensor<Float>.self).count == 2)
    }

    @Test("generic reflection optimizer trains a Dense end to end via autodiff")
    func trainsEndToEnd() {
        // Fit y = 3x + 1 with a Dense(1 -> 1), starting from zero.
        var model = Dense(
            weight: Tensor<Float>([0], shape: [1, 1]),
            bias: Tensor<Float>([0], shape: [1]))
        let x = Tensor<Float>([0, 1, 2, 3], shape: [4, 1])
        let y = Tensor<Float>([1, 4, 7, 10], shape: [4, 1])   // 3x + 1
        let n = Tensor<Float>.full([], 4.0)

        func loss(_ m: Dense) -> Float {
            let r = m(x) - y
            return ((r * r).sum() / n).scalars()[0]
        }
        let initialLoss = loss(model)

        for _ in 0..<800 {
            // Real reverse-mode autodiff wrt the whole model -> Dense.TangentVector.
            let grad = gradient(at: model) { m -> Tensor<Float> in
                let r = m(x) - y
                return (r * r).sum() / n
            }
            // Generic optimizer walks the tangent's Tensor slots via reflection.
            sgdUpdate(&model, gradient: grad, learningRate: 0.1)
            // Materialize params each step so the lazy graph doesn't grow unbounded.
            model.weight = Tensor<Float>(model.weight.scalars(), shape: model.weight.shape)
            model.bias = Tensor<Float>(model.bias.scalars(), shape: model.bias.shape)
        }

        let finalLoss = loss(model)
        let w = model.weight.scalars()[0]
        let b = model.bias.scalars()[0]

        #expect(finalLoss < initialLoss * 0.001)   // converged
        #expect(abs(w - 3) < 0.1)                   // weight -> 3
        #expect(abs(b - 1) < 0.1)                   // bias   -> 1
    }
}
