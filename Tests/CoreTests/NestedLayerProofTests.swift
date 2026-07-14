// Magma - Nested value-semantic model proof (Design A de-risk, part 2)
// Push the de-risk past flat models: a *nested* model (an MLP that contains two
// Dense sub-layers), whose Tensor slots the generic optimizer reaches by
// recursive reflection, trained by a *stateful* (momentum) optimizer via real
// autodiff. Exercises recursion into sub-layers and per-tensor optimizer state,
// with real matmul layers. CPU-materialized.

import Testing
import _Differentiation
@testable import Magma

private struct Dense: Differentiable, KeyPathIterable {
    var weight: Tensor<Float>
    var bias: Tensor<Float>
    @differentiable(reverse)
    func callAsFunction(_ x: Tensor<Float>) -> Tensor<Float> {
        x.matmul(weight) + bias
    }
}

// A nested model: two Dense sub-layers (1 -> 4 -> 1).
private struct MLP: Differentiable, KeyPathIterable {
    var layer1: Dense
    var layer2: Dense
    @differentiable(reverse)
    func callAsFunction(_ x: Tensor<Float>) -> Tensor<Float> {
        layer2(layer1(x))
    }
}

@Suite("Nested Value-Semantic Model Proof", .serialized)
struct NestedLayerProofTests {

    private func makeMLP() -> MLP {
        MLP(
            layer1: Dense(weight: Tensor<Float>([Float](repeating: 0.1, count: 4), shape: [1, 4]),
                          bias: Tensor<Float>.zeros([4])),
            layer2: Dense(weight: Tensor<Float>([Float](repeating: 0.1, count: 4), shape: [4, 1]),
                          bias: Tensor<Float>.zeros([1])))
    }

    @Test("recursion reaches every Tensor slot through nested sub-layers")
    func recursionEnumeratesNested() {
        let mlp = makeMLP()
        // layer1.weight, layer1.bias, layer2.weight, layer2.bias
        #expect(mlp.recursivelyWritableTensorKeyPaths().count == 4)
        // The synthesized (nested) tangent, walked without a conformance.
        let grad = gradient(at: mlp) { m in m(Tensor<Float>([1], shape: [1, 1])).sum() }
        #expect(recursivelyTensorKeyPaths(of: grad).count == 4)
    }

    @Test("MomentumSGD trains the nested MLP end to end via autodiff")
    func trainsNestedWithMomentum() {
        var model = makeMLP()
        let x = Tensor<Float>([0, 1, 2, 3], shape: [4, 1])
        let y = Tensor<Float>([1, 4, 7, 10], shape: [4, 1])   // 3x + 1
        let n = Tensor<Float>.full([], 4.0)

        func loss(_ m: MLP) -> Float {
            let r = m(x) - y
            return ((r * r).sum() / n).scalars()[0]
        }
        let initialLoss = loss(model)

        var opt = MomentumSGD<MLP>(learningRate: 0.01, momentum: 0.9)
        for _ in 0..<600 {
            let grad = gradient(at: model) { m -> Tensor<Float> in
                let r = m(x) - y
                return (r * r).sum() / n
            }
            opt.update(&model, gradient: grad)   // reflection-driven, stateful, real matmul grads
        }

        let finalLoss = loss(model)
        #expect(finalLoss < initialLoss * 0.1)   // converged substantially
        #expect(finalLoss < 1.0)
    }
}
