// Magma - Nested value-semantic model proof (Design A de-risk, part 2)
// Push the de-risk past flat models: a *nested* model (a net containing two
// sub-layers), whose Tensor slots the generic optimizer reaches by recursive
// reflection, trained by a *stateful* (momentum) optimizer via real autodiff.
// This exercises the two pieces beyond the flat SGD proof — recursion into
// sub-layers and per-tensor optimizer state.
//
// Uses elementwise affine sub-layers (not matmul): a separate, pre-existing bug
// in the lazy emitter surfaces when materializing a *nested*-matmul gradient
// (`StableHLOEmitter: Missing inputs for matmul`; the flat single-matmul Dense
// trains fine — see ValueLayerProofTests). That is orthogonal to Design A, so we
// avoid it here to keep the Design-A proof green. CPU-materialized.

import Testing
import _Differentiation
@testable import Magma

// Elementwise affine sub-layer: y = x * scale + shift (scale/shift broadcast).
private struct Affine: Differentiable, KeyPathIterable {
    var scale: Tensor<Float>
    var shift: Tensor<Float>
    @differentiable(reverse)
    func callAsFunction(_ x: Tensor<Float>) -> Tensor<Float> {
        x * scale + shift
    }
}

// A nested model: two sub-layers. Composed affines still fit y = 3x + 1.
private struct Net: Differentiable, KeyPathIterable {
    var l1: Affine
    var l2: Affine
    @differentiable(reverse)
    func callAsFunction(_ x: Tensor<Float>) -> Tensor<Float> {
        l2(l1(x))
    }
}

@Suite("Nested Value-Semantic Model Proof", .serialized)
struct NestedLayerProofTests {

    private func makeNet() -> Net {
        Net(
            l1: Affine(scale: Tensor<Float>([0.5], shape: [1]), shift: Tensor<Float>([0], shape: [1])),
            l2: Affine(scale: Tensor<Float>([0.5], shape: [1]), shift: Tensor<Float>([0], shape: [1])))
    }

    @Test("recursion reaches every Tensor slot through nested sub-layers")
    func recursionEnumeratesNested() {
        let net = makeNet()
        // l1.scale, l1.shift, l2.scale, l2.shift
        #expect(net.recursivelyWritableTensorKeyPaths().count == 4)
        // The synthesized (nested) tangent, walked without a conformance.
        let grad = gradient(at: net) { m in m(Tensor<Float>([1], shape: [1, 1])).sum() }
        #expect(recursivelyTensorKeyPaths(of: grad).count == 4)
    }

    @Test("MomentumSGD trains the nested model end to end via autodiff")
    func trainsNestedWithMomentum() {
        var model = makeNet()
        let x = Tensor<Float>([0, 1, 2, 3], shape: [4, 1])
        let y = Tensor<Float>([1, 4, 7, 10], shape: [4, 1])   // 3x + 1
        let n = Tensor<Float>.full([], 4.0)

        func loss(_ m: Net) -> Float {
            let r = m(x) - y
            return ((r * r).sum() / n).scalars()[0]
        }
        let initialLoss = loss(model)

        var opt = MomentumSGD<Net>(learningRate: 0.02, momentum: 0.9)
        for _ in 0..<400 {
            let grad = gradient(at: model) { m -> Tensor<Float> in
                let r = m(x) - y
                return (r * r).sum() / n
            }
            opt.update(&model, gradient: grad)   // reflection-driven, stateful
        }

        let finalLoss = loss(model)
        #expect(finalLoss < initialLoss * 0.05)   // converged substantially
        #expect(finalLoss < 0.5)
    }
}
