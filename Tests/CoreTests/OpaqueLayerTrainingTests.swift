// Magma - Training a model held as `some Layer`
// The ergonomic spelling the compiler bug appeared to forbid: build a model with
// a `-> some Layer` factory, hold it as an opaque type, and train it. Directly
// calling `gradient(at:)` on such a value crashes the compiler
// (Documentation/KNOWN_COMPILER_ISSUES.md), but routing the differentiation
// through the generic `modelGradient` helper works — this test proves it on the
// real Tensor-based Layer types, end to end.

import Testing
import _Differentiation
@testable import Magma

@Suite("Opaque `some Layer` Training", .serialized)
struct OpaqueLayerTrainingTests {

    // Returns an opaque `some Layer` — the model's concrete (nested Sequential2)
    // type is hidden from the caller, which is what used to force a crash.
    private func makeMLP() -> some Layer {
        sequential {
            Linear(weight: Tensor<Float>([Float](repeating: 0.5, count: 8), shape: [1, 8]),
                   bias: Tensor<Float>.zeros([8]))
            ReLU()
            Linear(weight: Tensor<Float>([Float](repeating: 0.3, count: 8), shape: [8, 1]),
                   bias: Tensor<Float>.zeros([1]))
        }
    }

    @Test("modelGradient differentiates an opaque model without crashing the compiler")
    func opaqueGradient() {
        let model = makeMLP()   // static type: some Layer
        let x = Tensor<Float>([1], shape: [1, 1])
        let y = Tensor<Float>([4], shape: [1, 1])
        let g = modelGradient(of: model, input: x, target: y) { pred, tgt in
            let r = pred - tgt
            return (r * r).sum()
        }
        // Reflection reaches the same 4 tensor slots through the opaque tangent.
        #expect(recursivelyTensorKeyPaths(of: g).count == 4)
    }

    @Test("Adam trains an opaque `some Layer` model end to end")
    func trainsOpaqueModel() {
        var model = makeMLP()   // held as `some Layer`
        let x = Tensor<Float>([0, 1, 2, 3], shape: [4, 1])
        let y = Tensor<Float>([1, 4, 7, 10], shape: [4, 1])   // 3x + 1
        let n = Tensor<Float>.full([], 4.0)

        let mse: @differentiable(reverse) (Tensor<Float>, Tensor<Float>) -> Tensor<Float> = { pred, tgt in
            let r = pred - tgt
            return (r * r).sum() / n
        }

        let initialLoss = modelValueWithGradient(of: model, input: x, target: y, lossFn: mse)
            .value.scalars()[0]

        var opt = Adam(learningRate: 0.05)
        for _ in 0..<400 {
            let g = modelGradient(of: model, input: x, target: y, lossFn: mse)
            opt.update(&model, gradient: g)   // update over the opaque type (reflection only)
        }

        let finalLoss = modelValueWithGradient(of: model, input: x, target: y, lossFn: mse)
            .value.scalars()[0]
        #expect(finalLoss < initialLoss * 0.1)
        #expect(finalLoss < 1.0)
    }
}
