// Magma - Value-semantic Conv2d layer (Design A, part B: "port Conv")
// The differentiable conv2d wired into a value-semantic `Layer`, trained by the
// generic reflection `Adam` — no Parameter list, no per-layer optimizer code.
// Target is generated from known conv parameters so recovery is guaranteed;
// convergence exercises both the weight and bias gradients end to end.
//
// The model is spelled with its concrete type (differentiating `some Layer`
// currently crashes the SIL pipeline; see ValueLayerMLPTests).

import Testing
import _Differentiation
@testable import Magma

@Suite("Value-Semantic Conv2d + Adam", .serialized)
struct ConvLayerTests {

    @Test("reflection finds the conv layer's weight+bias slots (config is @noDerivative)")
    func slotCount() {
        let layer = Conv2d(
            weight: Tensor<Float>.zeros([2, 2, 1, 1]),
            bias: Tensor<Float>.zeros([1]))
        #expect(layer.recursivelyWritableTensorKeyPaths().count == 2)
        let g = gradient(at: layer) { l in l(Tensor<Float>.zeros([1, 4, 4, 1])).sum() }
        #expect(recursivelyTensorKeyPaths(of: g).count == 2)
    }

    @Test("Adam trains Conv2d to recover known kernel+bias")
    func trainsConv() {
        // Deterministic input [1,4,4,1].
        let x = Tensor<Float>((0..<16).map { Float($0 % 5) * 0.2 - 0.4 }, shape: [1, 4, 4, 1])
        // Ground-truth conv params, and the target they produce (2x2 valid -> [1,3,3,1]).
        let trueW = Tensor<Float>([0.5, -0.3, 0.2, 0.7], shape: [2, 2, 1, 1])
        let trueB = Tensor<Float>([0.1], shape: [1])
        let target = (x.conv2d(trueW) + trueB.broadcast(to: [1, 3, 3, 1]))
        let targetVals = target.scalars()
        let y = Tensor<Float>(targetVals, shape: [1, 3, 3, 1])
        let n = Tensor<Float>.full([], Float(targetVals.count))

        // Start from a different init.
        var model = Conv2d(
            weight: Tensor<Float>([Float](repeating: 0.0, count: 4), shape: [2, 2, 1, 1]),
            bias: Tensor<Float>.zeros([1]))

        func loss(_ m: Conv2d) -> Float {
            let r = m(x) - y
            return ((r * r).sum() / n).scalars()[0]
        }
        let initialLoss = loss(model)

        var opt = Adam(learningRate: 0.05)
        for _ in 0..<400 {
            let grad = gradient(at: model) { m -> Tensor<Float> in
                let r = m(x) - y
                return (r * r).sum() / n
            }
            opt.update(&model, gradient: grad)
        }

        let finalLoss = loss(model)
        #expect(finalLoss < initialLoss * 0.01)   // recovers the parameters
        #expect(finalLoss < 1e-3)
    }
}
