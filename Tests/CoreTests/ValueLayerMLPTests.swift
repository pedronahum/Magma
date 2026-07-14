// Magma - Value-semantic layer library + Adam (Design A, part B)
// The migration target as a usable API: build an MLP with the typed differentiable
// `sequential { Linear; ReLU; Linear }` and train it with a generic reflection
// `Adam` — no Parameter list, no per-model optimizer code. CPU-materialized.
//
// The model is spelled with its concrete (nested Sequential2) type rather than
// `some Layer`: differentiating through an opaque type currently crashes the SIL
// pipeline, so the concrete type is used.

import Testing
import _Differentiation
@testable import Magma

// The concrete type the builder produces for a 3-layer chain.
private typealias MLP = Sequential2<Sequential2<Linear, ReLU>, Linear>

@Suite("Value-Semantic MLP + Adam", .serialized)
struct ValueLayerMLPTests {

    // A 1 -> 8 -> 1 ReLU MLP, positive init so ReLU is active for x >= 0.
    private func makeMLP() -> MLP {
        sequential {
            Linear(weight: Tensor<Float>([Float](repeating: 0.5, count: 8), shape: [1, 8]),
                   bias: Tensor<Float>.zeros([8]))
            ReLU()
            Linear(weight: Tensor<Float>([Float](repeating: 0.3, count: 8), shape: [8, 1]),
                   bias: Tensor<Float>.zeros([1]))
        }
    }

    @Test("sequential builds a typed differentiable model; reflection finds its 4 tensor slots")
    func modelStructure() {
        let mlp = makeMLP()
        // Linear1(weight,bias) + ReLU(none) + Linear2(weight,bias) = 4
        #expect(mlp.recursivelyWritableTensorKeyPaths().count == 4)
        let grad = gradient(at: mlp) { m in m(Tensor<Float>([1], shape: [1, 1])).sum() }
        #expect(recursivelyTensorKeyPaths(of: grad).count == 4)
    }

    @Test("Adam trains the Sequential MLP end to end via autodiff")
    func trainsWithAdam() {
        var model = makeMLP()
        let x = Tensor<Float>([0, 1, 2, 3], shape: [4, 1])
        let y = Tensor<Float>([1, 4, 7, 10], shape: [4, 1])   // 3x + 1
        let n = Tensor<Float>.full([], 4.0)

        func loss(_ m: MLP) -> Float {
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
            opt.update(&model, gradient: grad)   // generic, reflection-driven, stateful
        }

        let finalLoss = loss(model)
        #expect(finalLoss < initialLoss * 0.1)   // converged substantially
        #expect(finalLoss < 1.0)
    }
}
