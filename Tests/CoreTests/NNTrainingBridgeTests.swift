// Magma - Real autodiff training of an nn.* layer
// The existing "training" tests for the nn.*/optim.* API feed dummy gradients
// (Tensor.ones). This trains a real nn.Linear with real gradients: autodiff via
// `parameterGradients`, applied through the identity-keyed `optim.Adam.step`.
// Loss goes to ~0 recovering known weights — the reference-semantic path works
// end to end (manually), which is what the bridge exists to make usable.
//
// .serialized: materialization drives the PJRT backend.

import Testing
import _Differentiation
@testable import Magma

@Suite("nn.* autodiff training bridge", .serialized)
struct NNTrainingBridgeTests {

    @Test("optim.Adam trains an nn.Linear with real autodiff gradients")
    func trainsLinearForReal() {
        let batch = 8
        // Deterministic inputs [8, 2].
        let xv = (0..<(batch * 2)).map { Float($0 % 5) * 0.3 - 0.6 }
        let x = Tensor<Float>(xv, shape: [batch, 2])
        // Ground-truth linear map: weight [1,2] = [2, -1], bias [1] = [0.5].
        let trueW = Tensor<Float>([2, -1], shape: [1, 2])
        let trueB = Tensor<Float>([0.5], shape: [1])
        let y = x.matmul(trueW.transpose()) + trueB.broadcast(to: [batch, 1])
        let n = Tensor<Float>.full([], Float(batch))

        let fc = nn.Linear(inputSize: 2, outputSize: 1)   // params: [weight, bias]
        var opt = optim.Adam(parameters: fc.parameters(), lr: 0.2)

        // parameters() order is [weight, bias]; express the loss over their values.
        func lossOf(_ p: [Tensor<Float>]) -> Tensor<Float> {
            let pred = x.matmul(p[0].transpose()) + p[1].broadcast(to: [batch, 1])
            let r = pred - y
            return (r * r).sum() / n
        }

        let initialLoss = parameterGradients(of: fc.parameters(), loss: lossOf).value.scalars()[0]

        // Keep the step count small: the nn.* optimizers update `param.value`
        // lazily, so the traced graph deepens with each step and is compiled once
        // when the loss is finally read. A few dozen aggressive Adam steps already
        // drive this convex problem down by orders of magnitude.
        for _ in 0..<50 {
            let (_, grads) = parameterGradients(of: fc.parameters(), loss: lossOf)
            opt.step(grads)   // identity-keyed application of real gradients
        }

        let finalLoss = parameterGradients(of: fc.parameters(), loss: lossOf).value.scalars()[0]
        #expect(finalLoss < initialLoss * 0.1)    // real gradients drive the loss down
        #expect(finalLoss < 0.1)
    }

    @Test("parameterGradients accumulates gradients of a parameter listed twice")
    func tiedParameterAccumulates() {
        let p = Parameter(Tensor<Float>([1, 2, 3], shape: [3]))
        // loss = sum(v0 * v1) with v0 == v1 == p: d/dv0 = v1, d/dv1 = v0.
        let (value, grads) = parameterGradients(of: [p, p]) { v in (v[0] * v[1]).sum() }
        #expect(value.scalars() == [14])
        #expect(grads.count == 1)
        #expect(grads[p]?.scalars() == [2, 4, 6])
    }

    @Test("parameterGradients gives zeros for parameters the loss ignores")
    func unusedParametersGetZeroGradients() {
        let a = Parameter(Tensor<Float>([1, 2], shape: [2]))
        let b = Parameter(Tensor<Float>([3, 4, 5, 6], shape: [2, 2]))

        // Loss independent of every parameter: the tangent comes back empty.
        let constant = Tensor<Float>([5], shape: [1])
        let (_, noGrads) = parameterGradients(of: [a, b]) { _ in constant.sum() }
        #expect(noGrads[a]?.shape == [2])
        #expect(noGrads[a]?.scalars() == [0, 0])
        #expect(noGrads[b]?.shape == [2, 2])
        #expect(noGrads[b]?.scalars() == [0, 0, 0, 0])

        // Loss depending on `a` only: `b` still gets a correctly shaped zero.
        let (_, partial) = parameterGradients(of: [a, b]) { v in (v[0] * v[0]).sum() }
        #expect(partial[a]?.scalars() == [2, 4])
        #expect(partial[b]?.shape == [2, 2])
        #expect(partial[b]?.scalars() == [0, 0, 0, 0])

        // The result feeds an optimizer step without tripping its
        // "no gradient supplied" precondition, and leaves `b` unchanged.
        var opt = optim.SGD(parameters: [a, b], lr: 0.5)
        opt.step(partial)
        #expect(a.value.scalars() == [0, 0])
        #expect(b.value.scalars() == [3, 4, 5, 6])
    }
}
