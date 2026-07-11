// Magma - Adam numerical value test
// Verifies a materialized Adam step produces the correct update after the
// loop-invariant-constant hoisting refactor.
//
// .serialized: materialization creates a GPU PJRT client (see XLAGPUSmokeTests).

import Testing
@testable import Magma
@testable import LazyTensor

@Suite("Adam Step Value Tests", .serialized)
struct AdamStepValueTests {

    @Test("single Adam step matches closed form")
    func singleStep() {
        // p=1, g=1, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, t=1
        // m=0.1, v=0.001; mHat=1, vHat=1; update=0.1*1/(1+eps) ~ 0.1; p' ~ 0.9
        let p = Parameter(Tensor<Float>([1.0], shape: [1]))
        var opt = optim.Adam(parameters: [p], lr: 0.1)
        opt.step([Tensor<Float>([1.0], shape: [1])])

        let updated = p.value.scalars()[0]
        #expect(abs(updated - 0.9) < 1e-4)
    }

    @Test("two Adam steps stay finite and decrease")
    func twoSteps() {
        let p = Parameter(Tensor<Float>([2.0, -2.0], shape: [2]))
        var opt = optim.Adam(parameters: [p], lr: 0.1)
        opt.step([Tensor<Float>([1.0, -1.0], shape: [2])])
        opt.step([Tensor<Float>([1.0, -1.0], shape: [2])])

        let v = p.value.scalars()
        #expect(v.allSatisfy { $0.isFinite })
        #expect(v[0] < 2.0)          // decreased toward 0 for positive grad
        #expect(v[1] > -2.0)         // increased toward 0 for negative grad
    }
}
