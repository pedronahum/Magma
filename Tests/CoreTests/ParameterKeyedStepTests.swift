// Magma - Identity-keyed optimizer step
// The positional `step([Tensor])` matches gradients to parameters by array
// position, which silently trains the wrong tensor if the order is off. The
// `[Parameter: Tensor]` overload matches by identity instead — verified here to be
// independent of insertion order and to honor frozen (requiresGrad == false)
// parameters.
//
// .serialized: materialization drives the PJRT backend.

import Testing
@testable import Magma

@Suite("Identity-keyed optimizer step", .serialized)
struct ParameterKeyedStepTests {

    @Test("gradients are matched by identity, not by dictionary/insertion order")
    func orderIndependent() {
        let a = Parameter(Tensor<Float>([1, 2, 3], shape: [3]), name: "a")
        let b = Parameter(Tensor<Float>([10, 20, 30], shape: [3]), name: "b")
        var opt = optim.SGD(parameters: [a, b], lr: 1.0)

        // Build the map "out of order" relative to `parameters` — must not matter.
        let grads: [Parameter: Tensor<Float>] = [
            b: Tensor<Float>([2, 2, 2], shape: [3]),
            a: Tensor<Float>([1, 1, 1], shape: [3]),
        ]
        opt.step(grads)

        #expect(a.value.scalars() == [0, 1, 2])       // [1,2,3] - [1,1,1]
        #expect(b.value.scalars() == [8, 18, 28])      // [10,20,30] - [2,2,2]
    }

    @Test("frozen parameters may be omitted and are left untouched")
    func frozenOmitted() {
        let w = Parameter(Tensor<Float>([1, 1, 1], shape: [3]), name: "w")
        let frozen = Parameter(Tensor<Float>([5, 5, 5], shape: [3]),
                               requiresGrad: false, name: "frozen")
        var opt = optim.SGD(parameters: [w, frozen], lr: 1.0)

        // Only supply a gradient for the trainable parameter.
        opt.step([w: Tensor<Float>([1, 1, 1], shape: [3])])

        #expect(w.value.scalars() == [0, 0, 0])
        #expect(frozen.value.scalars() == [5, 5, 5])   // untouched
    }
}
