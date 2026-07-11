// Magma - where_/select materialization tests
// stablehlo.select needs an i1 predicate; conditions here are float (0/1), so
// select must coerce them. These materialize on GPU to lock in that fix
// (where_/maskedFill previously never compiled on the XLA path).
//
// .serialized: materialization creates a GPU PJRT client (see XLAGPUSmokeTests).

import Testing
@testable import Magma
@testable import LazyTensor

@Suite("Where/Select Tests", .serialized)
struct WhereSelectTests {

    @Test("where_ selects elementwise by a float mask")
    func whereSelects() {
        let x = Tensor<Float>([1, 2, 3, 4], shape: [4])
        let y = Tensor<Float>([10, 20, 30, 40], shape: [4])
        let mask = x.greaterThan(2.0)            // [0, 0, 1, 1]
        let result = Tensor.where_(mask, x, y).scalars()
        // mask true -> x, else y
        #expect(result == [10, 20, 3, 4])
    }

    @Test("maskedFill replaces where mask is set")
    func maskedFill() {
        let x = Tensor<Float>([1, 2, 3, 4], shape: [4])
        let mask = x.greaterThan(2.0)            // [0, 0, 1, 1]
        let fill = Tensor<Float>.full([4], -1, on: x.device)
        let result = x.maskedFill(mask: mask, value: fill).scalars()
        #expect(result == [1, 2, -1, -1])
    }
}
