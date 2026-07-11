// Magma - non-Float Tensor init tests
// Constructing a Tensor with a non-Float scalar type must convert numerically
// (not via String round-trip + force-unwrap).
//
// .serialized: values may materialize on the GPU (see XLAGPUSmokeTests).

import Testing
@testable import Magma
@testable import LazyTensor

@Suite("Non-Float Tensor Init Tests", .serialized)
struct NonFloatInitTests {

    @Test("Double tensor round-trips values")
    func doubleInit() {
        let t = Tensor<Double>([1.5, 2.5, 3.5], shape: [3])
        #expect(t.shape == [3])
        #expect(t.scalars() == [1.5, 2.5, 3.5])
    }

    @Test("Int32 tensor converts to storage without crashing")
    func int32Init() {
        let t = Tensor<Int32>([5, 7, 9], shape: [3])
        #expect(t.shape == [3])
        #expect(t.scalars() == [5, 7, 9])
    }

    @Test("Int tensor converts")
    func intInit() {
        let t = Tensor<Int>([-2, 0, 4], shape: [3])
        #expect(t.scalars() == [-2, 0, 4])
    }
}
