// Magma - Tensor initializer validation tests
//
// Tensor(_:shape:) and the shape-based factories must reject a data count that
// does not match the shape, and negative dimensions, at the call site instead
// of building a malformed graph that fails much later. The trapping cases run
// as exit tests.

import Testing
@testable import Magma

@Suite("Tensor initializer validation")
struct TensorInitValidationTests {

    @Test("A matching data count and shape build the tensor")
    func matchingCountIsAccepted() {
        let t = Tensor<Float>([1, 2, 3, 4, 5, 6], shape: [2, 3])
        #expect(t.shape == [2, 3])
        #expect(t.scalars() == [1, 2, 3, 4, 5, 6])
        let scalar = Tensor<Float>([7], shape: [])
        #expect(scalar.shape == [])
        #expect(scalar.scalars() == [7])
        let empty = Tensor<Float>([], shape: [0, 3])
        #expect(empty.elementCount == 0)
    }

    @Test("Too few values for the shape traps")
    func tooFewValuesTraps() async {
        await #expect(processExitsWith: .failure) {
            _ = Tensor<Float>([1, 2, 3], shape: [2, 2])
        }
    }

    @Test("Too many values for the shape traps")
    func tooManyValuesTraps() async {
        await #expect(processExitsWith: .failure) {
            _ = Tensor<Float>([1, 2, 3, 4, 5], shape: [2, 2])
        }
    }

    @Test("A scalar shape needs exactly one value")
    func scalarShapeNeedsOneValue() async {
        await #expect(processExitsWith: .failure) {
            _ = Tensor<Float>([], shape: [])
        }
    }

    @Test("Negative dimensions trap in Tensor(_:shape:) and the factories")
    func negativeDimensionTraps() async {
        await #expect(processExitsWith: .failure) {
            _ = Tensor<Float>([1, 2], shape: [-1, -2])
        }
        await #expect(processExitsWith: .failure) {
            _ = Tensor<Float>.zeros([2, -3])
        }
        await #expect(processExitsWith: .failure) {
            _ = Tensor<Float>.randn([-4])
        }
    }
}
