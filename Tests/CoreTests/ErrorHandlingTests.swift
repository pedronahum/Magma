// Magma - Error Handling Tests
// Tests for TensorError and debugging utilities

import Testing
@testable import Magma
@testable import StableHLO

// MARK: - TensorError Description Tests

@Suite("TensorError Description Tests")
struct TensorErrorDescriptionTests {

    @Test("Shape mismatch description")
    func shapeMismatchDescription() {
        let error = TensorError.shapeMismatch(
            operation: "add",
            lhsShape: [2, 3],
            rhsShape: [4, 5]
        )

        let desc = error.description
        #expect(desc.contains("Shape mismatch"))
        #expect(desc.contains("add"))
        #expect(desc.contains("[2, 3]"))
        #expect(desc.contains("[4, 5]"))
    }

    @Test("Shape mismatch with message")
    func shapeMismatchWithMessage() {
        let error = TensorError.shapeMismatch(
            operation: "concat",
            lhsShape: [2, 3],
            rhsShape: [2, 4],
            message: "All dimensions except concat dimension must match"
        )

        let desc = error.description
        #expect(desc.contains("All dimensions except"))
    }

    @Test("Incompatible shape description")
    func incompatibleShapeDescription() {
        let error = TensorError.incompatibleShape(
            operation: "matmul",
            shape: [3],
            requirement: "Requires at least 2D tensors"
        )

        let desc = error.description
        #expect(desc.contains("Incompatible shape"))
        #expect(desc.contains("matmul"))
        #expect(desc.contains("[3]"))
        #expect(desc.contains("Requires at least 2D"))
    }

    @Test("Invalid reshape description")
    func invalidReshapeDescription() {
        let error = TensorError.invalidReshape(
            from: [2, 3],
            to: [2, 4]
        )

        let desc = error.description
        #expect(desc.contains("Invalid reshape"))
        #expect(desc.contains("6 elements"))
        #expect(desc.contains("8 elements"))
    }

    @Test("Invalid broadcast description")
    func invalidBroadcastDescription() {
        let error = TensorError.invalidBroadcast(
            from: [2, 3],
            to: [4, 5]
        )

        let desc = error.description
        #expect(desc.contains("Invalid broadcast"))
        #expect(desc.contains("[2, 3]"))
        #expect(desc.contains("[4, 5]"))
    }

    @Test("Axis out of bounds description")
    func axisOutOfBoundsDescription() {
        let error = TensorError.axisOutOfBounds(
            axis: 5,
            rank: 3
        )

        let desc = error.description
        #expect(desc.contains("Axis out of bounds"))
        #expect(desc.contains("5"))
        #expect(desc.contains("rank: 3"))
        #expect(desc.contains("[-3, 2]"))  // Valid range
    }

    @Test("Index out of bounds description")
    func indexOutOfBoundsDescription() {
        let error = TensorError.indexOutOfBounds(
            index: 10,
            dimension: 0,
            size: 5
        )

        let desc = error.description
        #expect(desc.contains("Index out of bounds"))
        #expect(desc.contains("10"))
        #expect(desc.contains("size 5"))
    }

    @Test("Device mismatch description")
    func deviceMismatchDescription() {
        let error = TensorError.deviceMismatch(
            lhsDevice: "CPU:0",
            rhsDevice: "GPU:0",
            operation: "add"
        )

        let desc = error.description
        #expect(desc.contains("Device mismatch"))
        #expect(desc.contains("CPU:0"))
        #expect(desc.contains("GPU:0"))
    }

    @Test("Type mismatch description")
    func typeMismatchDescription() {
        let error = TensorError.typeMismatch(
            lhsType: .float32,
            rhsType: .int32,
            operation: "add"
        )

        let desc = error.description
        #expect(desc.contains("Type mismatch"))
        #expect(desc.contains("float32"))
        #expect(desc.contains("int32"))
    }

    @Test("Empty tensor description")
    func emptyTensorDescription() {
        let error = TensorError.emptyTensor(operation: "sum")

        let desc = error.description
        #expect(desc.contains("Empty tensor"))
        #expect(desc.contains("sum"))
    }

    @Test("Not scalar description")
    func notScalarDescription() {
        let error = TensorError.notScalar(shape: [2, 3])

        let desc = error.description
        #expect(desc.contains("Expected scalar"))
        #expect(desc.contains("[2, 3]"))
    }
}

// MARK: - TensorDebug Tests

@Suite("TensorDebug Tests")
struct TensorDebugTests {

    @Test("Are broadcastable true")
    func areBroadcastableTrue() {
        // Same shape
        #expect(TensorDebug.areBroadcastable([2, 3], [2, 3]))

        // Scalar with anything
        #expect(TensorDebug.areBroadcastable([], [2, 3]))
        #expect(TensorDebug.areBroadcastable([2, 3], []))

        // Different ranks with 1
        #expect(TensorDebug.areBroadcastable([1, 3], [2, 3]))
        #expect(TensorDebug.areBroadcastable([2, 1], [2, 3]))

        // Higher rank
        #expect(TensorDebug.areBroadcastable([3], [2, 3]))
        #expect(TensorDebug.areBroadcastable([1, 1, 3], [2, 3]))
    }

    @Test("Are broadcastable false")
    func areBroadcastableFalse() {
        // Incompatible dimensions
        #expect(!TensorDebug.areBroadcastable([2, 3], [4, 5]))
        #expect(!TensorDebug.areBroadcastable([2, 3], [2, 4]))
        #expect(!TensorDebug.areBroadcastable([3, 3], [2, 3]))
    }

    @Test("Broadcast result shape")
    func broadcastResultShape() {
        // Same shape
        #expect(TensorDebug.broadcastResultShape([2, 3], [2, 3]) == [2, 3])

        // Broadcasting
        #expect(TensorDebug.broadcastResultShape([1, 3], [2, 3]) == [2, 3])
        #expect(TensorDebug.broadcastResultShape([2, 1], [2, 3]) == [2, 3])
        #expect(TensorDebug.broadcastResultShape([3], [2, 3]) == [2, 3])

        // Invalid
        #expect(TensorDebug.broadcastResultShape([2, 3], [4, 5]) == nil)
    }

    @Test("Is valid reshape")
    func isValidReshape() {
        #expect(TensorDebug.isValidReshape(from: [2, 3], to: [6]))
        #expect(TensorDebug.isValidReshape(from: [2, 3], to: [3, 2]))
        #expect(TensorDebug.isValidReshape(from: [2, 3], to: [1, 6]))

        #expect(!TensorDebug.isValidReshape(from: [2, 3], to: [5]))
        #expect(!TensorDebug.isValidReshape(from: [2, 3], to: [2, 4]))
    }
}

// MARK: - TensorAssert Tests

@Suite("TensorAssert Tests")
struct TensorAssertTests {

    @Test("Broadcastable pass")
    func broadcastablePass() {
        // Should not crash
        TensorAssert.broadcastable([2, 3], [2, 3], operation: "add")
        TensorAssert.broadcastable([1, 3], [2, 3], operation: "add")
        TensorAssert.broadcastable([3], [2, 3], operation: "add")
    }

    @Test("Shapes equal pass")
    func shapesEqualPass() {
        // Should not crash
        TensorAssert.shapesEqual([2, 3], [2, 3], operation: "assign")
        TensorAssert.shapesEqual([], [], operation: "assign")
    }

    @Test("Valid axis pass")
    func validAxisPass() {
        // Should not crash
        TensorAssert.validAxis(0, rank: 3)
        TensorAssert.validAxis(2, rank: 3)
        TensorAssert.validAxis(-1, rank: 3)  // Last axis
        TensorAssert.validAxis(-3, rank: 3)  // First axis
    }

    @Test("Valid reshape pass")
    func validReshapePass() {
        // Should not crash
        TensorAssert.validReshape(from: [2, 3], to: [6])
        TensorAssert.validReshape(from: [2, 3, 4], to: [24])
        TensorAssert.validReshape(from: [], to: [1])
    }

    @Test("Valid matmul pass")
    func validMatmulPass() {
        // Should not crash
        TensorAssert.validMatmul(lhs: [2, 3], rhs: [3, 4])
        TensorAssert.validMatmul(lhs: [5, 2, 3], rhs: [5, 3, 4])
    }

    @Test("Non-empty pass")
    func nonEmptyPass() {
        // Should not crash
        TensorAssert.nonEmpty(shape: [2, 3], operation: "sum")
        TensorAssert.nonEmpty(shape: [], operation: "sum")  // Scalar is 1 element
        TensorAssert.nonEmpty(shape: [1], operation: "sum")
    }

    @Test("Is scalar pass")
    func isScalarPass() {
        // Should not crash
        TensorAssert.isScalar(shape: [])
        TensorAssert.isScalar(shape: [1])
        TensorAssert.isScalar(shape: [1, 1, 1])
    }
}

// MARK: - Matmul Shape Help Tests

@Suite("Matmul Shape Help Tests")
struct MatmulShapeHelpTests {

    @Test("Matmul shape mismatch help")
    func matmulShapeMismatchHelp() {
        let error = TensorError.shapeMismatch(
            operation: "matmul",
            lhsShape: [2, 3],
            rhsShape: [4, 5]
        )

        let desc = error.description
        // Should contain helpful matmul info
        #expect(desc.contains("inner dimensions"))
    }
}

// MARK: - Error Conformances

@Suite("Error Conformance Tests")
struct ErrorConformanceTests {

    @Test("TensorError conforms to Error")
    func tensorErrorConformsToError() {
        let error: Error = TensorError.emptyTensor(operation: "test")
        #expect(error != nil)
    }

    @Test("TensorError conforms to CustomStringConvertible")
    func tensorErrorConformsToCustomStringConvertible() {
        let error: CustomStringConvertible = TensorError.emptyTensor(operation: "test")
        #expect(!error.description.isEmpty)
    }
}
