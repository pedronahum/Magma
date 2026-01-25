// Magma - Error Handling Tests
// Tests for TensorError and debugging utilities

import XCTest
@testable import Magma
@testable import StableHLO

// MARK: - TensorError Description Tests

final class TensorErrorDescriptionTests: XCTestCase {

    func testShapeMismatchDescription() {
        let error = TensorError.shapeMismatch(
            operation: "add",
            lhsShape: [2, 3],
            rhsShape: [4, 5]
        )

        let desc = error.description
        XCTAssertTrue(desc.contains("Shape mismatch"))
        XCTAssertTrue(desc.contains("add"))
        XCTAssertTrue(desc.contains("[2, 3]"))
        XCTAssertTrue(desc.contains("[4, 5]"))
    }

    func testShapeMismatchWithMessage() {
        let error = TensorError.shapeMismatch(
            operation: "concat",
            lhsShape: [2, 3],
            rhsShape: [2, 4],
            message: "All dimensions except concat dimension must match"
        )

        let desc = error.description
        XCTAssertTrue(desc.contains("All dimensions except"))
    }

    func testIncompatibleShapeDescription() {
        let error = TensorError.incompatibleShape(
            operation: "matmul",
            shape: [3],
            requirement: "Requires at least 2D tensors"
        )

        let desc = error.description
        XCTAssertTrue(desc.contains("Incompatible shape"))
        XCTAssertTrue(desc.contains("matmul"))
        XCTAssertTrue(desc.contains("[3]"))
        XCTAssertTrue(desc.contains("Requires at least 2D"))
    }

    func testInvalidReshapeDescription() {
        let error = TensorError.invalidReshape(
            from: [2, 3],
            to: [2, 4]
        )

        let desc = error.description
        XCTAssertTrue(desc.contains("Invalid reshape"))
        XCTAssertTrue(desc.contains("6 elements"))
        XCTAssertTrue(desc.contains("8 elements"))
    }

    func testInvalidBroadcastDescription() {
        let error = TensorError.invalidBroadcast(
            from: [2, 3],
            to: [4, 5]
        )

        let desc = error.description
        XCTAssertTrue(desc.contains("Invalid broadcast"))
        XCTAssertTrue(desc.contains("[2, 3]"))
        XCTAssertTrue(desc.contains("[4, 5]"))
    }

    func testAxisOutOfBoundsDescription() {
        let error = TensorError.axisOutOfBounds(
            axis: 5,
            rank: 3
        )

        let desc = error.description
        XCTAssertTrue(desc.contains("Axis out of bounds"))
        XCTAssertTrue(desc.contains("5"))
        XCTAssertTrue(desc.contains("rank: 3"))
        XCTAssertTrue(desc.contains("[-3, 2]"))  // Valid range
    }

    func testIndexOutOfBoundsDescription() {
        let error = TensorError.indexOutOfBounds(
            index: 10,
            dimension: 0,
            size: 5
        )

        let desc = error.description
        XCTAssertTrue(desc.contains("Index out of bounds"))
        XCTAssertTrue(desc.contains("10"))
        XCTAssertTrue(desc.contains("size 5"))
    }

    func testDeviceMismatchDescription() {
        let error = TensorError.deviceMismatch(
            lhsDevice: "CPU:0",
            rhsDevice: "GPU:0",
            operation: "add"
        )

        let desc = error.description
        XCTAssertTrue(desc.contains("Device mismatch"))
        XCTAssertTrue(desc.contains("CPU:0"))
        XCTAssertTrue(desc.contains("GPU:0"))
    }

    func testTypeMismatchDescription() {
        let error = TensorError.typeMismatch(
            lhsType: .float32,
            rhsType: .int32,
            operation: "add"
        )

        let desc = error.description
        XCTAssertTrue(desc.contains("Type mismatch"))
        XCTAssertTrue(desc.contains("float32"))
        XCTAssertTrue(desc.contains("int32"))
    }

    func testEmptyTensorDescription() {
        let error = TensorError.emptyTensor(operation: "sum")

        let desc = error.description
        XCTAssertTrue(desc.contains("Empty tensor"))
        XCTAssertTrue(desc.contains("sum"))
    }

    func testNotScalarDescription() {
        let error = TensorError.notScalar(shape: [2, 3])

        let desc = error.description
        XCTAssertTrue(desc.contains("Expected scalar"))
        XCTAssertTrue(desc.contains("[2, 3]"))
    }
}

// MARK: - TensorDebug Tests

final class TensorDebugTests: XCTestCase {

    func testAreBroadcastableTrue() {
        // Same shape
        XCTAssertTrue(TensorDebug.areBroadcastable([2, 3], [2, 3]))

        // Scalar with anything
        XCTAssertTrue(TensorDebug.areBroadcastable([], [2, 3]))
        XCTAssertTrue(TensorDebug.areBroadcastable([2, 3], []))

        // Different ranks with 1
        XCTAssertTrue(TensorDebug.areBroadcastable([1, 3], [2, 3]))
        XCTAssertTrue(TensorDebug.areBroadcastable([2, 1], [2, 3]))

        // Higher rank
        XCTAssertTrue(TensorDebug.areBroadcastable([3], [2, 3]))
        XCTAssertTrue(TensorDebug.areBroadcastable([1, 1, 3], [2, 3]))
    }

    func testAreBroadcastableFalse() {
        // Incompatible dimensions
        XCTAssertFalse(TensorDebug.areBroadcastable([2, 3], [4, 5]))
        XCTAssertFalse(TensorDebug.areBroadcastable([2, 3], [2, 4]))
        XCTAssertFalse(TensorDebug.areBroadcastable([3, 3], [2, 3]))
    }

    func testBroadcastResultShape() {
        // Same shape
        XCTAssertEqual(TensorDebug.broadcastResultShape([2, 3], [2, 3]), [2, 3])

        // Broadcasting
        XCTAssertEqual(TensorDebug.broadcastResultShape([1, 3], [2, 3]), [2, 3])
        XCTAssertEqual(TensorDebug.broadcastResultShape([2, 1], [2, 3]), [2, 3])
        XCTAssertEqual(TensorDebug.broadcastResultShape([3], [2, 3]), [2, 3])

        // Invalid
        XCTAssertNil(TensorDebug.broadcastResultShape([2, 3], [4, 5]))
    }

    func testIsValidReshape() {
        XCTAssertTrue(TensorDebug.isValidReshape(from: [2, 3], to: [6]))
        XCTAssertTrue(TensorDebug.isValidReshape(from: [2, 3], to: [3, 2]))
        XCTAssertTrue(TensorDebug.isValidReshape(from: [2, 3], to: [1, 6]))

        XCTAssertFalse(TensorDebug.isValidReshape(from: [2, 3], to: [5]))
        XCTAssertFalse(TensorDebug.isValidReshape(from: [2, 3], to: [2, 4]))
    }
}

// MARK: - TensorAssert Tests

final class TensorAssertTests: XCTestCase {

    func testBroadcastablePass() {
        // Should not crash
        TensorAssert.broadcastable([2, 3], [2, 3], operation: "add")
        TensorAssert.broadcastable([1, 3], [2, 3], operation: "add")
        TensorAssert.broadcastable([3], [2, 3], operation: "add")
    }

    func testShapesEqualPass() {
        // Should not crash
        TensorAssert.shapesEqual([2, 3], [2, 3], operation: "assign")
        TensorAssert.shapesEqual([], [], operation: "assign")
    }

    func testValidAxisPass() {
        // Should not crash
        TensorAssert.validAxis(0, rank: 3)
        TensorAssert.validAxis(2, rank: 3)
        TensorAssert.validAxis(-1, rank: 3)  // Last axis
        TensorAssert.validAxis(-3, rank: 3)  // First axis
    }

    func testValidReshapePass() {
        // Should not crash
        TensorAssert.validReshape(from: [2, 3], to: [6])
        TensorAssert.validReshape(from: [2, 3, 4], to: [24])
        TensorAssert.validReshape(from: [], to: [1])
    }

    func testValidMatmulPass() {
        // Should not crash
        TensorAssert.validMatmul(lhs: [2, 3], rhs: [3, 4])
        TensorAssert.validMatmul(lhs: [5, 2, 3], rhs: [5, 3, 4])
    }

    func testNonEmptyPass() {
        // Should not crash
        TensorAssert.nonEmpty(shape: [2, 3], operation: "sum")
        TensorAssert.nonEmpty(shape: [], operation: "sum")  // Scalar is 1 element
        TensorAssert.nonEmpty(shape: [1], operation: "sum")
    }

    func testIsScalarPass() {
        // Should not crash
        TensorAssert.isScalar(shape: [])
        TensorAssert.isScalar(shape: [1])
        TensorAssert.isScalar(shape: [1, 1, 1])
    }
}

// MARK: - Matmul Shape Help Tests

final class MatmulShapeHelpTests: XCTestCase {

    func testMatmulShapeMismatchHelp() {
        let error = TensorError.shapeMismatch(
            operation: "matmul",
            lhsShape: [2, 3],
            rhsShape: [4, 5]
        )

        let desc = error.description
        // Should contain helpful matmul info
        XCTAssertTrue(desc.contains("inner dimensions"))
    }
}

// MARK: - Error Conformances

final class ErrorConformanceTests: XCTestCase {

    func testTensorErrorConformsToError() {
        let error: Error = TensorError.emptyTensor(operation: "test")
        XCTAssertNotNil(error)
    }

    func testTensorErrorConformsToCustomStringConvertible() {
        let error: CustomStringConvertible = TensorError.emptyTensor(operation: "test")
        XCTAssertFalse(error.description.isEmpty)
    }
}
