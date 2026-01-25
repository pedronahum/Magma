// Magma - Torch Tests
// End-to-end tests for the PyTorch-compatible API
// These require XLA for full functionality

import XCTest
@testable import Magma
@testable import LazyTensor
@testable import StableHLO

final class TensorCreationTests: XCTestCase {

    func testZeros() {
        let t = Tensor<Float>.zeros([2, 3])
        XCTAssertEqual(t.shape, [2, 3])
        XCTAssertEqual(t.dtype, .float32)
        XCTAssertEqual(t.rank, 2)
        XCTAssertEqual(t.elementCount, 6)
    }

    func testOnes() {
        let t = Tensor<Float>.ones([3, 4, 5])
        XCTAssertEqual(t.shape, [3, 4, 5])
        XCTAssertEqual(t.elementCount, 60)
    }

    func testDevice() {
        let t = Tensor<Float>.zeros([2, 2], on: .default)
        XCTAssertEqual(t.device, .default)
        XCTAssertEqual(t.device.backend, .cpu)
    }
}

final class TensorArithmeticTests: XCTestCase {

    func testAdd() {
        let a = Tensor<Float>.zeros([2, 3])
        let b = Tensor<Float>.ones([2, 3])
        let c = a + b

        XCTAssertEqual(c.shape, [2, 3])
        XCTAssertEqual(c.dtype, .float32)

        // Verify IR node is created
        if case .operation(let op, let inputs, _) = c.handle.irNode {
            XCTAssertEqual(op, .add)
            XCTAssertEqual(inputs.count, 2)
        } else {
            XCTFail("Expected operation node")
        }
    }

    func testSubtract() {
        let a = Tensor<Float>.zeros([2, 3])
        let b = Tensor<Float>.ones([2, 3])
        let c = a - b

        XCTAssertEqual(c.shape, [2, 3])
    }

    func testMultiply() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.ones([2, 3])
        let c = a * b

        XCTAssertEqual(c.shape, [2, 3])
    }

    func testDivide() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.ones([2, 3])
        let c = a / b

        XCTAssertEqual(c.shape, [2, 3])
    }
}

final class TensorMatrixTests: XCTestCase {

    func testMatmul() {
        let a = Tensor<Float>.zeros([32, 784])
        let b = Tensor<Float>.zeros([784, 256])
        let c = a.matmul(b)

        XCTAssertEqual(c.shape, [32, 256])
    }

    func testTranspose() {
        let a = Tensor<Float>.zeros([3, 4])
        let b = a.transpose()

        XCTAssertEqual(b.shape, [4, 3])
    }

    func testReshape() {
        let a = Tensor<Float>.zeros([2, 3, 4])
        let b = a.reshape([6, 4])

        XCTAssertEqual(b.shape, [6, 4])
        XCTAssertEqual(b.elementCount, 24)
    }
}

final class TensorReductionTests: XCTestCase {

    func testSum() {
        let a = Tensor<Float>.ones([2, 3])
        let b = a.sum()

        XCTAssertEqual(b.shape, [])  // Scalar
        XCTAssertEqual(b.rank, 0)
    }

    func testMean() {
        let a = Tensor<Float>.ones([2, 3])
        let b = a.mean()

        XCTAssertEqual(b.shape, [])
    }

    func testMax() {
        let a = Tensor<Float>.ones([2, 3])
        let b = a.max()

        XCTAssertEqual(b.shape, [])
    }
}

final class TensorActivationTests: XCTestCase {

    func testRelu() {
        let a = Tensor<Float>.zeros([2, 3])
        let b = a.relu()

        XCTAssertEqual(b.shape, [2, 3])

        if case .operation(let op, _, _) = b.handle.irNode {
            XCTAssertEqual(op, .relu)
        } else {
            XCTFail("Expected relu operation")
        }
    }

    func testSigmoid() {
        let a = Tensor<Float>.zeros([2, 3])
        let b = a.sigmoid()

        XCTAssertEqual(b.shape, [2, 3])
    }

    func testTanh() {
        let a = Tensor<Float>.zeros([2, 3])
        let b = a.tanh()

        XCTAssertEqual(b.shape, [2, 3])
    }
}

final class TensorFunctionalTests: XCTestCase {

    func testFunctionalRelu() {
        let x = Tensor<Float>.zeros([2, 3])
        let y = nn.functional.relu(x)

        XCTAssertEqual(y.shape, x.shape)
    }

    func testFunctionalSigmoid() {
        let x = Tensor<Float>.zeros([2, 3])
        let y = nn.functional.sigmoid(x)

        XCTAssertEqual(y.shape, x.shape)
    }
}

final class TensorChainedOperationsTests: XCTestCase {

    func testLinearLayerSimulation() {
        // Simulates: y = relu(x @ w + b)
        let x = Tensor<Float>.zeros([32, 784])
        let w = Tensor<Float>.zeros([784, 256])
        let b = Tensor<Float>.zeros([32, 256])

        let y = (x.matmul(w) + b).relu()

        XCTAssertEqual(y.shape, [32, 256])
    }

    func testMultipleOperations() {
        let a = Tensor<Float>.zeros([10, 20])
        let b = Tensor<Float>.ones([10, 20])

        let result = ((a + b) * b).mean()

        XCTAssertEqual(result.shape, [])
    }
}

// MARK: - Scan and While Loop Tests

final class ScanTests: XCTestCase {

    func testScanCounter() {
        // Test basic scan with simple counter
        let initial = Tensor<Float>([0.0], shape: [1])
        let iterations = 10

        let final = scan(iterations: iterations, initialState: initial) { state in
            state + Tensor<Float>([1.0], shape: [1])
        }

        XCTAssertEqual(final.shape, [1])

        // Verify IR structure
        // For now scan uses unrolling, so we should see the result
        final.markForMaterialization()
        LazyTensorBarrier()

        let values = final.scalars()
        XCTAssertEqual(values.count, 1)
        XCTAssertEqual(values[0], Float(iterations), accuracy: 0.001)
    }

    func testScanDoubling() {
        // Test scan with multiplication
        let initial = Tensor<Float>([1.0], shape: [1])

        let final = scan(iterations: 5, initialState: initial) { state in
            state * Tensor<Float>([2.0], shape: [1])
        }

        final.markForMaterialization()
        LazyTensorBarrier()

        let values = final.scalars()
        XCTAssertEqual(values[0], 32.0, accuracy: 0.001) // 1 * 2^5 = 32
    }

    func testScanXLATensor() {
        // Test the XLA tensor scan variant
        let initial = Tensor<Float>([0.0], shape: [1])

        let final = scanXLATensor(iterations: 10, initial: initial) { x in
            x + Tensor<Float>([1.0], shape: [1])
        }

        XCTAssertEqual(final.shape, [1])

        // This uses traced while loop - the IR should contain a whileLoopTraced node
        if case .whileLoopTraced(let iterations, _, _, _, _) = final.handle.irNode {
            XCTAssertEqual(iterations, 10)
        } else {
            XCTFail("Expected whileLoopTraced node")
        }
    }

    func testScanXLATensorExecution() {
        // Test that XLA while loop actually executes correctly
        let initial = Tensor<Float>([1.0], shape: [1])

        let final = scanXLATensor(iterations: 5, initial: initial) { x in
            x * Tensor<Float>([2.0], shape: [1])
        }

        // Verify IR structure
        if case .whileLoopTraced(let iterations, _, _, _, _) = final.handle.irNode {
            XCTAssertEqual(iterations, 5)
        } else {
            XCTFail("Expected whileLoopTraced node")
        }

        // Execute and verify result
        final.markForMaterialization()
        LazyTensorBarrier()

        let values = final.scalars()
        XCTAssertEqual(values.count, 1)
        XCTAssertEqual(values[0], 32.0, accuracy: 0.001) // 1 * 2^5 = 32
    }
}

// MARK: - Tensor Helper Operation Tests (Shape Verification - No XLA Required)

final class TensorEyeTests: XCTestCase {

    func testEyeShape() {
        let I = Tensor<Float>.eye(3)
        XCTAssertEqual(I.shape, [3, 3])
        XCTAssertEqual(I.dtype, .float32)
    }

    func testEyeLargerShape() {
        let I = Tensor<Float>.eye(5)
        XCTAssertEqual(I.shape, [5, 5])
        XCTAssertEqual(I.dtype, .float32)
    }

    func testEyeIRStructure() {
        // Verify eye creates comparison-based IR
        let I = Tensor<Float>.eye(3)
        // Eye uses equalTo which creates a comparison operation
        if case .operation(let op, _, _) = I.handle.irNode {
            XCTAssertEqual(op, .equal)  // equalTo uses .equal operation
        } else {
            XCTFail("Expected operation node for eye")
        }
    }
}

final class TensorLinspaceTests: XCTestCase {

    func testLinspaceShape() {
        let x = Tensor<Float>.linspace(0, 1, steps: 5)
        XCTAssertEqual(x.shape, [5])
    }

    func testLinspaceShapeLarge() {
        let x = Tensor<Float>.linspace(0, 100, steps: 101)
        XCTAssertEqual(x.shape, [101])
    }

    func testLinspaceSingleStepShape() {
        let x = Tensor<Float>.linspace(5, 10, steps: 1)
        XCTAssertEqual(x.shape, [1])
    }

    func testLinspaceIRStructure() {
        let x = Tensor<Float>.linspace(0, 1, steps: 5)
        // linspace uses arange + arithmetic operations
        if case .operation(let op, _, _) = x.handle.irNode {
            XCTAssertEqual(op, .add)  // Final operation is addition
        } else {
            XCTFail("Expected operation node for linspace")
        }
    }
}

final class TensorTriangularTests: XCTestCase {

    func testTrilShape() {
        let x = Tensor<Float>.ones([3, 3])
        let lower = x.tril()
        XCTAssertEqual(lower.shape, [3, 3])
    }

    func testTriuShape() {
        let x = Tensor<Float>.ones([3, 3])
        let upper = x.triu()
        XCTAssertEqual(upper.shape, [3, 3])
    }

    func testTrilNonSquareShape() {
        let x = Tensor<Float>.ones([2, 4])
        let lower = x.tril()
        XCTAssertEqual(lower.shape, [2, 4])
    }

    func testTriuNonSquareShape() {
        let x = Tensor<Float>.ones([4, 2])
        let upper = x.triu()
        XCTAssertEqual(upper.shape, [4, 2])
    }

    func testTrilIRStructure() {
        let x = Tensor<Float>.ones([3, 3])
        let lower = x.tril()
        // tril uses multiplication with mask
        if case .operation(let op, _, _) = lower.handle.irNode {
            XCTAssertEqual(op, .multiply)
        } else {
            XCTFail("Expected multiply operation for tril")
        }
    }
}

final class TensorArgmaxArgminTests: XCTestCase {

    func testArgmaxGlobalShape() {
        let x = Tensor<Float>([1, 3, 2, 3], shape: [4])
        let idx = x.argmax()
        // Global argmax should return scalar-like shape
        XCTAssertEqual(idx.shape, [])  // Scalar
    }

    func testArgmaxDimShape() {
        // [[1, 2, 3], [6, 5, 4]]
        let x = Tensor<Float>([1, 2, 3, 6, 5, 4], shape: [2, 3])
        let rowMax = x.argmax(dim: 1)  // Max along columns for each row
        XCTAssertEqual(rowMax.shape, [2])  // One index per row
    }

    func testArgmaxDim0Shape() {
        // [[1, 2, 3], [6, 5, 4]]
        let x = Tensor<Float>([1, 2, 3, 6, 5, 4], shape: [2, 3])
        let colMax = x.argmax(dim: 0)  // Max along rows for each column
        XCTAssertEqual(colMax.shape, [3])  // One index per column
    }

    func testArgminGlobalShape() {
        let x = Tensor<Float>([3, 1, 2, 1], shape: [4])
        let idx = x.argmin()
        XCTAssertEqual(idx.shape, [])  // Scalar
    }

    func testArgminDimShape() {
        // [[3, 2, 1], [4, 5, 6]]
        let x = Tensor<Float>([3, 2, 1, 4, 5, 6], shape: [2, 3])
        let rowMin = x.argmin(dim: 1)
        XCTAssertEqual(rowMin.shape, [2])
    }

    func testArgmaxNegativeDimShape() {
        let x = Tensor<Float>([1, 2, 3, 6, 5, 4], shape: [2, 3])
        let rowMax = x.argmax(dim: -1)  // Same as dim: 1
        XCTAssertEqual(rowMax.shape, [2])
    }

    func testArgmax3DShape() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let result = x.argmax(dim: 1)
        XCTAssertEqual(result.shape, [2, 4])  // Remove dim 1
    }
}

final class TensorSqueezeTests: XCTestCase {

    func testSqueezeDim() {
        let x = Tensor<Float>.ones([2, 1, 3])
        let squeezed = x.squeeze(dim: 1)
        XCTAssertEqual(squeezed.shape, [2, 3])
    }

    func testSqueezeAll() {
        let x = Tensor<Float>.ones([1, 2, 1, 3, 1])
        let squeezed = x.squeeze()
        XCTAssertEqual(squeezed.shape, [2, 3])
    }

    func testSqueezeNoDimOne() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let squeezed = x.squeeze()
        XCTAssertEqual(squeezed.shape, [2, 3, 4])  // No change
    }

    func testSqueezeNegativeDim() {
        let x = Tensor<Float>.ones([2, 3, 1])
        let squeezed = x.squeeze(dim: -1)
        XCTAssertEqual(squeezed.shape, [2, 3])
    }

    func testExpandingShapeAndSqueeze() {
        let x = Tensor<Float>.ones([2, 3])
        let expanded = x.expandingShape(at: 1)
        XCTAssertEqual(expanded.shape, [2, 1, 3])
        let squeezed = expanded.squeeze(dim: 1)
        XCTAssertEqual(squeezed.shape, [2, 3])
    }

    func testSqueezeFirstDim() {
        let x = Tensor<Float>.ones([1, 3, 4])
        let squeezed = x.squeeze(dim: 0)
        XCTAssertEqual(squeezed.shape, [3, 4])
    }
}

final class TensorMinAlongAxesTests: XCTestCase {

    func testMinAlongAxisShape() {
        let x = Tensor<Float>([3, 1, 4, 1, 5, 9], shape: [2, 3])
        let minVals = x.min(alongAxes: [1])  // Min along last axis
        XCTAssertEqual(minVals.shape, [2])
    }

    func testMinAlongAxis0Shape() {
        let x = Tensor<Float>([3, 1, 4, 1, 5, 9], shape: [2, 3])
        let minVals = x.min(alongAxes: [0])  // Min along first axis
        XCTAssertEqual(minVals.shape, [3])
    }

    func testMinAlongMultipleAxesShape() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let minVals = x.min(alongAxes: [0, 2])
        XCTAssertEqual(minVals.shape, [3])
    }

    func testMinIRStructure() {
        let x = Tensor<Float>([3, 1, 4, 1, 5, 9], shape: [2, 3])
        let minVals = x.min(alongAxes: [1])
        if case .operation(let op, _, _) = minVals.handle.irNode {
            XCTAssertEqual(op, .reduceMin)
        } else {
            XCTFail("Expected reduceMin operation")
        }
    }
}
