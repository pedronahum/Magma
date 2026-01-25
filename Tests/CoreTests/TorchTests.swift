// Magma - Torch Tests
// End-to-end tests for the PyTorch-compatible API
// These require XLA for full functionality

import Testing
@testable import Magma
@testable import LazyTensor
@testable import StableHLO

@Suite("Tensor Creation Tests")
struct TensorCreationTests {

    @Test("Zeros")
    func zeros() {
        let t = Tensor<Float>.zeros([2, 3])
        #expect(t.shape == [2, 3])
        #expect(t.dtype == .float32)
        #expect(t.rank == 2)
        #expect(t.elementCount == 6)
    }

    @Test("Ones")
    func ones() {
        let t = Tensor<Float>.ones([3, 4, 5])
        #expect(t.shape == [3, 4, 5])
        #expect(t.elementCount == 60)
    }

    @Test("Device")
    func device() {
        let t = Tensor<Float>.zeros([2, 2], on: .default)
        #expect(t.device == .default)
        #expect(t.device.backend == .cpu)
    }
}

@Suite("Tensor Arithmetic Tests")
struct TensorArithmeticTests {

    @Test("Add")
    func add() {
        let a = Tensor<Float>.zeros([2, 3])
        let b = Tensor<Float>.ones([2, 3])
        let c = a + b

        #expect(c.shape == [2, 3])
        #expect(c.dtype == .float32)

        // Verify IR node is created
        if case .operation(let op, let inputs, _) = c.handle.irNode {
            #expect(op == .add)
            #expect(inputs.count == 2)
        } else {
            Issue.record("Expected operation node")
        }
    }

    @Test("Subtract")
    func subtract() {
        let a = Tensor<Float>.zeros([2, 3])
        let b = Tensor<Float>.ones([2, 3])
        let c = a - b

        #expect(c.shape == [2, 3])
    }

    @Test("Multiply")
    func multiply() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.ones([2, 3])
        let c = a * b

        #expect(c.shape == [2, 3])
    }

    @Test("Divide")
    func divide() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.ones([2, 3])
        let c = a / b

        #expect(c.shape == [2, 3])
    }
}

@Suite("Tensor Matrix Tests")
struct TensorMatrixTests {

    @Test("Matmul")
    func matmul() {
        let a = Tensor<Float>.zeros([32, 784])
        let b = Tensor<Float>.zeros([784, 256])
        let c = a.matmul(b)

        #expect(c.shape == [32, 256])
    }

    @Test("Transpose")
    func transpose() {
        let a = Tensor<Float>.zeros([3, 4])
        let b = a.transpose()

        #expect(b.shape == [4, 3])
    }

    @Test("Reshape")
    func reshape() {
        let a = Tensor<Float>.zeros([2, 3, 4])
        let b = a.reshape([6, 4])

        #expect(b.shape == [6, 4])
        #expect(b.elementCount == 24)
    }
}

@Suite("Tensor Reduction Tests")
struct TensorReductionTests {

    @Test("Sum")
    func sum() {
        let a = Tensor<Float>.ones([2, 3])
        let b = a.sum()

        #expect(b.shape == [])  // Scalar
        #expect(b.rank == 0)
    }

    @Test("Mean")
    func mean() {
        let a = Tensor<Float>.ones([2, 3])
        let b = a.mean()

        #expect(b.shape == [])
    }

    @Test("Max")
    func max() {
        let a = Tensor<Float>.ones([2, 3])
        let b = a.max()

        #expect(b.shape == [])
    }
}

@Suite("Tensor Activation Tests")
struct TensorActivationTests {

    @Test("ReLU")
    func relu() {
        let a = Tensor<Float>.zeros([2, 3])
        let b = a.relu()

        #expect(b.shape == [2, 3])

        if case .operation(let op, _, _) = b.handle.irNode {
            #expect(op == .relu)
        } else {
            Issue.record("Expected relu operation")
        }
    }

    @Test("Sigmoid")
    func sigmoid() {
        let a = Tensor<Float>.zeros([2, 3])
        let b = a.sigmoid()

        #expect(b.shape == [2, 3])
    }

    @Test("Tanh")
    func tanh() {
        let a = Tensor<Float>.zeros([2, 3])
        let b = a.tanh()

        #expect(b.shape == [2, 3])
    }
}

@Suite("Tensor Functional Tests")
struct TensorFunctionalTests {

    @Test("Functional ReLU")
    func functionalRelu() {
        let x = Tensor<Float>.zeros([2, 3])
        let y = nn.functional.relu(x)

        #expect(y.shape == x.shape)
    }

    @Test("Functional Sigmoid")
    func functionalSigmoid() {
        let x = Tensor<Float>.zeros([2, 3])
        let y = nn.functional.sigmoid(x)

        #expect(y.shape == x.shape)
    }
}

@Suite("Tensor Chained Operations Tests")
struct TensorChainedOperationsTests {

    @Test("Linear layer simulation")
    func linearLayerSimulation() {
        // Simulates: y = relu(x @ w + b)
        let x = Tensor<Float>.zeros([32, 784])
        let w = Tensor<Float>.zeros([784, 256])
        let b = Tensor<Float>.zeros([32, 256])

        let y = (x.matmul(w) + b).relu()

        #expect(y.shape == [32, 256])
    }

    @Test("Multiple operations")
    func multipleOperations() {
        let a = Tensor<Float>.zeros([10, 20])
        let b = Tensor<Float>.ones([10, 20])

        let result = ((a + b) * b).mean()

        #expect(result.shape == [])
    }
}

// MARK: - Scan and While Loop Tests

@Suite("Scan Tests")
struct ScanTests {

    @Test("Scan counter")
    func scanCounter() {
        // Test basic scan with simple counter
        let initial = Tensor<Float>([0.0], shape: [1])
        let iterations = 10

        let final = scan(iterations: iterations, initialState: initial) { state in
            state + Tensor<Float>([1.0], shape: [1])
        }

        #expect(final.shape == [1])

        // Verify IR structure
        // For now scan uses unrolling, so we should see the result
        final.markForMaterialization()
        LazyTensorBarrier()

        let values = final.scalars()
        #expect(values.count == 1)
        #expect(abs(values[0] - Float(iterations)) < 0.001)
    }

    @Test("Scan doubling")
    func scanDoubling() {
        // Test scan with multiplication
        let initial = Tensor<Float>([1.0], shape: [1])

        let final = scan(iterations: 5, initialState: initial) { state in
            state * Tensor<Float>([2.0], shape: [1])
        }

        final.markForMaterialization()
        LazyTensorBarrier()

        let values = final.scalars()
        #expect(abs(values[0] - 32.0) < 0.001) // 1 * 2^5 = 32
    }

    @Test("Scan XLA tensor")
    func testScanXLATensor() {
        // Test the XLA tensor scan variant
        let initial = Tensor<Float>([0.0], shape: [1])

        let final = Magma.scanXLATensor(iterations: 10, initial: initial) { x in
            x + Tensor<Float>([1.0], shape: [1])
        }

        #expect(final.shape == [1])

        // This uses traced while loop - the IR should contain a whileLoopTraced node
        if case .whileLoopTraced(let iterations, _, _, _, _) = final.handle.irNode {
            #expect(iterations == 10)
        } else {
            Issue.record("Expected whileLoopTraced node")
        }
    }

    @Test("Scan XLA tensor execution")
    func testScanXLATensorExecution() {
        // Test that XLA while loop actually executes correctly
        let initial = Tensor<Float>([1.0], shape: [1])

        let final = Magma.scanXLATensor(iterations: 5, initial: initial) { x in
            x * Tensor<Float>([2.0], shape: [1])
        }

        // Verify IR structure
        if case .whileLoopTraced(let iterations, _, _, _, _) = final.handle.irNode {
            #expect(iterations == 5)
        } else {
            Issue.record("Expected whileLoopTraced node")
        }

        // Execute and verify result
        final.markForMaterialization()
        LazyTensorBarrier()

        let values = final.scalars()
        #expect(values.count == 1)
        #expect(abs(values[0] - 32.0) < 0.001) // 1 * 2^5 = 32
    }
}

// MARK: - Tensor Helper Operation Tests (Shape Verification - No XLA Required)

@Suite("Tensor Eye Tests")
struct TensorEyeTests {

    @Test("Eye shape")
    func eyeShape() {
        let I = Tensor<Float>.eye(3)
        #expect(I.shape == [3, 3])
        #expect(I.dtype == .float32)
    }

    @Test("Eye larger shape")
    func eyeLargerShape() {
        let I = Tensor<Float>.eye(5)
        #expect(I.shape == [5, 5])
        #expect(I.dtype == .float32)
    }

    @Test("Eye IR structure")
    func eyeIRStructure() {
        // Verify eye creates comparison-based IR
        let I = Tensor<Float>.eye(3)
        // Eye uses equalTo which creates a comparison operation
        if case .operation(let op, _, _) = I.handle.irNode {
            #expect(op == .equal)  // equalTo uses .equal operation
        } else {
            Issue.record("Expected operation node for eye")
        }
    }
}

@Suite("Tensor Linspace Tests")
struct TensorLinspaceTests {

    @Test("Linspace shape")
    func linspaceShape() {
        let x = Tensor<Float>.linspace(0, 1, steps: 5)
        #expect(x.shape == [5])
    }

    @Test("Linspace shape large")
    func linspaceShapeLarge() {
        let x = Tensor<Float>.linspace(0, 100, steps: 101)
        #expect(x.shape == [101])
    }

    @Test("Linspace single step shape")
    func linspaceSingleStepShape() {
        let x = Tensor<Float>.linspace(5, 10, steps: 1)
        #expect(x.shape == [1])
    }

    @Test("Linspace IR structure")
    func linspaceIRStructure() {
        let x = Tensor<Float>.linspace(0, 1, steps: 5)
        // linspace uses arange + arithmetic operations
        if case .operation(let op, _, _) = x.handle.irNode {
            #expect(op == .add)  // Final operation is addition
        } else {
            Issue.record("Expected operation node for linspace")
        }
    }
}

@Suite("Tensor Triangular Tests")
struct TensorTriangularTests {

    @Test("Tril shape")
    func trilShape() {
        let x = Tensor<Float>.ones([3, 3])
        let lower = x.tril()
        #expect(lower.shape == [3, 3])
    }

    @Test("Triu shape")
    func triuShape() {
        let x = Tensor<Float>.ones([3, 3])
        let upper = x.triu()
        #expect(upper.shape == [3, 3])
    }

    @Test("Tril non-square shape")
    func trilNonSquareShape() {
        let x = Tensor<Float>.ones([2, 4])
        let lower = x.tril()
        #expect(lower.shape == [2, 4])
    }

    @Test("Triu non-square shape")
    func triuNonSquareShape() {
        let x = Tensor<Float>.ones([4, 2])
        let upper = x.triu()
        #expect(upper.shape == [4, 2])
    }

    @Test("Tril IR structure")
    func trilIRStructure() {
        let x = Tensor<Float>.ones([3, 3])
        let lower = x.tril()
        // tril uses multiplication with mask
        if case .operation(let op, _, _) = lower.handle.irNode {
            #expect(op == .multiply)
        } else {
            Issue.record("Expected multiply operation for tril")
        }
    }
}

@Suite("Tensor Argmax Argmin Tests")
struct TensorArgmaxArgminTests {

    @Test("Argmax global shape")
    func argmaxGlobalShape() {
        let x = Tensor<Float>([1, 3, 2, 3], shape: [4])
        let idx = x.argmax()
        // Global argmax should return scalar-like shape
        #expect(idx.shape == [])  // Scalar
    }

    @Test("Argmax dim shape")
    func argmaxDimShape() {
        // [[1, 2, 3], [6, 5, 4]]
        let x = Tensor<Float>([1, 2, 3, 6, 5, 4], shape: [2, 3])
        let rowMax = x.argmax(dim: 1)  // Max along columns for each row
        #expect(rowMax.shape == [2])  // One index per row
    }

    @Test("Argmax dim 0 shape")
    func argmaxDim0Shape() {
        // [[1, 2, 3], [6, 5, 4]]
        let x = Tensor<Float>([1, 2, 3, 6, 5, 4], shape: [2, 3])
        let colMax = x.argmax(dim: 0)  // Max along rows for each column
        #expect(colMax.shape == [3])  // One index per column
    }

    @Test("Argmin global shape")
    func argminGlobalShape() {
        let x = Tensor<Float>([3, 1, 2, 1], shape: [4])
        let idx = x.argmin()
        #expect(idx.shape == [])  // Scalar
    }

    @Test("Argmin dim shape")
    func argminDimShape() {
        // [[3, 2, 1], [4, 5, 6]]
        let x = Tensor<Float>([3, 2, 1, 4, 5, 6], shape: [2, 3])
        let rowMin = x.argmin(dim: 1)
        #expect(rowMin.shape == [2])
    }

    @Test("Argmax negative dim shape")
    func argmaxNegativeDimShape() {
        let x = Tensor<Float>([1, 2, 3, 6, 5, 4], shape: [2, 3])
        let rowMax = x.argmax(dim: -1)  // Same as dim: 1
        #expect(rowMax.shape == [2])
    }

    @Test("Argmax 3D shape")
    func argmax3DShape() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let result = x.argmax(dim: 1)
        #expect(result.shape == [2, 4])  // Remove dim 1
    }
}

@Suite("Tensor Squeeze Tests")
struct TensorSqueezeTests {

    @Test("Squeeze dim")
    func squeezeDim() {
        let x = Tensor<Float>.ones([2, 1, 3])
        let squeezed = x.squeeze(dim: 1)
        #expect(squeezed.shape == [2, 3])
    }

    @Test("Squeeze all")
    func squeezeAll() {
        let x = Tensor<Float>.ones([1, 2, 1, 3, 1])
        let squeezed = x.squeeze()
        #expect(squeezed.shape == [2, 3])
    }

    @Test("Squeeze no dim one")
    func squeezeNoDimOne() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let squeezed = x.squeeze()
        #expect(squeezed.shape == [2, 3, 4])  // No change
    }

    @Test("Squeeze negative dim")
    func squeezeNegativeDim() {
        let x = Tensor<Float>.ones([2, 3, 1])
        let squeezed = x.squeeze(dim: -1)
        #expect(squeezed.shape == [2, 3])
    }

    @Test("Expanding shape and squeeze")
    func expandingShapeAndSqueeze() {
        let x = Tensor<Float>.ones([2, 3])
        let expanded = x.expandingShape(at: 1)
        #expect(expanded.shape == [2, 1, 3])
        let squeezed = expanded.squeeze(dim: 1)
        #expect(squeezed.shape == [2, 3])
    }

    @Test("Squeeze first dim")
    func squeezeFirstDim() {
        let x = Tensor<Float>.ones([1, 3, 4])
        let squeezed = x.squeeze(dim: 0)
        #expect(squeezed.shape == [3, 4])
    }
}

@Suite("Tensor Min Along Axes Tests")
struct TensorMinAlongAxesTests {

    @Test("Min along axis shape")
    func minAlongAxisShape() {
        let x = Tensor<Float>([3, 1, 4, 1, 5, 9], shape: [2, 3])
        let minVals = x.min(alongAxes: [1])  // Min along last axis
        #expect(minVals.shape == [2])
    }

    @Test("Min along axis 0 shape")
    func minAlongAxis0Shape() {
        let x = Tensor<Float>([3, 1, 4, 1, 5, 9], shape: [2, 3])
        let minVals = x.min(alongAxes: [0])  // Min along first axis
        #expect(minVals.shape == [3])
    }

    @Test("Min along multiple axes shape")
    func minAlongMultipleAxesShape() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let minVals = x.min(alongAxes: [0, 2])
        #expect(minVals.shape == [3])
    }

    @Test("Min IR structure")
    func minIRStructure() {
        let x = Tensor<Float>([3, 1, 4, 1, 5, 9], shape: [2, 3])
        let minVals = x.min(alongAxes: [1])
        if case .operation(let op, _, _) = minVals.handle.irNode {
            #expect(op == .reduceMin)
        } else {
            Issue.record("Expected reduceMin operation")
        }
    }
}
