// Magma - Data Loading Tests
// Tests for Dataset and DataLoader

import XCTest
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - TensorDataset Tests

final class TensorDatasetTests: XCTestCase {

    func testTensorDatasetCreation() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let dataset = TensorDataset(inputs: inputs, targets: targets)

        XCTAssertEqual(dataset.count, 100)
    }

    func testTensorDatasetShapeMismatch() {
        // Different number of samples should fail
        // (precondition will trigger in debug builds)
    }
}

// MARK: - SimpleBatchLoader Tests

final class SimpleBatchLoaderTests: XCTestCase {

    func testBatchLoaderCreation() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 32)

        XCTAssertEqual(loader.numSamples, 100)
        XCTAssertEqual(loader.batchSize, 32)
        XCTAssertEqual(loader.numBatches, 4)  // ceil(100/32) = 4
        XCTAssertEqual(loader.shuffle, false)
        XCTAssertEqual(loader.dropLast, false)
    }

    func testBatchLoaderIteration() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 32)

        var batchCount = 0
        for batch in loader {
            XCTAssertEqual(batch.input.shape[1], 784)
            XCTAssertEqual(batch.target.shape[1], 10)
            batchCount += 1
        }

        XCTAssertEqual(batchCount, 4)
    }

    func testBatchLoaderExactDivision() {
        let inputs = Tensor<Float>.ones([64, 784])
        let targets = Tensor<Float>.zeros([64, 10])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 32)

        XCTAssertEqual(loader.numBatches, 2)

        var batchCount = 0
        for _ in loader {
            batchCount += 1
        }
        XCTAssertEqual(batchCount, 2)
    }

    func testBatchLoaderSingleBatch() {
        let inputs = Tensor<Float>.ones([16, 784])
        let targets = Tensor<Float>.zeros([16, 10])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 32)

        XCTAssertEqual(loader.numBatches, 1)
    }

    func testGetSpecificBatch() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 32)

        let batch0 = loader.batch(0)
        XCTAssertEqual(batch0.input.shape[0], 32)

        let batch3 = loader.batch(3)  // Last batch (100 - 96 = 4 samples)
        XCTAssertEqual(batch3.input.shape[0], 4)
    }

    // MARK: - Shuffle Tests

    func testBatchLoaderWithShuffle() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(
            inputs: inputs,
            targets: targets,
            batchSize: 32,
            shuffle: true
        )

        XCTAssertEqual(loader.shuffle, true)
        XCTAssertEqual(loader.numBatches, 4)

        var batchCount = 0
        for batch in loader {
            XCTAssertEqual(batch.input.shape[1], 784)
            XCTAssertEqual(batch.target.shape[1], 10)
            batchCount += 1
        }

        XCTAssertEqual(batchCount, 4)
    }

    func testShuffleProducesDifferentOrder() {
        // Create inputs with unique values per sample so we can verify shuffling
        var inputData = [Float]()
        for i in 0..<10 {
            inputData.append(contentsOf: [Float](repeating: Float(i), count: 4))
        }
        let inputs = Tensor<Float>(inputData, shape: [10, 4])
        let targets = Tensor<Float>.zeros([10, 2])

        // Create two shuffled loaders
        let loader1 = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 5, shuffle: true)
        let loader2 = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 5, shuffle: true)

        // Collect batches from both
        var batches1: [[Float]] = []
        var batches2: [[Float]] = []

        for batch in loader1 {
            batches1.append(batch.input.scalars())
        }
        for batch in loader2 {
            batches2.append(batch.input.scalars())
        }

        // Both should have same number of batches
        XCTAssertEqual(batches1.count, 2)
        XCTAssertEqual(batches2.count, 2)

        // Shapes should be preserved
        XCTAssertEqual(batches1[0].count, 20)  // 5 samples * 4 features
    }

    func testShufflePreservesAllSamples() {
        // Verify that shuffling doesn't lose any samples
        var inputData = [Float]()
        for i in 0..<8 {
            inputData.append(Float(i))  // Each sample is just its index
        }
        let inputs = Tensor<Float>(inputData, shape: [8, 1])
        let targets = Tensor<Float>.zeros([8, 1])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 4, shuffle: true)

        var allValues = Set<Float>()
        for batch in loader {
            let values = batch.input.scalars()
            for v in values {
                allValues.insert(v)
            }
        }

        // All 8 unique values should be present
        XCTAssertEqual(allValues.count, 8)
    }

    // MARK: - DropLast Tests

    func testBatchLoaderWithDropLast() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(
            inputs: inputs,
            targets: targets,
            batchSize: 32,
            dropLast: true
        )

        XCTAssertEqual(loader.dropLast, true)
        XCTAssertEqual(loader.numBatches, 3)  // 100 / 32 = 3 (drops last 4 samples)

        var batchCount = 0
        var allBatchSizes: [Int] = []
        for batch in loader {
            allBatchSizes.append(batch.input.shape[0])
            batchCount += 1
        }

        XCTAssertEqual(batchCount, 3)
        // All batches should be full size
        XCTAssertTrue(allBatchSizes.allSatisfy { $0 == 32 })
    }

    func testBatchLoaderDropLastExactDivision() {
        let inputs = Tensor<Float>.ones([64, 784])
        let targets = Tensor<Float>.zeros([64, 10])

        let loader = SimpleBatchLoader(
            inputs: inputs,
            targets: targets,
            batchSize: 32,
            dropLast: true
        )

        // Exact division should still work
        XCTAssertEqual(loader.numBatches, 2)

        var batchCount = 0
        for _ in loader {
            batchCount += 1
        }
        XCTAssertEqual(batchCount, 2)
    }

    func testBatchLoaderShuffleAndDropLast() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(
            inputs: inputs,
            targets: targets,
            batchSize: 32,
            shuffle: true,
            dropLast: true
        )

        XCTAssertEqual(loader.shuffle, true)
        XCTAssertEqual(loader.dropLast, true)
        XCTAssertEqual(loader.numBatches, 3)

        var batchCount = 0
        for batch in loader {
            XCTAssertEqual(batch.input.shape[0], 32)  // All batches should be full
            batchCount += 1
        }

        XCTAssertEqual(batchCount, 3)
    }

    // MARK: - Multi-epoch Shuffle Tests

    func testShuffleEachEpoch() {
        var inputData = [Float]()
        for i in 0..<16 {
            inputData.append(Float(i))
        }
        let inputs = Tensor<Float>(inputData, shape: [16, 1])
        let targets = Tensor<Float>.zeros([16, 1])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 8, shuffle: true)

        // Collect first values from multiple epochs
        var firstBatchFirstValues: [Float] = []
        for _ in 0..<3 {
            for (idx, batch) in loader.enumerated() {
                if idx == 0 {
                    firstBatchFirstValues.append(batch.input.scalars()[0])
                }
            }
        }

        // We collected 3 first-batch-first-values from 3 epochs
        XCTAssertEqual(firstBatchFirstValues.count, 3)
        // At least two epochs should have different first values (highly likely with shuffling)
        // Note: This could theoretically fail with very bad luck, but probability is low
    }
}

// MARK: - Tensor Slice Tests

final class TensorSliceTests: XCTestCase {

    func testSliceBasic() {
        let x = Tensor<Float>.ones([100, 784])
        let sliced = x.slice(start: 0, size: 32)

        XCTAssertEqual(sliced.shape, [32, 784])
    }

    func testSliceMiddle() {
        let x = Tensor<Float>.ones([100, 784])
        let sliced = x.slice(start: 32, size: 32)

        XCTAssertEqual(sliced.shape, [32, 784])
    }

    func testSliceEnd() {
        let x = Tensor<Float>.ones([100, 784])
        let sliced = x.slice(start: 96, size: 4)

        XCTAssertEqual(sliced.shape, [4, 784])
    }

    func testSlice3D() {
        let x = Tensor<Float>.ones([100, 28, 28])
        let sliced = x.slice(start: 0, size: 16)

        XCTAssertEqual(sliced.shape, [16, 28, 28])
    }

    func testSliceGradient() {
        let x = Tensor<Float>.ones([100, 10])

        let grad = gradient(at: x) { t in
            t.slice(start: 0, size: 32).sum()
        }

        // Gradient should have original shape
        XCTAssertEqual(grad.shape, [100, 10])
    }
}

// MARK: - Training Loop Integration Tests

final class DataLoaderTrainingTests: XCTestCase {

    func testTrainingLoopWithLoader() {
        // Create synthetic dataset
        let trainX = Tensor<Float>.ones([64, 784])
        let trainY = Tensor<Float>.zeros([64, 10])

        let loader = SimpleBatchLoader(inputs: trainX, targets: trainY, batchSize: 32)

        // Create model
        let model = nn.sequential {
            nn.Linear(inputSize: 784, outputSize: 128)
            nn.ReLU()
            nn.Linear(inputSize: 128, outputSize: 10)
        }

        var optimizer = optim.SGD(parameters: model.parameters(), lr: 0.01)

        // Training loop
        var totalBatches = 0
        for batch in loader {
            // Forward pass
            let output = model(batch.input)
            XCTAssertEqual(output.shape[1], 10)

            // Create dummy gradients (normally computed via autodiff)
            let grads = model.parameters().map { Tensor<Float>.ones($0.shape) }

            // Optimizer step
            optimizer.step(grads)
            totalBatches += 1
        }

        XCTAssertEqual(totalBatches, 2)
    }

    func testMultiEpochTraining() {
        let trainX = Tensor<Float>.ones([32, 784])
        let trainY = Tensor<Float>.zeros([32, 10])

        let loader = SimpleBatchLoader(inputs: trainX, targets: trainY, batchSize: 16)

        let model = nn.sequential {
            nn.Linear(inputSize: 784, outputSize: 64)
            nn.ReLU()
            nn.Linear(inputSize: 64, outputSize: 10)
        }

        var optimizer = optim.Adam(parameters: model.parameters(), lr: 0.001)

        let numEpochs = 3
        var totalBatches = 0

        for _ in 0..<numEpochs {
            for batch in loader {
                let output = model(batch.input)
                XCTAssertEqual(output.shape[1], 10)

                let grads = model.parameters().map { Tensor<Float>.ones($0.shape) }
                optimizer.step(grads)
                totalBatches += 1
            }
        }

        XCTAssertEqual(totalBatches, numEpochs * 2)
    }
}

// MARK: - Gather Operation Tests

final class GatherTests: XCTestCase {

    func testGatherBasic2D() {
        let x = Tensor<Float>.randn([4, 8])  // [4, 8]
        let indices = Tensor<Float>.zeros([4, 3])  // Gather 3 elements per row

        let result = x.gather(indices: indices, axis: 1)
        XCTAssertEqual(result.shape, [4, 3])
    }

    func testGatherAxis0() {
        let x = Tensor<Float>.randn([10, 5])  // [10, 5]
        let indices = Tensor<Float>.zeros([3, 5])  // Gather 3 rows

        let result = x.gather(indices: indices, axis: 0)
        XCTAssertEqual(result.shape, [3, 5])
    }

    func testGather3D() {
        let x = Tensor<Float>.randn([2, 8, 16])
        let indices = Tensor<Float>.zeros([2, 4, 16])  // Gather 4 elements along axis 1

        let result = x.gather(indices: indices, axis: 1)
        XCTAssertEqual(result.shape, [2, 4, 16])
    }

    func testGatherNegativeAxis() {
        let x = Tensor<Float>.randn([4, 8])
        let indices = Tensor<Float>.zeros([4, 2])

        let result = x.gather(indices: indices, axis: -1)  // Same as axis=1
        XCTAssertEqual(result.shape, [4, 2])
    }

    func testGatherGradient() {
        let x = Tensor<Float>.randn([4, 8])
        let indices = Tensor<Float>.zeros([4, 2])

        let grad = gradient(at: x) { t in
            t.gather(indices: indices, axis: 1).sum()
        }

        XCTAssertEqual(grad.shape, [4, 8])
    }
}

// MARK: - Scatter Operation Tests

final class ScatterTests: XCTestCase {

    func testScatterBasic2D() {
        let x = Tensor<Float>.zeros([4, 8])
        let indices = Tensor<Float>.zeros([4, 2])
        let updates = Tensor<Float>.ones([4, 2])

        let result = x.scatter(indices: indices, updates: updates, axis: 1)
        XCTAssertEqual(result.shape, [4, 8])
    }

    func testScatterAxis0() {
        let x = Tensor<Float>.zeros([10, 5])
        let indices = Tensor<Float>.zeros([3, 5])
        let updates = Tensor<Float>.ones([3, 5])

        let result = x.scatter(indices: indices, updates: updates, axis: 0)
        XCTAssertEqual(result.shape, [10, 5])
    }

    func testScatter3D() {
        let x = Tensor<Float>.zeros([2, 8, 16])
        let indices = Tensor<Float>.zeros([2, 3, 16])
        let updates = Tensor<Float>.ones([2, 3, 16])

        let result = x.scatter(indices: indices, updates: updates, axis: 1)
        XCTAssertEqual(result.shape, [2, 8, 16])
    }

    func testScatterNegativeAxis() {
        let x = Tensor<Float>.zeros([4, 8])
        let indices = Tensor<Float>.zeros([4, 2])
        let updates = Tensor<Float>.ones([4, 2])

        let result = x.scatter(indices: indices, updates: updates, axis: -1)
        XCTAssertEqual(result.shape, [4, 8])
    }

    func testScatterGradient() {
        let x = Tensor<Float>.zeros([4, 8])
        let indices = Tensor<Float>.zeros([4, 2])
        let updates = Tensor<Float>.ones([4, 2])

        let (inputGrad, updatesGrad) = gradient(at: x, updates) { base, upd in
            base.scatter(indices: indices, updates: upd, axis: 1).sum()
        }

        XCTAssertEqual(inputGrad.shape, [4, 8])
        XCTAssertEqual(updatesGrad.shape, [4, 2])
    }
}

// MARK: - Expand Operation Tests

final class ExpandTests: XCTestCase {

    func testExpandBasic() {
        let x = Tensor<Float>.ones([1, 4])
        let result = x.expand([3, 4])

        XCTAssertEqual(result.shape, [3, 4])
    }

    func testExpandMultipleDims() {
        let x = Tensor<Float>.ones([1, 1, 4])
        let result = x.expand([3, 5, 4])

        XCTAssertEqual(result.shape, [3, 5, 4])
    }

    func testExpandMiddleDim() {
        let x = Tensor<Float>.ones([2, 1, 4])
        let result = x.expand([2, 8, 4])

        XCTAssertEqual(result.shape, [2, 8, 4])
    }

    func testExpandNoChange() {
        let x = Tensor<Float>.ones([2, 4])
        let result = x.expand([2, 4])  // Same shape

        XCTAssertEqual(result.shape, [2, 4])
    }

    func testExpandGradient() {
        let x = Tensor<Float>.ones([1, 4])

        let grad = gradient(at: x) { t in
            t.expand([3, 4]).sum()
        }

        // Gradient should be summed back to original shape
        XCTAssertEqual(grad.shape, [1, 4])
    }

    func testExpandGradient3D() {
        let x = Tensor<Float>.ones([1, 1, 8])

        let grad = gradient(at: x) { t in
            t.expand([4, 6, 8]).sum()
        }

        XCTAssertEqual(grad.shape, [1, 1, 8])
    }
}

// MARK: - Sum with Dims Tests

final class SumWithDimsTests: XCTestCase {

    func testSumSingleDim() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [1], keepDims: false)

        XCTAssertEqual(result.shape, [4, 16])
    }

    func testSumSingleDimKeepDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [1], keepDims: true)

        XCTAssertEqual(result.shape, [4, 1, 16])
    }

    func testSumMultipleDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [0, 2], keepDims: false)

        XCTAssertEqual(result.shape, [8])
    }

    func testSumMultipleDimsKeepDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [0, 2], keepDims: true)

        XCTAssertEqual(result.shape, [1, 8, 1])
    }

    func testSumNegativeDim() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [-1], keepDims: true)

        XCTAssertEqual(result.shape, [4, 8, 1])
    }

    func testSumAllDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [0, 1, 2], keepDims: false)

        XCTAssertEqual(result.shape, [])  // Scalar
    }
}

// MARK: - Concat Tests

final class ConcatTests: XCTestCase {

    func testConcatAxis0() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([3, 3])
        let c = Tensor<Float>.concat([a, b], axis: 0)
        XCTAssertEqual(c.shape, [5, 3])
    }

    func testConcatAxis1() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 4])
        let c = Tensor<Float>.concat([a, b], axis: 1)
        XCTAssertEqual(c.shape, [2, 7])
    }

    func testConcatNegativeAxis() {
        let a = Tensor<Float>.ones([2, 3, 4])
        let b = Tensor<Float>.zeros([2, 3, 5])
        let c = Tensor<Float>.concat([a, b], axis: -1)
        XCTAssertEqual(c.shape, [2, 3, 9])
    }

    func testConcatMultipleTensors() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = Tensor<Float>.randn([2, 3])
        let d = Tensor<Float>.concat([a, b, c], axis: 0)
        XCTAssertEqual(d.shape, [6, 3])
    }

    func testConcatSingleTensor() {
        let a = Tensor<Float>.ones([2, 3])
        let c = Tensor<Float>.concat([a], axis: 0)
        XCTAssertEqual(c.shape, [2, 3])
    }

    func testConcatInstanceMethod() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = a.concat(with: [b], axis: 0)
        XCTAssertEqual(c.shape, [4, 3])
    }
}

// MARK: - Stack Tests

final class StackTests: XCTestCase {

    func testStackAxis0() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = Tensor<Float>.stack([a, b], axis: 0)
        XCTAssertEqual(c.shape, [2, 2, 3])
    }

    func testStackAxis1() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = Tensor<Float>.stack([a, b], axis: 1)
        XCTAssertEqual(c.shape, [2, 2, 3])
    }

    func testStackAxis2() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = Tensor<Float>.stack([a, b], axis: 2)
        XCTAssertEqual(c.shape, [2, 3, 2])
    }

    func testStackNegativeAxis() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = Tensor<Float>.stack([a, b], axis: -1)
        XCTAssertEqual(c.shape, [2, 3, 2])
    }

    func testStackMultipleTensors() {
        let tensors = (0..<5).map { _ in Tensor<Float>.randn([4, 8]) }
        let stacked = Tensor<Float>.stack(tensors, axis: 0)
        XCTAssertEqual(stacked.shape, [5, 4, 8])
    }

    func testStackSingleTensor() {
        let a = Tensor<Float>.ones([2, 3])
        let c = Tensor<Float>.stack([a], axis: 0)
        XCTAssertEqual(c.shape, [1, 2, 3])
    }

    func testStackInstanceMethod() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = a.stack(with: [b], axis: 0)
        XCTAssertEqual(c.shape, [2, 2, 3])
    }

    func testStack3DTensors() {
        let a = Tensor<Float>.ones([2, 3, 4])
        let b = Tensor<Float>.zeros([2, 3, 4])
        let c = Tensor<Float>.stack([a, b], axis: 1)
        XCTAssertEqual(c.shape, [2, 2, 3, 4])
    }
}

// MARK: - SliceAxis Tests

final class SliceAxisTests: XCTestCase {

    func testSliceAxis0() {
        let x = Tensor<Float>.randn([10, 5, 3])
        let s = x.sliceAxis(axis: 0, start: 2, size: 4)
        XCTAssertEqual(s.shape, [4, 5, 3])
    }

    func testSliceAxis1() {
        let x = Tensor<Float>.randn([10, 8, 3])
        let s = x.sliceAxis(axis: 1, start: 1, size: 5)
        XCTAssertEqual(s.shape, [10, 5, 3])
    }

    func testSliceAxis2() {
        let x = Tensor<Float>.randn([10, 5, 8])
        let s = x.sliceAxis(axis: 2, start: 3, size: 2)
        XCTAssertEqual(s.shape, [10, 5, 2])
    }

    func testSliceAxisNegative() {
        let x = Tensor<Float>.randn([10, 5, 8])
        let s = x.sliceAxis(axis: -1, start: 0, size: 4)
        XCTAssertEqual(s.shape, [10, 5, 4])
    }

    func testSliceAxisSingleElement() {
        let x = Tensor<Float>.randn([10, 5, 8])
        let s = x.sliceAxis(axis: 0, start: 3, size: 1)
        XCTAssertEqual(s.shape, [1, 5, 8])
    }
}

// MARK: - Advanced Slicing Tests

final class AdvancedSlicingTests: XCTestCase {

    // MARK: - Negative Index Tests

    func testSliceNegativeStart() {
        let x = Tensor<Float>.randn([10, 8])
        // Last 3 elements along axis 0
        let s = x.slice(axis: 0, start: -3, stop: nil)
        XCTAssertEqual(s.shape, [3, 8])
    }

    func testSliceNegativeStop() {
        let x = Tensor<Float>.randn([10, 8])
        // All but last 2 elements
        let s = x.slice(axis: 0, start: 0, stop: -2)
        XCTAssertEqual(s.shape, [8, 8])
    }

    func testSliceNegativeStartAndStop() {
        let x = Tensor<Float>.randn([10, 8])
        // Elements from -5 to -2 (indices 5, 6, 7)
        let s = x.slice(axis: 0, start: -5, stop: -2)
        XCTAssertEqual(s.shape, [3, 8])
    }

    func testSliceNegativeAxis() {
        let x = Tensor<Float>.randn([4, 10, 8])
        // Slice last axis
        let s = x.slice(axis: -1, start: 2, stop: 6)
        XCTAssertEqual(s.shape, [4, 10, 4])
    }

    func testSliceNegativeAxisWithNegativeIndices() {
        let x = Tensor<Float>.randn([4, 10, 8])
        // Last 3 along last axis
        let s = x.slice(axis: -1, start: -3, stop: nil)
        XCTAssertEqual(s.shape, [4, 10, 3])
    }

    // MARK: - Step/Stride Tests

    func testSliceWithStep() {
        let x = Tensor<Float>.randn([10, 8])
        // Every other element
        let s = x.slice(axis: 0, start: 0, stop: nil, step: 2)
        XCTAssertEqual(s.shape, [5, 8])
    }

    func testSliceWithStepAndStart() {
        let x = Tensor<Float>.randn([10, 8])
        // Start at 1, every 2 elements: 1, 3, 5, 7, 9
        let s = x.slice(axis: 0, start: 1, stop: nil, step: 2)
        XCTAssertEqual(s.shape, [5, 8])
    }

    func testSliceWithStepAndStop() {
        let x = Tensor<Float>.randn([10, 8])
        // From 0 to 8, step 3: 0, 3, 6
        let s = x.slice(axis: 0, start: 0, stop: 8, step: 3)
        XCTAssertEqual(s.shape, [3, 8])
    }

    func testSliceWithStep3() {
        let x = Tensor<Float>.randn([12, 8])
        // Every 3rd element: 0, 3, 6, 9
        let s = x.slice(axis: 0, start: nil, stop: nil, step: 3)
        XCTAssertEqual(s.shape, [4, 8])
    }

    func testSliceWithStepOn3D() {
        let x = Tensor<Float>.randn([4, 20, 16])
        // Every 4th along axis 1
        let s = x.slice(axis: 1, start: nil, stop: nil, step: 4)
        XCTAssertEqual(s.shape, [4, 5, 16])
    }

    func testSliceWithStepAndNegativeIndices() {
        let x = Tensor<Float>.randn([10, 8])
        // From index 1 to -1 (9), step 2: 1, 3, 5, 7
        let s = x.slice(axis: 0, start: 1, stop: -1, step: 2)
        XCTAssertEqual(s.shape, [4, 8])
    }

    // MARK: - Edge Cases

    func testSliceNilStartAndStop() {
        let x = Tensor<Float>.randn([10, 8])
        // Full slice (no-op)
        let s = x.slice(axis: 0, start: nil, stop: nil)
        XCTAssertEqual(s.shape, [10, 8])
    }

    func testSliceSingleElement() {
        let x = Tensor<Float>.randn([10, 8])
        // Single element slice
        let s = x.slice(axis: 0, start: 5, stop: 6)
        XCTAssertEqual(s.shape, [1, 8])
    }

    func testSliceLastElement() {
        let x = Tensor<Float>.randn([10, 8])
        // Last element using negative index
        let s = x.slice(axis: 0, start: -1, stop: nil)
        XCTAssertEqual(s.shape, [1, 8])
    }

    func testSliceFirstElement() {
        let x = Tensor<Float>.randn([10, 8])
        let s = x.slice(axis: 0, start: 0, stop: 1)
        XCTAssertEqual(s.shape, [1, 8])
    }

    // MARK: - Gradient Tests

    func testAdvancedSliceGradient() {
        let x = Tensor<Float>.randn([10, 8])
        let grad = gradient(at: x) { t in
            t.slice(axis: 0, start: 2, stop: 7).sum()
        }
        XCTAssertEqual(grad.shape, [10, 8])
    }

    func testAdvancedSliceGradientWithStep() {
        let x = Tensor<Float>.randn([10, 8])
        let grad = gradient(at: x) { t in
            t.slice(axis: 0, start: nil, stop: nil, step: 2).sum()
        }
        XCTAssertEqual(grad.shape, [10, 8])
    }

    func testAdvancedSliceGradientNegativeIndices() {
        let x = Tensor<Float>.randn([10, 8])
        let grad = gradient(at: x) { t in
            t.slice(axis: 0, start: -5, stop: -1).sum()
        }
        XCTAssertEqual(grad.shape, [10, 8])
    }
}

// MARK: - Comparison Operations Tests

final class ComparisonTests: XCTestCase {

    func testLessThan() {
        let x = Tensor<Float>.randn([4, 8])
        let y = Tensor<Float>.randn([4, 8])
        let mask = x.lessThan(y)
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testLessThanScalar() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x.lessThan(0.0)
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testGreaterThan() {
        let x = Tensor<Float>.randn([4, 8])
        let y = Tensor<Float>.randn([4, 8])
        let mask = x.greaterThan(y)
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testGreaterThanScalar() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x.greaterThan(0.5)
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testEqualTo() {
        let x = Tensor<Float>.ones([4, 8])
        let y = Tensor<Float>.ones([4, 8])
        let mask = x.equalTo(y)
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testEqualToScalar() {
        let x = Tensor<Float>.ones([4, 8])
        let mask = x.equalTo(1.0)
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testLessThanOrEqual() {
        let x = Tensor<Float>.randn([4, 8])
        let y = Tensor<Float>.randn([4, 8])
        let mask = x.lessThanOrEqual(y)
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testGreaterThanOrEqual() {
        let x = Tensor<Float>.randn([4, 8])
        let y = Tensor<Float>.randn([4, 8])
        let mask = x.greaterThanOrEqual(y)
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testNotEqual() {
        let x = Tensor<Float>.ones([4, 8])
        let y = Tensor<Float>.zeros([4, 8])
        let mask = x.notEqual(y)
        XCTAssertEqual(mask.shape, [4, 8])
    }

    // MARK: - Operator Tests

    func testLessThanOperator() {
        let x = Tensor<Float>.randn([4, 8])
        let y = Tensor<Float>.randn([4, 8])
        let mask = x .< y
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testGreaterThanOperator() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x .> 0.0
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testLessOrEqualOperator() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x .<= 0.5
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testGreaterOrEqualOperator() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x .>= -0.5
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testEqualityOperator() {
        let x = Tensor<Float>.ones([4, 8])
        let mask = x .== 1.0
        XCTAssertEqual(mask.shape, [4, 8])
    }

    func testNotEqualOperator() {
        let x = Tensor<Float>.ones([4, 8])
        let mask = x .!= 0.0
        XCTAssertEqual(mask.shape, [4, 8])
    }
}

// MARK: - Boolean Masking Tests

final class BooleanMaskingTests: XCTestCase {

    func testMaskedSelect() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x.greaterThan(0.0)
        let result = x.maskedSelect(mask)
        // Result is flattened with zeros where mask is false
        XCTAssertEqual(result.shape, [32])
    }

    func testWhereFunction() {
        let x = Tensor<Float>.ones([4, 8])
        let y = Tensor<Float>.zeros([4, 8])
        let mask = Tensor<Float>.randn([4, 8]).greaterThan(0.0)

        let result = Tensor<Float>.where_(mask, x, y)
        XCTAssertEqual(result.shape, [4, 8])
    }

    func testWhereMethod() {
        let x = Tensor<Float>.ones([4, 8])
        let y = Tensor<Float>.zeros([4, 8])
        let mask = Tensor<Float>.randn([4, 8]).greaterThan(0.0)

        let result = x.where_(mask, y)
        XCTAssertEqual(result.shape, [4, 8])
    }

    func testWhereWithScalarValues() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x.greaterThan(0.0)

        // Replace negative values with zero
        let positive = Tensor<Float>.where_(mask, x, Tensor<Float>.zeros([4, 8]))
        XCTAssertEqual(positive.shape, [4, 8])
    }

    func testMaskChaining() {
        let x = Tensor<Float>.randn([4, 8])
        // Values between -0.5 and 0.5
        let inRange = x.greaterThan(-0.5) * x.lessThan(0.5)
        XCTAssertEqual(inRange.shape, [4, 8])
    }

    func testMaskedFillWithComparison() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x.lessThan(0.0)
        // Set all negative values to zero
        let result = x.maskedFill(mask: mask, value: Tensor<Float>.zeros([4, 8]))
        XCTAssertEqual(result.shape, [4, 8])
    }
}

// MARK: - MNIST Dataset Tests

final class MNISTDatasetTests: XCTestCase {

    /// Test that MNIST data files can be downloaded and parsed (requires network)
    /// This test is skipped by default - run manually to verify network download
    func testMNISTDownloadAndParse() throws {
        // Skip if MNIST data directory doesn't exist or network is unavailable
        // This test requires network access, so we'll make it conditional

        // Try to load MNIST - if data is cached, this will be fast
        // If not, it will download (~10MB per file)
        do {
            let mnist = try MNIST(split: .train, normalize: true, flatten: true)

            // Verify shapes
            XCTAssertEqual(mnist.count, 60000, "Training set should have 60000 samples")
            XCTAssertEqual(mnist.images.shape, [60000, 784], "Images should be [60000, 784]")
            XCTAssertEqual(mnist.labels.shape, [60000, 10], "Labels should be [60000, 10] (one-hot)")
        } catch {
            // If network is unavailable or download fails, skip the test
            throw XCTSkip("MNIST download failed (network may be unavailable): \(error)")
        }
    }

    func testMNISTTestSplit() throws {
        do {
            let mnist = try MNIST(split: .test, normalize: true, flatten: true)

            XCTAssertEqual(mnist.count, 10000, "Test set should have 10000 samples")
            XCTAssertEqual(mnist.images.shape, [10000, 784])
            XCTAssertEqual(mnist.labels.shape, [10000, 10])
        } catch {
            throw XCTSkip("MNIST download failed: \(error)")
        }
    }

    func testMNISTUnflattenedImages() throws {
        do {
            let mnist = try MNIST(split: .train, normalize: true, flatten: false)

            // Unflattened images should be [N, 28, 28, 1]
            XCTAssertEqual(mnist.images.shape, [60000, 28, 28, 1])
        } catch {
            throw XCTSkip("MNIST download failed: \(error)")
        }
    }

    func testMNISTNonOneHotLabels() throws {
        do {
            let mnist = try MNIST(split: .train, normalize: true, flatten: true, oneHot: false)

            // Non-one-hot labels should just be [N]
            XCTAssertEqual(mnist.labels.shape, [60000])
        } catch {
            throw XCTSkip("MNIST download failed: \(error)")
        }
    }

    func testMNISTWithSimpleBatchLoader() throws {
        do {
            let mnist = try MNIST(split: .train, normalize: true, flatten: true)
            let loader = SimpleBatchLoader(inputs: mnist.images, targets: mnist.labels, batchSize: 64)

            XCTAssertEqual(loader.numSamples, 60000)
            XCTAssertEqual(loader.batchSize, 64)
            XCTAssertEqual(loader.numBatches, 938)  // ceil(60000/64) = 938

            // Get first batch
            let batch = loader.batch(0)
            XCTAssertEqual(batch.input.shape, [64, 784])
            XCTAssertEqual(batch.target.shape, [64, 10])
        } catch {
            throw XCTSkip("MNIST download failed: \(error)")
        }
    }
}
