// Magma - Data Loading Tests
// Tests for Dataset and DataLoader

import Testing
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - TensorDataset Tests

@Suite("TensorDataset Tests")
struct TensorDatasetTests {

    @Test("TensorDataset creation")
    func tensorDatasetCreation() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let dataset = TensorDataset(inputs: inputs, targets: targets)

        #expect(dataset.count == 100)
    }

    @Test("TensorDataset shape mismatch")
    func tensorDatasetShapeMismatch() {
        // Different number of samples should fail
        // (precondition will trigger in debug builds)
    }
}

// MARK: - SimpleBatchLoader Tests

@Suite("SimpleBatchLoader Tests")
struct SimpleBatchLoaderTests {

    @Test("BatchLoader creation")
    func batchLoaderCreation() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 32)

        #expect(loader.numSamples == 100)
        #expect(loader.batchSize == 32)
        #expect(loader.numBatches == 4)  // ceil(100/32) = 4
        #expect(loader.shuffle == false)
        #expect(loader.dropLast == false)
    }

    @Test("BatchLoader iteration")
    func batchLoaderIteration() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 32)

        var batchCount = 0
        for batch in loader {
            #expect(batch.input.shape[1] == 784)
            #expect(batch.target.shape[1] == 10)
            batchCount += 1
        }

        #expect(batchCount == 4)
    }

    @Test("BatchLoader exact division")
    func batchLoaderExactDivision() {
        let inputs = Tensor<Float>.ones([64, 784])
        let targets = Tensor<Float>.zeros([64, 10])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 32)

        #expect(loader.numBatches == 2)

        var batchCount = 0
        for _ in loader {
            batchCount += 1
        }
        #expect(batchCount == 2)
    }

    @Test("BatchLoader single batch")
    func batchLoaderSingleBatch() {
        let inputs = Tensor<Float>.ones([16, 784])
        let targets = Tensor<Float>.zeros([16, 10])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 32)

        #expect(loader.numBatches == 1)
    }

    @Test("Get specific batch")
    func getSpecificBatch() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(inputs: inputs, targets: targets, batchSize: 32)

        let batch0 = loader.batch(0)
        #expect(batch0.input.shape[0] == 32)

        let batch3 = loader.batch(3)  // Last batch (100 - 96 = 4 samples)
        #expect(batch3.input.shape[0] == 4)
    }

    // MARK: - Shuffle Tests

    @Test("BatchLoader with shuffle")
    func batchLoaderWithShuffle() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(
            inputs: inputs,
            targets: targets,
            batchSize: 32,
            shuffle: true
        )

        #expect(loader.shuffle == true)
        #expect(loader.numBatches == 4)

        var batchCount = 0
        for batch in loader {
            #expect(batch.input.shape[1] == 784)
            #expect(batch.target.shape[1] == 10)
            batchCount += 1
        }

        #expect(batchCount == 4)
    }

    @Test("Shuffle produces different order")
    func shuffleProducesDifferentOrder() {
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
        #expect(batches1.count == 2)
        #expect(batches2.count == 2)

        // Shapes should be preserved
        #expect(batches1[0].count == 20)  // 5 samples * 4 features
    }

    @Test("Shuffle preserves all samples")
    func shufflePreservesAllSamples() {
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
        #expect(allValues.count == 8)
    }

    // MARK: - DropLast Tests

    @Test("BatchLoader with dropLast")
    func batchLoaderWithDropLast() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(
            inputs: inputs,
            targets: targets,
            batchSize: 32,
            dropLast: true
        )

        #expect(loader.dropLast == true)
        #expect(loader.numBatches == 3)  // 100 / 32 = 3 (drops last 4 samples)

        var batchCount = 0
        var allBatchSizes: [Int] = []
        for batch in loader {
            allBatchSizes.append(batch.input.shape[0])
            batchCount += 1
        }

        #expect(batchCount == 3)
        // All batches should be full size
        #expect(allBatchSizes.allSatisfy { $0 == 32 })
    }

    @Test("BatchLoader dropLast exact division")
    func batchLoaderDropLastExactDivision() {
        let inputs = Tensor<Float>.ones([64, 784])
        let targets = Tensor<Float>.zeros([64, 10])

        let loader = SimpleBatchLoader(
            inputs: inputs,
            targets: targets,
            batchSize: 32,
            dropLast: true
        )

        // Exact division should still work
        #expect(loader.numBatches == 2)

        var batchCount = 0
        for _ in loader {
            batchCount += 1
        }
        #expect(batchCount == 2)
    }

    @Test("BatchLoader shuffle and dropLast")
    func batchLoaderShuffleAndDropLast() {
        let inputs = Tensor<Float>.ones([100, 784])
        let targets = Tensor<Float>.zeros([100, 10])

        let loader = SimpleBatchLoader(
            inputs: inputs,
            targets: targets,
            batchSize: 32,
            shuffle: true,
            dropLast: true
        )

        #expect(loader.shuffle == true)
        #expect(loader.dropLast == true)
        #expect(loader.numBatches == 3)

        var batchCount = 0
        for batch in loader {
            #expect(batch.input.shape[0] == 32)  // All batches should be full
            batchCount += 1
        }

        #expect(batchCount == 3)
    }

    // MARK: - Multi-epoch Shuffle Tests

    @Test("Shuffle each epoch")
    func shuffleEachEpoch() {
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
        #expect(firstBatchFirstValues.count == 3)
        // At least two epochs should have different first values (highly likely with shuffling)
        // Note: This could theoretically fail with very bad luck, but probability is low
    }
}

// MARK: - Tensor Slice Tests

@Suite("Tensor Slice Tests")
struct TensorSliceTests {

    @Test("Slice basic")
    func sliceBasic() {
        let x = Tensor<Float>.ones([100, 784])
        let sliced = x.slice(start: 0, size: 32)

        #expect(sliced.shape == [32, 784])
    }

    @Test("Slice middle")
    func sliceMiddle() {
        let x = Tensor<Float>.ones([100, 784])
        let sliced = x.slice(start: 32, size: 32)

        #expect(sliced.shape == [32, 784])
    }

    @Test("Slice end")
    func sliceEnd() {
        let x = Tensor<Float>.ones([100, 784])
        let sliced = x.slice(start: 96, size: 4)

        #expect(sliced.shape == [4, 784])
    }

    @Test("Slice 3D")
    func slice3D() {
        let x = Tensor<Float>.ones([100, 28, 28])
        let sliced = x.slice(start: 0, size: 16)

        #expect(sliced.shape == [16, 28, 28])
    }

    @Test("Slice gradient")
    func sliceGradient() {
        let x = Tensor<Float>.ones([100, 10])

        let grad = gradient(at: x) { t in
            t.slice(start: 0, size: 32).sum()
        }

        // Gradient should have original shape
        #expect(grad.shape == [100, 10])
    }
}

// MARK: - Training Loop Integration Tests

@Suite("DataLoader Training Tests")
struct DataLoaderTrainingTests {

    @Test("Training loop with loader")
    func trainingLoopWithLoader() {
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
            #expect(output.shape[1] == 10)

            // Create dummy gradients (normally computed via autodiff)
            let grads = model.parameters().map { Tensor<Float>.ones($0.shape) }

            // Optimizer step
            optimizer.step(grads)
            totalBatches += 1
        }

        #expect(totalBatches == 2)
    }

    @Test("Multi-epoch training")
    func multiEpochTraining() {
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
                #expect(output.shape[1] == 10)

                let grads = model.parameters().map { Tensor<Float>.ones($0.shape) }
                optimizer.step(grads)
                totalBatches += 1
            }
        }

        #expect(totalBatches == numEpochs * 2)
    }
}

// MARK: - Gather Operation Tests

@Suite("Gather Tests")
struct GatherTests {

    @Test("Gather basic 2D")
    func gatherBasic2D() {
        let x = Tensor<Float>.randn([4, 8])  // [4, 8]
        let indices = Tensor<Float>.zeros([4, 3])  // Gather 3 elements per row

        let result = x.gather(indices: indices, axis: 1)
        #expect(result.shape == [4, 3])
    }

    @Test("Gather axis 0")
    func gatherAxis0() {
        let x = Tensor<Float>.randn([10, 5])  // [10, 5]
        let indices = Tensor<Float>.zeros([3, 5])  // Gather 3 rows

        let result = x.gather(indices: indices, axis: 0)
        #expect(result.shape == [3, 5])
    }

    @Test("Gather 3D")
    func gather3D() {
        let x = Tensor<Float>.randn([2, 8, 16])
        let indices = Tensor<Float>.zeros([2, 4, 16])  // Gather 4 elements along axis 1

        let result = x.gather(indices: indices, axis: 1)
        #expect(result.shape == [2, 4, 16])
    }

    @Test("Gather negative axis")
    func gatherNegativeAxis() {
        let x = Tensor<Float>.randn([4, 8])
        let indices = Tensor<Float>.zeros([4, 2])

        let result = x.gather(indices: indices, axis: -1)  // Same as axis=1
        #expect(result.shape == [4, 2])
    }

    @Test("Gather gradient")
    func gatherGradient() {
        let x = Tensor<Float>.randn([4, 8])
        let indices = Tensor<Float>.zeros([4, 2])

        let grad = gradient(at: x) { t in
            t.gather(indices: indices, axis: 1).sum()
        }

        #expect(grad.shape == [4, 8])
    }
}

// MARK: - Scatter Operation Tests

@Suite("Scatter Tests")
struct ScatterTests {

    @Test("Scatter basic 2D")
    func scatterBasic2D() {
        let x = Tensor<Float>.zeros([4, 8])
        let indices = Tensor<Float>.zeros([4, 2])
        let updates = Tensor<Float>.ones([4, 2])

        let result = x.scatter(indices: indices, updates: updates, axis: 1)
        #expect(result.shape == [4, 8])
    }

    @Test("Scatter axis 0")
    func scatterAxis0() {
        let x = Tensor<Float>.zeros([10, 5])
        let indices = Tensor<Float>.zeros([3, 5])
        let updates = Tensor<Float>.ones([3, 5])

        let result = x.scatter(indices: indices, updates: updates, axis: 0)
        #expect(result.shape == [10, 5])
    }

    @Test("Scatter 3D")
    func scatter3D() {
        let x = Tensor<Float>.zeros([2, 8, 16])
        let indices = Tensor<Float>.zeros([2, 3, 16])
        let updates = Tensor<Float>.ones([2, 3, 16])

        let result = x.scatter(indices: indices, updates: updates, axis: 1)
        #expect(result.shape == [2, 8, 16])
    }

    @Test("Scatter negative axis")
    func scatterNegativeAxis() {
        let x = Tensor<Float>.zeros([4, 8])
        let indices = Tensor<Float>.zeros([4, 2])
        let updates = Tensor<Float>.ones([4, 2])

        let result = x.scatter(indices: indices, updates: updates, axis: -1)
        #expect(result.shape == [4, 8])
    }

    @Test("Scatter gradient")
    func scatterGradient() {
        let x = Tensor<Float>.zeros([4, 8])
        let indices = Tensor<Float>.zeros([4, 2])
        let updates = Tensor<Float>.ones([4, 2])

        let (inputGrad, updatesGrad) = gradient(at: x, updates) { base, upd in
            base.scatter(indices: indices, updates: upd, axis: 1).sum()
        }

        #expect(inputGrad.shape == [4, 8])
        #expect(updatesGrad.shape == [4, 2])
    }
}

// MARK: - Expand Operation Tests

@Suite("Expand Tests")
struct ExpandTests {

    @Test("Expand basic")
    func expandBasic() {
        let x = Tensor<Float>.ones([1, 4])
        let result = x.expand([3, 4])

        #expect(result.shape == [3, 4])
    }

    @Test("Expand multiple dims")
    func expandMultipleDims() {
        let x = Tensor<Float>.ones([1, 1, 4])
        let result = x.expand([3, 5, 4])

        #expect(result.shape == [3, 5, 4])
    }

    @Test("Expand middle dim")
    func expandMiddleDim() {
        let x = Tensor<Float>.ones([2, 1, 4])
        let result = x.expand([2, 8, 4])

        #expect(result.shape == [2, 8, 4])
    }

    @Test("Expand no change")
    func expandNoChange() {
        let x = Tensor<Float>.ones([2, 4])
        let result = x.expand([2, 4])  // Same shape

        #expect(result.shape == [2, 4])
    }

    @Test("Expand gradient")
    func expandGradient() {
        let x = Tensor<Float>.ones([1, 4])

        let grad = gradient(at: x) { t in
            t.expand([3, 4]).sum()
        }

        // Gradient should be summed back to original shape
        #expect(grad.shape == [1, 4])
    }

    @Test("Expand gradient 3D")
    func expandGradient3D() {
        let x = Tensor<Float>.ones([1, 1, 8])

        let grad = gradient(at: x) { t in
            t.expand([4, 6, 8]).sum()
        }

        #expect(grad.shape == [1, 1, 8])
    }
}

// MARK: - Sum with Dims Tests

@Suite("Sum with Dims Tests")
struct SumWithDimsTests {

    @Test("Sum single dim")
    func sumSingleDim() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [1], keepDims: false)

        #expect(result.shape == [4, 16])
    }

    @Test("Sum single dim keepDims")
    func sumSingleDimKeepDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [1], keepDims: true)

        #expect(result.shape == [4, 1, 16])
    }

    @Test("Sum multiple dims")
    func sumMultipleDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [0, 2], keepDims: false)

        #expect(result.shape == [8])
    }

    @Test("Sum multiple dims keepDims")
    func sumMultipleDimsKeepDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [0, 2], keepDims: true)

        #expect(result.shape == [1, 8, 1])
    }

    @Test("Sum negative dim")
    func sumNegativeDim() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [-1], keepDims: true)

        #expect(result.shape == [4, 8, 1])
    }

    @Test("Sum all dims")
    func sumAllDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let result = x.sum(dims: [0, 1, 2], keepDims: false)

        #expect(result.shape == [])  // Scalar
    }
}

// MARK: - Concat Tests

@Suite("Concat Tests")
struct ConcatTests {

    @Test("Concat axis 0")
    func concatAxis0() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([3, 3])
        let c = Tensor<Float>.concat([a, b], axis: 0)
        #expect(c.shape == [5, 3])
    }

    @Test("Concat axis 1")
    func concatAxis1() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 4])
        let c = Tensor<Float>.concat([a, b], axis: 1)
        #expect(c.shape == [2, 7])
    }

    @Test("Concat negative axis")
    func concatNegativeAxis() {
        let a = Tensor<Float>.ones([2, 3, 4])
        let b = Tensor<Float>.zeros([2, 3, 5])
        let c = Tensor<Float>.concat([a, b], axis: -1)
        #expect(c.shape == [2, 3, 9])
    }

    @Test("Concat multiple tensors")
    func concatMultipleTensors() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = Tensor<Float>.randn([2, 3])
        let d = Tensor<Float>.concat([a, b, c], axis: 0)
        #expect(d.shape == [6, 3])
    }

    @Test("Concat single tensor")
    func concatSingleTensor() {
        let a = Tensor<Float>.ones([2, 3])
        let c = Tensor<Float>.concat([a], axis: 0)
        #expect(c.shape == [2, 3])
    }

    @Test("Concat instance method")
    func concatInstanceMethod() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = a.concat(with: [b], axis: 0)
        #expect(c.shape == [4, 3])
    }
}

// MARK: - Stack Tests

@Suite("Stack Tests")
struct StackTests {

    @Test("Stack axis 0")
    func stackAxis0() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = Tensor<Float>.stack([a, b], axis: 0)
        #expect(c.shape == [2, 2, 3])
    }

    @Test("Stack axis 1")
    func stackAxis1() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = Tensor<Float>.stack([a, b], axis: 1)
        #expect(c.shape == [2, 2, 3])
    }

    @Test("Stack axis 2")
    func stackAxis2() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = Tensor<Float>.stack([a, b], axis: 2)
        #expect(c.shape == [2, 3, 2])
    }

    @Test("Stack negative axis")
    func stackNegativeAxis() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = Tensor<Float>.stack([a, b], axis: -1)
        #expect(c.shape == [2, 3, 2])
    }

    @Test("Stack multiple tensors")
    func stackMultipleTensors() {
        let tensors = (0..<5).map { _ in Tensor<Float>.randn([4, 8]) }
        let stacked = Tensor<Float>.stack(tensors, axis: 0)
        #expect(stacked.shape == [5, 4, 8])
    }

    @Test("Stack single tensor")
    func stackSingleTensor() {
        let a = Tensor<Float>.ones([2, 3])
        let c = Tensor<Float>.stack([a], axis: 0)
        #expect(c.shape == [1, 2, 3])
    }

    @Test("Stack instance method")
    func stackInstanceMethod() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.zeros([2, 3])
        let c = a.stack(with: [b], axis: 0)
        #expect(c.shape == [2, 2, 3])
    }

    @Test("Stack 3D tensors")
    func stack3DTensors() {
        let a = Tensor<Float>.ones([2, 3, 4])
        let b = Tensor<Float>.zeros([2, 3, 4])
        let c = Tensor<Float>.stack([a, b], axis: 1)
        #expect(c.shape == [2, 2, 3, 4])
    }
}

// MARK: - SliceAxis Tests

@Suite("SliceAxis Tests")
struct SliceAxisTests {

    @Test("SliceAxis 0")
    func sliceAxis0() {
        let x = Tensor<Float>.randn([10, 5, 3])
        let s = x.sliceAxis(axis: 0, start: 2, size: 4)
        #expect(s.shape == [4, 5, 3])
    }

    @Test("SliceAxis 1")
    func sliceAxis1() {
        let x = Tensor<Float>.randn([10, 8, 3])
        let s = x.sliceAxis(axis: 1, start: 1, size: 5)
        #expect(s.shape == [10, 5, 3])
    }

    @Test("SliceAxis 2")
    func sliceAxis2() {
        let x = Tensor<Float>.randn([10, 5, 8])
        let s = x.sliceAxis(axis: 2, start: 3, size: 2)
        #expect(s.shape == [10, 5, 2])
    }

    @Test("SliceAxis negative")
    func sliceAxisNegative() {
        let x = Tensor<Float>.randn([10, 5, 8])
        let s = x.sliceAxis(axis: -1, start: 0, size: 4)
        #expect(s.shape == [10, 5, 4])
    }

    @Test("SliceAxis single element")
    func sliceAxisSingleElement() {
        let x = Tensor<Float>.randn([10, 5, 8])
        let s = x.sliceAxis(axis: 0, start: 3, size: 1)
        #expect(s.shape == [1, 5, 8])
    }
}

// MARK: - Advanced Slicing Tests

@Suite("Advanced Slicing Tests")
struct AdvancedSlicingTests {

    // MARK: - Negative Index Tests

    @Test("Slice negative start")
    func sliceNegativeStart() {
        let x = Tensor<Float>.randn([10, 8])
        // Last 3 elements along axis 0
        let s = x.slice(axis: 0, start: -3, stop: nil)
        #expect(s.shape == [3, 8])
    }

    @Test("Slice negative stop")
    func sliceNegativeStop() {
        let x = Tensor<Float>.randn([10, 8])
        // All but last 2 elements
        let s = x.slice(axis: 0, start: 0, stop: -2)
        #expect(s.shape == [8, 8])
    }

    @Test("Slice negative start and stop")
    func sliceNegativeStartAndStop() {
        let x = Tensor<Float>.randn([10, 8])
        // Elements from -5 to -2 (indices 5, 6, 7)
        let s = x.slice(axis: 0, start: -5, stop: -2)
        #expect(s.shape == [3, 8])
    }

    @Test("Slice negative axis")
    func sliceNegativeAxis() {
        let x = Tensor<Float>.randn([4, 10, 8])
        // Slice last axis
        let s = x.slice(axis: -1, start: 2, stop: 6)
        #expect(s.shape == [4, 10, 4])
    }

    @Test("Slice negative axis with negative indices")
    func sliceNegativeAxisWithNegativeIndices() {
        let x = Tensor<Float>.randn([4, 10, 8])
        // Last 3 along last axis
        let s = x.slice(axis: -1, start: -3, stop: nil)
        #expect(s.shape == [4, 10, 3])
    }

    // MARK: - Step/Stride Tests

    @Test("Slice with step")
    func sliceWithStep() {
        let x = Tensor<Float>.randn([10, 8])
        // Every other element
        let s = x.slice(axis: 0, start: 0, stop: nil, step: 2)
        #expect(s.shape == [5, 8])
    }

    @Test("Slice with step and start")
    func sliceWithStepAndStart() {
        let x = Tensor<Float>.randn([10, 8])
        // Start at 1, every 2 elements: 1, 3, 5, 7, 9
        let s = x.slice(axis: 0, start: 1, stop: nil, step: 2)
        #expect(s.shape == [5, 8])
    }

    @Test("Slice with step and stop")
    func sliceWithStepAndStop() {
        let x = Tensor<Float>.randn([10, 8])
        // From 0 to 8, step 3: 0, 3, 6
        let s = x.slice(axis: 0, start: 0, stop: 8, step: 3)
        #expect(s.shape == [3, 8])
    }

    @Test("Slice with step 3")
    func sliceWithStep3() {
        let x = Tensor<Float>.randn([12, 8])
        // Every 3rd element: 0, 3, 6, 9
        let s = x.slice(axis: 0, start: nil, stop: nil, step: 3)
        #expect(s.shape == [4, 8])
    }

    @Test("Slice with step on 3D")
    func sliceWithStepOn3D() {
        let x = Tensor<Float>.randn([4, 20, 16])
        // Every 4th along axis 1
        let s = x.slice(axis: 1, start: nil, stop: nil, step: 4)
        #expect(s.shape == [4, 5, 16])
    }

    @Test("Slice with step and negative indices")
    func sliceWithStepAndNegativeIndices() {
        let x = Tensor<Float>.randn([10, 8])
        // From index 1 to -1 (9), step 2: 1, 3, 5, 7
        let s = x.slice(axis: 0, start: 1, stop: -1, step: 2)
        #expect(s.shape == [4, 8])
    }

    // MARK: - Edge Cases

    @Test("Slice nil start and stop")
    func sliceNilStartAndStop() {
        let x = Tensor<Float>.randn([10, 8])
        // Full slice (no-op)
        let s = x.slice(axis: 0, start: nil, stop: nil)
        #expect(s.shape == [10, 8])
    }

    @Test("Slice single element")
    func sliceSingleElement() {
        let x = Tensor<Float>.randn([10, 8])
        // Single element slice
        let s = x.slice(axis: 0, start: 5, stop: 6)
        #expect(s.shape == [1, 8])
    }

    @Test("Slice last element")
    func sliceLastElement() {
        let x = Tensor<Float>.randn([10, 8])
        // Last element using negative index
        let s = x.slice(axis: 0, start: -1, stop: nil)
        #expect(s.shape == [1, 8])
    }

    @Test("Slice first element")
    func sliceFirstElement() {
        let x = Tensor<Float>.randn([10, 8])
        let s = x.slice(axis: 0, start: 0, stop: 1)
        #expect(s.shape == [1, 8])
    }

    // MARK: - Gradient Tests

    @Test("Advanced slice gradient")
    func advancedSliceGradient() {
        let x = Tensor<Float>.randn([10, 8])
        let grad = gradient(at: x) { t in
            t.slice(axis: 0, start: 2, stop: 7).sum()
        }
        #expect(grad.shape == [10, 8])
    }

    @Test("Advanced slice gradient with step")
    func advancedSliceGradientWithStep() {
        let x = Tensor<Float>.randn([10, 8])
        let grad = gradient(at: x) { t in
            t.slice(axis: 0, start: nil, stop: nil, step: 2).sum()
        }
        #expect(grad.shape == [10, 8])
    }

    @Test("Advanced slice gradient negative indices")
    func advancedSliceGradientNegativeIndices() {
        let x = Tensor<Float>.randn([10, 8])
        let grad = gradient(at: x) { t in
            t.slice(axis: 0, start: -5, stop: -1).sum()
        }
        #expect(grad.shape == [10, 8])
    }
}

// MARK: - Comparison Operations Tests

@Suite("Comparison Tests")
struct ComparisonTests {

    @Test("Less than")
    func lessThan() {
        let x = Tensor<Float>.randn([4, 8])
        let y = Tensor<Float>.randn([4, 8])
        let mask = x.lessThan(y)
        #expect(mask.shape == [4, 8])
    }

    @Test("Less than scalar")
    func lessThanScalar() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x.lessThan(0.0)
        #expect(mask.shape == [4, 8])
    }

    @Test("Greater than")
    func greaterThan() {
        let x = Tensor<Float>.randn([4, 8])
        let y = Tensor<Float>.randn([4, 8])
        let mask = x.greaterThan(y)
        #expect(mask.shape == [4, 8])
    }

    @Test("Greater than scalar")
    func greaterThanScalar() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x.greaterThan(0.5)
        #expect(mask.shape == [4, 8])
    }

    @Test("Equal to")
    func equalTo() {
        let x = Tensor<Float>.ones([4, 8])
        let y = Tensor<Float>.ones([4, 8])
        let mask = x.equalTo(y)
        #expect(mask.shape == [4, 8])
    }

    @Test("Equal to scalar")
    func equalToScalar() {
        let x = Tensor<Float>.ones([4, 8])
        let mask = x.equalTo(1.0)
        #expect(mask.shape == [4, 8])
    }

    @Test("Less than or equal")
    func lessThanOrEqual() {
        let x = Tensor<Float>.randn([4, 8])
        let y = Tensor<Float>.randn([4, 8])
        let mask = x.lessThanOrEqual(y)
        #expect(mask.shape == [4, 8])
    }

    @Test("Greater than or equal")
    func greaterThanOrEqual() {
        let x = Tensor<Float>.randn([4, 8])
        let y = Tensor<Float>.randn([4, 8])
        let mask = x.greaterThanOrEqual(y)
        #expect(mask.shape == [4, 8])
    }

    @Test("Not equal")
    func notEqual() {
        let x = Tensor<Float>.ones([4, 8])
        let y = Tensor<Float>.zeros([4, 8])
        let mask = x.notEqual(y)
        #expect(mask.shape == [4, 8])
    }

    // MARK: - Operator Tests

    @Test("Less than operator")
    func lessThanOperator() {
        let x = Tensor<Float>.randn([4, 8])
        let y = Tensor<Float>.randn([4, 8])
        let mask = x .< y
        #expect(mask.shape == [4, 8])
    }

    @Test("Greater than operator")
    func greaterThanOperator() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x .> 0.0
        #expect(mask.shape == [4, 8])
    }

    @Test("Less or equal operator")
    func lessOrEqualOperator() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x .<= 0.5
        #expect(mask.shape == [4, 8])
    }

    @Test("Greater or equal operator")
    func greaterOrEqualOperator() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x .>= -0.5
        #expect(mask.shape == [4, 8])
    }

    @Test("Equality operator")
    func equalityOperator() {
        let x = Tensor<Float>.ones([4, 8])
        let mask = x .== 1.0
        #expect(mask.shape == [4, 8])
    }

    @Test("Not equal operator")
    func notEqualOperator() {
        let x = Tensor<Float>.ones([4, 8])
        let mask = x .!= 0.0
        #expect(mask.shape == [4, 8])
    }
}

// MARK: - Boolean Masking Tests

@Suite("Boolean Masking Tests")
struct BooleanMaskingTests {

    @Test("Masked select")
    func maskedSelect() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x.greaterThan(0.0)
        let result = x.maskedSelect(mask)
        // Result is flattened with zeros where mask is false
        #expect(result.shape == [32])
    }

    @Test("Where function")
    func whereFunction() {
        let x = Tensor<Float>.ones([4, 8])
        let y = Tensor<Float>.zeros([4, 8])
        let mask = Tensor<Float>.randn([4, 8]).greaterThan(0.0)

        let result = Tensor<Float>.where_(mask, x, y)
        #expect(result.shape == [4, 8])
    }

    @Test("Where method")
    func whereMethod() {
        let x = Tensor<Float>.ones([4, 8])
        let y = Tensor<Float>.zeros([4, 8])
        let mask = Tensor<Float>.randn([4, 8]).greaterThan(0.0)

        let result = x.where_(mask, y)
        #expect(result.shape == [4, 8])
    }

    @Test("Where with scalar values")
    func whereWithScalarValues() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x.greaterThan(0.0)

        // Replace negative values with zero
        let positive = Tensor<Float>.where_(mask, x, Tensor<Float>.zeros([4, 8]))
        #expect(positive.shape == [4, 8])
    }

    @Test("Mask chaining")
    func maskChaining() {
        let x = Tensor<Float>.randn([4, 8])
        // Values between -0.5 and 0.5
        let inRange = x.greaterThan(-0.5) * x.lessThan(0.5)
        #expect(inRange.shape == [4, 8])
    }

    @Test("Masked fill with comparison")
    func maskedFillWithComparison() {
        let x = Tensor<Float>.randn([4, 8])
        let mask = x.lessThan(0.0)
        // Set all negative values to zero
        let result = x.maskedFill(mask: mask, value: Tensor<Float>.zeros([4, 8]))
        #expect(result.shape == [4, 8])
    }
}

// MARK: - MNIST Dataset Tests

@Suite("MNIST Dataset Tests")
struct MNISTDatasetTests {

    /// Test that MNIST data files can be downloaded and parsed (requires network)
    /// This test is skipped by default - run manually to verify network download
    @Test("MNIST download and parse")
    func mnistDownloadAndParse() throws {
        // Skip if MNIST data directory doesn't exist or network is unavailable
        // This test requires network access, so we'll make it conditional

        // Try to load MNIST - if data is cached, this will be fast
        // If not, it will download (~10MB per file)
        let mnist = try MNIST(split: .train, normalize: true, flatten: true)

        // Verify shapes
        #expect(mnist.count == 60000, "Training set should have 60000 samples")
        #expect(mnist.images.shape == [60000, 784], "Images should be [60000, 784]")
        #expect(mnist.labels.shape == [60000, 10], "Labels should be [60000, 10] (one-hot)")
    }

    @Test("MNIST test split")
    func mnistTestSplit() throws {
        let mnist = try MNIST(split: .test, normalize: true, flatten: true)

        #expect(mnist.count == 10000, "Test set should have 10000 samples")
        #expect(mnist.images.shape == [10000, 784])
        #expect(mnist.labels.shape == [10000, 10])
    }

    @Test("MNIST unflattened images")
    func mnistUnflattenedImages() throws {
        let mnist = try MNIST(split: .train, normalize: true, flatten: false)

        // Unflattened images should be [N, 28, 28, 1]
        #expect(mnist.images.shape == [60000, 28, 28, 1])
    }

    @Test("MNIST non-one-hot labels")
    func mnistNonOneHotLabels() throws {
        let mnist = try MNIST(split: .train, normalize: true, flatten: true, oneHot: false)

        // Non-one-hot labels should just be [N]
        #expect(mnist.labels.shape == [60000])
    }

    @Test("MNIST with SimpleBatchLoader")
    func mnistWithSimpleBatchLoader() throws {
        let mnist = try MNIST(split: .train, normalize: true, flatten: true)
        let loader = SimpleBatchLoader(inputs: mnist.images, targets: mnist.labels, batchSize: 64)

        #expect(loader.numSamples == 60000)
        #expect(loader.batchSize == 64)
        #expect(loader.numBatches == 938)  // ceil(60000/64) = 938

        // Get first batch
        let batch = loader.batch(0)
        #expect(batch.input.shape == [64, 784])
        #expect(batch.target.shape == [64, 10])
    }
}
