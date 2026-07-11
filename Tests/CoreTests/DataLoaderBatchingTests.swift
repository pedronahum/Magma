// Magma - DataLoader batching tests
// Regression for the previously-stubbed generic Dataset/DataLoader path, which
// returned the whole dataset for every "batch". Verifies real indexed batching.
//
// .serialized: materializing tensors creates a GPU PJRT client (see
// XLAGPUSmokeTests).

import Testing
@testable import Magma
@testable import LazyTensor

@Suite("DataLoader Batching Tests", .serialized)
struct DataLoaderBatchingTests {

    @Test("DataLoader yields correctly batched rows")
    func batchesCorrectRows() {
        // Row i encoded as [i, i]; target i.
        let inputs = Tensor<Float>([0, 0,  1, 1,  2, 2,  3, 3], shape: [4, 2])
        let targets = Tensor<Float>([0, 1, 2, 3], shape: [4, 1])
        let loader = DataLoader(dataset: TensorDataset(inputs: inputs, targets: targets),
                                batchSize: 2)   // no shuffle -> contiguous slices

        var shapes: [[Int]] = []
        var inputVals: [[Float]] = []
        var targetVals: [[Float]] = []
        for b in loader {
            shapes.append(b.input.shape)
            inputVals.append(b.input.scalars())
            targetVals.append(b.target.scalars())
        }

        #expect(shapes.count == 2)                 // NOT one giant "batch"
        #expect(shapes == [[2, 2], [2, 2]])
        #expect(inputVals[0] == [0, 0, 1, 1])
        #expect(inputVals[1] == [2, 2, 3, 3])
        #expect(targetVals[0] == [0, 1])
        #expect(targetVals[1] == [2, 3])
    }

    @Test("last partial batch is smaller, not the whole set")
    func partialLastBatch() {
        let inputs = Tensor<Float>([0, 1, 2, 3, 4], shape: [5, 1])
        let targets = Tensor<Float>([0, 1, 2, 3, 4], shape: [5, 1])
        let loader = DataLoader(dataset: TensorDataset(inputs: inputs, targets: targets),
                                batchSize: 2)

        let sizes = loader.map { $0.input.shape[0] }
        #expect(sizes == [2, 2, 1])
    }

    @Test("TensorDataset subscript returns a single sample")
    func subscriptSingleSample() {
        let inputs = Tensor<Float>([10, 11,  20, 21,  30, 31], shape: [3, 2])
        let targets = Tensor<Float>([1, 2, 3], shape: [3, 1])
        let ds = TensorDataset(inputs: inputs, targets: targets)

        let s = ds[1]
        #expect(s.input.shape == [1, 2])
        #expect(s.input.scalars() == [20, 21])
        #expect(s.target.scalars() == [2])
    }

    @Test("data.stack stacks along a new leading axis")
    func stackNewAxis() {
        let a = Tensor<Float>([1, 2], shape: [2])
        let b = Tensor<Float>([3, 4], shape: [2])
        let stacked = data.stack([a, b])
        #expect(stacked.shape == [2, 2])
        #expect(stacked.scalars() == [1, 2, 3, 4])
    }
}
