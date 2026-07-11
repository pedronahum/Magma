// Magma - Data Loading Module
// Dataset and DataLoader for training neural networks

import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import _Differentiation
import LazyTensor
import StableHLO
import XLARuntime

// MARK: - Dataset Protocol

/// Protocol for datasets that can be used with DataLoader.
///
/// A dataset provides access to samples by index and knows its total size.
///
/// Example:
/// ```swift
/// struct MyDataset: Dataset {
///     typealias Element = (Tensor<Float>, Tensor<Float>)
///
///     var count: Int { return 1000 }
///
///     subscript(index: Int) -> Element {
///         let x = loadInput(index)
///         let y = loadLabel(index)
///         return (x, y)
///     }
/// }
/// ```
public protocol Dataset {
    associatedtype Element

    /// Total number of samples in the dataset.
    var count: Int { get }

    /// Access a single sample by index.
    subscript(index: Int) -> Element { get }

    /// Assemble a batch from the given sample indices.
    ///
    /// A default implementation (for the `(input, target)` element type)
    /// gathers each sample via `subscript` and concatenates along the batch
    /// axis; datasets backed by contiguous tensors override this for efficiency.
    func batch(indices: [Int]) -> Element
}

// MARK: - TensorDataset

/// A simple dataset wrapping tensors.
///
/// Stores input and target tensors and provides samples by index.
///
/// Example:
/// ```swift
/// let inputs = Tensor<Float>.randn([1000, 784])
/// let targets = Tensor<Float>.randn([1000, 10])
/// let dataset = TensorDataset(inputs: inputs, targets: targets)
/// ```
public struct TensorDataset: Dataset {
    public typealias Element = (input: Tensor<Float>, target: Tensor<Float>)

    /// Input tensor with shape [numSamples, ...]
    public let inputs: Tensor<Float>

    /// Target tensor with shape [numSamples, ...]
    public let targets: Tensor<Float>

    /// Number of samples
    public var count: Int { inputs.shape[0] }

    /// Creates a tensor dataset.
    ///
    /// - Parameters:
    ///   - inputs: Input tensor, first dimension is batch.
    ///   - targets: Target tensor, first dimension is batch.
    public init(inputs: Tensor<Float>, targets: Tensor<Float>) {
        precondition(inputs.shape[0] == targets.shape[0],
                    "Inputs and targets must have same number of samples")
        self.inputs = inputs
        self.targets = targets
    }

    public subscript(index: Int) -> Element {
        precondition(index >= 0 && index < count, "Index out of bounds")
        // A single sample, keeping a leading batch dimension of 1 so samples
        // concatenate cleanly into a batch.
        return (inputs.slice(start: index, size: 1), targets.slice(start: index, size: 1))
    }

    /// Efficient batch assembly. Contiguous ascending indices (the common
    /// non-shuffled loader case) use a single slice; arbitrary/shuffled index
    /// sets use gather. Mirrors SimpleBatchLoader.
    public func batch(indices: [Int]) -> Element {
        precondition(!indices.isEmpty, "batch(indices:) requires at least one index")
        if let first = indices.first,
           indices == Array(first..<(first + indices.count)) {
            return (inputs.slice(start: first, size: indices.count),
                    targets.slice(start: first, size: indices.count))
        }
        let indexTensor = Tensor<Float>(indices.map { Float($0) }, shape: [indices.count], on: inputs.device)
        return (inputs.gather(indices: indexTensor, axis: 0),
                targets.gather(indices: indexTensor, axis: 0))
    }
}

extension Dataset where Element == (input: Tensor<Float>, target: Tensor<Float>) {
    /// Default batching: gather each sample and concatenate along the batch axis.
    public func batch(indices: [Int]) -> Element {
        precondition(!indices.isEmpty, "batch(indices:) requires at least one index")
        let samples = indices.map { self[$0] }
        let inputs = samples[0].input.concat(with: samples.dropFirst().map { $0.input }, axis: 0)
        let targets = samples[0].target.concat(with: samples.dropFirst().map { $0.target }, axis: 0)
        return (inputs, targets)
    }
}

// MARK: - DataLoader

/// Iterates over a dataset in batches.
///
/// Provides batched iteration with optional shuffling.
///
/// Example:
/// ```swift
/// let loader = DataLoader(dataset: dataset, batchSize: 32, shuffle: true)
/// for (inputs, targets) in loader {
///     let output = model(inputs)
///     let loss = criterion(output, targets)
///     // ...
/// }
/// ```
public struct DataLoader<D: Dataset>: Sequence where D.Element == (input: Tensor<Float>, target: Tensor<Float>) {
    /// The underlying dataset
    public let dataset: D

    /// Number of samples per batch
    public let batchSize: Int

    /// Whether to shuffle indices each epoch
    public let shuffle: Bool

    /// Whether to drop the last incomplete batch
    public let dropLast: Bool

    /// Creates a data loader.
    ///
    /// - Parameters:
    ///   - dataset: The dataset to iterate over.
    ///   - batchSize: Number of samples per batch.
    ///   - shuffle: Whether to shuffle each epoch. Defaults to false.
    ///   - dropLast: Whether to drop incomplete last batch. Defaults to false.
    public init(
        dataset: D,
        batchSize: Int,
        shuffle: Bool = false,
        dropLast: Bool = false
    ) {
        precondition(batchSize > 0, "Batch size must be positive")
        self.dataset = dataset
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.dropLast = dropLast
    }

    /// Number of batches
    public var count: Int {
        if dropLast {
            return dataset.count / batchSize
        } else {
            return (dataset.count + batchSize - 1) / batchSize
        }
    }

    public func makeIterator() -> DataLoaderIterator<D> {
        DataLoaderIterator(loader: self)
    }
}

/// Iterator for DataLoader
public struct DataLoaderIterator<D: Dataset>: IteratorProtocol where D.Element == (input: Tensor<Float>, target: Tensor<Float>) {
    private let loader: DataLoader<D>
    private var indices: [Int]
    private var currentIndex: Int = 0

    init(loader: DataLoader<D>) {
        self.loader = loader
        self.indices = Array(0..<loader.dataset.count)
        if loader.shuffle {
            self.indices.shuffle()
        }
    }

    public mutating func next() -> (input: Tensor<Float>, target: Tensor<Float>)? {
        guard currentIndex < indices.count else { return nil }

        let endIndex = min(currentIndex + loader.batchSize, indices.count)
        let batchSize = endIndex - currentIndex

        // Skip incomplete batch if dropLast
        if loader.dropLast && batchSize < loader.batchSize {
            return nil
        }

        // Get batch indices
        let batchIndices = Array(indices[currentIndex..<endIndex])
        currentIndex = endIndex

        // Assemble the full batch from the selected indices (respects shuffle).
        return loader.dataset.batch(indices: batchIndices)
    }
}

// MARK: - Batch Utilities

/// Namespace for data utilities
public enum data {}

extension data {
    /// Stack multiple tensors into a batch.
    ///
    /// - Parameter tensors: Array of tensors with the same shape.
    /// - Returns: Stacked tensor with new batch dimension.
    public static func stack(_ tensors: [Tensor<Float>]) -> Tensor<Float> {
        guard !tensors.isEmpty else {
            fatalError("Cannot stack empty array")
        }
        // Stack along a new leading axis (e.g. [D] tensors -> [N, D]).
        return Tensor<Float>.stack(tensors, axis: 0)
    }
}

// MARK: - SimpleBatchLoader

/// A simple batch loader that works with pre-batched tensors.
///
/// This is a simpler alternative to DataLoader when data is already
/// organized as batched tensors. Supports shuffling for randomized training.
///
/// Example:
/// ```swift
/// let loader = SimpleBatchLoader(
///     inputs: trainX,    // [numSamples, ...]
///     targets: trainY,   // [numSamples, ...]
///     batchSize: 32,
///     shuffle: true      // Randomize batch order each epoch
/// )
///
/// for epoch in 0..<numEpochs {
///     for batch in loader {
///         let loss = trainStep(model, batch.input, batch.target)
///     }
/// }
/// ```
public struct SimpleBatchLoader: Sequence {
    public typealias Batch = (input: Tensor<Float>, target: Tensor<Float>)

    /// All input data
    public let inputs: Tensor<Float>

    /// All target data
    public let targets: Tensor<Float>

    /// Batch size
    public let batchSize: Int

    /// Whether to shuffle indices each iteration
    public let shuffle: Bool

    /// Whether to drop the last incomplete batch
    public let dropLast: Bool

    /// Number of samples
    public var numSamples: Int { inputs.shape[0] }

    /// Number of batches
    public var numBatches: Int {
        if dropLast {
            return numSamples / batchSize
        } else {
            return (numSamples + batchSize - 1) / batchSize
        }
    }

    /// Creates a simple batch loader.
    ///
    /// - Parameters:
    ///   - inputs: Input tensor with first dimension as batch.
    ///   - targets: Target tensor with first dimension as batch.
    ///   - batchSize: Number of samples per batch.
    ///   - shuffle: Whether to shuffle samples each epoch. Defaults to false.
    ///   - dropLast: Whether to drop the last incomplete batch. Defaults to false.
    public init(
        inputs: Tensor<Float>,
        targets: Tensor<Float>,
        batchSize: Int,
        shuffle: Bool = false,
        dropLast: Bool = false
    ) {
        precondition(inputs.shape[0] == targets.shape[0],
            "SimpleBatchLoader: inputs and targets must have same number of samples. " +
            "Got inputs with \(inputs.shape[0]) samples and targets with \(targets.shape[0]) samples.")
        precondition(batchSize > 0,
            "SimpleBatchLoader: batch size must be positive, got \(batchSize).")
        self.inputs = inputs
        self.targets = targets
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.dropLast = dropLast
    }

    public func makeIterator() -> SimpleBatchIterator {
        SimpleBatchIterator(loader: self)
    }

    /// Get a specific batch by index (uses sequential ordering, ignores shuffle).
    public func batch(_ index: Int) -> Batch {
        precondition(index >= 0 && index < numBatches,
            "SimpleBatchLoader: batch index \(index) out of bounds. Valid range: 0..<\(numBatches).")

        let startIdx = index * batchSize
        let endIdx = Swift.min(startIdx + batchSize, numSamples)
        let actualBatchSize = endIdx - startIdx

        // Create batch tensors using slice operation
        let inputBatch = inputs.slice(start: startIdx, size: actualBatchSize)
        let targetBatch = targets.slice(start: startIdx, size: actualBatchSize)

        return (inputBatch, targetBatch)
    }

    /// Get a batch using shuffled indices.
    internal func batch(indices: [Int]) -> Batch {
        // For shuffled batches, we need to gather from the indices
        // This uses the gather operation for efficient GPU/TPU execution
        let indexTensor = Tensor<Float>(indices.map { Float($0) }, shape: [indices.count], on: inputs.device)

        let inputBatch = inputs.gather(indices: indexTensor, axis: 0)
        let targetBatch = targets.gather(indices: indexTensor, axis: 0)

        return (inputBatch, targetBatch)
    }
}

/// Iterator for SimpleBatchLoader
public struct SimpleBatchIterator: IteratorProtocol {
    private let loader: SimpleBatchLoader
    private var indices: [Int]
    private var currentIndex: Int = 0

    init(loader: SimpleBatchLoader) {
        self.loader = loader
        self.indices = Array(0..<loader.numSamples)
        if loader.shuffle {
            self.indices.shuffle()
        }
    }

    public mutating func next() -> SimpleBatchLoader.Batch? {
        guard currentIndex < indices.count else { return nil }

        let endIndex = Swift.min(currentIndex + loader.batchSize, indices.count)
        let batchSize = endIndex - currentIndex

        // Skip incomplete batch if dropLast is enabled
        if loader.dropLast && batchSize < loader.batchSize {
            return nil
        }

        let batchIndices = Array(indices[currentIndex..<endIndex])
        currentIndex = endIndex

        if loader.shuffle {
            // Use gather-based batch for shuffled indices
            return loader.batch(indices: batchIndices)
        } else {
            // Use efficient slice-based batch for sequential access
            let startIdx = batchIndices[0]
            return loader.batch(startIdx / loader.batchSize)
        }
    }
}

// MARK: - Tensor Slice Extension

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {
    /// Slice tensor along first dimension.
    ///
    /// - Parameters:
    ///   - start: Starting index.
    ///   - size: Number of elements to take.
    /// - Returns: Sliced tensor.
    public func slice(start: Int, size: Int) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        var newShape = shape
        newShape[0] = size

        let handle = LazyTensorHandle(
            id: id,
            shape: newShape,
            dtype: dtype,
            device: device
        )

        // Create slice operation
        var startIndices = Array(repeating: 0, count: rank)
        startIndices[0] = start

        var limitIndices = shape
        limitIndices[0] = start + size

        let strides = Array(repeating: 1, count: rank)

        handle.irNode = .operation(
            op: .slice,
            inputs: [self.handle],
            attributes: [
                "start": startIndices,
                "limit": limitIndices,
                "strides": strides
            ]
        )

        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Slice a tensor along a specific axis
    ///
    /// - Parameters:
    ///   - axis: The axis to slice along (0-indexed, supports negative indexing).
    ///   - start: Starting index along the axis.
    ///   - size: Number of elements to take along the axis.
    /// - Returns: Sliced tensor with reduced size on the specified axis.
    public func sliceAxis(axis: Int, start: Int, size: Int) -> Tensor {
        let normalizedAxis = axis < 0 ? rank + axis : axis
        precondition(normalizedAxis >= 0 && normalizedAxis < rank, "Axis out of range")
        precondition(start >= 0 && start + size <= shape[normalizedAxis], "Slice range out of bounds")

        let id = TensorRegistry.shared.nextTensorId()
        var newShape = shape
        newShape[normalizedAxis] = size

        let handle = LazyTensorHandle(
            id: id,
            shape: newShape,
            dtype: dtype,
            device: device
        )

        // Create slice operation with proper start/limit for all dimensions
        var startIndices = Array(repeating: 0, count: rank)
        startIndices[normalizedAxis] = start

        var limitIndices = shape
        limitIndices[normalizedAxis] = start + size

        let strides = Array(repeating: 1, count: rank)

        handle.irNode = .operation(
            op: .slice,
            inputs: [self.handle],
            attributes: [
                "start": startIndices,
                "limit": limitIndices,
                "strides": strides
            ]
        )

        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// VJP for sliceAxis
    @derivative(of: sliceAxis)
    public func vjpSliceAxis(axis: Int, start: Int, size: Int) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let normalizedAxis = axis < 0 ? rank + axis : axis
        let originalShape = self.shape
        let result = self.sliceAxis(axis: axis, start: start, size: size)
        return (result, { v in
            v.padToShapeAxis(originalShape, axis: normalizedAxis, startIndex: start)
        })
    }

    /// Pad tensor to target shape along a specific axis (helper for sliceAxis gradient)
    internal func padToShapeAxis(_ targetShape: [Int], axis: Int, startIndex: Int) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: targetShape,
            dtype: dtype,
            device: device
        )

        var lowPad = Array(repeating: 0, count: rank)
        lowPad[axis] = startIndex

        var highPad = Array(repeating: 0, count: rank)
        highPad[axis] = targetShape[axis] - startIndex - shape[axis]

        handle.irNode = .operation(
            op: .pad,
            inputs: [self.handle],
            attributes: [
                "lowPad": lowPad,
                "highPad": highPad,
                "interiorPad": Array(repeating: 0, count: rank)
            ]
        )

        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    // MARK: - Advanced Slicing

    /// Slice tensor along a specific axis with start, stop, and step.
    ///
    /// Supports negative indices (counting from end) and strided access.
    /// Similar to Python/NumPy slicing: `tensor[start:stop:step]`
    ///
    /// - Parameters:
    ///   - axis: The axis to slice along (supports negative indexing).
    ///   - start: Starting index (nil = beginning, negative = from end).
    ///   - stop: Ending index (nil = end, negative = from end).
    ///   - step: Step size (default 1, must be positive for now).
    /// - Returns: Sliced tensor.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>.randn([10, 20, 30])
    /// let y = x.slice(axis: 0, start: 2, stop: 8, step: 2)  // Every other from 2 to 8
    /// let z = x.slice(axis: -1, start: -5, stop: nil)  // Last 5 along final axis
    /// ```
    public func slice(axis: Int, start: Int? = nil, stop: Int? = nil, step: Int = 1) -> Tensor {
        precondition(step > 0, "Step must be positive (negative steps not yet supported)")

        let normalizedAxis = axis < 0 ? rank + axis : axis
        precondition(normalizedAxis >= 0 && normalizedAxis < rank, "Axis out of range")

        let dimSize = shape[normalizedAxis]

        // Normalize start index
        var startIdx: Int
        if let s = start {
            startIdx = s < 0 ? dimSize + s : s
            startIdx = Swift.max(0, Swift.min(startIdx, dimSize))
        } else {
            startIdx = 0
        }

        // Normalize stop index
        var stopIdx: Int
        if let s = stop {
            stopIdx = s < 0 ? dimSize + s : s
            stopIdx = Swift.max(0, Swift.min(stopIdx, dimSize))
        } else {
            stopIdx = dimSize
        }

        // Calculate output size
        let sliceSize = Swift.max(0, (stopIdx - startIdx + step - 1) / step)

        if sliceSize == 0 {
            // Return empty tensor - create with zero size on that axis
            var newShape = shape
            newShape[normalizedAxis] = 0
            return Tensor.zeros(newShape, on: device)
        }

        let id = TensorRegistry.shared.nextTensorId()
        var newShape = shape
        newShape[normalizedAxis] = sliceSize

        let handle = LazyTensorHandle(
            id: id,
            shape: newShape,
            dtype: dtype,
            device: device
        )

        // Build start/limit/strides for all dimensions
        var startIndices = Array(repeating: 0, count: rank)
        startIndices[normalizedAxis] = startIdx

        var limitIndices = shape
        limitIndices[normalizedAxis] = stopIdx

        var strides = Array(repeating: 1, count: rank)
        strides[normalizedAxis] = step

        handle.irNode = .operation(
            op: .slice,
            inputs: [self.handle],
            attributes: [
                "start": startIndices,
                "limit": limitIndices,
                "strides": strides
            ]
        )

        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// VJP for advanced slice with step
    @derivative(of: slice(axis:start:stop:step:))
    public func vjpSliceAdvanced(axis: Int, start: Int? = nil, stop: Int? = nil, step: Int = 1)
        -> (value: Tensor, pullback: (Tensor) -> Tensor)
    {
        let normalizedAxis = axis < 0 ? rank + axis : axis
        let dimSize = shape[normalizedAxis]

        // Normalize indices
        var startIdx = start ?? 0
        if startIdx < 0 { startIdx = dimSize + startIdx }
        startIdx = Swift.max(0, Swift.min(startIdx, dimSize))

        var stopIdx = stop ?? dimSize
        if stopIdx < 0 { stopIdx = dimSize + stopIdx }
        stopIdx = Swift.max(0, Swift.min(stopIdx, dimSize))

        let originalShape = self.shape
        let result = self.slice(axis: axis, start: start, stop: stop, step: step)

        return (result, { v in
            if step == 1 {
                // Simple case: just pad
                return v.padToShapeAxis(originalShape, axis: normalizedAxis, startIndex: startIdx)
            } else {
                // Strided case: need to scatter values back
                // Create indices for where each output element came from
                let sliceSize = v.shape[normalizedAxis]
                var indices: [Float] = []
                for i in 0..<sliceSize {
                    indices.append(Float(startIdx + i * step))
                }

                // Create 1D index tensor for scatter
                // scatter expects: indices [n], updates [..., n, ...] where n is at axis position
                let indexTensor = Tensor<Float>(indices, shape: [sliceSize], on: v.device)

                // Scatter gradient back
                return Tensor<Scalar>.zeros(originalShape, on: v.device)
                    .scatter(indices: indexTensor, updates: v, axis: normalizedAxis)
            }
        })
    }

    /// VJP for slice
    @derivative(of: slice)
    public func vjpSlice(start: Int, size: Int) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let originalShape = self.shape
        let result = self.slice(start: start, size: size)
        return (result, { v in
            // Pad gradient back to original shape
            v.padToShape(originalShape, startIndex: start)
        })
    }

    /// Pad tensor to target shape (helper for slice gradient)
    internal func padToShape(_ targetShape: [Int], startIndex: Int) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: targetShape,
            dtype: dtype,
            device: device
        )

        // Create pad operation
        var lowPad = Array(repeating: 0, count: rank)
        lowPad[0] = startIndex

        var highPad = Array(repeating: 0, count: rank)
        highPad[0] = targetShape[0] - startIndex - shape[0]

        handle.irNode = .operation(
            op: .pad,
            inputs: [self.handle],
            attributes: [
                "lowPad": lowPad,
                "highPad": highPad,
                "interiorPad": Array(repeating: 0, count: rank)
            ]
        )

        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    // MARK: - Gather Operation

    /// Gather slices from tensor along an axis using indices.
    ///
    /// Similar to `torch.index_select` - selects slices along the specified axis.
    /// For each index in indices, selects the corresponding slice from self.
    ///
    /// - Parameters:
    ///   - indices: 1D index tensor specifying which slices to gather.
    ///   - axis: Axis along which to gather (default: 0).
    /// - Returns: Gathered tensor with shape where the gather axis is replaced by indices shape.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>.randn([5, 4])  // [5, 4]
    /// let idx = Tensor<Float>([0, 2, 4], shape: [3])  // [3]
    /// let gathered = x.gather(indices: idx, axis: 0)  // [3, 4] - selects rows 0, 2, 4
    /// ```
    public func gather(indices: Tensor<Float>, axis: Int = 0) -> Tensor {
        let normalizedAxis = axis < 0 ? rank + axis : axis
        precondition(normalizedAxis >= 0 && normalizedAxis < rank, "axis out of range")

        // Only 1-D indices are supported: this is index-select along `axis`
        // (the axis dimension is replaced by the number of indices). The prior
        // multi-dimensional path produced incorrect values (the gather
        // dimension numbers were wrong), so it now fails loudly instead of
        // silently returning garbage. All in-tree callers (data loaders,
        // embedding lookup, image flips) use 1-D indices.
        precondition(indices.rank == 1,
            "gather supports only 1-D indices (index-select along axis); got indices shape \(indices.shape).")
        var resultShape = self.shape
        resultShape[normalizedAxis] = indices.shape[0]

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: dtype,
            device: device
        )

        handle.irNode = .operation(
            op: .gather,
            inputs: [self.handle, indices.handle],
            attributes: ["axis": normalizedAxis]
        )

        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// VJP for gather operation
    @derivative(of: gather, wrt: self)
    public func vjpGather(indices: Tensor<Float>, axis: Int = 0) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.gather(indices: indices, axis: axis)
        let originalShape = self.shape
        return (result, { [selfDevice = self.device] v in
            // Use oneHot + matmul instead of scatter to avoid region-based MLIR ops.
            // This also correctly accumulates gradients for duplicate indices.
            let n = originalShape[axis]
            let oneHot = Tensor<Float>.oneHot(indices, numClasses: n, on: selfDevice)
            // oneHot: [numIndices, n], v: [numIndices, D]
            // grad = oneHot^T @ v = [n, D]
            let vFloat = Tensor<Float>(handle: v.handle)
            let grad = oneHot.transpose().matmul(vFloat)
            return Tensor<Scalar>(handle: grad.handle)
        })
    }

    // MARK: - Scatter Operation

    /// Scatter slices into tensor at positions specified by indices.
    ///
    /// Similar to inverse of gather - scatters slices from updates into self at positions given by indices.
    ///
    /// - Parameters:
    ///   - indices: 1D index tensor specifying where to scatter.
    ///   - updates: Tensor of slices to scatter.
    ///   - axis: Axis along which to scatter (default: 0).
    /// - Returns: New tensor with scattered values.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>.zeros([5, 4])  // [5, 4]
    /// let idx = Tensor<Float>([0, 2, 4], shape: [3])  // [3] - indices
    /// let updates = Tensor<Float>.ones([3, 4])  // [3, 4] - slices to scatter
    /// let result = x.scatter(indices: idx, updates: updates, axis: 0)  // rows 0, 2, 4 become ones
    /// ```
    public func scatter(indices: Tensor<Float>, updates: Tensor, axis: Int = 0) -> Tensor {
        let normalizedAxis = axis < 0 ? rank + axis : axis
        precondition(normalizedAxis >= 0 && normalizedAxis < rank, "axis out of range")

        // Validate updates shape based on scatter mode:
        // Mode 1 (slice scatter): indices is 1D, updates has shape [..., indices.count, ...]
        // Mode 2 (element scatter): updates has same shape as indices
        // Mode 3 (multi-dim gather inverse): updates has shape with axis replaced by indices.shape
        var expectedSliceScatterShape = self.shape
        if indices.rank == 1 {
            expectedSliceScatterShape[normalizedAxis] = indices.shape[0]
        } else {
            expectedSliceScatterShape[normalizedAxis] = indices.elementCount
        }

        // Mode 3: For multi-dim indices, updates may have indices.shape inserted at axis
        var expectedMultiDimScatterShape = self.shape
        expectedMultiDimScatterShape.remove(at: normalizedAxis)
        expectedMultiDimScatterShape.insert(contentsOf: indices.shape, at: normalizedAxis)

        let validSliceScatter = updates.shape == expectedSliceScatterShape
        let validElementScatter = updates.shape == indices.shape
        let validMultiDimScatter = updates.shape == expectedMultiDimScatterShape
        precondition(validSliceScatter || validElementScatter || validMultiDimScatter,
                     "updates shape \(updates.shape) incompatible with scatter into \(shape) at axis \(normalizedAxis)")

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,  // Result has same shape as input
            dtype: dtype,
            device: device
        )

        handle.irNode = .operation(
            op: .scatter,
            inputs: [self.handle, indices.handle, updates.handle],
            attributes: ["axis": normalizedAxis]
        )

        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// VJP for scatter operation (with respect to self and updates, indices are not differentiable)
    @derivative(of: scatter, wrt: (self, updates))
    public func vjpScatter(indices: Tensor<Float>, updates: Tensor, axis: Int = 0) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        let result = self.scatter(indices: indices, updates: updates, axis: axis)
        let isElementScatter = updates.shape == indices.shape
        let updatesShape = updates.shape
        let device = self.device
        return (result, { v in
            // Gradient for input: just pass through the gradient (updates overwrite)
            let inputGrad = v
            // Gradient for updates: gather from the output gradient at scatter positions
            let updatesGrad: Tensor<Scalar>
            if isElementScatter {
                // Element scatter mode: gather individual elements from gradient
                // For now, use slice gather and then reshape if shapes differ
                let gathered = v.gather(indices: indices, axis: axis)
                if gathered.shape == updatesShape {
                    updatesGrad = gathered
                } else {
                    // Shapes don't match due to gather semantics - use zeros as conservative approx
                    // TODO: Implement proper gatherElements for element-wise gathering
                    updatesGrad = Tensor<Scalar>.zeros(updatesShape, on: device)
                }
            } else {
                // Slice scatter mode: use regular gather
                updatesGrad = v.gather(indices: indices, axis: axis)
            }
            return (inputGrad, updatesGrad)
        })
    }

    // MARK: - Expand Operation

    /// Expand tensor to a larger size by broadcasting.
    ///
    /// Similar to PyTorch's `expand`. Only dimensions of size 1 can be expanded.
    ///
    /// - Parameter targetShape: The target shape to expand to.
    /// - Returns: Expanded tensor.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>.ones([1, 4])  // [1, 4]
    /// let expanded = x.expand([3, 4])  // [3, 4]
    /// ```
    public func expand(_ targetShape: [Int]) -> Tensor {
        precondition(targetShape.count == rank, "expand requires same number of dimensions")

        // Validate that expansion is valid
        for i in 0..<rank {
            if shape[i] != targetShape[i] {
                precondition(shape[i] == 1, "Cannot expand dimension \(i) from \(shape[i]) to \(targetShape[i]); only size-1 dimensions can be expanded")
            }
        }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: targetShape,
            dtype: dtype,
            device: device
        )

        handle.irNode = .operation(
            op: .broadcast,
            inputs: [self.handle],
            attributes: ["shape": targetShape]
        )

        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// VJP for expand operation
    @derivative(of: expand)
    public func vjpExpand(_ targetShape: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.expand(targetShape)
        let originalShape = self.shape
        return (result, { v in
            // Sum along expanded dimensions
            var axes: [Int] = []
            for i in 0..<originalShape.count {
                if originalShape[i] == 1 && targetShape[i] > 1 {
                    axes.append(i)
                }
            }
            if axes.isEmpty {
                return v
            }
            // Reduce sum along expanded axes
            return v.sum(dims: axes, keepDims: true)
        })
    }

    /// Sum along specified dimensions.
    ///
    /// - Parameters:
    ///   - dims: Dimensions to sum along.
    ///   - keepDims: Whether to keep the reduced dimensions as size 1.
    /// - Returns: Reduced tensor.
    public func sum(dims: [Int], keepDims: Bool = false) -> Tensor {
        let normalizedDims = dims.map { $0 < 0 ? rank + $0 : $0 }

        var resultShape: [Int]
        if keepDims {
            resultShape = shape
            for dim in normalizedDims {
                resultShape[dim] = 1
            }
        } else {
            resultShape = shape.enumerated()
                .filter { !normalizedDims.contains($0.offset) }
                .map { $0.element }
        }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: dtype,
            device: device
        )

        handle.irNode = .operation(
            op: .reduceSum,
            inputs: [self.handle],
            attributes: ["axes": normalizedDims, "keepDims": keepDims]
        )

        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    // MARK: - Concatenate Operation

    /// Concatenate tensors along an existing axis.
    ///
    /// All tensors must have the same shape except in the concatenation dimension.
    ///
    /// - Parameters:
    ///   - tensors: Array of tensors to concatenate.
    ///   - axis: Axis along which to concatenate (default: 0).
    /// - Returns: Concatenated tensor.
    ///
    /// Example:
    /// ```swift
    /// let a = Tensor<Float>.ones([2, 3])
    /// let b = Tensor<Float>.zeros([2, 3])
    /// let c = Tensor.concat([a, b], axis: 0)  // [4, 3]
    /// let d = Tensor.concat([a, b], axis: 1)  // [2, 6]
    /// ```
    public static func concat(_ tensors: [Tensor], axis: Int = 0) -> Tensor {
        precondition(!tensors.isEmpty,
            "concat: cannot concatenate empty array of tensors.")

        let first = tensors[0]
        let normalizedAxis = axis < 0 ? first.rank + axis : axis
        precondition(normalizedAxis >= 0 && normalizedAxis < first.rank,
            "concat: axis \(axis) out of range for tensor with rank \(first.rank). " +
            "Valid range: \(-first.rank) to \(first.rank - 1).")

        // Validate shapes match except on concat dimension
        for (idx, tensor) in tensors.dropFirst().enumerated() {
            precondition(tensor.rank == first.rank,
                "concat: all tensors must have same rank. " +
                "Tensor 0 has rank \(first.rank), tensor \(idx + 1) has rank \(tensor.rank).")
            for i in 0..<first.rank {
                if i != normalizedAxis {
                    precondition(tensor.shape[i] == first.shape[i],
                        "concat: shape mismatch at dimension \(i). " +
                        "Tensor 0 has shape \(first.shape), tensor \(idx + 1) has shape \(tensor.shape). " +
                        "All dimensions except axis \(normalizedAxis) must match.")
                }
            }
        }

        // Compute result shape
        var resultShape = first.shape
        resultShape[normalizedAxis] = tensors.map { $0.shape[normalizedAxis] }.reduce(0, +)

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: first.dtype,
            device: first.device
        )

        handle.irNode = .operation(
            op: .concatenate,
            inputs: tensors.map { $0.handle },
            attributes: ["dimension": normalizedAxis]
        )

        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Concatenate this tensor with others along an axis.
    ///
    /// Convenience method that calls the static `concat`.
    public func concat(with others: [Tensor], axis: Int = 0) -> Tensor {
        Tensor.concat([self] + others, axis: axis)
    }

    // MARK: - Stack Operation

    /// Stack tensors along a new axis.
    ///
    /// All tensors must have the same shape. The result has one more dimension
    /// than the input tensors.
    ///
    /// - Parameters:
    ///   - tensors: Array of tensors to stack.
    ///   - axis: Axis at which to insert the new dimension (default: 0).
    /// - Returns: Stacked tensor with shape [n, ...] or [..., n] depending on axis.
    ///
    /// Example:
    /// ```swift
    /// let a = Tensor<Float>.ones([2, 3])
    /// let b = Tensor<Float>.zeros([2, 3])
    /// let c = Tensor.stack([a, b], axis: 0)  // [2, 2, 3]
    /// let d = Tensor.stack([a, b], axis: 1)  // [2, 2, 3]
    /// let e = Tensor.stack([a, b], axis: 2)  // [2, 3, 2]
    /// ```
    public static func stack(_ tensors: [Tensor], axis: Int = 0) -> Tensor {
        precondition(!tensors.isEmpty,
            "stack: cannot stack empty array of tensors.")

        let first = tensors[0]
        let normalizedAxis = axis < 0 ? first.rank + 1 + axis : axis
        precondition(normalizedAxis >= 0 && normalizedAxis <= first.rank,
            "stack: axis \(axis) out of range for tensor with rank \(first.rank). " +
            "Valid range: \(-(first.rank + 1)) to \(first.rank) (result will have rank \(first.rank + 1)).")

        // Validate all shapes match
        for (idx, tensor) in tensors.dropFirst().enumerated() {
            precondition(tensor.shape == first.shape,
                "stack: all tensors must have same shape. " +
                "Tensor 0 has shape \(first.shape), tensor \(idx + 1) has shape \(tensor.shape).")
        }

        // Unsqueeze each tensor at the target axis, then concatenate
        let unsqueezed = tensors.map { tensor -> Tensor in
            var newShape = tensor.shape
            newShape.insert(1, at: normalizedAxis)
            return tensor.reshape(newShape)
        }

        return Tensor.concat(unsqueezed, axis: normalizedAxis)
    }

    /// Stack this tensor with others along a new axis.
    ///
    /// Convenience method that calls the static `stack`.
    public func stack(with others: [Tensor], axis: Int = 0) -> Tensor {
        Tensor.stack([self] + others, axis: axis)
    }
}

// MARK: - Element Indexing

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {
    /// Access a single element from a 1D tensor by index.
    ///
    /// Returns a scalar tensor (0-dimensional) containing the element at the given index.
    /// This operation is differentiable.
    ///
    /// - Parameter index: The index of the element to access (supports negative indexing).
    /// - Returns: A scalar tensor containing the element.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>([1.0, 2.0, 3.0, 4.0, 5.0], shape: [5])
    /// let elem = x[2]  // Tensor containing 3.0
    /// let last = x[-1]  // Tensor containing 5.0 (negative indexing)
    /// ```
    @differentiable(reverse)
    public subscript(index: Int) -> Tensor {
        let normalizedIndex = index < 0 ? shape[0] + index : index
        precondition(normalizedIndex >= 0 && normalizedIndex < shape[0], "Index \(index) out of bounds for dimension 0 with size \(shape[0])")

        // Slice to get a single element, then squeeze to scalar
        let sliced = self.sliceAxis(axis: 0, start: normalizedIndex, size: 1)

        // Remove the first dimension to get the element
        if rank == 1 {
            // For 1D tensor, result is a scalar
            return sliced.reshape([])
        } else {
            // For ND tensor, result has rank N-1
            let newShape = Array(shape.dropFirst())
            return sliced.reshape(newShape)
        }
    }

    /// VJP for single-element subscript indexing
    @derivative(of: subscript(_:))
    public func vjpSubscript(index: Int) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let normalizedIndex = index < 0 ? shape[0] + index : index
        let result = self[index]
        let originalShape = self.shape

        return (result, { v in
            // Gradient needs to go back to the original position
            // Create a zero tensor of original shape and scatter the gradient
            if self.rank == 1 {
                // For 1D: reshape gradient to [1] and pad
                let gradWithDim = v.reshape([1])
                return gradWithDim.padToShapeAxis(originalShape, axis: 0, startIndex: normalizedIndex)
            } else {
                // For ND: reshape gradient to [1, ...rest] and pad
                var gradShape = [1] + v.shape
                let gradWithDim = v.reshape(gradShape)
                return gradWithDim.padToShapeAxis(originalShape, axis: 0, startIndex: normalizedIndex)
            }
        })
    }

    /// Access a row from a 2D tensor by index.
    ///
    /// Returns a 1D tensor containing the specified row.
    /// This operation is differentiable.
    ///
    /// - Parameter row: The row index to access (supports negative indexing).
    /// - Returns: A 1D tensor containing the row.
    ///
    /// Example:
    /// ```swift
    /// let matrix = Tensor<Float>([[1, 2, 3], [4, 5, 6]], shape: [2, 3])
    /// let row = matrix.row(0)  // Tensor [1, 2, 3]
    /// let lastRow = matrix.row(-1)  // Tensor [4, 5, 6]
    /// ```
    @differentiable(reverse)
    public func row(_ index: Int) -> Tensor {
        precondition(rank >= 1, "row() requires at least 1D tensor")
        return self[index]
    }

    /// VJP for row access
    @derivative(of: row)
    public func vjpRow(_ index: Int) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.row(index)
        let originalShape = self.shape
        let normalizedIndex = index < 0 ? shape[0] + index : index

        return (result, { v in
            // Reshape gradient to have leading dimension of 1
            var gradShape = [1] + v.shape
            // Ensure gradShape matches originalShape dimensions
            if gradShape.count < originalShape.count {
                gradShape = [1] + Array(repeating: 1, count: originalShape.count - gradShape.count - 1) + v.shape
            }
            let gradWithDim = v.reshape(gradShape)
            return gradWithDim.padToShapeAxis(originalShape, axis: 0, startIndex: normalizedIndex)
        })
    }

    /// Access an element from a 2D tensor by row and column indices.
    ///
    /// Returns a scalar tensor containing the element at [row, col].
    /// This operation is differentiable.
    ///
    /// - Parameters:
    ///   - row: The row index (supports negative indexing).
    ///   - col: The column index (supports negative indexing).
    /// - Returns: A scalar tensor containing the element.
    ///
    /// Example:
    /// ```swift
    /// let matrix = Tensor<Float>([[1, 2, 3], [4, 5, 6]], shape: [2, 3])
    /// let elem = matrix[0, 1]  // Tensor containing 2.0
    /// let corner = matrix[-1, -1]  // Tensor containing 6.0
    /// ```
    @differentiable(reverse)
    public subscript(row: Int, col: Int) -> Tensor {
        precondition(rank == 2, "2D subscript requires 2D tensor, got rank \(rank)")
        let rowTensor = self[row]  // Get the row (1D)
        return rowTensor[col]  // Get the element (scalar)
    }

    /// VJP for 2D subscript
    @derivative(of: subscript(_:_:))
    public func vjpSubscript2D(row: Int, col: Int) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self[row, col]
        let originalShape = self.shape
        let normalizedRow = row < 0 ? shape[0] + row : row
        let normalizedCol = col < 0 ? shape[1] + col : col

        return (result, { v in
            // Create zero tensor and place gradient at [row, col]
            // v is scalar, need to place it at position [row, col] in original shape
            let gradRow = v.reshape([1]).padToShapeAxis([originalShape[1]], axis: 0, startIndex: normalizedCol)
            let gradFull = gradRow.reshape([1, originalShape[1]]).padToShapeAxis(originalShape, axis: 0, startIndex: normalizedRow)
            return gradFull
        })
    }
}

// MARK: - MNIST Dataset

/// MNIST dataset loader that downloads and parses the MNIST data.
///
/// The MNIST database of handwritten digits has 60,000 training images
/// and 10,000 test images. Each image is 28x28 grayscale.
///
/// Example:
/// ```swift
/// let mnist = try MNIST(split: .train, normalize: true)
/// let loader = SimpleBatchLoader(inputs: mnist.images, targets: mnist.labels, batchSize: 32)
/// for batch in loader {
///     let output = model(batch.input)
///     // ...
/// }
/// ```
public struct MNIST {
    /// Dataset split
    public enum Split {
        case train
        case test
    }

    /// Image tensor with shape [N, 784] (flattened) or [N, 28, 28, 1] (2D)
    public let images: Tensor<Float>

    /// Label tensor with shape [N, 10] (one-hot encoded)
    public let labels: Tensor<Float>

    /// Number of samples
    public var count: Int { images.shape[0] }

    /// Base URL for MNIST data (using a reliable mirror)
    private static let baseURL = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    /// MNIST file names
    private static let trainImagesFile = "train-images-idx3-ubyte.gz"
    private static let trainLabelsFile = "train-labels-idx1-ubyte.gz"
    private static let testImagesFile = "t10k-images-idx3-ubyte.gz"
    private static let testLabelsFile = "t10k-labels-idx1-ubyte.gz"

    /// Creates an MNIST dataset.
    ///
    /// - Parameters:
    ///   - split: Training or test split.
    ///   - normalize: Whether to normalize pixel values to [0, 1]. Default true.
    ///   - flatten: Whether to flatten images to [N, 784]. Default true.
    ///   - oneHot: Whether to one-hot encode labels. Default true.
    ///   - dataDir: Directory to cache downloaded data. Default ~/.magma/data/mnist.
    ///   - device: Device to place tensors on.
    /// - Throws: If data download or parsing fails.
    public init(
        split: Split = .train,
        normalize: Bool = true,
        flatten: Bool = true,
        oneHot: Bool = true,
        dataDir: String? = nil,
        device: Device = .default
    ) throws {
        let cacheDir = dataDir ?? MNIST.defaultDataDir()

        // Ensure directory exists
        try FileManager.default.createDirectory(
            atPath: cacheDir,
            withIntermediateDirectories: true,
            attributes: nil
        )

        // Select files based on split
        let (imagesFile, labelsFile) = switch split {
        case .train:
            (MNIST.trainImagesFile, MNIST.trainLabelsFile)
        case .test:
            (MNIST.testImagesFile, MNIST.testLabelsFile)
        }

        // Download and decompress if needed
        let imagesPath = try MNIST.ensureFile(imagesFile, in: cacheDir)
        let labelsPath = try MNIST.ensureFile(labelsFile, in: cacheDir)

        // Parse IDX files
        let (imageData, numImages, rows, cols) = try MNIST.parseIDX3(path: imagesPath)
        let labelData = try MNIST.parseIDX1(path: labelsPath)

        precondition(imageData.count == numImages * rows * cols,
                    "Image data size mismatch")
        precondition(labelData.count == numImages,
                    "Label data size mismatch")

        // Convert to float and optionally normalize
        var floatImages = imageData.map { Float($0) }
        if normalize {
            floatImages = floatImages.map { $0 / 255.0 }
        }

        // Create image tensor
        if flatten {
            self.images = Tensor<Float>(floatImages, shape: [numImages, rows * cols], on: device)
        } else {
            self.images = Tensor<Float>(floatImages, shape: [numImages, rows, cols, 1], on: device)
        }

        // Create label tensor
        if oneHot {
            // One-hot encode: [N] -> [N, 10]
            var oneHotData = [Float](repeating: 0.0, count: numImages * 10)
            for (i, label) in labelData.enumerated() {
                oneHotData[i * 10 + Int(label)] = 1.0
            }
            self.labels = Tensor<Float>(oneHotData, shape: [numImages, 10], on: device)
        } else {
            // Just convert to float: [N]
            let floatLabels = labelData.map { Float($0) }
            self.labels = Tensor<Float>(floatLabels, shape: [numImages], on: device)
        }
    }

    /// Default data directory
    private static func defaultDataDir() -> String {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/.magma/data/mnist"
    }

    /// Ensure a file exists, downloading and decompressing if needed
    private static func ensureFile(_ filename: String, in directory: String) throws -> String {
        // Remove .gz extension for decompressed path
        let decompressedName = filename.hasSuffix(".gz")
            ? String(filename.dropLast(3))
            : filename
        let decompressedPath = "\(directory)/\(decompressedName)"

        // Check if decompressed file exists
        if FileManager.default.fileExists(atPath: decompressedPath) {
            return decompressedPath
        }

        // Check if compressed file exists
        let compressedPath = "\(directory)/\(filename)"
        if !FileManager.default.fileExists(atPath: compressedPath) {
            // Download the file
            try downloadFile(filename, to: compressedPath)
        }

        // Decompress if needed
        if filename.hasSuffix(".gz") {
            try decompressGzip(from: compressedPath, to: decompressedPath)
        }

        return decompressedPath
    }

    /// Download a file from the MNIST server
    private static func downloadFile(_ filename: String, to path: String) throws {
        let urlString = baseURL + filename
        guard let url = URL(string: urlString) else {
            throw MNISTError.invalidURL(urlString)
        }

        print("Downloading \(filename)...")

        // Synchronous download using URLSession
        let semaphore = DispatchSemaphore(value: 0)
        var downloadError: Error?
        var downloadedData: Data?

        let task = URLSession.shared.dataTask(with: url) { data, response, error in
            if let error = error {
                downloadError = error
            } else if let httpResponse = response as? HTTPURLResponse,
                      httpResponse.statusCode != 200 {
                downloadError = MNISTError.httpError(httpResponse.statusCode)
            } else {
                downloadedData = data
            }
            semaphore.signal()
        }
        task.resume()
        semaphore.wait()

        if let error = downloadError {
            throw error
        }

        guard let data = downloadedData else {
            throw MNISTError.noData
        }

        try data.write(to: URL(fileURLWithPath: path))
        print("Downloaded \(filename) (\(data.count) bytes)")
    }

    /// Decompress gzip file
    private static func decompressGzip(from sourcePath: String, to destPath: String) throws {
        // Use the gzip command-line tool for simplicity
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/gzip")
        process.arguments = ["-dk", sourcePath]

        try process.run()
        process.waitUntilExit()

        if process.terminationStatus != 0 {
            throw MNISTError.decompressFailed
        }

        // gzip -dk creates file without .gz extension in same directory
        // Move if needed (it should already be in the right place)
    }

    /// Parse IDX3 file format (images)
    /// Format: [magic:4][numImages:4][rows:4][cols:4][pixel data...]
    private static func parseIDX3(path: String) throws -> (data: [UInt8], numImages: Int, rows: Int, cols: Int) {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))

        guard data.count >= 16 else {
            throw MNISTError.invalidFormat("File too small for IDX3 header")
        }

        // Read header (big-endian)
        let magic = data.prefix(4).withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        guard magic == 0x00000803 else {
            throw MNISTError.invalidFormat("Invalid magic number: \(magic)")
        }

        let numImages = Int(data[4...7].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian })
        let rows = Int(data[8...11].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian })
        let cols = Int(data[12...15].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian })

        let pixelData = Array(data.dropFirst(16))

        return (pixelData, numImages, rows, cols)
    }

    /// Parse IDX1 file format (labels)
    /// Format: [magic:4][numItems:4][label data...]
    private static func parseIDX1(path: String) throws -> [UInt8] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))

        guard data.count >= 8 else {
            throw MNISTError.invalidFormat("File too small for IDX1 header")
        }

        // Read header (big-endian)
        let magic = data.prefix(4).withUnsafeBytes { $0.load(as: UInt32.self).bigEndian }
        guard magic == 0x00000801 else {
            throw MNISTError.invalidFormat("Invalid magic number: \(magic)")
        }

        let numItems = Int(data[4...7].withUnsafeBytes { $0.load(as: UInt32.self).bigEndian })
        let labelData = Array(data.dropFirst(8))

        guard labelData.count == numItems else {
            throw MNISTError.invalidFormat("Label count mismatch: expected \(numItems), got \(labelData.count)")
        }

        return labelData
    }
}

/// Errors that can occur when loading MNIST
public enum MNISTError: Error, CustomStringConvertible {
    case invalidURL(String)
    case httpError(Int)
    case noData
    case decompressFailed
    case invalidFormat(String)

    public var description: String {
        switch self {
        case .invalidURL(let url):
            return "Invalid URL: \(url)"
        case .httpError(let code):
            return "HTTP error: \(code)"
        case .noData:
            return "No data received"
        case .decompressFailed:
            return "Failed to decompress gzip file"
        case .invalidFormat(let msg):
            return "Invalid file format: \(msg)"
        }
    }
}

// MARK: - Memory-Mapped Token Dataset

/// A memory-mapped dataset for large token files (GPT-2 style training).
///
/// Efficiently loads pre-tokenized data stored as binary UInt16 token IDs.
/// Uses memory mapping for random access without loading the entire file.
///
/// Data format: Raw binary file of UInt16 little-endian token IDs.
/// Create with Python: `np.array(tokens, dtype=np.uint16).tofile('train.bin')`
///
/// Example:
/// ```swift
/// let dataset = try MemoryMappedTokenDataset(
///     path: "train.bin",
///     sequenceLength: 1024
/// )
///
/// for batch in dataset.batches(batchSize: 8, device: metal) {
///     let (inputs, targets) = batch
///     // inputs:  [8, 1024] - token IDs
///     // targets: [8, 1024] - shifted by 1
/// }
/// ```
public class MemoryMappedTokenDataset {
    /// File handle for the memory-mapped file
    private let fileHandle: FileHandle

    /// Memory-mapped data pointer
    private let data: UnsafeRawPointer

    /// Total number of tokens in the file
    public let tokenCount: Int

    /// Sequence length for each sample
    public let sequenceLength: Int

    /// Number of complete sequences available
    public var count: Int {
        // We need sequenceLength + 1 tokens per sample (input + 1 for target shift)
        max(0, tokenCount - sequenceLength)
    }

    /// Creates a memory-mapped token dataset.
    ///
    /// - Parameters:
    ///   - path: Path to the binary token file.
    ///   - sequenceLength: Length of each sequence (context window).
    public init(path: String, sequenceLength: Int) throws {
        self.sequenceLength = sequenceLength

        // Open file
        guard let handle = FileHandle(forReadingAtPath: path) else {
            throw TokenDatasetError.fileNotFound(path)
        }
        self.fileHandle = handle

        // Get file size
        let fileSize = handle.seekToEndOfFile()
        handle.seek(toFileOffset: 0)

        // Token count = file size / 2 (UInt16 = 2 bytes)
        self.tokenCount = Int(fileSize) / 2

        guard tokenCount > sequenceLength else {
            throw TokenDatasetError.fileTooSmall(
                expected: sequenceLength + 1,
                got: tokenCount
            )
        }

        // Memory map the file
        let fd = handle.fileDescriptor
        let ptr = mmap(nil, Int(fileSize), PROT_READ, MAP_PRIVATE, fd, 0)

        guard ptr != MAP_FAILED else {
            throw TokenDatasetError.mmapFailed
        }

        self.data = UnsafeRawPointer(ptr!)
    }

    deinit {
        // Unmap memory
        let fileSize = tokenCount * 2
        munmap(UnsafeMutableRawPointer(mutating: data), fileSize)
        try? fileHandle.close()
    }

    /// Get tokens at a specific index range.
    ///
    /// - Parameters:
    ///   - start: Starting token index.
    ///   - length: Number of tokens to read.
    /// - Returns: Array of token IDs.
    public func getTokens(start: Int, length: Int) -> [Int32] {
        precondition(start >= 0 && start + length <= tokenCount,
                    "Token range out of bounds")

        let ptr = data.assumingMemoryBound(to: UInt16.self)
        var tokens: [Int32] = []
        tokens.reserveCapacity(length)

        for i in 0..<length {
            tokens.append(Int32(ptr[start + i]))
        }

        return tokens
    }

    /// Get a single sample (input, target) pair.
    ///
    /// - Parameter index: Sample index (0 to count-1).
    /// - Returns: Tuple of (input tokens, target tokens) each of length sequenceLength.
    public func getSample(at index: Int) -> (input: [Int32], target: [Int32]) {
        precondition(index >= 0 && index < count, "Index out of bounds")

        // Input: tokens[index : index + seqLen]
        // Target: tokens[index + 1 : index + seqLen + 1]
        let allTokens = getTokens(start: index, length: sequenceLength + 1)

        let input = Array(allTokens.prefix(sequenceLength))
        let target = Array(allTokens.suffix(sequenceLength))

        return (input, target)
    }

    /// Generate batches of data.
    ///
    /// - Parameters:
    ///   - batchSize: Number of sequences per batch.
    ///   - device: Device to create tensors on.
    ///   - shuffle: Whether to shuffle sample order.
    /// - Returns: Iterator yielding (input, target) tensor pairs.
    public func batches(
        batchSize: Int,
        device: Device = .default,
        shuffle: Bool = true
    ) -> TokenBatchIterator {
        TokenBatchIterator(
            dataset: self,
            batchSize: batchSize,
            device: device,
            shuffle: shuffle
        )
    }

    /// Iterator for token batches.
    public struct TokenBatchIterator: IteratorProtocol, Sequence {
        private let dataset: MemoryMappedTokenDataset
        private let batchSize: Int
        private let device: Device
        private var indices: [Int]
        private var currentIndex: Int = 0

        init(dataset: MemoryMappedTokenDataset, batchSize: Int, device: Device, shuffle: Bool) {
            self.dataset = dataset
            self.batchSize = batchSize
            self.device = device
            self.indices = Array(0..<dataset.count)
            if shuffle {
                self.indices.shuffle()
            }
        }

        public mutating func next() -> (input: Tensor<Float>, target: Tensor<Float>)? {
            guard currentIndex + batchSize <= indices.count else { return nil }

            var inputBatch: [Float] = []
            var targetBatch: [Float] = []
            inputBatch.reserveCapacity(batchSize * dataset.sequenceLength)
            targetBatch.reserveCapacity(batchSize * dataset.sequenceLength)

            for i in 0..<batchSize {
                let idx = indices[currentIndex + i]
                let (input, target) = dataset.getSample(at: idx)
                inputBatch.append(contentsOf: input.map { Float($0) })
                targetBatch.append(contentsOf: target.map { Float($0) })
            }

            currentIndex += batchSize

            let inputTensor = Tensor<Float>(
                inputBatch,
                shape: [batchSize, dataset.sequenceLength],
                on: device
            )
            let targetTensor = Tensor<Float>(
                targetBatch,
                shape: [batchSize, dataset.sequenceLength],
                on: device
            )

            return (inputTensor, targetTensor)
        }
    }
}

/// Errors for token dataset operations
public enum TokenDatasetError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case fileTooSmall(expected: Int, got: Int)
    case mmapFailed

    public var description: String {
        switch self {
        case .fileNotFound(let path):
            return "Token file not found: \(path)"
        case .fileTooSmall(let expected, let got):
            return "Token file too small: need at least \(expected) tokens, got \(got)"
        case .mmapFailed:
            return "Failed to memory-map file"
        }
    }
}

// MARK: - Language Model Batch Utilities

/// Utilities for creating language model training batches.
public enum LanguageModelBatch {
    /// Create input/target pairs from a sequence of tokens with shifting.
    ///
    /// For language models, target is the input shifted by 1 position.
    ///
    /// - Parameters:
    ///   - tokens: 1D tensor of token IDs.
    ///   - sequenceLength: Length of each sequence.
    ///   - device: Device to create tensors on.
    /// - Returns: Tuple of (input, target) tensors.
    ///
    /// Example:
    /// ```swift
    /// // tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], seqLen = 4
    /// // input  = [0, 1, 2, 3]
    /// // target = [1, 2, 3, 4]
    /// let (x, y) = LanguageModelBatch.createPair(tokens, sequenceLength: 4)
    /// ```
    public static func createPair(
        _ tokens: Tensor<Float>,
        sequenceLength: Int,
        device: Device = .default
    ) -> (input: Tensor<Float>, target: Tensor<Float>) {
        precondition(tokens.rank == 1, "Tokens must be 1D")
        precondition(tokens.elementCount > sequenceLength,
                    "Need more tokens than sequence length")

        // Extract values
        let values = tokens.scalars()

        let input = Array(values.prefix(sequenceLength))
        let target = Array(values.dropFirst().prefix(sequenceLength))

        return (
            Tensor<Float>(input, shape: [sequenceLength], on: device),
            Tensor<Float>(target, shape: [sequenceLength], on: device)
        )
    }

    /// Create random batches from a long token sequence.
    ///
    /// - Parameters:
    ///   - tokens: 1D array of token IDs.
    ///   - batchSize: Number of sequences per batch.
    ///   - sequenceLength: Length of each sequence.
    ///   - device: Device to create tensors on.
    /// - Returns: Tuple of (input, target) batch tensors.
    public static func randomBatch(
        tokens: [Int32],
        batchSize: Int,
        sequenceLength: Int,
        device: Device = .default
    ) -> (input: Tensor<Float>, target: Tensor<Float>) {
        let maxStart = tokens.count - sequenceLength - 1

        var inputBatch: [Float] = []
        var targetBatch: [Float] = []
        inputBatch.reserveCapacity(batchSize * sequenceLength)
        targetBatch.reserveCapacity(batchSize * sequenceLength)

        for _ in 0..<batchSize {
            let start = Int.random(in: 0...maxStart)
            for j in 0..<sequenceLength {
                inputBatch.append(Float(tokens[start + j]))
                targetBatch.append(Float(tokens[start + j + 1]))
            }
        }

        return (
            Tensor<Float>(inputBatch, shape: [batchSize, sequenceLength], on: device),
            Tensor<Float>(targetBatch, shape: [batchSize, sequenceLength], on: device)
        )
    }
}
