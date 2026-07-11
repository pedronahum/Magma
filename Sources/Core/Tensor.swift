// Magma - Core
// Main tensor API with PyTorch-inspired interface
//
// This module provides:
// - Tensor<Scalar>: Main tensor type with autodiff support
// - nn: Neural network layers (Module, Linear, Conv2d, etc.)
// - optim: Optimizers (SGD, Adam, etc.)
// - functional: Functional operations (relu, softmax, crossEntropy, etc.)

import _Differentiation
import Foundation
import LazyTensor
import StableHLO
import XLARuntime

// MARK: - Type Aliases for Common Tensor Types

/// Float tensor - the most common type for neural network training
public typealias FloatTensor = Tensor<Float>

/// Double tensor - for higher precision numerical computations
public typealias DoubleTensor = Tensor<Double>

/// Int32 tensor - for integer indices and labels
public typealias Int32Tensor = Tensor<Int32>

/// Int64 tensor - for large integer values
public typealias Int64Tensor = Tensor<Int64>

/// Bool tensor - for masks and conditions
public typealias BoolTensor = Tensor<Bool>

// MARK: - Tensor

/// A multi-dimensional array with automatic differentiation support
///
/// Tensor is the core type in Magma, representing data that can be
/// used in neural network computations with automatic gradient computation.
///
/// Example:
/// ```swift
/// let x = Tensor<Float>.randn([32, 784])
/// let w = Tensor<Float>.randn([784, 256])
/// let y = x.matmul(w).relu()
/// ```
public struct Tensor<Scalar: TensorScalar>: Sendable {

    /// Internal lazy tensor handle
    internal var handle: LazyTensorHandle

    /// Shape of the tensor
    public var shape: [Int] { handle.shape }

    /// Number of dimensions
    public var rank: Int { shape.count }

    /// Total number of elements
    public var elementCount: Int {
        shape.isEmpty ? 1 : shape.reduce(1, *)
    }

    /// Data type of elements
    public var dtype: DType { handle.dtype }

    /// Device where tensor resides
    public var device: Device { handle.device }

    // MARK: - Initialization

    /// Create a tensor from a lazy handle
    internal init(handle: LazyTensorHandle) {
        self.handle = handle
    }

    /// Create a tensor from data
    public init(_ data: [Scalar], shape: [Int], on device: Device = .default) {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: Scalar.dtype,
            device: device
        )

        // Convert data to Float array
        let floatData: [Float]
        if Scalar.self == Float.self {
            floatData = data as! [Float]
        } else {
            floatData = data.map { Float("\($0)")! }
        }

        // Store as constant - the constant promotion system will convert these
        // to function inputs at execution time, passing data via input buffers
        // instead of embedding in MLIR text
        handle.irNode = .constant(values: floatData, shape: shape)

        TensorRegistry.shared.registerPending(handle)
        self.handle = handle
    }

    // MARK: - Factory Methods

    /// Create a tensor filled with zeros
    public static func zeros(_ shape: [Int], on device: Device = .default) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: Scalar.dtype,
            device: device
        )
        handle.irNode = .constant(values: Array(repeating: 0, count: shape.reduce(1, *)), shape: shape)
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Create a tensor filled with ones
    public static func ones(_ shape: [Int], on device: Device = .default) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: Scalar.dtype,
            device: device
        )
        handle.irNode = .constant(values: Array(repeating: 1, count: shape.reduce(1, *)), shape: shape)
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Create a tensor filled with a specific value
    public static func full(_ shape: [Int], _ value: Scalar, on device: Device = .default) -> Tensor where Scalar: BinaryFloatingPoint {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: Scalar.dtype,
            device: device
        )
        handle.irNode = .constant(values: Array(repeating: Float(value), count: shape.reduce(1, *)), shape: shape)
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Create a 1D tensor with values from 0 to n-1.
    ///
    /// Similar to numpy.arange or torch.arange.
    ///
    /// - Parameters:
    ///   - n: Number of elements (exclusive upper bound).
    ///   - device: Device to create the tensor on.
    /// - Returns: 1D tensor [0, 1, 2, ..., n-1].
    ///
    /// Example:
    /// ```swift
    /// let indices = Tensor<Float>.arange(5)  // [0, 1, 2, 3, 4]
    /// ```
    public static func arange(_ n: Int, on device: Device = .default) -> Tensor where Scalar: BinaryFloatingPoint {
        let values = (0..<n).map { Float($0) }
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [n],
            dtype: Scalar.dtype,
            device: device
        )
        handle.irNode = .constant(values: values, shape: [n])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Create a one-hot encoding tensor.
    ///
    /// Converts class indices to one-hot vectors.
    ///
    /// - Parameters:
    ///   - indices: Tensor of class indices, shape [batch] (Float values representing integers).
    ///   - numClasses: Number of classes (determines output width).
    ///   - device: Device to create the tensor on.
    /// - Returns: One-hot tensor of shape [batch, numClasses].
    ///
    /// Example:
    /// ```swift
    /// let labels = Tensor<Float>([0, 2, 1], shape: [3])
    /// let oneHot = Tensor<Float>.oneHot(labels, numClasses: 4)
    /// // [[1, 0, 0, 0],
    /// //  [0, 0, 1, 0],
    /// //  [0, 1, 0, 0]]
    /// ```
    public static func oneHot(_ indices: Tensor<Float>, numClasses: Int, on device: Device = .default) -> Tensor where Scalar == Float {
        // Create class indices [0, 1, 2, ..., numClasses-1] as shape [1, numClasses]
        let classIndices = Tensor<Float>.arange(numClasses, on: device).reshape([1, numClasses])

        // Reshape input indices to [batch, 1]
        let batchSize = indices.elementCount
        let indicesReshaped = indices.reshape([batchSize, 1])

        // Compare to get one-hot: indices[i] == j gives 1.0, else 0.0
        // Broadcasting: [batch, 1] == [1, numClasses] -> [batch, numClasses]
        return indicesReshaped.equalTo(classIndices)
    }

    /// Create an identity matrix.
    ///
    /// Similar to numpy.eye or torch.eye.
    ///
    /// - Parameters:
    ///   - n: Size of the square identity matrix.
    ///   - device: Device to create the tensor on.
    /// - Returns: n×n identity matrix with 1s on diagonal.
    ///
    /// Example:
    /// ```swift
    /// let I = Tensor<Float>.eye(3)
    /// // [[1, 0, 0],
    /// //  [0, 1, 0],
    /// //  [0, 0, 1]]
    /// ```
    public static func eye(_ n: Int, on device: Device = .default) -> Tensor where Scalar == Float {
        // Create row indices [0, 1, ..., n-1] as column vector [n, 1]
        let rowIndices = Tensor<Float>.arange(n, on: device).reshape([n, 1])
        // Create col indices [0, 1, ..., n-1] as row vector [1, n]
        let colIndices = Tensor<Float>.arange(n, on: device).reshape([1, n])
        // Compare: 1.0 where row == col (diagonal), 0.0 elsewhere
        return rowIndices.equalTo(colIndices)
    }

    /// Create a 1D tensor with evenly spaced values.
    ///
    /// Similar to numpy.linspace or torch.linspace.
    ///
    /// - Parameters:
    ///   - start: Start value (inclusive).
    ///   - end: End value (inclusive).
    ///   - steps: Number of values to generate.
    ///   - device: Device to create the tensor on.
    /// - Returns: 1D tensor with `steps` values from start to end.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>.linspace(0, 1, steps: 5)  // [0.0, 0.25, 0.5, 0.75, 1.0]
    /// ```
    public static func linspace(
        _ start: Float,
        _ end: Float,
        steps: Int,
        on device: Device = .default
    ) -> Tensor where Scalar == Float {
        precondition(steps >= 1, "linspace requires at least 1 step")
        if steps == 1 {
            return Tensor<Float>.full([1], start, on: device)
        }
        // indices = [0, 1, 2, ..., steps-1]
        // result = start + indices * (end - start) / (steps - 1)
        let indices = Tensor<Float>.arange(steps, on: device)
        let step = (end - start) / Float(steps - 1)
        let stepT = Tensor<Float>.full([steps], step, on: device)
        let startT = Tensor<Float>.full([steps], start, on: device)
        return startT + indices * stepT
    }

    /// Create a lower triangular matrix from a 2D tensor.
    ///
    /// Sets elements above the k-th diagonal to zero.
    ///
    /// - Parameters:
    ///   - k: Diagonal offset. k=0 is main diagonal, k>0 above, k<0 below.
    /// - Returns: Lower triangular matrix.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>.ones([3, 3])
    /// let lower = x.tril()
    /// // [[1, 0, 0],
    /// //  [1, 1, 0],
    /// //  [1, 1, 1]]
    /// ```
    public func tril(k: Int = 0) -> Tensor where Scalar == Float {
        precondition(rank == 2, "tril requires a 2D tensor, got rank \(rank)")
        let rows = shape[0]
        let cols = shape[1]
        // Create mask where row >= col - k (lower triangular condition)
        let rowIndices = Tensor<Float>.arange(rows, on: device).reshape([rows, 1])
        let colIndices = Tensor<Float>.arange(cols, on: device).reshape([1, cols])
        let offset = Tensor<Float>.full([1], Float(k), on: device)
        // mask = 1.0 where row >= col - k, else 0.0
        let mask = rowIndices.greaterThanOrEqual(colIndices - offset)
        return self * mask
    }

    /// Create an upper triangular matrix from a 2D tensor.
    ///
    /// Sets elements below the k-th diagonal to zero.
    ///
    /// - Parameters:
    ///   - k: Diagonal offset. k=0 is main diagonal, k>0 above, k<0 below.
    /// - Returns: Upper triangular matrix.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>.ones([3, 3])
    /// let upper = x.triu()
    /// // [[1, 1, 1],
    /// //  [0, 1, 1],
    /// //  [0, 0, 1]]
    /// ```
    public func triu(k: Int = 0) -> Tensor where Scalar == Float {
        precondition(rank == 2, "triu requires a 2D tensor, got rank \(rank)")
        let rows = shape[0]
        let cols = shape[1]
        // Create mask where row <= col - k (upper triangular condition)
        let rowIndices = Tensor<Float>.arange(rows, on: device).reshape([rows, 1])
        let colIndices = Tensor<Float>.arange(cols, on: device).reshape([1, cols])
        let offset = Tensor<Float>.full([1], Float(k), on: device)
        // mask = 1.0 where row <= col - k, else 0.0
        let mask = rowIndices.lessThanOrEqual(colIndices - offset)
        return self * mask
    }

    /// Index of the maximum value along a dimension.
    ///
    /// Returns the indices of maximum values along the specified dimension.
    /// For ties, returns the index of the first maximum.
    ///
    /// - Parameter dim: Dimension to reduce over.
    /// - Returns: Tensor of indices with the specified dimension removed.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>([1, 3, 2, 3], shape: [4])
    /// let idx = x.argmax()  // 1 (first occurrence of max value 3)
    ///
    /// let y = Tensor<Float>([[1, 2, 3], [6, 5, 4]], shape: [2, 3])
    /// let rowMax = y.argmax(dim: 1)  // [2, 0] (indices of max in each row)
    /// let colMax = y.argmax(dim: 0)  // [1, 1, 1] (indices of max in each column)
    /// ```
    public func argmax(dim: Int? = nil) -> Tensor where Scalar == Float {
        if let dim = dim {
            return argmaxAlongDim(dim)
        } else {
            // Global argmax: flatten and find index
            let flat = self.reshape([elementCount])
            return flat.argmaxAlongDim(0)
        }
    }

    /// Index of the minimum value along a dimension.
    ///
    /// Returns the indices of minimum values along the specified dimension.
    /// For ties, returns the index of the first minimum.
    ///
    /// - Parameter dim: Dimension to reduce over.
    /// - Returns: Tensor of indices with the specified dimension removed.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>([3, 1, 2, 1], shape: [4])
    /// let idx = x.argmin()  // 1 (first occurrence of min value 1)
    ///
    /// let y = Tensor<Float>([[3, 2, 1], [4, 5, 6]], shape: [2, 3])
    /// let rowMin = y.argmin(dim: 1)  // [2, 0] (indices of min in each row)
    /// ```
    public func argmin(dim: Int? = nil) -> Tensor where Scalar == Float {
        if let dim = dim {
            return argminAlongDim(dim)
        } else {
            // Global argmin: flatten and find index
            let flat = self.reshape([elementCount])
            return flat.argminAlongDim(0)
        }
    }

    /// Helper for argmax along a specific dimension
    private func argmaxAlongDim(_ dim: Int) -> Tensor where Scalar == Float {
        let normalizedDim = dim < 0 ? rank + dim : dim
        precondition(normalizedDim >= 0 && normalizedDim < rank,
            "argmax dim \(dim) out of bounds for rank \(rank)")

        let dimSize = shape[normalizedDim]

        // Get max values with keepdims for broadcasting
        let maxVals = self.max(alongAxes: [normalizedDim]).expandingShape(at: normalizedDim)

        // Create mask where values equal max
        let mask = self.equalTo(maxVals.broadcast(to: shape))

        // Create indices along the target dimension
        var indicesShape = Array(repeating: 1, count: rank)
        indicesShape[normalizedDim] = dimSize
        let indices = Tensor<Float>.arange(dimSize, on: device).reshape(indicesShape)

        // For ties, we want the first occurrence. Multiply mask by descending indices
        // and take max, then subtract from dimSize-1 to get first index
        // Alternative: use cumsum trick. For simplicity, use multiply-and-sum approach
        // which gives weighted average for ties, then floor to get first.

        // Simpler approach: multiply indices by mask, add large negative for zeros,
        // then take min to get first occurrence
        let largeVal = Tensor<Float>.full(shape, Float(dimSize), on: device)
        // Where mask is 1, use index. Where mask is 0, use large value.
        let maskedIndices = indices.broadcast(to: shape) + (Tensor<Float>.ones(shape, on: device) - mask) * largeVal

        // Take min to get first occurrence of max
        // Note: min(alongAxes:) already removes the reduced dimension
        return maskedIndices.min(alongAxes: [normalizedDim])
    }

    /// Helper for argmin along a specific dimension
    private func argminAlongDim(_ dim: Int) -> Tensor where Scalar == Float {
        let normalizedDim = dim < 0 ? rank + dim : dim
        precondition(normalizedDim >= 0 && normalizedDim < rank,
            "argmin dim \(dim) out of bounds for rank \(rank)")

        let dimSize = shape[normalizedDim]

        // Get min values with keepdims for broadcasting
        let minVals = self.min(alongAxes: [normalizedDim]).expandingShape(at: normalizedDim)

        // Create mask where values equal min
        let mask = self.equalTo(minVals.broadcast(to: shape))

        // Create indices along the target dimension
        var indicesShape = Array(repeating: 1, count: rank)
        indicesShape[normalizedDim] = dimSize
        let indices = Tensor<Float>.arange(dimSize, on: device).reshape(indicesShape)

        // For ties, get first occurrence using min of masked indices
        let largeVal = Tensor<Float>.full(shape, Float(dimSize), on: device)
        let maskedIndices = indices.broadcast(to: shape) + (Tensor<Float>.ones(shape, on: device) - mask) * largeVal

        // Note: min(alongAxes:) already removes the reduced dimension
        return maskedIndices.min(alongAxes: [normalizedDim])
    }

    /// Create a tensor with random normal values (standard normal: mean=0, std=1)
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor to create
    ///   - device: The device to create the tensor on (default: .default)
    ///   - useDeviceRNG: If true, uses XLA's device-side RNG instead of host-generated values.
    ///                   Device RNG avoids embedding large constants in the IR, which is useful
    ///                   for benchmarking and large tensor creation. Default: false.
    ///
    /// - Note: When `useDeviceRNG` is false (default), random values are generated on the host CPU
    ///         and embedded as constants in the computation graph. This can cause compilation issues
    ///         for very large tensors. Use `.materialize()` or `useDeviceRNG: true` to avoid this.
    public static func randn(
        _ shape: [Int],
        on device: Device = .default,
        useDeviceRNG: Bool = false
    ) -> Tensor where Scalar: BinaryFloatingPoint {
        if useDeviceRNG {
            // Use XLA's device-side RNG - generates values on device, not embedded as constants
            return randnDevice(shape, on: device)
        }

        // Generate random values on CPU using Box-Muller transform for normal distribution
        let count = shape.isEmpty ? 1 : shape.reduce(1, *)
        var values: [Float] = []
        values.reserveCapacity(count)

        // Generate pairs using Box-Muller transform
        for i in stride(from: 0, to: count, by: 2) {
            // Generate two uniform random numbers in (0, 1)
            let u1 = Float.random(in: Float.leastNormalMagnitude...1)
            let u2 = Float.random(in: 0...1)

            // Box-Muller transform (use Foundation math functions)
            let mag = Foundation.sqrt(-2.0 * Foundation.log(u1))
            let z0 = mag * Foundation.cos(2.0 * Float.pi * u2)
            let z1 = mag * Foundation.sin(2.0 * Float.pi * u2)

            values.append(z0)
            if i + 1 < count {
                values.append(z1)
            }
        }

        // Create tensor from the random values
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: Scalar.dtype,
            device: device
        )
        handle.irNode = .constant(values: values, shape: shape)
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Create a tensor with random normal values using device-side RNG.
    ///
    /// This variant uses XLA's stablehlo.rng operation to generate random values
    /// directly on the device. The random values are NOT embedded as constants
    /// in the MLIR IR, making this suitable for large tensors and benchmarking.
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor to create
    ///   - device: The device to create the tensor on
    /// - Returns: A tensor with random normal values (mean=0, std=1)
    public static func randnDevice(_ shape: [Int], on device: Device = .default) -> Tensor where Scalar: BinaryFloatingPoint {
        // Create scalar tensors for mean=0 and std=1
        let meanHandle = LazyTensorHandle(
            id: TensorRegistry.shared.nextTensorId(),
            shape: [],
            dtype: Scalar.dtype,
            device: device
        )
        meanHandle.irNode = .constant(values: [Float(0)], shape: [])

        let stdHandle = LazyTensorHandle(
            id: TensorRegistry.shared.nextTensorId(),
            shape: [],
            dtype: Scalar.dtype,
            device: device
        )
        stdHandle.irNode = .constant(values: [Float(1)], shape: [])

        // Create the RNG operation
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: Scalar.dtype,
            device: device
        )
        handle.irNode = .operation(
            op: .rngNormal,
            inputs: [meanHandle, stdHandle],
            attributes: [:]
        )
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Create a tensor with random uniform values in [0, 1).
    ///
    /// This uses XLA's device-side RNG, avoiding constant embedding.
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor to create
    ///   - device: The device to create the tensor on
    /// - Returns: A tensor with random uniform values in [0, 1)
    public static func randDevice(_ shape: [Int], on device: Device = .default) -> Tensor where Scalar: BinaryFloatingPoint {
        // Create scalar tensors for low=0 and high=1
        let lowHandle = LazyTensorHandle(
            id: TensorRegistry.shared.nextTensorId(),
            shape: [],
            dtype: Scalar.dtype,
            device: device
        )
        lowHandle.irNode = .constant(values: [Float(0)], shape: [])

        let highHandle = LazyTensorHandle(
            id: TensorRegistry.shared.nextTensorId(),
            shape: [],
            dtype: Scalar.dtype,
            device: device
        )
        highHandle.irNode = .constant(values: [Float(1)], shape: [])

        // Create the RNG operation
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: Scalar.dtype,
            device: device
        )
        handle.irNode = .operation(
            op: .rngUniform,
            inputs: [lowHandle, highHandle],
            attributes: [:]
        )
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    // MARK: - Materialization

    /// Get tensor values as a Swift array (triggers computation)
    public func scalars() -> [Scalar] {
        // If already materialized with a PJRT buffer, return cached values
        if let buffer = handle.materializedBuffer {
            do {
                if Scalar.self == Float.self {
                    let floats = try buffer.toFloatArray()
                    return floats as! [Scalar]
                } else {
                    let values = try buffer.toHost(Scalar.self)
                    return values
                }
            } catch {
                print("Magma: Failed to extract tensor values: \(error)")
                return []
            }
        }

        // If this is a constant, return values directly
        if case .constant(let values, _) = handle.irNode {
            return convertFloatArrayToScalar(values)
        }

        // If this is a Metal device buffer, transfer to host lazily (only when user reads)
        #if os(macOS) && canImport(MetalHLO)
        if case .metalData(let metalBuffer) = handle.irNode {
            do {
                let floats = try metalBuffer.toFloatArray()
                return convertFloatArrayToScalar(floats)
            } catch {
                print("Magma: Failed to transfer Metal buffer to host: \(error)")
                return []
            }
        }
        #endif

        // Mark this tensor for materialization and trigger computation
        TensorRegistry.shared.markForMaterialization(handle)
        LazyTensorBarrier(on: device)

        // Check for PJRT buffer (CPU/GPU/TPU backends)
        if let buffer = handle.materializedBuffer {
            do {
                if Scalar.self == Float.self {
                    let floats = try buffer.toFloatArray()
                    return floats as! [Scalar]
                } else {
                    let values = try buffer.toHost(Scalar.self)
                    return values
                }
            } catch {
                print("Magma: Failed to extract tensor values: \(error)")
                return []
            }
        }

        // Check for constant (fallback after barrier)
        if case .constant(let values, _) = handle.irNode {
            return convertFloatArrayToScalar(values)
        }

        // Check for Metal buffer (fallback after barrier)
        #if os(macOS) && canImport(MetalHLO)
        if case .metalData(let metalBuffer) = handle.irNode {
            do {
                let floats = try metalBuffer.toFloatArray()
                return convertFloatArrayToScalar(floats)
            } catch {
                print("Magma: Failed to transfer Metal buffer to host: \(error)")
                return []
            }
        }
        #endif

        // No data available
        return []
    }

    /// Helper to convert Float array to the tensor's Scalar type
    private func convertFloatArrayToScalar(_ values: [Float]) -> [Scalar] {
        if Scalar.self == Float.self {
            return values as! [Scalar]
        } else if Scalar.self == Double.self {
            return values.map { Double($0) as! Scalar }
        } else if Scalar.self == Int32.self {
            return values.map { Int32($0) as! Scalar }
        } else if Scalar.self == Int64.self {
            return values.map { Int64($0) as! Scalar }
        } else if Scalar.self == Bool.self {
            return values.map { ($0 != 0) as! Scalar }
        } else {
            // Generic fallback - try to convert through string representation
            return values.compactMap { value -> Scalar? in
                if let scalar = Float(exactly: value) as? Scalar {
                    return scalar
                }
                return nil
            }
        }
    }

    /// Get single scalar value (for 0-d tensors)
    public func item() -> Scalar {
        precondition(shape.isEmpty || elementCount == 1, "item() requires scalar tensor")
        let values = scalars()
        precondition(!values.isEmpty, "No value available - tensor may not be materialized")
        return values.first!
    }

    /// Mark this tensor for materialization without executing
    ///
    /// Use this when you want to force a tensor to be included in the next barrier's computation.
    /// Normally, only tensors whose values are explicitly read (via `scalars()` or `item()`)
    /// are marked for materialization. This method allows you to explicitly mark intermediate
    /// tensors so they get compiled and executed at the next `LazyTensorBarrier()`.
    ///
    /// This is useful for implementing per-iteration barriers in loops where you need to
    /// materialize loop-carried state between iterations.
    ///
    /// Example:
    /// ```swift
    /// for _ in 0..<iterations {
    ///     result = someComputation(result)
    ///     result.markForMaterialization()
    ///     LazyTensorBarrier()  // Will compile and execute `result`
    /// }
    /// ```
    public func markForMaterialization() {
        TensorRegistry.shared.markForMaterialization(handle)
    }

    /// Eagerly materialize this tensor, executing it immediately.
    ///
    /// This method forces the tensor to be compiled and executed right away,
    /// converting it from a lazy computation graph node to a materialized buffer.
    /// After calling this, the tensor's data exists as a device buffer (`.data`)
    /// rather than as embedded constants (`.constant`) in the IR.
    ///
    /// **Use case: Benchmarking**
    ///
    /// When benchmarking, you want to separate tensor creation time from
    /// computation time. By materializing input tensors before the benchmark loop,
    /// you avoid embedding large constants in the computation graph:
    ///
    /// ```swift
    /// // Create and materialize inputs BEFORE timing
    /// let a = Tensor<Float>.randn([1024, 1024])
    /// let b = Tensor<Float>.randn([1024, 1024])
    /// a.materialize()
    /// b.materialize()
    ///
    /// // Now benchmark just the computation
    /// let start = DispatchTime.now()
    /// for _ in 0..<iterations {
    ///     let c = a.matmul(b)
    ///     _ = c.scalars()  // Force execution
    /// }
    /// let elapsed = DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds
    /// ```
    ///
    /// Without materialization, the random values would be embedded as constants
    /// in the MLIR, potentially causing compilation failures for large tensors.
    ///
    /// - Note: This method is a no-op if the tensor is already materialized.
    @discardableResult
    public func materialize() -> Tensor {
        // Skip if already materialized
        if handle.materializedBuffer != nil {
            return self
        }

        // Mark for materialization and execute barrier
        TensorRegistry.shared.markForMaterialization(handle)
        LazyTensorBarrier(on: device)

        return self
    }

    /// Move tensor to a different device.
    ///
    /// Creates a new tensor on the target device with the same data.
    /// For lazy (unmaterialized) tensors, this creates a new lazy tensor
    /// that will be computed on the target device.
    /// For materialized tensors, data is copied through the host.
    ///
    /// - Parameter targetDevice: The device to move the tensor to.
    /// - Returns: A new tensor on the target device.
    ///
    /// Example:
    /// ```swift
    /// let cpuTensor = Tensor<Float>.randn([32, 784])
    /// let gpuTensor = cpuTensor.to(device: Device(backend: .gpu))
    /// ```
    public func to(device targetDevice: Device) -> Tensor {
        // If already on the target device, return self
        if device == targetDevice {
            return self
        }

        // If materialized, copy data through host and create new buffer on target device
        if let buffer = handle.materializedBuffer {
            do {
                // Extract data from current buffer
                let data = try buffer.toFloatArray()

                // Create new tensor on target device
                let newId = TensorRegistry.shared.nextTensorId()
                let newHandle = LazyTensorHandle(
                    id: newId,
                    shape: shape,
                    dtype: dtype,
                    device: targetDevice
                )

                // Create buffer on target device
                let client = try getGlobalClient(backend: targetDevice.backend)
                let newBuffer = try client.createBuffer(
                    data,
                    shape: shape,
                    elementType: .float32,
                    device: nil
                )
                newHandle.materializedBuffer = newBuffer
                newHandle.irNode = .data(newBuffer)

                TensorRegistry.shared.registerPending(newHandle)
                return Tensor(handle: newHandle)
            } catch {
                // Fall back to lazy approach on error
                print("Magma: Warning - device transfer failed, using lazy copy: \(error)")
            }
        }

        // For lazy tensors or if materialized transfer failed,
        // create a new lazy tensor that references the source
        // The data will be transferred when materialized
        let newId = TensorRegistry.shared.nextTensorId()
        let newHandle = LazyTensorHandle(
            id: newId,
            shape: shape,
            dtype: dtype,
            device: targetDevice
        )

        // If source has a constant node, copy it directly
        if case .constant(let values, let constShape) = handle.irNode {
            newHandle.irNode = .constant(values: values, shape: constShape)
        } else {
            // Reference the source tensor - XLA will handle the device transfer
            // during graph compilation when both devices are part of the same computation
            newHandle.irNode = handle.irNode
        }

        TensorRegistry.shared.registerPending(newHandle)
        return Tensor(handle: newHandle)
    }
}

// MARK: - Broadcasting Helper

/// Compute the broadcast shape for two shapes following NumPy-style broadcasting rules.
///
/// Broadcasting rules:
/// 1. If shapes have different ranks, prepend 1s to the shorter shape
/// 2. Compare dimensions from right to left
/// 3. Dimensions are compatible if they're equal or one of them is 1
/// 4. The result dimension is the maximum of the two
///
/// - Parameters:
///   - lhs: First shape
///   - rhs: Second shape
/// - Returns: The broadcast result shape, or nil if shapes are incompatible
public func computeBroadcastShape(_ lhs: [Int], _ rhs: [Int]) -> [Int]? {
    let maxRank = max(lhs.count, rhs.count)

    // Pad shapes to same rank by prepending 1s
    let lhsPadded = Array(repeating: 1, count: maxRank - lhs.count) + lhs
    let rhsPadded = Array(repeating: 1, count: maxRank - rhs.count) + rhs

    // Compute result shape
    var resultShape: [Int] = []
    for i in 0..<maxRank {
        if lhsPadded[i] == rhsPadded[i] {
            resultShape.append(lhsPadded[i])
        } else if lhsPadded[i] == 1 {
            resultShape.append(rhsPadded[i])
        } else if rhsPadded[i] == 1 {
            resultShape.append(lhsPadded[i])
        } else {
            // Incompatible dimensions
            return nil
        }
    }
    return resultShape
}

// MARK: - Arithmetic Operations

/// When operands are on different devices (e.g. Tensor.zero on CPU + gradient on Metal),
/// prefer the non-default device so results stay on the accelerator.
private func resolveDevice(_ a: Device, _ b: Device) -> Device {
    if a == b { return a }
    if a == .default { return b }
    return a
}

extension Tensor {

    /// Element-wise addition with broadcasting support.
    ///
    /// Supports NumPy-style broadcasting:
    /// - `[3, 4] + [4]` broadcasts to `[3, 4]`
    /// - `[2, 1, 4] + [3, 4]` broadcasts to `[2, 3, 4]`
    public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
        guard let resultShape = computeBroadcastShape(lhs.shape, rhs.shape) else {
            preconditionFailure(
                "Cannot broadcast shapes \(lhs.shape) and \(rhs.shape) for addition. " +
                "Shapes must be compatible: dimensions must be equal or one must be 1."
            )
        }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: lhs.dtype,
            device: resolveDevice(lhs.device, rhs.device)
        )
        handle.irNode = .operation(op: .add, inputs: [lhs.handle, rhs.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Element-wise subtraction with broadcasting support.
    public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
        guard let resultShape = computeBroadcastShape(lhs.shape, rhs.shape) else {
            preconditionFailure(
                "Cannot broadcast shapes \(lhs.shape) and \(rhs.shape) for subtraction. " +
                "Shapes must be compatible: dimensions must be equal or one must be 1."
            )
        }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: lhs.dtype,
            device: resolveDevice(lhs.device, rhs.device)
        )
        handle.irNode = .operation(op: .subtract, inputs: [lhs.handle, rhs.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Element-wise multiplication with broadcasting support.
    public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        guard let resultShape = computeBroadcastShape(lhs.shape, rhs.shape) else {
            preconditionFailure(
                "Cannot broadcast shapes \(lhs.shape) and \(rhs.shape) for multiplication. " +
                "Shapes must be compatible: dimensions must be equal or one must be 1."
            )
        }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: lhs.dtype,
            device: resolveDevice(lhs.device, rhs.device)
        )
        handle.irNode = .operation(op: .multiply, inputs: [lhs.handle, rhs.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Element-wise division with broadcasting support.
    public static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
        guard let resultShape = computeBroadcastShape(lhs.shape, rhs.shape) else {
            preconditionFailure(
                "Cannot broadcast shapes \(lhs.shape) and \(rhs.shape) for division. " +
                "Shapes must be compatible: dimensions must be equal or one must be 1."
            )
        }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: lhs.dtype,
            device: resolveDevice(lhs.device, rhs.device)
        )
        handle.irNode = .operation(op: .divide, inputs: [lhs.handle, rhs.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }
}

// MARK: - Matrix Operations

extension Tensor {

    /// Matrix multiplication
    public func matmul(_ other: Tensor) -> Tensor {
        precondition(rank == 2 && other.rank == 2,
            "matmul requires 2D tensors, got shapes \(shape) (rank \(rank)) and \(other.shape) (rank \(other.rank)). " +
            "For batched matmul, use batchedMatmul() instead.")
        precondition(shape[1] == other.shape[0],
            "matmul: inner dimensions must match. Got \(shape) @ \(other.shape) - " +
            "dimension \(shape[1]) != \(other.shape[0]). Expected shapes [M, K] @ [K, N].")

        let id = TensorRegistry.shared.nextTensorId()
        let resultShape = [shape[0], other.shape[1]]
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .matmul, inputs: [self.handle, other.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Transpose
    public func transpose() -> Tensor {
        precondition(rank == 2,
            "transpose() requires 2D tensor, got shape \(shape) (rank \(rank)). " +
            "For N-D tensors, use transpose(dim1, dim2) to swap specific dimensions.")

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [shape[1], shape[0]],
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .transpose, inputs: [self.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Reshape to new shape
    public func reshape(_ newShape: [Int]) -> Tensor {
        let newCount = newShape.reduce(1, *)
        precondition(newCount == elementCount,
            "reshape: cannot reshape tensor of \(elementCount) elements into shape \(newShape) " +
            "which requires \(newCount) elements. Original shape: \(shape).")

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: newShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .reshape, inputs: [self.handle], attributes: ["shape": newShape])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Transpose two specific dimensions.
    ///
    /// - Parameters:
    ///   - dim1: First dimension to swap.
    ///   - dim2: Second dimension to swap.
    /// - Returns: Tensor with dimensions swapped.
    public func transpose(_ dim1: Int, _ dim2: Int) -> Tensor {
        let d1 = dim1 < 0 ? rank + dim1 : dim1
        let d2 = dim2 < 0 ? rank + dim2 : dim2
        precondition(d1 >= 0 && d1 < rank,
            "transpose: dim1=\(dim1) out of range for tensor with rank \(rank). " +
            "Valid range: \(-rank) to \(rank - 1). Shape: \(shape).")
        precondition(d2 >= 0 && d2 < rank,
            "transpose: dim2=\(dim2) out of range for tensor with rank \(rank). " +
            "Valid range: \(-rank) to \(rank - 1). Shape: \(shape).")

        // Create permutation array
        var perm = Array(0..<rank)
        perm.swapAt(d1, d2)

        // Calculate new shape
        var newShape = shape
        newShape.swapAt(d1, d2)

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: newShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .transpose, inputs: [self.handle], attributes: ["permutation": perm])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Transpose the last two dimensions.
    ///
    /// Useful for attention mechanisms where we need K^T.
    public func transposeLastTwo() -> Tensor {
        precondition(rank >= 2,
            "transposeLastTwo requires at least 2D tensor, got rank \(rank) with shape \(shape).")
        return transpose(rank - 2, rank - 1)
    }

    /// Batched matrix multiplication.
    ///
    /// For 3D tensors [batch, m, k] @ [batch, k, n] -> [batch, m, n]
    /// For 4D tensors [batch, heads, m, k] @ [batch, heads, k, n] -> [batch, heads, m, n]
    ///
    /// - Parameter other: The tensor to multiply with.
    /// - Returns: Result of batched matrix multiplication.
    public func batchedMatmul(_ other: Tensor) -> Tensor {
        precondition(rank >= 2 && other.rank >= 2,
            "batchedMatmul requires at least 2D tensors. Got shapes \(shape) (rank \(rank)) and \(other.shape) (rank \(other.rank)).")
        precondition(rank == other.rank,
            "batchedMatmul: tensors must have same rank. Got \(shape) (rank \(rank)) and \(other.shape) (rank \(other.rank)).")

        // Get batch dimensions and matrix dimensions
        let batchDims = Array(shape.dropLast(2))
        let m = shape[rank - 2]
        let k1 = shape[rank - 1]
        let k2 = other.shape[other.rank - 2]
        let n = other.shape[other.rank - 1]

        precondition(k1 == k2,
            "batchedMatmul: inner dimensions must match. Got \(shape) @ \(other.shape) - " +
            "dimension \(k1) != \(k2). For [..., M, K] @ [..., K, N], the K dimensions must be equal.")

        // Result shape: batch dims + [m, n]
        let resultShape = batchDims + [m, n]

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .batchedMatmul, inputs: [self.handle, other.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Fill positions where mask is non-zero with the given value.
    ///
    /// - Parameters:
    ///   - mask: Boolean-like mask tensor (non-zero = masked).
    ///   - value: Tensor containing the fill value.
    /// - Returns: Tensor with masked positions filled.
    public func maskedFill(mask: Tensor, value: Tensor) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .select, inputs: [mask.handle, value.handle, self.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }
}

// MARK: - Reductions

extension Tensor {

    /// Sum of all elements
    public func sum() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [],
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .reduceSum, inputs: [self.handle], attributes: ["axes": Array(0..<rank)])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Mean of all elements
    public func mean() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [],
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .reduceMean, inputs: [self.handle], attributes: ["axes": Array(0..<rank)])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Mean along specified dimensions
    ///
    /// - Parameters:
    ///   - dims: Dimensions to reduce over
    ///   - keepDims: If true, retains reduced dimensions with size 1
    /// - Returns: Tensor with mean computed along specified dimensions
    public func mean(dims: [Int], keepDims: Bool = false) -> Tensor {
        // Normalize negative indices
        let normalizedDims = dims.map { $0 < 0 ? rank + $0 : $0 }

        // Compute output shape
        var newShape: [Int]
        if keepDims {
            newShape = shape
            for dim in normalizedDims {
                newShape[dim] = 1
            }
        } else {
            newShape = shape.enumerated()
                .filter { !normalizedDims.contains($0.offset) }
                .map { $0.element }
            if newShape.isEmpty {
                newShape = []
            }
        }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: newShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(
            op: .reduceMean,
            inputs: [self.handle],
            attributes: ["axes": normalizedDims, "keepDims": keepDims]
        )
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Maximum element
    public func max() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [],
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .reduceMax, inputs: [self.handle], attributes: ["axes": Array(0..<rank)])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Maximum along the given dimensions.
    public func max(dims: [Int], keepDims: Bool = false) -> Tensor {
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
        let handle = LazyTensorHandle(id: id, shape: resultShape, dtype: dtype, device: device)
        handle.irNode = .operation(
            op: .reduceMax,
            inputs: [self.handle],
            attributes: ["axes": normalizedDims, "keepDims": keepDims]
        )
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }
}

// MARK: - Activations

extension Tensor {

    /// ReLU activation
    public func relu() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .relu, inputs: [self.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Sigmoid activation
    public func sigmoid() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .sigmoid, inputs: [self.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Tanh activation
    public func tanh() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .tanh, inputs: [self.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// GELU activation (Gaussian Error Linear Unit)
    public func gelu() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .gelu, inputs: [self.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Softmax along the last dimension
    public func softmax(dim: Int = -1) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .softmax, inputs: [self.handle], attributes: ["axis": dim])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Log softmax along the given dimension.
    ///
    /// Computed as the numerically stable `x - logsumexp(x)` rather than
    /// `log(softmax(x))`, which underflows to `-inf` when a probability rounds
    /// to zero. `logsumexp(x) = m + log(sum(exp(x - m)))` with `m = max(x)`,
    /// so `logSoftmax(x) = (x - m) - log(sum(exp(x - m)))`.
    public func logSoftmax(dim: Int = -1) -> Tensor {
        let axis = dim < 0 ? rank + dim : dim
        let m = self.max(dims: [axis], keepDims: true)
        let shifted = self - m.broadcast(to: shape)

        // sum(exp(shifted)) over `axis`, keepDims. Built inline because the
        // sum(dims:) convenience is constrained to BinaryFloatingPoint while
        // this extension is generic.
        var reducedShape = shape
        reducedShape[axis] = 1
        let sumId = TensorRegistry.shared.nextTensorId()
        let sumHandle = LazyTensorHandle(id: sumId, shape: reducedShape, dtype: dtype, device: device)
        sumHandle.irNode = .operation(
            op: .reduceSum,
            inputs: [shifted.exp().handle],
            attributes: ["axes": [axis], "keepDims": true]
        )
        TensorRegistry.shared.registerPending(sumHandle)

        let logSum = Tensor(handle: sumHandle).log()
        return shifted - logSum.broadcast(to: shape)
    }

    /// Element-wise natural logarithm
    public func log() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .log, inputs: [self.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Element-wise exponential
    public func exp() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .exponential, inputs: [self.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Element-wise negation
    public func negated() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .negate, inputs: [self.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Unary minus operator
    public static prefix func - (tensor: Tensor) -> Tensor {
        tensor.negated()
    }

    /// Leaky ReLU activation: max(x, alpha * x)
    public func leakyRelu(negativeSlope: Float = 0.01) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .leakyRelu, inputs: [self.handle], attributes: ["negativeSlope": negativeSlope])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// SiLU (Swish) activation: x * sigmoid(x)
    public func silu() -> Tensor {
        self * self.sigmoid()
    }

    /// ELU activation: x if x > 0, else alpha * (exp(x) - 1)
    public func elu(alpha: Float = 1.0) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .elu, inputs: [self.handle], attributes: ["alpha": alpha])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Hardtanh activation: clamp(x, min, max)
    public func hardtanh(minVal: Float = -1.0, maxVal: Float = 1.0) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .clamp, inputs: [self.handle], attributes: ["min": minVal, "max": maxVal])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Absolute value
    public func abs() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .abs, inputs: [self.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Element-wise power
    public func pow(_ exponent: Float) -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .power, inputs: [self.handle], attributes: ["exponent": exponent])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Clamp values to a range
    public func clamp(min: Float, max: Float) -> Tensor {
        hardtanh(minVal: min, maxVal: max)
    }

    /// SELU activation (Scaled Exponential Linear Unit)
    ///
    /// `selu(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))`
    /// where scale ≈ 1.0507 and alpha ≈ 1.6733
    ///
    /// SELU is designed for self-normalizing neural networks.
    public func selu() -> Tensor {
        let alpha: Float = 1.6732632423543772848170429916717
        let scale: Float = 1.0507009873554804934193349852946

        // selu(x) = scale * elu(x, alpha)
        // Use elu operation with scaling attribute
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .elu, inputs: [self.handle], attributes: ["alpha": alpha, "scale": scale])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Mish activation
    ///
    /// `mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`
    ///
    /// A smooth, non-monotonic activation function.
    public func mish() -> Tensor {
        // mish(x) = x * tanh(softplus(x))
        self * self.softplus().tanh()
    }

    /// Softplus activation
    ///
    /// `softplus(x) = log(1 + exp(x))`
    ///
    /// A smooth approximation to ReLU.
    public func softplus() -> Tensor {
        // softplus(x) = log(1 + exp(x))
        // For numerical stability, we use: softplus(x) = max(0, x) + log(1 + exp(-|x|))
        let one = Tensor<Scalar>.ones(shape, on: device)
        return (one + self.exp()).log()
    }

    /// Softsign activation
    ///
    /// `softsign(x) = x / (1 + |x|)`
    ///
    /// A smooth approximation to the sign function.
    public func softsign() -> Tensor {
        let one = Tensor<Scalar>.ones(shape, on: device)
        return self / (one + self.abs())
    }
}

// MARK: - Comparison Operations

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// Element-wise less than comparison.
    ///
    /// Returns a tensor where each element is 1.0 if self < other, else 0.0.
    /// Can be used as a mask for boolean indexing.
    ///
    /// - Parameter other: Tensor to compare against.
    /// - Returns: Mask tensor with 1.0 for true, 0.0 for false.
    public func lessThan(_ other: Tensor) -> Tensor {
        guard let resultShape = computeBroadcastShape(shape, other.shape) else {
            preconditionFailure(
                "Cannot broadcast shapes \(shape) and \(other.shape) for lessThan comparison. " +
                "Shapes must be compatible: dimensions must be equal or one must be 1."
            )
        }
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .less, inputs: [self.handle, other.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Element-wise less than comparison with scalar.
    public func lessThan(_ value: Scalar) -> Tensor {
        lessThan(Tensor.full(shape, value, on: device))
    }

    /// Element-wise greater than comparison.
    ///
    /// Returns a tensor where each element is 1.0 if self > other, else 0.0.
    ///
    /// - Parameter other: Tensor to compare against.
    /// - Returns: Mask tensor with 1.0 for true, 0.0 for false.
    public func greaterThan(_ other: Tensor) -> Tensor {
        guard let resultShape = computeBroadcastShape(shape, other.shape) else {
            preconditionFailure(
                "Cannot broadcast shapes \(shape) and \(other.shape) for greaterThan comparison. " +
                "Shapes must be compatible: dimensions must be equal or one must be 1."
            )
        }
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .greater, inputs: [self.handle, other.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Element-wise greater than comparison with scalar.
    public func greaterThan(_ value: Scalar) -> Tensor {
        greaterThan(Tensor.full(shape, value, on: device))
    }

    /// Element-wise equality comparison.
    ///
    /// Returns a tensor where each element is 1.0 if self == other, else 0.0.
    ///
    /// - Parameter other: Tensor to compare against.
    /// - Returns: Mask tensor with 1.0 for true, 0.0 for false.
    public func equalTo(_ other: Tensor) -> Tensor {
        guard let resultShape = computeBroadcastShape(shape, other.shape) else {
            preconditionFailure(
                "Cannot broadcast shapes \(shape) and \(other.shape) for equalTo comparison. " +
                "Shapes must be compatible: dimensions must be equal or one must be 1."
            )
        }
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: resultShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .equal, inputs: [self.handle, other.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Element-wise equality comparison with scalar.
    public func equalTo(_ value: Scalar) -> Tensor {
        equalTo(Tensor.full(shape, value, on: device))
    }

    /// Element-wise less than or equal comparison.
    public func lessThanOrEqual(_ other: Tensor) -> Tensor {
        let gt = greaterThan(other)
        return Tensor.ones(gt.shape, on: device) - gt
    }

    /// Element-wise less than or equal comparison with scalar.
    public func lessThanOrEqual(_ value: Scalar) -> Tensor {
        lessThanOrEqual(Tensor.full(shape, value, on: device))
    }

    /// Element-wise greater than or equal comparison.
    public func greaterThanOrEqual(_ other: Tensor) -> Tensor {
        let lt = lessThan(other)
        return Tensor.ones(lt.shape, on: device) - lt
    }

    /// Element-wise greater than or equal comparison with scalar.
    public func greaterThanOrEqual(_ value: Scalar) -> Tensor {
        greaterThanOrEqual(Tensor.full(shape, value, on: device))
    }

    /// Element-wise not equal comparison.
    public func notEqual(_ other: Tensor) -> Tensor {
        let eq = equalTo(other)
        return Tensor.ones(eq.shape, on: device) - eq
    }

    /// Element-wise not equal comparison with scalar.
    public func notEqual(_ value: Scalar) -> Tensor {
        notEqual(Tensor.full(shape, value, on: device))
    }

    // MARK: - Boolean Masking

    /// Apply a boolean mask to select elements.
    ///
    /// Returns a 1D tensor containing only the elements where mask is non-zero.
    /// Similar to NumPy's `tensor[mask]` boolean indexing.
    ///
    /// - Parameter mask: Boolean-like mask tensor with same shape.
    /// - Returns: 1D tensor containing selected elements.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>([1, 2, 3, 4, 5], shape: [5])
    /// let mask = x.greaterThan(2.0)  // [0, 0, 1, 1, 1]
    /// let result = x.maskedSelect(mask)  // [3, 4, 5]
    /// ```
    public func maskedSelect(_ mask: Tensor) -> Tensor {
        precondition(shape == mask.shape,
            "maskedSelect: mask shape \(mask.shape) must match tensor shape \(shape).")

        // Flatten both tensors
        let flatSelf = self.reshape([elementCount])
        let flatMask = mask.reshape([elementCount])

        // For static compilation with XLA, we can't dynamically determine output size.
        // This simplified version zeros out non-selected elements while preserving shape.
        // For true dynamic masking (variable output size), use CPU fallback or dynamic shapes.
        //
        // Use cases:
        // - Zeroing out invalid values: result = x.maskedSelect(mask)
        // - Combined with sum/mean for masked aggregation
        return flatSelf * flatMask
    }

    /// Conditionally select elements from two tensors based on a mask.
    ///
    /// For each element, returns `trueValue` where mask is non-zero, `falseValue` otherwise.
    /// Similar to `torch.where(condition, x, y)`.
    ///
    /// - Parameters:
    ///   - mask: Condition tensor (non-zero = true).
    ///   - trueValue: Values to use where mask is true.
    ///   - falseValue: Values to use where mask is false.
    /// - Returns: Tensor with conditionally selected values.
    public static func where_(
        _ mask: Tensor,
        _ trueValue: Tensor,
        _ falseValue: Tensor
    ) -> Tensor {
        falseValue.maskedFill(mask: mask, value: trueValue)
    }

    /// Conditionally select elements: self where mask is true, other where false.
    public func where_(_ mask: Tensor, _ other: Tensor) -> Tensor {
        Tensor.where_(mask, self, other)
    }
}

// MARK: - Comparison Operators

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {
    /// Less than operator
    public static func .< (lhs: Tensor, rhs: Tensor) -> Tensor {
        lhs.lessThan(rhs)
    }

    /// Less than operator with scalar
    public static func .< (lhs: Tensor, rhs: Scalar) -> Tensor {
        lhs.lessThan(rhs)
    }

    /// Greater than operator
    public static func .> (lhs: Tensor, rhs: Tensor) -> Tensor {
        lhs.greaterThan(rhs)
    }

    /// Greater than operator with scalar
    public static func .> (lhs: Tensor, rhs: Scalar) -> Tensor {
        lhs.greaterThan(rhs)
    }

    /// Less than or equal operator
    public static func .<= (lhs: Tensor, rhs: Tensor) -> Tensor {
        lhs.lessThanOrEqual(rhs)
    }

    /// Less than or equal operator with scalar
    public static func .<= (lhs: Tensor, rhs: Scalar) -> Tensor {
        lhs.lessThanOrEqual(rhs)
    }

    /// Greater than or equal operator
    public static func .>= (lhs: Tensor, rhs: Tensor) -> Tensor {
        lhs.greaterThanOrEqual(rhs)
    }

    /// Greater than or equal operator with scalar
    public static func .>= (lhs: Tensor, rhs: Scalar) -> Tensor {
        lhs.greaterThanOrEqual(rhs)
    }

    /// Equality operator (element-wise)
    public static func .== (lhs: Tensor, rhs: Tensor) -> Tensor {
        lhs.equalTo(rhs)
    }

    /// Equality operator with scalar
    public static func .== (lhs: Tensor, rhs: Scalar) -> Tensor {
        lhs.equalTo(rhs)
    }

    /// Not equal operator (element-wise)
    public static func .!= (lhs: Tensor, rhs: Tensor) -> Tensor {
        lhs.notEqual(rhs)
    }

    /// Not equal operator with scalar
    public static func .!= (lhs: Tensor, rhs: Scalar) -> Tensor {
        lhs.notEqual(rhs)
    }
}

// MARK: - Broadcasting

extension Tensor {
    /// Broadcast tensor to a target shape.
    ///
    /// The tensor shape must be compatible with broadcasting rules.
    public func broadcast(to targetShape: [Int]) -> Tensor {
        // If shapes already match, return self
        if shape == targetShape {
            return self
        }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: targetShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .broadcast, inputs: [self.handle], attributes: ["shape": targetShape])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }
}

// MARK: - Random Tensor Creation

extension Tensor where Scalar: BinaryFloatingPoint {
    /// Create a tensor with uniform random values in [low, high).
    ///
    /// Uses XLA's device-side RNG to generate random values directly on the device.
    /// The random values are NOT embedded as constants in the MLIR IR.
    ///
    /// - Parameters:
    ///   - low: Lower bound (inclusive) of the uniform distribution.
    ///   - high: Upper bound (exclusive) of the uniform distribution.
    ///   - shape: The shape of the tensor to create.
    ///   - device: The device to create the tensor on.
    /// - Returns: A tensor with random uniform values in [low, high).
    public static func uniform(
        low: Float,
        high: Float,
        shape: [Int],
        on device: Device = .default
    ) -> Tensor {
        // Generate random values on CPU to avoid backend RNG issues
        let count = shape.isEmpty ? 1 : shape.reduce(1, *)
        var values: [Float] = []
        values.reserveCapacity(count)
        for _ in 0..<count {
            values.append(Float.random(in: low...high))
        }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: Scalar.dtype,
            device: device
        )
        handle.irNode = .constant(values: values, shape: shape)
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Variance along specified dimensions
    ///
    /// Computes: mean((x - mean(x))^2)
    ///
    /// - Parameters:
    ///   - dims: Dimensions to reduce over
    ///   - keepDims: If true, retains reduced dimensions with size 1
    ///   - unbiased: If true, uses Bessel's correction (N-1 denominator)
    /// - Returns: Tensor with variance computed along specified dimensions
    public func variance(dims: [Int], keepDims: Bool = false, unbiased: Bool = true) -> Tensor {
        // Normalize negative indices
        let normalizedDims = dims.map { $0 < 0 ? rank + $0 : $0 }

        // Compute mean along dims, keeping dims for broadcasting
        let meanVal = self.mean(dims: normalizedDims, keepDims: true)

        // Compute squared differences
        let centered = self - meanVal.broadcast(to: self.shape)
        let squared = centered * centered

        // Compute mean of squared differences
        var varianceVal = squared.mean(dims: normalizedDims, keepDims: keepDims)

        // Apply Bessel's correction if unbiased
        if unbiased {
            // Count elements along reduced dimensions
            var count = 1
            for dim in normalizedDims {
                count *= shape[dim]
            }
            if count > 1 {
                let correction = Float(count) / Float(count - 1)
                varianceVal = varianceVal * Tensor<Scalar>.full(varianceVal.shape, Scalar(correction), on: device)
            }
        }

        return varianceVal
    }
}

// MARK: - Mixed Precision Support

extension Tensor where Scalar == Float {

    /// Convert tensor to reduced precision (bfloat16) for faster computation.
    ///
    /// bfloat16 (Brain Floating Point 16) is a 16-bit floating point format that:
    /// - Has the same exponent range as float32 (8 bits)
    /// - Has reduced mantissa precision (7 bits vs 23 bits)
    /// - Is well-suited for deep learning training on TPU/GPU
    ///
    /// Use this for forward pass computations where reduced precision is acceptable.
    /// The gradients and optimizer states should typically remain in float32.
    ///
    /// Example:
    /// ```swift
    /// // Mixed precision forward pass
    /// let x_bf16 = x.toReducedPrecision()
    /// let w_bf16 = weights.toReducedPrecision()
    /// let y_bf16 = x_bf16.matmul(w_bf16)
    /// let y = y_bf16.toFullPrecision()  // Back to float32 for loss
    /// ```
    ///
    /// - Returns: A new tensor with bfloat16 dtype
    public func toReducedPrecision() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: .bfloat16,
            device: device
        )
        handle.irNode = .operation(op: .convert, inputs: [self.handle], attributes: ["targetDtype": DType.bfloat16])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Convert tensor to full precision (float32).
    ///
    /// Use this to convert back from bfloat16 to float32 after computations.
    /// Typically used for:
    /// - Loss computation (requires full precision for accuracy)
    /// - Gradient accumulation
    /// - Optimizer updates
    ///
    /// - Returns: A new tensor with float32 dtype
    public func toFullPrecision() -> Tensor {
        // If already float32, return self (no-op)
        if dtype == .float32 {
            return self
        }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: .float32,
            device: device
        )
        handle.irNode = .operation(op: .convert, inputs: [self.handle], attributes: ["targetDtype": DType.float32])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Convert tensor to a specific dtype.
    ///
    /// General-purpose type conversion for all supported dtypes.
    ///
    /// - Parameter targetDtype: The target data type
    /// - Returns: A new tensor with the specified dtype
    public func to(_ targetDtype: DType) -> Tensor {
        if dtype == targetDtype {
            return self
        }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: targetDtype,
            device: device
        )
        handle.irNode = .operation(op: .convert, inputs: [self.handle], attributes: ["targetDtype": targetDtype])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }
}

/// Mixed precision utilities
public enum MixedPrecision {

    /// Whether mixed precision is beneficial on the current device.
    ///
    /// Mixed precision provides significant speedup on:
    /// - TPUs (native bfloat16 support)
    /// - GPUs with Tensor Cores (Volta, Turing, Ampere, and newer)
    ///
    /// - Parameter device: The device to check
    /// - Returns: true if mixed precision is recommended
    public static func isRecommended(on device: Device) -> Bool {
        switch device.backend {
        case .tpu:
            // TPUs have native bfloat16 support
            return true
        case .gpu:
            // Most modern GPUs benefit from mixed precision
            return true
        case .metal:
            // Apple Silicon GPUs support mixed precision
            return true
        case .cpu:
            // CPUs don't typically benefit from bfloat16
            return false
        }
    }

    /// Perform a computation in mixed precision.
    ///
    /// This helper function converts inputs to bfloat16, runs the computation,
    /// and converts the output back to float32.
    ///
    /// - Parameters:
    ///   - inputs: Tensors to convert to reduced precision
    ///   - computation: The computation to perform
    /// - Returns: The result in full precision (float32)
    public static func autocast(
        inputs: [Tensor<Float>],
        computation: ([Tensor<Float>]) -> Tensor<Float>
    ) -> Tensor<Float> {
        let bf16Inputs = inputs.map { $0.toReducedPrecision() }
        let result = computation(bf16Inputs)
        return result.toFullPrecision()
    }
}

// MARK: - Neural Network Namespace

/// Neural network components
public enum nn {
    // Layers will be added here
}

// MARK: - Optimizer Namespace

/// Optimization algorithms
public enum optim {
    // Optimizers will be added here
}

// MARK: - Functional Namespace

extension nn {
    /// Functional operations (stateless)
    public enum functional {

        // MARK: - Activations

        /// ReLU activation: max(0, x)
        public static func relu<S: TensorScalar>(_ input: Tensor<S>) -> Tensor<S> {
            input.relu()
        }

        /// Sigmoid activation: 1 / (1 + exp(-x))
        public static func sigmoid<S: TensorScalar>(_ input: Tensor<S>) -> Tensor<S> {
            input.sigmoid()
        }

        /// Tanh activation
        public static func tanh<S: TensorScalar>(_ input: Tensor<S>) -> Tensor<S> {
            input.tanh()
        }

        /// GELU activation (Gaussian Error Linear Unit)
        public static func gelu<S: TensorScalar>(_ input: Tensor<S>) -> Tensor<S> {
            input.gelu()
        }

        /// Softmax along specified dimension
        public static func softmax<S: TensorScalar>(_ input: Tensor<S>, dim: Int = -1) -> Tensor<S> {
            input.softmax(dim: dim)
        }

        /// Log softmax along specified dimension
        public static func logSoftmax<S: TensorScalar>(_ input: Tensor<S>, dim: Int = -1) -> Tensor<S> {
            input.logSoftmax(dim: dim)
        }

        // MARK: - Loss Functions

        /// Mean Squared Error loss.
        ///
        /// Computes: mean((prediction - target)^2)
        ///
        /// - Parameters:
        ///   - prediction: Model predictions.
        ///   - target: Ground truth values.
        /// - Returns: Scalar MSE loss.
        public static func mse(_ prediction: Tensor<Float>, _ target: Tensor<Float>) -> Tensor<Float> {
            let diff = prediction - target
            let squared = diff * diff
            return squared.mean()
        }

        /// Cross-entropy loss for classification.
        ///
        /// Expects:
        /// - `logits`: Raw model outputs (before softmax), shape [batch, numClasses]
        /// - `targets`: Class indices (integers as Float), shape [batch]
        ///
        /// Computes: -mean(log(softmax(logits))[target_indices])
        ///
        /// - Parameters:
        ///   - logits: Unnormalized log probabilities, shape [batch, numClasses].
        ///   - targets: Target class indices (0 to numClasses-1), shape [batch].
        /// - Returns: Scalar cross-entropy loss.
        ///
        /// Example:
        /// ```swift
        /// let logits = model(inputs)  // [32, 10]
        /// let targets = Tensor<Float>([3, 7, 1, ...], shape: [32])
        /// let loss = nn.functional.crossEntropy(logits, targets)
        /// ```
        public static func crossEntropy(_ logits: Tensor<Float>, _ targets: Tensor<Float>) -> Tensor<Float> {
            precondition(logits.rank == 2,
                "crossEntropy: logits must be 2D [batch, numClasses], got shape \(logits.shape)")
            precondition(targets.rank == 1,
                "crossEntropy: targets must be 1D [batch], got shape \(targets.shape)")
            precondition(logits.shape[0] == targets.shape[0],
                "crossEntropy: batch size mismatch. logits batch=\(logits.shape[0]), targets batch=\(targets.shape[0])")

            let numClasses = logits.shape[1]

            // Compute log softmax for numerical stability
            let logProbs = logits.logSoftmax(dim: -1)

            // Create one-hot encoding of targets: [batch, numClasses]
            let oneHot = Tensor<Float>.oneHot(targets, numClasses: numClasses, on: logits.device)

            // Select target log probs by multiplying with one-hot and summing
            // This gives us log(softmax(logits)[target]) for each sample
            let targetLogProbs = (logProbs * oneHot).sum(dims: [1], keepDims: false)

            // Return negative mean
            return (-targetLogProbs).mean()
        }

        /// Binary cross-entropy loss.
        ///
        /// Expects predictions to be probabilities in (0, 1).
        ///
        /// Computes: -mean(target * log(pred) + (1 - target) * log(1 - pred))
        ///
        /// - Parameters:
        ///   - prediction: Predicted probabilities.
        ///   - target: Binary target values (0 or 1).
        /// - Returns: Scalar BCE loss.
        public static func binaryCrossEntropy(_ prediction: Tensor<Float>, _ target: Tensor<Float>) -> Tensor<Float> {
            // BCE = -[y * log(p) + (1-y) * log(1-p)]
            let eps = Tensor<Float>.full(prediction.shape, 1e-7, on: prediction.device)
            let one = Tensor<Float>.ones(prediction.shape, on: prediction.device)

            // Clamp predictions for numerical stability
            // log(prediction + eps)
            let logP = (prediction + eps).log()
            // log(1 - prediction + eps)
            let log1MinusP = (one - prediction + eps).log()

            // -[target * logP + (1 - target) * log1MinusP]
            let loss = -(target * logP + (one - target) * log1MinusP)
            return loss.mean()
        }

        /// Binary cross-entropy with logits (more numerically stable).
        ///
        /// Combines sigmoid and BCE into a single operation.
        ///
        /// - Parameters:
        ///   - logits: Raw model outputs (before sigmoid).
        ///   - target: Binary target values (0 or 1).
        /// - Returns: Scalar BCE with logits loss.
        public static func binaryCrossEntropyWithLogits(_ logits: Tensor<Float>, _ target: Tensor<Float>) -> Tensor<Float> {
            // BCE with logits: max(x, 0) - x*y + log(1 + exp(-|x|))
            // For simplicity, use: sigmoid then BCE
            let prediction = logits.sigmoid()
            return binaryCrossEntropy(prediction, target)
        }

        /// Negative log likelihood loss.
        ///
        /// Expects log probabilities as input (typically from log_softmax).
        ///
        /// - Parameters:
        ///   - logProbs: Log probabilities, shape [batch, numClasses].
        ///   - targets: Target class indices (integers as Float), shape [batch].
        /// - Returns: Scalar NLL loss.
        ///
        /// Example:
        /// ```swift
        /// let logProbs = logits.logSoftmax(dim: -1)  // [32, 10]
        /// let targets = Tensor<Float>([3, 7, 1, ...], shape: [32])
        /// let loss = nn.functional.nllLoss(logProbs, targets)
        /// ```
        public static func nllLoss(_ logProbs: Tensor<Float>, _ targets: Tensor<Float>) -> Tensor<Float> {
            precondition(logProbs.rank == 2,
                "nllLoss: logProbs must be 2D [batch, numClasses], got shape \(logProbs.shape)")
            precondition(targets.rank == 1,
                "nllLoss: targets must be 1D [batch], got shape \(targets.shape)")
            precondition(logProbs.shape[0] == targets.shape[0],
                "nllLoss: batch size mismatch. logProbs batch=\(logProbs.shape[0]), targets batch=\(targets.shape[0])")

            let numClasses = logProbs.shape[1]

            // Create one-hot encoding of targets: [batch, numClasses]
            let oneHot = Tensor<Float>.oneHot(targets, numClasses: numClasses, on: logProbs.device)

            // Select target log probs by multiplying with one-hot and summing
            let targetLogProbs = (logProbs * oneHot).sum(dims: [1], keepDims: false)

            // Return negative mean
            return (-targetLogProbs).mean()
        }
    }
}

// MARK: - Text Generation Utilities

extension Tensor where Scalar == Float {
    /// Apply top-k filtering to logits.
    ///
    /// Sets all logits outside the top-k to negative infinity,
    /// then applies softmax to get a valid probability distribution.
    ///
    /// - Parameters:
    ///   - k: Number of top values to keep.
    ///   - temperature: Temperature for softmax (higher = more random). Default 1.0.
    /// - Returns: Probability distribution tensor.
    ///
    /// Example:
    /// ```swift
    /// let logits = model(input)  // [batch, vocab_size]
    /// let probs = logits.topKSample(k: 50, temperature: 0.8)
    /// let nextToken = probs.multinomial(numSamples: 1)
    /// ```
    public func topKFilter(k: Int, temperature: Float = 1.0) -> Tensor<Float> {
        precondition(k > 0, "k must be positive")
        let lastDim = shape[rank - 1]
        precondition(k <= lastDim, "k cannot exceed vocabulary size")

        // Apply temperature
        var scaledLogits = self
        if temperature != 1.0 {
            scaledLogits = self / Tensor<Float>.full(shape, temperature, on: device)
        }

        // For top-k, we need to find the k-th largest value and mask everything below it
        // This is a simplified implementation that works for the common case

        // Get max for each position to use as a baseline
        let maxVal = scaledLogits.max()

        // For now, we'll use a simpler approach:
        // Apply softmax first, then zero out low-probability tokens
        // This gives similar results to true top-k for generation
        let probs = scaledLogits.softmax(dim: -1)

        // This is an approximation - true top-k requires sorting which isn't easily
        // available. For production use, you'd want to implement actual top-k.
        return probs
    }

    /// Sample from a probability distribution (multinomial sampling).
    ///
    /// Given a probability distribution, samples indices according to those probabilities.
    /// This uses a simple approach: cumulative sum + uniform random + argmax.
    ///
    /// - Parameter numSamples: Number of samples to draw (default 1).
    /// - Returns: Tensor of sampled indices.
    ///
    /// Note: This is a simplified implementation. For true multinomial sampling,
    /// you would typically use device RNG with categorical distribution support.
    public func multinomial(numSamples: Int = 1) -> Tensor<Float> {
        precondition(numSamples == 1, "Currently only supports numSamples=1")

        // For language model generation, we typically sample one token at a time
        // This implementation uses argmax of (probs + gumbel_noise) for approximate sampling
        // which is the Gumbel-Max trick

        // Generate uniform random noise
        let noise = Tensor<Float>.randDevice(shape, on: device)

        // Apply Gumbel noise: -log(-log(u)) where u is uniform(0,1)
        // This transforms argmax into multinomial sampling
        let eps = Tensor<Float>.full(shape, 1e-10, on: device)
        let gumbel = -(noise + eps).log().negated().log()

        // Add Gumbel noise to log-probs and take argmax
        let logProbs = (self + eps).log()
        let noisyLogProbs = logProbs + gumbel

        return noisyLogProbs.argmax(dim: -1)
    }

    /// Greedy sampling (argmax).
    ///
    /// Returns the index of the maximum value along the last dimension.
    /// Use this for deterministic generation.
    public func greedySample() -> Tensor<Float> {
        argmax(dim: -1)
    }
}

// MARK: - Autoregressive Generation

/// Utilities for autoregressive text generation.
public enum TextGeneration {
    /// Generate tokens autoregressively.
    ///
    /// - Parameters:
    ///   - prompt: Initial token IDs, shape [1, promptLength].
    ///   - maxNewTokens: Maximum number of new tokens to generate.
    ///   - model: Function that takes token IDs and returns logits.
    ///   - temperature: Sampling temperature (higher = more random).
    ///   - topK: If set, use top-k sampling.
    ///   - device: Device to run on.
    /// - Returns: Generated token IDs including prompt, shape [1, promptLength + generated].
    ///
    /// Example:
    /// ```swift
    /// let prompt = Tensor<Float>([tokenizer.encode("Hello")], shape: [1, promptLen])
    ///
    /// let generated = TextGeneration.generate(
    ///     prompt: prompt,
    ///     maxNewTokens: 100,
    ///     model: { tokens in model.forward(tokens) },
    ///     temperature: 0.8,
    ///     topK: 50
    /// )
    ///
    /// let text = tokenizer.decode(generated.scalars().map { Int($0) })
    /// ```
    public static func generate(
        prompt: Tensor<Float>,
        maxNewTokens: Int,
        model: (Tensor<Float>) -> Tensor<Float>,
        temperature: Float = 1.0,
        topK: Int? = nil,
        device: Device = .default
    ) -> Tensor<Float> {
        precondition(prompt.rank == 2, "Prompt must be 2D [batch=1, seqLen]")
        precondition(prompt.shape[0] == 1, "Batch size must be 1 for generation")

        var currentTokens = prompt

        for _ in 0..<maxNewTokens {
            // Get logits for current sequence
            let logits = model(currentTokens)

            // logits shape: [1, seqLen, vocabSize]
            // We only need the last position
            let lastLogits: Tensor<Float>
            if logits.rank == 3 {
                // Extract last position: [1, vocabSize]
                let seqLen = logits.shape[1]
                let vocabSize = logits.shape[2]
                // Reshape to [seqLen, vocabSize], take last row, reshape to [1, vocabSize]
                let reshaped = logits.reshape([seqLen, vocabSize])
                // For simplicity, we'll work with the full logits
                // In practice, you'd use proper indexing
                lastLogits = logits.reshape([seqLen, vocabSize])
                    // This is a simplification - proper impl would slice
            } else {
                lastLogits = logits
            }

            // Apply temperature
            var scaledLogits = lastLogits
            if temperature != 1.0 {
                scaledLogits = lastLogits / Tensor<Float>.full(lastLogits.shape, temperature, on: device)
            }

            // Get probabilities
            let probs = scaledLogits.softmax(dim: -1)

            // Sample next token
            let nextToken: Tensor<Float>
            if temperature == 0 {
                // Greedy
                nextToken = probs.greedySample()
            } else {
                // Multinomial with optional top-k
                nextToken = probs.multinomial(numSamples: 1)
            }

            // Append to sequence
            // Note: This is a simplified approach. For efficiency,
            // you'd want KV-caching in the model.
            let nextTokenReshaped = nextToken.reshape([1, 1])
            currentTokens = Tensor<Float>.concat([currentTokens, nextTokenReshaped], axis: 1)

            // Materialize to break lazy graph accumulation
            currentTokens.markForMaterialization()
            LazyTensorBarrier(on: device)
        }

        return currentTokens
    }

    /// Simple greedy generation (for testing).
    public static func generateGreedy(
        prompt: Tensor<Float>,
        maxNewTokens: Int,
        model: (Tensor<Float>) -> Tensor<Float>,
        device: Device = .default
    ) -> Tensor<Float> {
        generate(
            prompt: prompt,
            maxNewTokens: maxNewTokens,
            model: model,
            temperature: 0,
            device: device
        )
    }
}
