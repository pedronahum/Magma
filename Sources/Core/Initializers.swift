// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
// Copyright 2024-2026 Pedro N. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Weight initialization methods for neural networks.
// Ported from Swift for TensorFlow (S4TF).

import Foundation
import _Differentiation
import LazyTensor
import StableHLO
import XLARuntime

// MARK: - Fan Calculation

/// Helper to compute fan-in and fan-out for weight tensors.
///
/// For weight matrices, fan-in is the number of input units and fan-out is the number of output units.
/// For convolutional filters, these are scaled by the receptive field size.
public struct FanCalculator {
    /// Computes the fan-in and fan-out for a given shape.
    ///
    /// - Parameter shape: The shape of the weight tensor.
    ///   - For 2D tensors [in, out]: fan_in = in, fan_out = out
    ///   - For 4D convolution filters [H, W, in, out]: fan_in = H*W*in, fan_out = H*W*out
    /// - Returns: A tuple of (fanIn, fanOut).
    public static func compute(_ shape: [Int]) -> (fanIn: Int, fanOut: Int) {
        precondition(shape.count >= 2,
            "Fans cannot be computed for tensors with fewer than 2 dimensions. Got: \(shape.count)")

        // For 2D tensor (Dense/Embedding weights)
        if shape.count == 2 {
            return (shape[0], shape[1])
        }

        // For tensors with rank > 2 (convolution filters)
        // Shape is typically [H, W, C_in, C_out] or [D, H, W, C_in, C_out]
        let spatialDims = shape.dropLast(2)
        let spatialSize = spatialDims.reduce(1, *)
        let inputChannels = shape[shape.count - 2]
        let outputChannels = shape[shape.count - 1]

        return (inputChannels * spatialSize, outputChannels * spatialSize)
    }
}

// MARK: - Variance Scaling Initializers

extension Tensor where Scalar == Float {

    // MARK: - Random Uniform in Range

    /// Creates a tensor with random uniform values in the specified range.
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - lowerBound: The lower bound of the uniform distribution.
    ///   - upperBound: The upper bound of the uniform distribution.
    ///   - device: The device to create the tensor on.
    /// - Returns: A tensor with random values uniformly distributed in [lowerBound, upperBound).
    public static func randomUniform(
        _ shape: [Int],
        lowerBound: Float = 0,
        upperBound: Float = 1,
        on device: Device = .default
    ) -> Tensor {
        // Generate uniform in [0, 1), then scale and shift
        let uniform = randDevice(shape, on: device)
        let range = Tensor([upperBound - lowerBound], shape: [], on: device)
        let low = Tensor([lowerBound], shape: [], on: device)
        return uniform * range + low
    }

    // MARK: - Truncated Normal

    /// Creates a tensor with random values from a truncated normal distribution.
    ///
    /// Values are drawn from a normal distribution and rejected if they fall more than
    /// 2 standard deviations from the mean. This is useful for weight initialization
    /// to avoid extreme values.
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - mean: The mean of the normal distribution. Default is 0.
    ///   - stddev: The standard deviation of the normal distribution. Default is 1.
    ///   - device: The device to create the tensor on.
    /// - Returns: A tensor with truncated normal values.
    public static func truncatedNormal(
        _ shape: [Int],
        mean: Float = 0,
        stddev: Float = 1,
        on device: Device = .default
    ) -> Tensor {
        // Generate normal samples and reject those outside [-2*stddev, 2*stddev]
        // For efficiency, we use rejection sampling with CPU-generated values
        let count = shape.isEmpty ? 1 : shape.reduce(1, *)
        var values: [Float] = []
        values.reserveCapacity(count)

        let lowerBound = mean - 2 * stddev
        let upperBound = mean + 2 * stddev

        while values.count < count {
            // Generate pairs using Box-Muller transform
            let u1 = Float.random(in: Float.leastNormalMagnitude...1)
            let u2 = Float.random(in: 0...1)
            let mag = Foundation.sqrt(-2.0 * Foundation.log(u1))
            let z0 = mag * Foundation.cos(2.0 * Float.pi * u2) * stddev + mean
            let z1 = mag * Foundation.sin(2.0 * Float.pi * u2) * stddev + mean

            // Accept only values within bounds
            if z0 >= lowerBound && z0 <= upperBound {
                values.append(z0)
            }
            if values.count < count && z1 >= lowerBound && z1 <= upperBound {
                values.append(z1)
            }
        }

        // Create tensor from the truncated normal values
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: .float32,
            device: device
        )
        handle.irNode = IRNode.constant(values: values, shape: shape)
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    // MARK: - Glorot (Xavier) Initialization

    /// Creates a tensor using Glorot (Xavier) uniform initialization.
    ///
    /// Draws random samples from a uniform distribution between `-limit` and `limit`
    /// where `limit = sqrt(6 / (fanIn + fanOut))`.
    ///
    /// This initialization is designed to keep the scale of gradients roughly the same
    /// across all layers. It's well-suited for tanh and sigmoid activations.
    ///
    /// Reference: ["Understanding the difficulty of training deep feedforward neural networks"](
    /// http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - device: The device to create the tensor on.
    /// - Returns: A tensor with Glorot uniform initialization.
    public static func glorotUniform(
        _ shape: [Int],
        on device: Device = .default
    ) -> Tensor {
        let (fanIn, fanOut) = FanCalculator.compute(shape)
        let limit = Foundation.sqrt(6.0 / Float(fanIn + fanOut))
        return randomUniform(shape, lowerBound: -limit, upperBound: limit, on: device)
    }

    /// Creates a tensor using Glorot (Xavier) normal initialization.
    ///
    /// Draws random samples from a truncated normal distribution centered on 0
    /// with standard deviation `sqrt(2 / (fanIn + fanOut))`.
    ///
    /// Reference: ["Understanding the difficulty of training deep feedforward neural networks"](
    /// http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - device: The device to create the tensor on.
    /// - Returns: A tensor with Glorot normal initialization.
    public static func glorotNormal(
        _ shape: [Int],
        on device: Device = .default
    ) -> Tensor {
        let (fanIn, fanOut) = FanCalculator.compute(shape)
        var stddev = Foundation.sqrt(2.0 / Float(fanIn + fanOut))
        // Standard deviation of truncated standard normal between -2 and 2 standard deviations
        let truncationDeviation: Float = 0.87962566103423978
        stddev /= truncationDeviation  // Smooths the tails of the clipped normal
        return truncatedNormal(shape, mean: 0, stddev: stddev, on: device)
    }

    // MARK: - He (Kaiming) Initialization

    /// Creates a tensor using He (Kaiming) uniform initialization.
    ///
    /// Draws random samples from a uniform distribution between `-limit` and `limit`
    /// where `limit = sqrt(6 / fanIn)`.
    ///
    /// This initialization is designed specifically for ReLU activations and its variants.
    ///
    /// Reference: ["Delving Deep into Rectifiers: Surpassing Human-Level Performance
    /// on ImageNet Classification"](https://arxiv.org/abs/1502.01852)
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - device: The device to create the tensor on.
    /// - Returns: A tensor with He uniform initialization.
    public static func heUniform(
        _ shape: [Int],
        on device: Device = .default
    ) -> Tensor {
        let (fanIn, _) = FanCalculator.compute(shape)
        let limit = Foundation.sqrt(6.0 / Float(fanIn))
        return randomUniform(shape, lowerBound: -limit, upperBound: limit, on: device)
    }

    /// Creates a tensor using He (Kaiming) normal initialization.
    ///
    /// Draws random samples from a truncated normal distribution centered on 0
    /// with standard deviation `sqrt(2 / fanIn)`.
    ///
    /// This initialization is designed specifically for ReLU activations and its variants.
    ///
    /// Reference: ["Delving Deep into Rectifiers: Surpassing Human-Level Performance
    /// on ImageNet Classification"](https://arxiv.org/abs/1502.01852)
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - device: The device to create the tensor on.
    /// - Returns: A tensor with He normal initialization.
    public static func heNormal(
        _ shape: [Int],
        on device: Device = .default
    ) -> Tensor {
        let (fanIn, _) = FanCalculator.compute(shape)
        var stddev = Foundation.sqrt(2.0 / Float(fanIn))
        let truncationDeviation: Float = 0.87962566103423978
        stddev /= truncationDeviation
        return truncatedNormal(shape, mean: 0, stddev: stddev, on: device)
    }

    // MARK: - LeCun Initialization

    /// Creates a tensor using LeCun uniform initialization.
    ///
    /// Draws random samples from a uniform distribution between `-limit` and `limit`
    /// where `limit = sqrt(3 / fanIn)`.
    ///
    /// Reference: ["Efficient BackProp"](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - device: The device to create the tensor on.
    /// - Returns: A tensor with LeCun uniform initialization.
    public static func lecunUniform(
        _ shape: [Int],
        on device: Device = .default
    ) -> Tensor {
        let (fanIn, _) = FanCalculator.compute(shape)
        let limit = Foundation.sqrt(3.0 / Float(fanIn))
        return randomUniform(shape, lowerBound: -limit, upperBound: limit, on: device)
    }

    /// Creates a tensor using LeCun normal initialization.
    ///
    /// Draws random samples from a truncated normal distribution centered on 0
    /// with standard deviation `sqrt(1 / fanIn)`.
    ///
    /// Reference: ["Efficient BackProp"](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor.
    ///   - device: The device to create the tensor on.
    /// - Returns: A tensor with LeCun normal initialization.
    public static func lecunNormal(
        _ shape: [Int],
        on device: Device = .default
    ) -> Tensor {
        let (fanIn, _) = FanCalculator.compute(shape)
        var stddev = Foundation.sqrt(1.0 / Float(fanIn))
        let truncationDeviation: Float = 0.87962566103423978
        stddev /= truncationDeviation
        return truncatedNormal(shape, mean: 0, stddev: stddev, on: device)
    }

    // MARK: - Orthogonal Initialization

    /// Creates a tensor using orthogonal initialization.
    ///
    /// For 2D tensors, this creates an orthogonal matrix via QR decomposition.
    /// For higher-dimensional tensors, the leading dimensions are flattened.
    ///
    /// Orthogonal initialization helps preserve gradient magnitudes during backpropagation,
    /// which can help with training very deep networks.
    ///
    /// - Parameters:
    ///   - shape: The shape of the tensor (must have at least 2 dimensions).
    ///   - gain: A multiplicative factor to apply to the orthogonal tensor. Default is 1.
    ///   - device: The device to create the tensor on.
    /// - Returns: A tensor with orthogonal initialization.
    public static func orthogonal(
        _ shape: [Int],
        gain: Float = 1.0,
        on device: Device = .default
    ) -> Tensor {
        precondition(shape.count >= 2,
            "Orthogonal initialization requires at least 2 dimensions. Got: \(shape.count)")

        // Flatten all but the last dimension
        let rows = shape.dropLast().reduce(1, *)
        let cols = shape.last!

        // Generate random matrix with appropriate shape for QR decomposition
        let flatShape: [Int]
        let needsTranspose: Bool
        if rows < cols {
            flatShape = [cols, rows]
            needsTranspose = true
        } else {
            flatShape = [rows, cols]
            needsTranspose = false
        }

        // Generate random normal matrix
        let normal = Tensor.randn(flatShape, on: device)

        // Perform QR decomposition
        // Note: This requires the QR decomposition op to be available
        // For now, we use a simplified Gram-Schmidt-like approach on CPU
        let orthogonalMatrix = gramSchmidtOrthogonalize(normal, rows: flatShape[0], cols: flatShape[1])

        // Transpose if needed
        var result = needsTranspose ? orthogonalMatrix.transpose() : orthogonalMatrix

        // Reshape to target shape
        result = result.reshape(shape)

        // Apply gain
        if gain != 1.0 {
            let gainTensor = Tensor([gain], shape: [], on: device)
            result = result * gainTensor
        }

        return result
    }
}

// MARK: - Helper Functions

/// Performs Gram-Schmidt orthogonalization on a matrix.
///
/// This is a CPU implementation used as a fallback when QR decomposition
/// is not available on device.
private func gramSchmidtOrthogonalize(_ matrix: Tensor<Float>, rows: Int, cols: Int) -> Tensor<Float> {
    // Get the values from the matrix
    var values = matrix.scalars()

    // Perform modified Gram-Schmidt
    for j in 0..<cols {
        // Normalize column j
        var norm: Float = 0
        for i in 0..<rows {
            let val = values[i * cols + j]
            norm += val * val
        }
        norm = Foundation.sqrt(norm)

        if norm > 1e-10 {
            for i in 0..<rows {
                values[i * cols + j] /= norm
            }
        }

        // Orthogonalize remaining columns against column j
        for k in (j + 1)..<cols {
            // Compute dot product of columns j and k
            var dot: Float = 0
            for i in 0..<rows {
                dot += values[i * cols + j] * values[i * cols + k]
            }

            // Subtract projection
            for i in 0..<rows {
                values[i * cols + k] -= dot * values[i * cols + j]
            }
        }
    }

    // Create result tensor
    return Tensor(values, shape: [rows, cols], on: matrix.device)
}

// MARK: - Naming Conventions
//
// The initializers above follow different naming conventions from different frameworks:
//
// | Magma     | PyTorch         | TensorFlow/Keras    |
// |-----------------|-----------------|---------------------|
// | glorotUniform   | xavier_uniform_ | GlorotUniform       |
// | glorotNormal    | xavier_normal_  | GlorotNormal        |
// | heUniform       | kaiming_uniform_| HeUniform           |
// | heNormal        | kaiming_normal_ | HeNormal            |
// | lecunUniform    | -               | LecunUniform        |
// | lecunNormal     | -               | LecunNormal         |
// | truncatedNormal | -               | TruncatedNormal     |
// | orthogonal      | orthogonal_     | Orthogonal          |
//
// Note: Glorot = Xavier, He = Kaiming (named after researchers)
