// Magma - Data Transforms
// PyTorch-compatible transforms for data augmentation and preprocessing
//
// API Reference: https://pytorch.org/vision/stable/transforms.html
//
// This module provides transforms that can be composed together for data
// preprocessing and augmentation pipelines, following PyTorch's torchvision.transforms API.

import Foundation
import LazyTensor

// MARK: - Transform Protocol

/// A transform that can be applied to tensors.
///
/// This protocol mirrors PyTorch's `torchvision.transforms.v2.Transform` base class.
/// Transforms can be composed together using `Compose`.
///
/// Example:
/// ```swift
/// let transform = transforms.Compose([
///     transforms.Normalize(mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]),
///     transforms.RandomHorizontalFlip(p: 0.5)
/// ])
/// let augmented = transform(image)
/// ```
public protocol Transform {
    /// Apply the transform to a tensor.
    ///
    /// - Parameter input: The input tensor to transform.
    /// - Returns: The transformed tensor.
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float>
}

// MARK: - Transforms Namespace

/// Namespace for transform classes, mirroring `torchvision.transforms`.
///
/// Usage:
/// ```swift
/// let normalize = transforms.Normalize(mean: [0.5], std: [0.5])
/// let flip = transforms.RandomHorizontalFlip()
/// let composed = transforms.Compose([normalize, flip])
/// ```
public enum transforms {
    // Transforms are defined as nested types within this enum
}

// MARK: - Composition Transforms

extension transforms {

    /// Composes several transforms together.
    ///
    /// Equivalent to `torchvision.transforms.Compose`.
    ///
    /// Example:
    /// ```swift
    /// let transform = transforms.Compose([
    ///     transforms.Normalize(mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]),
    ///     transforms.RandomHorizontalFlip(p: 0.5),
    ///     transforms.RandomCrop(size: (224, 224))
    /// ])
    /// ```
    public struct Compose: Transform {
        /// The list of transforms to apply in order.
        public let transforms: [Transform]

        /// Creates a composed transform.
        ///
        /// - Parameter transforms: List of transforms to compose.
        public init(_ transforms: [Transform]) {
            self.transforms = transforms
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            var result = input
            for transform in transforms {
                result = transform(result)
            }
            return result
        }
    }

    /// Apply a list of transformations with a given probability.
    ///
    /// Equivalent to `torchvision.transforms.RandomApply`.
    ///
    /// Example:
    /// ```swift
    /// let transform = transforms.RandomApply([
    ///     transforms.ColorJitter(brightness: 0.2)
    /// ], p: 0.5)
    /// ```
    public struct RandomApply: Transform {
        /// The transforms to potentially apply.
        public let transforms: [Transform]
        /// Probability of applying the transforms.
        public let p: Float

        /// Creates a random apply transform.
        ///
        /// - Parameters:
        ///   - transforms: List of transforms to apply.
        ///   - p: Probability of applying transforms. Defaults to 0.5.
        public init(_ transforms: [Transform], p: Float = 0.5) {
            precondition(p >= 0 && p <= 1,
                "RandomApply: probability p must be in [0, 1], got \(p).")
            self.transforms = transforms
            self.p = p
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            if Float.random(in: 0..<1) < p {
                var result = input
                for transform in transforms {
                    result = transform(result)
                }
                return result
            }
            return input
        }
    }

    /// Apply single transformation randomly picked from a list.
    ///
    /// Equivalent to `torchvision.transforms.RandomChoice`.
    public struct RandomChoice: Transform {
        /// The transforms to choose from.
        public let transforms: [Transform]

        /// Creates a random choice transform.
        ///
        /// - Parameter transforms: List of transforms to choose from.
        public init(_ transforms: [Transform]) {
            precondition(!transforms.isEmpty,
                "RandomChoice: transforms list cannot be empty.")
            self.transforms = transforms
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            let idx = Int.random(in: 0..<transforms.count)
            return transforms[idx](input)
        }
    }
}

// MARK: - Normalization Transforms

extension transforms {

    /// Normalize a tensor with mean and standard deviation.
    ///
    /// Equivalent to `torchvision.transforms.Normalize`.
    ///
    /// For each channel: `output[c] = (input[c] - mean[c]) / std[c]`
    ///
    /// Example:
    /// ```swift
    /// // ImageNet normalization
    /// let normalize = transforms.Normalize(
    ///     mean: [0.485, 0.456, 0.406],
    ///     std: [0.229, 0.224, 0.225]
    /// )
    ///
    /// // Single-channel (grayscale) normalization
    /// let normalizeGray = transforms.Normalize(mean: [0.5], std: [0.5])
    /// ```
    public struct Normalize: Transform {
        /// Mean values for each channel.
        public let mean: [Float]
        /// Standard deviation values for each channel.
        public let std: [Float]
        /// Whether to apply in-place (currently always false in Magma).
        public let inplace: Bool

        /// Creates a normalize transform.
        ///
        /// - Parameters:
        ///   - mean: Sequence of means for each channel.
        ///   - std: Sequence of standard deviations for each channel.
        ///   - inplace: Whether to perform in-place. Defaults to false.
        public init(mean: [Float], std: [Float], inplace: Bool = false) {
            precondition(mean.count == std.count,
                "Normalize: mean and std must have same length. Got mean=\(mean.count), std=\(std.count).")
            precondition(std.allSatisfy { $0 != 0 },
                "Normalize: std values cannot be zero.")
            self.mean = mean
            self.std = std
            self.inplace = inplace
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            // Input expected shape: [batch, height, width, channels] or [height, width, channels]
            // or [batch, channels] for 1D data
            let rank = input.rank
            let numChannels: Int

            if rank >= 3 {
                // Image tensor: channels is last dimension (NHWC format)
                numChannels = input.shape[rank - 1]
            } else if rank == 2 {
                // [batch, features] - treat features as channels
                numChannels = input.shape[1]
            } else {
                // Scalar or 1D - use first mean/std
                numChannels = 1
            }

            precondition(mean.count == 1 || mean.count == numChannels,
                "Normalize: mean length (\(mean.count)) must be 1 or match number of channels (\(numChannels)).")

            // Create mean and std tensors that broadcast correctly
            let meanTensor: Tensor<Float>
            let stdTensor: Tensor<Float>

            if mean.count == 1 {
                // Broadcast single value to all channels
                meanTensor = Tensor<Float>.full(input.shape, mean[0], on: input.device)
                stdTensor = Tensor<Float>.full(input.shape, std[0], on: input.device)
            } else {
                // Per-channel normalization
                // Create tensors that broadcast: shape [1, 1, ..., numChannels]
                var broadcastShape = [Int](repeating: 1, count: rank)
                broadcastShape[rank - 1] = numChannels
                meanTensor = Tensor<Float>(mean, shape: broadcastShape, on: input.device)
                    .broadcast(to: input.shape)
                stdTensor = Tensor<Float>(std, shape: broadcastShape, on: input.device)
                    .broadcast(to: input.shape)
            }

            return (input - meanTensor) / stdTensor
        }
    }

    /// Denormalize a tensor (inverse of Normalize).
    ///
    /// For each channel: `output[c] = input[c] * std[c] + mean[c]`
    ///
    /// This is useful for visualization after normalization.
    public struct Denormalize: Transform {
        /// Mean values used in normalization.
        public let mean: [Float]
        /// Standard deviation values used in normalization.
        public let std: [Float]

        /// Creates a denormalize transform.
        ///
        /// - Parameters:
        ///   - mean: Sequence of means for each channel.
        ///   - std: Sequence of standard deviations for each channel.
        public init(mean: [Float], std: [Float]) {
            precondition(mean.count == std.count,
                "Denormalize: mean and std must have same length.")
            self.mean = mean
            self.std = std
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            let rank = input.rank
            let numChannels = rank >= 3 ? input.shape[rank - 1] : (rank == 2 ? input.shape[1] : 1)

            let meanTensor: Tensor<Float>
            let stdTensor: Tensor<Float>

            if mean.count == 1 {
                meanTensor = Tensor<Float>.full(input.shape, mean[0], on: input.device)
                stdTensor = Tensor<Float>.full(input.shape, std[0], on: input.device)
            } else {
                var broadcastShape = [Int](repeating: 1, count: rank)
                broadcastShape[rank - 1] = numChannels
                meanTensor = Tensor<Float>(mean, shape: broadcastShape, on: input.device)
                    .broadcast(to: input.shape)
                stdTensor = Tensor<Float>(std, shape: broadcastShape, on: input.device)
                    .broadcast(to: input.shape)
            }

            return input * stdTensor + meanTensor
        }
    }
}

// MARK: - Geometric Transforms

extension transforms {

    /// Horizontally flip the input with a given probability.
    ///
    /// Equivalent to `torchvision.transforms.RandomHorizontalFlip`.
    ///
    /// Example:
    /// ```swift
    /// let flip = transforms.RandomHorizontalFlip(p: 0.5)
    /// let flipped = flip(image)  // 50% chance of being flipped
    /// ```
    public struct RandomHorizontalFlip: Transform {
        /// Probability of flipping.
        public let p: Float

        /// Creates a random horizontal flip transform.
        ///
        /// - Parameter p: Probability of the image being flipped. Defaults to 0.5.
        public init(p: Float = 0.5) {
            precondition(p >= 0 && p <= 1,
                "RandomHorizontalFlip: probability p must be in [0, 1], got \(p).")
            self.p = p
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            if Float.random(in: 0..<1) < p {
                return horizontalFlip(input)
            }
            return input
        }
    }

    /// Vertically flip the input with a given probability.
    ///
    /// Equivalent to `torchvision.transforms.RandomVerticalFlip`.
    public struct RandomVerticalFlip: Transform {
        /// Probability of flipping.
        public let p: Float

        /// Creates a random vertical flip transform.
        ///
        /// - Parameter p: Probability of the image being flipped. Defaults to 0.5.
        public init(p: Float = 0.5) {
            precondition(p >= 0 && p <= 1,
                "RandomVerticalFlip: probability p must be in [0, 1], got \(p).")
            self.p = p
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            if Float.random(in: 0..<1) < p {
                return verticalFlip(input)
            }
            return input
        }
    }

    /// Crop the input at the center.
    ///
    /// Equivalent to `torchvision.transforms.CenterCrop`.
    ///
    /// Example:
    /// ```swift
    /// let crop = transforms.CenterCrop(size: (224, 224))
    /// let cropped = crop(image)  // [batch, 224, 224, channels]
    /// ```
    public struct CenterCrop: Transform {
        /// Target output size (height, width).
        public let size: (Int, Int)

        /// Creates a center crop transform.
        ///
        /// - Parameter size: Desired output size (height, width).
        public init(size: (Int, Int)) {
            precondition(size.0 > 0 && size.1 > 0,
                "CenterCrop: size must be positive, got \(size).")
            self.size = size
        }

        /// Creates a center crop transform with square output.
        ///
        /// - Parameter size: Desired output size (both height and width).
        public init(size: Int) {
            self.init(size: (size, size))
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            // Expected shape: [batch, height, width, channels] or [height, width, channels]
            let rank = input.rank
            precondition(rank >= 3,
                "CenterCrop: input must be at least 3D [H, W, C], got rank \(rank).")

            let hasBatch = rank == 4
            let heightIdx = hasBatch ? 1 : 0
            let widthIdx = hasBatch ? 2 : 1

            let inputHeight = input.shape[heightIdx]
            let inputWidth = input.shape[widthIdx]
            let (targetHeight, targetWidth) = size

            precondition(targetHeight <= inputHeight && targetWidth <= inputWidth,
                "CenterCrop: crop size \(size) larger than input size (\(inputHeight), \(inputWidth)).")

            let top = (inputHeight - targetHeight) / 2
            let left = (inputWidth - targetWidth) / 2

            return cropImage(input, top: top, left: left, height: targetHeight, width: targetWidth)
        }
    }

    /// Crop the input at a random location.
    ///
    /// Equivalent to `torchvision.transforms.RandomCrop`.
    ///
    /// Example:
    /// ```swift
    /// let crop = transforms.RandomCrop(size: (224, 224))
    /// let cropped = crop(image)
    /// ```
    public struct RandomCrop: Transform {
        /// Target output size (height, width).
        public let size: (Int, Int)
        /// Padding to add before cropping (optional).
        public let padding: (Int, Int, Int, Int)?

        /// Creates a random crop transform.
        ///
        /// - Parameters:
        ///   - size: Desired output size (height, width).
        ///   - padding: Optional padding (top, bottom, left, right) to add before cropping.
        public init(size: (Int, Int), padding: (Int, Int, Int, Int)? = nil) {
            precondition(size.0 > 0 && size.1 > 0,
                "RandomCrop: size must be positive, got \(size).")
            self.size = size
            self.padding = padding
        }

        /// Creates a random crop transform with square output.
        ///
        /// - Parameter size: Desired output size (both height and width).
        public init(size: Int, padding: (Int, Int, Int, Int)? = nil) {
            self.init(size: (size, size), padding: padding)
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            var tensor = input

            // Apply padding if specified
            if let pad = padding {
                tensor = padImage(tensor, top: pad.0, bottom: pad.1, left: pad.2, right: pad.3)
            }

            let rank = tensor.rank
            precondition(rank >= 3,
                "RandomCrop: input must be at least 3D [H, W, C], got rank \(rank).")

            let hasBatch = rank == 4
            let heightIdx = hasBatch ? 1 : 0
            let widthIdx = hasBatch ? 2 : 1

            let inputHeight = tensor.shape[heightIdx]
            let inputWidth = tensor.shape[widthIdx]
            let (targetHeight, targetWidth) = size

            precondition(targetHeight <= inputHeight && targetWidth <= inputWidth,
                "RandomCrop: crop size \(size) larger than input size (\(inputHeight), \(inputWidth)). " +
                "Consider adding padding.")

            let maxTop = inputHeight - targetHeight
            let maxLeft = inputWidth - targetWidth
            let top = Int.random(in: 0...maxTop)
            let left = Int.random(in: 0...maxLeft)

            return cropImage(tensor, top: top, left: left, height: targetHeight, width: targetWidth)
        }
    }

    /// Pad the input on all sides.
    ///
    /// Equivalent to `torchvision.transforms.Pad`.
    public struct Pad: Transform {
        /// Padding amounts (top, bottom, left, right).
        public let padding: (Int, Int, Int, Int)
        /// Fill value for padding.
        public let fill: Float

        /// Creates a pad transform.
        ///
        /// - Parameters:
        ///   - padding: Padding on each side (top, bottom, left, right).
        ///   - fill: Fill value for constant padding. Defaults to 0.
        public init(padding: (Int, Int, Int, Int), fill: Float = 0) {
            self.padding = padding
            self.fill = fill
        }

        /// Creates a pad transform with uniform padding.
        ///
        /// - Parameters:
        ///   - padding: Padding on all sides.
        ///   - fill: Fill value for constant padding. Defaults to 0.
        public init(padding: Int, fill: Float = 0) {
            self.init(padding: (padding, padding, padding, padding), fill: fill)
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            padImage(input, top: padding.0, bottom: padding.1, left: padding.2, right: padding.3, fill: fill)
        }
    }
}

// MARK: - Color Transforms

extension transforms {

    /// Convert image to grayscale.
    ///
    /// Equivalent to `torchvision.transforms.Grayscale`.
    public struct Grayscale: Transform {
        /// Number of output channels (1 or 3).
        public let numOutputChannels: Int

        /// Creates a grayscale transform.
        ///
        /// - Parameter numOutputChannels: Number of channels in output (1 or 3). Defaults to 1.
        public init(numOutputChannels: Int = 1) {
            precondition(numOutputChannels == 1 || numOutputChannels == 3,
                "Grayscale: numOutputChannels must be 1 or 3, got \(numOutputChannels).")
            self.numOutputChannels = numOutputChannels
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            // Expected shape: [..., H, W, C] where C is 3 (RGB)
            let rank = input.rank
            precondition(rank >= 3,
                "Grayscale: input must be at least 3D, got rank \(rank).")

            let numChannels = input.shape[rank - 1]
            guard numChannels == 3 else {
                // Already grayscale or single channel
                if numOutputChannels == 3 && numChannels == 1 {
                    // Replicate to 3 channels
                    return Tensor<Float>.concat([input, input, input], axis: rank - 1)
                }
                return input
            }

            // RGB to grayscale using standard weights: 0.299*R + 0.587*G + 0.114*B
            let weights = Tensor<Float>([0.299, 0.587, 0.114], shape: [1, 1, 3], on: input.device)
            let broadcastWeights = weights.broadcast(to: input.shape)
            let weighted = input * broadcastWeights
            let gray = weighted.sum(dims: [rank - 1], keepDims: true)

            if numOutputChannels == 3 {
                return Tensor<Float>.concat([gray, gray, gray], axis: rank - 1)
            }
            return gray
        }
    }

    /// Randomly convert image to grayscale with a probability.
    ///
    /// Equivalent to `torchvision.transforms.RandomGrayscale`.
    public struct RandomGrayscale: Transform {
        /// Probability of conversion.
        public let p: Float

        /// Creates a random grayscale transform.
        ///
        /// - Parameter p: Probability of the image being converted. Defaults to 0.1.
        public init(p: Float = 0.1) {
            precondition(p >= 0 && p <= 1,
                "RandomGrayscale: probability p must be in [0, 1], got \(p).")
            self.p = p
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            if Float.random(in: 0..<1) < p {
                let numChannels = input.shape[input.rank - 1]
                let grayscale = Grayscale(numOutputChannels: numChannels)
                return grayscale(input)
            }
            return input
        }
    }

    /// Invert the colors of the image.
    ///
    /// Equivalent to `torchvision.transforms.RandomInvert`.
    public struct RandomInvert: Transform {
        /// Probability of inversion.
        public let p: Float

        /// Creates a random invert transform.
        ///
        /// - Parameter p: Probability of the image being inverted. Defaults to 0.5.
        public init(p: Float = 0.5) {
            precondition(p >= 0 && p <= 1,
                "RandomInvert: probability p must be in [0, 1], got \(p).")
            self.p = p
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            if Float.random(in: 0..<1) < p {
                // Assumes input is in [0, 1] range
                return Tensor<Float>.full(input.shape, 1.0, on: input.device) - input
            }
            return input
        }
    }
}

// MARK: - Miscellaneous Transforms

extension transforms {

    /// Apply a user-defined function as a transform.
    ///
    /// Equivalent to `torchvision.transforms.Lambda`.
    ///
    /// Example:
    /// ```swift
    /// let clamp = transforms.Lambda { $0.clamp(min: 0, max: 1) }
    /// ```
    public struct Lambda: Transform {
        /// The function to apply.
        public let lambd: (Tensor<Float>) -> Tensor<Float>

        /// Creates a lambda transform.
        ///
        /// - Parameter lambd: Function to apply to the tensor.
        public init(_ lambd: @escaping (Tensor<Float>) -> Tensor<Float>) {
            self.lambd = lambd
        }

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            lambd(input)
        }
    }

    /// Identity transform that returns input unchanged.
    ///
    /// Useful as a placeholder or for conditional transform pipelines.
    public struct Identity: Transform {
        public init() {}

        public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
            input
        }
    }
}

// MARK: - Functional Transforms (transforms.functional equivalent)

/// Functional transforms that can be called directly on tensors.
///
/// These mirror `torchvision.transforms.functional`.
public enum functional {

    /// Horizontally flip a tensor.
    ///
    /// - Parameter input: Tensor with shape [..., H, W, C].
    /// - Returns: Horizontally flipped tensor.
    public static func hflip(_ input: Tensor<Float>) -> Tensor<Float> {
        horizontalFlip(input)
    }

    /// Vertically flip a tensor.
    ///
    /// - Parameter input: Tensor with shape [..., H, W, C].
    /// - Returns: Vertically flipped tensor.
    public static func vflip(_ input: Tensor<Float>) -> Tensor<Float> {
        verticalFlip(input)
    }

    /// Crop a tensor.
    ///
    /// - Parameters:
    ///   - input: Input tensor with shape [..., H, W, C].
    ///   - top: Top pixel coordinate.
    ///   - left: Left pixel coordinate.
    ///   - height: Height of crop.
    ///   - width: Width of crop.
    /// - Returns: Cropped tensor.
    public static func crop(
        _ input: Tensor<Float>,
        top: Int,
        left: Int,
        height: Int,
        width: Int
    ) -> Tensor<Float> {
        cropImage(input, top: top, left: left, height: height, width: width)
    }

    /// Normalize a tensor.
    ///
    /// - Parameters:
    ///   - input: Input tensor.
    ///   - mean: Mean for each channel.
    ///   - std: Standard deviation for each channel.
    /// - Returns: Normalized tensor.
    public static func normalize(
        _ input: Tensor<Float>,
        mean: [Float],
        std: [Float]
    ) -> Tensor<Float> {
        transforms.Normalize(mean: mean, std: std)(input)
    }

    /// Pad a tensor.
    ///
    /// - Parameters:
    ///   - input: Input tensor.
    ///   - padding: Padding (top, bottom, left, right).
    ///   - fill: Fill value.
    /// - Returns: Padded tensor.
    public static func pad(
        _ input: Tensor<Float>,
        padding: (Int, Int, Int, Int),
        fill: Float = 0
    ) -> Tensor<Float> {
        padImage(input, top: padding.0, bottom: padding.1, left: padding.2, right: padding.3, fill: fill)
    }
}

// MARK: - Internal Helper Functions

/// Horizontally flip a tensor (reverse along width axis).
internal func horizontalFlip(_ input: Tensor<Float>) -> Tensor<Float> {
    // Shape: [..., H, W, C] - flip along W axis
    let rank = input.rank
    precondition(rank >= 3,
        "horizontalFlip: input must be at least 3D [H, W, C], got rank \(rank).")

    let widthAxis = rank - 2  // W is second-to-last in NHWC

    // Use gather with 1D reversed indices to flip along width axis
    let width = input.shape[widthAxis]
    var indices = [Float]()
    for i in stride(from: width - 1, through: 0, by: -1) {
        indices.append(Float(i))
    }

    // 1D index tensor for slice selection
    let indexTensor = Tensor<Float>(indices, shape: [width], on: input.device)
    return input.gather(indices: indexTensor, axis: widthAxis)
}

/// Vertically flip a tensor (reverse along height axis).
internal func verticalFlip(_ input: Tensor<Float>) -> Tensor<Float> {
    let rank = input.rank
    precondition(rank >= 3,
        "verticalFlip: input must be at least 3D [H, W, C], got rank \(rank).")

    let heightAxis = rank - 3  // H is third-to-last in NHWC (or second-to-last in HWC)

    // Use gather with 1D reversed indices to flip along height axis
    let height = input.shape[heightAxis]
    var indices = [Float]()
    for i in stride(from: height - 1, through: 0, by: -1) {
        indices.append(Float(i))
    }

    // 1D index tensor for slice selection
    let indexTensor = Tensor<Float>(indices, shape: [height], on: input.device)
    return input.gather(indices: indexTensor, axis: heightAxis)
}

/// Crop a tensor at specified location.
internal func cropImage(
    _ input: Tensor<Float>,
    top: Int,
    left: Int,
    height: Int,
    width: Int
) -> Tensor<Float> {
    let rank = input.rank
    precondition(rank >= 3,
        "crop: input must be at least 3D [H, W, C], got rank \(rank).")

    let hasBatch = rank == 4
    let heightAxis = hasBatch ? 1 : 0
    let widthAxis = hasBatch ? 2 : 1

    // Slice along height axis
    let afterHeightSlice = input.sliceAxis(axis: heightAxis, start: top, size: height)
    // Slice along width axis
    let result = afterHeightSlice.sliceAxis(axis: widthAxis, start: left, size: width)

    return result
}

/// Pad an image tensor.
internal func padImage(
    _ input: Tensor<Float>,
    top: Int,
    bottom: Int,
    left: Int,
    right: Int,
    fill: Float = 0
) -> Tensor<Float> {
    let rank = input.rank
    precondition(rank >= 3,
        "padImage: input must be at least 3D [H, W, C], got rank \(rank).")

    let hasBatch = rank == 4

    // Build padding configuration for pad operation
    var lowPad: [Int]
    var highPad: [Int]

    if hasBatch {
        // [batch, height, width, channels]
        lowPad = [0, top, left, 0]
        highPad = [0, bottom, right, 0]
    } else {
        // [height, width, channels]
        lowPad = [top, left, 0]
        highPad = [bottom, right, 0]
    }

    // Calculate new shape
    var newShape = input.shape
    if hasBatch {
        newShape[1] += top + bottom
        newShape[2] += left + right
    } else {
        newShape[0] += top + bottom
        newShape[1] += left + right
    }

    let id = TensorRegistry.shared.nextTensorId()
    let handle = LazyTensorHandle(
        id: id,
        shape: newShape,
        dtype: input.dtype,
        device: input.device
    )

    handle.irNode = .operation(
        op: .pad,
        inputs: [input.handle],
        attributes: [
            "lowPad": lowPad,
            "highPad": highPad,
            "interiorPad": Array(repeating: 0, count: rank),
            "paddingValue": fill
        ]
    )

    TensorRegistry.shared.registerPending(handle)
    return Tensor(handle: handle)
}
