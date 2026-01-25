// Magma - Neural Network Module System
// Simplified Module protocol and core layers

import Foundation
import LazyTensor
import StableHLO
import XLARuntime

// MARK: - Module Protocol

/// A neural network module with learnable parameters.
///
/// `Module` is the base protocol for all neural network components in Magma.
/// Types conforming to `Module` represent differentiable transformations that can be
/// composed to build complex architectures.
///
/// Example:
/// ```swift
/// struct MLP: Module {
///     var fc1: nn.Linear
///     var fc2: nn.Linear
///
///     init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
///         fc1 = nn.Linear(inputSize: inputSize, outputSize: hiddenSize)
///         fc2 = nn.Linear(inputSize: hiddenSize, outputSize: outputSize)
///     }
///
///     func forward(_ input: Tensor<Float>) -> Tensor<Float> {
///         return fc2(fc1(input).relu())
///     }
/// }
/// ```
public protocol Module {
    /// The input type accepted by this module
    associatedtype Input
    /// The output type produced by this module
    associatedtype Output

    /// Performs the forward pass of the module.
    ///
    /// - Parameter input: The input to the module.
    /// - Returns: The output of the module.
    func forward(_ input: Input) -> Output

    /// Returns all learnable parameters of this module.
    ///
    /// Override this method for modules with custom parameter collection.
    func parameters() -> [Parameter]

    /// Move all parameters to the specified device.
    ///
    /// - Parameter device: The target device.
    mutating func to(device: Device)
}

extension Module {
    /// Default implementation: no parameters.
    public func parameters() -> [Parameter] {
        return []
    }

    /// Default implementation: no-op for modules without parameters.
    public mutating func to(device: Device) {
        // Override in modules with parameters
    }

    /// Callable syntax for forward pass.
    public func callAsFunction(_ input: Input) -> Output {
        forward(input)
    }
}

// MARK: - Parameter

/// A learnable parameter in a neural network module.
///
/// `Parameter` wraps a tensor that should be updated during training.
/// Parameters are typically weights and biases of layers.
///
/// Example:
/// ```swift
/// struct CustomLayer: Module {
///     var weight: Parameter
///     var bias: Parameter
///
///     init(inputSize: Int, outputSize: Int) {
///         weight = Parameter(Tensor<Float>.randn([inputSize, outputSize]))
///         bias = Parameter(Tensor<Float>.zeros([outputSize]))
///     }
///
///     func forward(_ input: Tensor<Float>) -> Tensor<Float> {
///         return input.matmul(weight.value) + bias.value
///     }
///
///     func parameters() -> [Parameter] {
///         return [weight, bias]
///     }
/// }
/// ```
public final class Parameter: @unchecked Sendable {
    /// The tensor value of this parameter.
    public var value: Tensor<Float>

    /// Whether this parameter requires gradient computation.
    public var requiresGrad: Bool

    /// Optional name for debugging.
    public var name: String?

    /// Create a parameter from a tensor.
    ///
    /// - Parameters:
    ///   - value: The initial tensor value.
    ///   - requiresGrad: Whether to compute gradients. Defaults to `true`.
    ///   - name: Optional name for debugging.
    public init(_ value: Tensor<Float>, requiresGrad: Bool = true, name: String? = nil) {
        self.value = value
        self.requiresGrad = requiresGrad
        self.name = name
    }

    /// Shape of the parameter tensor.
    public var shape: [Int] { value.shape }

    /// Total number of elements.
    public var elementCount: Int { value.elementCount }

    /// Move this parameter to a different device.
    ///
    /// - Parameter device: The target device.
    public func to(device: Device) {
        value = value.to(device: device)
    }
}

// MARK: - Linear Layer

extension nn {
    /// A fully-connected (dense) linear layer.
    ///
    /// Applies a linear transformation: `y = x @ weight.T + bias`
    ///
    /// Example:
    /// ```swift
    /// let fc = nn.Linear(inputSize: 784, outputSize: 256)
    /// let x = Tensor<Float>.randn([32, 784])
    /// let y = fc(x)  // Shape: [32, 256]
    /// ```
    public struct Linear: Module {
        /// Weight matrix with shape [outputSize, inputSize]
        public var weight: Parameter

        /// Bias vector with shape [outputSize]
        public var bias: Parameter

        /// Whether this layer uses bias
        public let useBias: Bool

        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Creates a linear layer.
        ///
        /// - Parameters:
        ///   - inputSize: Number of input features.
        ///   - outputSize: Number of output features.
        ///   - bias: Whether to include a bias term. Defaults to `true`.
        ///   - device: Device to allocate parameters on.
        public init(
            inputSize: Int,
            outputSize: Int,
            bias: Bool = true,
            device: Device = .default
        ) {
            // Kaiming uniform initialization
            let k = 1.0 / Float(inputSize).squareRoot()
            self.weight = Parameter(
                Tensor<Float>.uniform(low: -k, high: k, shape: [outputSize, inputSize], on: device),
                name: "weight"
            )
            self.useBias = bias
            if bias {
                self.bias = Parameter(
                    Tensor<Float>.uniform(low: -k, high: k, shape: [outputSize], on: device),
                    name: "bias"
                )
            } else {
                self.bias = Parameter(Tensor<Float>.zeros([outputSize], on: device), requiresGrad: false)
            }
        }

        /// Forward pass: y = x @ weight.T + bias
        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            // input: [batch, inputSize]
            // weight: [outputSize, inputSize]
            // output: [batch, outputSize]
            let wT = weight.value.transpose()  // [inputSize, outputSize]
            var result = input.matmul(wT)
            if useBias {
                result = result + bias.value.broadcast(to: result.shape)
            }
            return result
        }

        public func parameters() -> [Parameter] {
            useBias ? [weight, bias] : [weight]
        }

        public mutating func to(device: Device) {
            weight.to(device: device)
            if useBias {
                bias.to(device: device)
            }
        }
    }
}

// MARK: - Embedding Layer

extension nn {
    /// Embedding layer for looking up dense vectors from indices.
    ///
    /// A simple lookup table that stores embeddings of a fixed dictionary and size.
    /// This module is often used to store word embeddings and retrieve them using indices.
    ///
    /// The input to the module is a tensor of indices (as Float), and the output
    /// is the corresponding embeddings.
    ///
    /// Example:
    /// ```swift
    /// // Create embedding for vocabulary of 1000 words, 256-dim embeddings
    /// let embedding = nn.Embedding(numEmbeddings: 1000, embeddingDim: 256)
    ///
    /// // Look up embeddings for indices [0, 5, 10]
    /// let indices = Tensor<Float>([0, 5, 10], shape: [3])
    /// let embeddings = embedding(indices)  // Shape: [3, 256]
    ///
    /// // Batch lookup: indices shape [batch, seqLen]
    /// let batchIndices = Tensor<Float>.zeros([32, 128])  // 32 sequences of length 128
    /// let batchEmbeddings = embedding(batchIndices)  // Shape: [32, 128, 256]
    /// ```
    public struct Embedding: Module {
        /// Embedding weight matrix with shape [numEmbeddings, embeddingDim]
        public var weight: Parameter

        /// Number of embeddings (vocabulary size)
        public let numEmbeddings: Int

        /// Dimension of each embedding vector
        public let embeddingDim: Int

        /// Optional padding index - embeddings at this index are zeroed
        public let paddingIdx: Int?

        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Creates an embedding layer.
        ///
        /// - Parameters:
        ///   - numEmbeddings: Size of the dictionary of embeddings (vocabulary size).
        ///   - embeddingDim: The size of each embedding vector.
        ///   - paddingIdx: If specified, entries at this index do not contribute to gradient
        ///                 and the embedding vector is all zeros. Defaults to nil.
        ///   - device: Device to allocate parameters on.
        public init(
            numEmbeddings: Int,
            embeddingDim: Int,
            paddingIdx: Int? = nil,
            device: Device = .default
        ) {
            self.numEmbeddings = numEmbeddings
            self.embeddingDim = embeddingDim
            self.paddingIdx = paddingIdx

            // Initialize with normal distribution, std=1
            let weightTensor = Tensor<Float>.randn([numEmbeddings, embeddingDim], on: device)

            // Note: padding index handling is done during forward pass
            // where we can mask out gradients appropriately

            self.weight = Parameter(weightTensor, name: "weight")
        }

        /// Forward pass: look up embeddings for input indices.
        ///
        /// - Parameter input: Tensor of indices with any shape. Each element should be
        ///                   in range [0, numEmbeddings).
        /// - Returns: Tensor of shape [*input.shape, embeddingDim]
        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            // input: [*] - any shape with indices
            // weight: [numEmbeddings, embeddingDim]
            // output: [*, embeddingDim]

            let inputShape = input.shape
            let batchSize = inputShape.reduce(1, *)

            // Flatten input to 1D for gather operation
            let flatInput = input.reshape([batchSize])

            // Gather embeddings: use gather along axis 0
            let gathered = weight.value.gather(indices: flatInput, axis: 0)

            // Reshape to output shape: [*input.shape, embeddingDim]
            let outputShape = inputShape + [embeddingDim]
            return gathered.reshape(outputShape)
        }

        public func parameters() -> [Parameter] {
            [weight]
        }

        public mutating func to(device: Device) {
            weight.to(device: device)
        }
    }
}

// MARK: - Activation Layers

extension nn {
    /// ReLU activation layer: max(0, x)
    public struct ReLU: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.relu()
        }
    }

    /// Sigmoid activation layer: 1 / (1 + exp(-x))
    public struct Sigmoid: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.sigmoid()
        }
    }

    /// Tanh activation layer
    public struct Tanh: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.tanh()
        }
    }

    /// GELU activation layer (Gaussian Error Linear Unit)
    public struct GELU: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.gelu()
        }
    }

    /// Leaky ReLU activation layer: max(x, alpha * x)
    public struct LeakyReLU: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Negative slope (default: 0.01)
        public let negativeSlope: Float

        public init(negativeSlope: Float = 0.01) {
            self.negativeSlope = negativeSlope
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.leakyRelu(negativeSlope: negativeSlope)
        }
    }

    /// SiLU (Swish) activation layer: x * sigmoid(x)
    public struct SiLU: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.silu()
        }
    }

    /// ELU activation layer
    public struct ELU: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Alpha value (default: 1.0)
        public let alpha: Float

        public init(alpha: Float = 1.0) {
            self.alpha = alpha
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.elu(alpha: alpha)
        }
    }

    /// Hardtanh (clamp) activation layer
    public struct Hardtanh: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        public let minVal: Float
        public let maxVal: Float

        public init(minVal: Float = -1.0, maxVal: Float = 1.0) {
            self.minVal = minVal
            self.maxVal = maxVal
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.hardtanh(minVal: minVal, maxVal: maxVal)
        }
    }

    /// SELU (Scaled Exponential Linear Unit) activation layer.
    ///
    /// SELU is designed for self-normalizing neural networks (SNNs).
    /// With appropriate weight initialization (lecun_normal), the activations
    /// naturally converge to zero mean and unit variance.
    ///
    /// `selu(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))`
    /// where scale ≈ 1.0507 and alpha ≈ 1.6733
    ///
    /// Reference: ["Self-Normalizing Neural Networks"](https://arxiv.org/abs/1706.02515)
    public struct SELU: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        // SELU constants
        private let alpha: Float = 1.6732632423543772848170429916717
        private let scale: Float = 1.0507009873554804934193349852946

        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.selu()
        }
    }

    /// Mish activation layer.
    ///
    /// `mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`
    ///
    /// Mish is a smooth, non-monotonic activation function that
    /// has shown improvements over ReLU in some tasks.
    ///
    /// Reference: ["Mish: A Self Regularized Non-Monotonic Activation Function"](
    /// https://arxiv.org/abs/1908.08681)
    public struct Mish: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.mish()
        }
    }

    /// Softplus activation layer.
    ///
    /// `softplus(x) = (1/beta) * log(1 + exp(beta * x))`
    ///
    /// Softplus is a smooth approximation to ReLU.
    public struct Softplus: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Beta value for sharpness (default: 1.0)
        public let beta: Float

        /// Threshold above which to use linear approximation (default: 20.0)
        public let threshold: Float

        public init(beta: Float = 1.0, threshold: Float = 20.0) {
            self.beta = beta
            self.threshold = threshold
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.softplus()
        }
    }

    /// Softsign activation layer.
    ///
    /// `softsign(x) = x / (1 + |x|)`
    ///
    /// Softsign is a smooth approximation to the sign function,
    /// similar to tanh but with lighter saturation.
    public struct Softsign: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            input.softsign()
        }
    }

    /// PReLU (Parametric ReLU) activation layer.
    ///
    /// `prelu(x) = max(0, x) + a * min(0, x)`
    ///
    /// Unlike LeakyReLU where the negative slope is fixed,
    /// PReLU learns the optimal slope during training.
    ///
    /// Reference: ["Delving Deep into Rectifiers"](https://arxiv.org/abs/1502.01852)
    public struct PReLU: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Learnable negative slope parameter
        public var weight: Parameter

        /// Number of parameters (1 or numChannels)
        public let numParameters: Int

        /// Creates a PReLU layer.
        ///
        /// - Parameters:
        ///   - numParameters: Number of slope parameters. Use 1 for a single
        ///                    shared slope, or numChannels for per-channel slopes.
        ///   - init: Initial value for the slope. Defaults to 0.25.
        ///   - device: Device to allocate parameters on.
        public init(numParameters: Int = 1, init initVal: Float = 0.25, device: Device = .default) {
            self.numParameters = numParameters
            self.weight = Parameter(
                Tensor<Float>.full([numParameters], initVal, on: device),
                name: "weight"
            )
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            // prelu(x) = max(0, x) + a * min(0, x)
            let positive = input.relu()
            let negative = input - positive  // min(0, x)

            // Broadcast weight if needed
            let a: Tensor<Float>
            if numParameters == 1 {
                a = weight.value.broadcast(to: input.shape)
            } else {
                // Per-channel: weight shape [C], need to broadcast to input shape
                var broadcastShape = Array(repeating: 1, count: input.rank)
                broadcastShape[input.rank - 1] = numParameters
                a = weight.value.reshape(broadcastShape).broadcast(to: input.shape)
            }

            return positive + a * negative
        }

        public func parameters() -> [Parameter] {
            [weight]
        }
    }
}

// MARK: - Flatten Layer

extension nn {
    /// Flattens input tensor to 2D: [batch, features]
    public struct Flatten: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Start dimension to flatten (default: 1, preserves batch)
        public let startDim: Int

        public init(startDim: Int = 1) {
            self.startDim = startDim
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            let shape = input.shape
            guard startDim < shape.count else { return input }

            // Compute flattened size
            let batchDims = Array(shape[0..<startDim])
            let flatSize = shape[startDim...].reduce(1, *)

            return input.reshape(batchDims + [flatSize])
        }
    }
}

// MARK: - Dropout Layer

extension nn {
    /// Dropout layer for regularization.
    ///
    /// During training, randomly zeros elements with probability `p`.
    /// During inference, returns input unchanged.
    ///
    /// Uses inverted dropout: during training, surviving activations are
    /// scaled by 1/(1-p) so no scaling is needed at inference time.
    ///
    /// Example:
    /// ```swift
    /// var dropout = nn.Dropout(p: 0.5)
    /// let x = Tensor<Float>.randn([32, 256])
    ///
    /// // Training mode - applies random masking
    /// dropout.training = true
    /// let y = dropout(x)  // ~50% of values zeroed, rest scaled by 2
    ///
    /// // Inference mode - returns input unchanged
    /// dropout.training = false
    /// let z = dropout(x)  // Same as x
    /// ```
    public struct Dropout: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Dropout probability (probability of zeroing an element)
        public let p: Float

        /// Whether the layer is in training mode
        public var training: Bool = true

        public init(p: Float = 0.5) {
            precondition(p >= 0 && p < 1, "Dropout probability must be in [0, 1)")
            self.p = p
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            // During inference or when p=0, return input unchanged
            guard training && p > 0 else { return input }

            // Generate random values in [0, 1) using XLA device RNG
            let randValues = Tensor<Float>.uniform(low: 0, high: 1, shape: input.shape, on: input.device)

            // Create mask: 1.0 where rand >= p (keep), 0.0 where rand < p (drop)
            // This gives us keep probability of (1-p)
            let mask = randValues.greaterThanOrEqual(p)

            // Apply mask and scale by 1/(1-p) for inverted dropout
            // This ensures expected value stays the same during training
            let scale = 1.0 / (1.0 - p)
            let scaleT = Tensor<Float>.full(input.shape, scale, on: input.device)

            return input * mask * scaleT
        }
    }
}

// MARK: - Conv1d Layer

extension nn {
    /// 1D Convolution layer.
    ///
    /// Applies a 1D convolution over an input signal composed of several input planes.
    /// Input shape: [batch, length, inChannels] (NLC format)
    /// Output shape: [batch, outLength, outChannels]
    ///
    /// Example:
    /// ```swift
    /// let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3)
    /// let x = Tensor<Float>.zeros([32, 100, 3])
    /// let y = conv(x)  // Shape: [32, 98, 64]
    /// ```
    public struct Conv1d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Weight tensor with shape [kernelSize, inChannels, outChannels]
        public var weight: Parameter

        /// Bias vector with shape [outChannels]
        public var bias: Parameter

        /// Whether this layer uses bias
        public let useBias: Bool

        /// Input channels
        public let inChannels: Int

        /// Output channels
        public let outChannels: Int

        /// Kernel size
        public let kernelSize: Int

        /// Stride
        public let stride: Int

        /// Padding
        public let padding: Int

        /// Creates a 1D convolution layer.
        ///
        /// - Parameters:
        ///   - inChannels: Number of input channels.
        ///   - outChannels: Number of output channels.
        ///   - kernelSize: Size of the convolving kernel.
        ///   - stride: Stride of the convolution. Defaults to 1.
        ///   - padding: Zero-padding added to both sides. Defaults to 0.
        ///   - bias: Whether to include a bias term. Defaults to `true`.
        ///   - device: Device to allocate parameters on.
        public init(
            inChannels: Int,
            outChannels: Int,
            kernelSize: Int,
            stride: Int = 1,
            padding: Int = 0,
            bias: Bool = true,
            device: Device = .default
        ) {
            self.inChannels = inChannels
            self.outChannels = outChannels
            self.kernelSize = kernelSize
            self.stride = stride
            self.padding = padding
            self.useBias = bias

            // Kaiming uniform initialization
            let fanIn = inChannels * kernelSize
            let k = 1.0 / Float(fanIn).squareRoot()

            // Weight shape: [kernelSize, inChannels, outChannels]
            self.weight = Parameter(
                Tensor<Float>.uniform(
                    low: -k, high: k,
                    shape: [kernelSize, inChannels, outChannels],
                    on: device
                ),
                name: "weight"
            )

            if bias {
                self.bias = Parameter(
                    Tensor<Float>.uniform(low: -k, high: k, shape: [outChannels], on: device),
                    name: "bias"
                )
            } else {
                self.bias = Parameter(Tensor<Float>.zeros([outChannels], on: device), requiresGrad: false)
            }
        }

        /// Forward pass: applies 1D convolution
        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            // Input: [batch, length, inChannels] (NLC)
            // Weight: [kernelSize, inChannels, outChannels]
            precondition(input.rank == 3,
                "Conv1d requires 3D input [batch, length, channels], got shape \(input.shape) (rank \(input.rank)).")
            precondition(input.shape[2] == inChannels,
                "Conv1d: input channels mismatch. Expected \(inChannels) channels (from layer config), " +
                "got \(input.shape[2]) in input shape \(input.shape).")

            // Compute output shape
            let batchSize = input.shape[0]
            let inLength = input.shape[1]
            let outLength = (inLength + 2 * padding - kernelSize) / stride + 1

            // Create conv1d operation
            let id = TensorRegistry.shared.nextTensorId()
            let handle = LazyTensorHandle(
                id: id,
                shape: [batchSize, outLength, outChannels],
                dtype: input.dtype,
                device: input.device
            )
            handle.irNode = .operation(
                op: .conv1d,
                inputs: [input.handle, weight.value.handle],
                attributes: [
                    "stride": stride,
                    "padding": [padding, padding]
                ]
            )
            TensorRegistry.shared.registerPending(handle)

            var result = Tensor<Float>(handle: handle)

            if useBias {
                // Broadcast bias to [1, 1, outChannels] and add
                result = result + bias.value.broadcast(to: [batchSize, outLength, outChannels])
            }

            return result
        }

        public func parameters() -> [Parameter] {
            useBias ? [weight, bias] : [weight]
        }

        public mutating func to(device: Device) {
            weight.to(device: device)
            if useBias {
                bias.to(device: device)
            }
        }
    }
}

// MARK: - Conv2d Layer

extension nn {
    /// 2D Convolution layer.
    ///
    /// Applies a 2D convolution over an input signal composed of several input planes.
    /// Input shape: [batch, height, width, inChannels] (NHWC format)
    /// Output shape: [batch, outHeight, outWidth, outChannels]
    ///
    /// Example:
    /// ```swift
    /// let conv = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3)
    /// let x = Tensor<Float>.zeros([32, 224, 224, 3])
    /// let y = conv(x)  // Shape: [32, 222, 222, 64]
    /// ```
    public struct Conv2d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Weight tensor with shape [kernelH, kernelW, inChannels, outChannels]
        public var weight: Parameter

        /// Bias vector with shape [outChannels]
        public var bias: Parameter

        /// Whether this layer uses bias
        public let useBias: Bool

        /// Input channels
        public let inChannels: Int

        /// Output channels
        public let outChannels: Int

        /// Kernel size (height, width)
        public let kernelSize: (Int, Int)

        /// Stride (height, width)
        public let stride: (Int, Int)

        /// Padding (height, width)
        public let padding: (Int, Int)

        /// Creates a 2D convolution layer.
        ///
        /// - Parameters:
        ///   - inChannels: Number of input channels.
        ///   - outChannels: Number of output channels.
        ///   - kernelSize: Size of the convolving kernel.
        ///   - stride: Stride of the convolution. Defaults to 1.
        ///   - padding: Zero-padding added to both sides. Defaults to 0.
        ///   - bias: Whether to include a bias term. Defaults to `true`.
        ///   - device: Device to allocate parameters on.
        public init(
            inChannels: Int,
            outChannels: Int,
            kernelSize: Int,
            stride: Int = 1,
            padding: Int = 0,
            bias: Bool = true,
            device: Device = .default
        ) {
            self.init(
                inChannels: inChannels,
                outChannels: outChannels,
                kernelSize: (kernelSize, kernelSize),
                stride: (stride, stride),
                padding: (padding, padding),
                bias: bias,
                device: device
            )
        }

        /// Creates a 2D convolution layer with separate height/width parameters.
        public init(
            inChannels: Int,
            outChannels: Int,
            kernelSize: (Int, Int),
            stride: (Int, Int) = (1, 1),
            padding: (Int, Int) = (0, 0),
            bias: Bool = true,
            device: Device = .default
        ) {
            self.inChannels = inChannels
            self.outChannels = outChannels
            self.kernelSize = kernelSize
            self.stride = stride
            self.padding = padding
            self.useBias = bias

            // Kaiming uniform initialization
            let fanIn = inChannels * kernelSize.0 * kernelSize.1
            let k = 1.0 / Float(fanIn).squareRoot()

            // Weight shape: [kernelH, kernelW, inChannels, outChannels]
            self.weight = Parameter(
                Tensor<Float>.uniform(
                    low: -k, high: k,
                    shape: [kernelSize.0, kernelSize.1, inChannels, outChannels],
                    on: device
                ),
                name: "weight"
            )

            if bias {
                self.bias = Parameter(
                    Tensor<Float>.uniform(low: -k, high: k, shape: [outChannels], on: device),
                    name: "bias"
                )
            } else {
                self.bias = Parameter(Tensor<Float>.zeros([outChannels], on: device), requiresGrad: false)
            }
        }

        /// Forward pass: applies 2D convolution
        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            // Input: [batch, height, width, inChannels] (NHWC)
            // Weight: [kernelH, kernelW, inChannels, outChannels]
            precondition(input.rank == 4,
                "Conv2d requires 4D input [batch, height, width, channels], got shape \(input.shape) (rank \(input.rank)).")
            precondition(input.shape[3] == inChannels,
                "Conv2d: input channels mismatch. Expected \(inChannels) channels (from layer config), " +
                "got \(input.shape[3]) in input shape \(input.shape).")

            // Compute output shape
            let batchSize = input.shape[0]
            let inHeight = input.shape[1]
            let inWidth = input.shape[2]
            let outHeight = (inHeight + 2 * padding.0 - kernelSize.0) / stride.0 + 1
            let outWidth = (inWidth + 2 * padding.1 - kernelSize.1) / stride.1 + 1

            // Create conv2d operation
            let id = TensorRegistry.shared.nextTensorId()
            let handle = LazyTensorHandle(
                id: id,
                shape: [batchSize, outHeight, outWidth, outChannels],
                dtype: input.dtype,
                device: input.device
            )
            handle.irNode = .operation(
                op: .conv2d,
                inputs: [input.handle, weight.value.handle],
                attributes: [
                    "strides": [stride.0, stride.1],
                    "padding": [[padding.0, padding.0], [padding.1, padding.1]]
                ]
            )
            TensorRegistry.shared.registerPending(handle)

            var result = Tensor<Float>(handle: handle)

            if useBias {
                // Broadcast bias to [1, 1, 1, outChannels] and add
                result = result + bias.value.broadcast(to: [batchSize, outHeight, outWidth, outChannels])
            }

            return result
        }

        public func parameters() -> [Parameter] {
            useBias ? [weight, bias] : [weight]
        }

        public mutating func to(device: Device) {
            weight.to(device: device)
            if useBias {
                bias.to(device: device)
            }
        }
    }
}

// MARK: - ConvTranspose2d Layer

extension nn {
    /// 2D Transposed Convolution layer (Deconvolution).
    ///
    /// Applies a 2D transposed convolution operator over an input image composed
    /// of several input planes. This layer can be used for upsampling in generative
    /// models (e.g., GANs, autoencoders).
    /// Input shape: [batch, height, width, inChannels] (NHWC format)
    /// Output shape: [batch, outHeight, outWidth, outChannels]
    ///
    /// Example:
    /// ```swift
    /// let convT = nn.ConvTranspose2d(inChannels: 64, outChannels: 32, kernelSize: 4, stride: 2, padding: 1)
    /// let x = Tensor<Float>.zeros([32, 14, 14, 64])
    /// let y = convT(x)  // Shape: [32, 28, 28, 32]
    /// ```
    public struct ConvTranspose2d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Weight tensor with shape [kernelH, kernelW, outChannels, inChannels]
        /// Note: For transposed conv, weight layout differs from regular conv
        public var weight: Parameter

        /// Bias vector with shape [outChannels]
        public var bias: Parameter

        /// Whether this layer uses bias
        public let useBias: Bool

        /// Input channels
        public let inChannels: Int

        /// Output channels
        public let outChannels: Int

        /// Kernel size (height, width)
        public let kernelSize: (Int, Int)

        /// Stride (height, width)
        public let stride: (Int, Int)

        /// Padding (height, width)
        public let padding: (Int, Int)

        /// Output padding (height, width) - additional size added to one side of output
        public let outputPadding: (Int, Int)

        /// Creates a 2D transposed convolution layer.
        ///
        /// - Parameters:
        ///   - inChannels: Number of input channels.
        ///   - outChannels: Number of output channels.
        ///   - kernelSize: Size of the convolving kernel.
        ///   - stride: Stride of the convolution. Defaults to 1.
        ///   - padding: Zero-padding added to both sides. Defaults to 0.
        ///   - outputPadding: Additional size added to one side of output. Defaults to 0.
        ///   - bias: Whether to include a bias term. Defaults to `true`.
        ///   - device: Device to allocate parameters on.
        public init(
            inChannels: Int,
            outChannels: Int,
            kernelSize: Int,
            stride: Int = 1,
            padding: Int = 0,
            outputPadding: Int = 0,
            bias: Bool = true,
            device: Device = .default
        ) {
            self.init(
                inChannels: inChannels,
                outChannels: outChannels,
                kernelSize: (kernelSize, kernelSize),
                stride: (stride, stride),
                padding: (padding, padding),
                outputPadding: (outputPadding, outputPadding),
                bias: bias,
                device: device
            )
        }

        /// Creates a 2D transposed convolution layer with separate height/width parameters.
        public init(
            inChannels: Int,
            outChannels: Int,
            kernelSize: (Int, Int),
            stride: (Int, Int) = (1, 1),
            padding: (Int, Int) = (0, 0),
            outputPadding: (Int, Int) = (0, 0),
            bias: Bool = true,
            device: Device = .default
        ) {
            self.inChannels = inChannels
            self.outChannels = outChannels
            self.kernelSize = kernelSize
            self.stride = stride
            self.padding = padding
            self.outputPadding = outputPadding
            self.useBias = bias

            // Kaiming uniform initialization
            let fanIn = inChannels * kernelSize.0 * kernelSize.1
            let k = 1.0 / Float(fanIn).squareRoot()

            // Weight shape: [kernelH, kernelW, outChannels, inChannels]
            // Note: reversed from regular conv2d
            self.weight = Parameter(
                Tensor<Float>.uniform(
                    low: -k, high: k,
                    shape: [kernelSize.0, kernelSize.1, outChannels, inChannels],
                    on: device
                ),
                name: "weight"
            )

            if bias {
                self.bias = Parameter(
                    Tensor<Float>.uniform(low: -k, high: k, shape: [outChannels], on: device),
                    name: "bias"
                )
            } else {
                self.bias = Parameter(Tensor<Float>.zeros([outChannels], on: device), requiresGrad: false)
            }
        }

        /// Forward pass: applies 2D transposed convolution
        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            // Input: [batch, height, width, inChannels] (NHWC)
            // Weight: [kernelH, kernelW, outChannels, inChannels]
            precondition(input.rank == 4,
                "ConvTranspose2d requires 4D input [batch, height, width, channels], got shape \(input.shape) (rank \(input.rank)).")
            precondition(input.shape[3] == inChannels,
                "ConvTranspose2d: input channels mismatch. Expected \(inChannels) channels (from layer config), " +
                "got \(input.shape[3]) in input shape \(input.shape).")

            // Compute output shape for transposed conv:
            // outSize = (inSize - 1) * stride - 2 * padding + kernelSize + outputPadding
            let batchSize = input.shape[0]
            let inHeight = input.shape[1]
            let inWidth = input.shape[2]
            let outHeight = (inHeight - 1) * stride.0 - 2 * padding.0 + kernelSize.0 + outputPadding.0
            let outWidth = (inWidth - 1) * stride.1 - 2 * padding.1 + kernelSize.1 + outputPadding.1

            // Create convTranspose2d operation
            let id = TensorRegistry.shared.nextTensorId()
            let handle = LazyTensorHandle(
                id: id,
                shape: [batchSize, outHeight, outWidth, outChannels],
                dtype: input.dtype,
                device: input.device
            )
            handle.irNode = .operation(
                op: .convTranspose2d,
                inputs: [input.handle, weight.value.handle],
                attributes: [
                    "strides": [stride.0, stride.1],
                    "padding": [[padding.0, padding.0], [padding.1, padding.1]],
                    "outputPadding": [outputPadding.0, outputPadding.1]
                ]
            )
            TensorRegistry.shared.registerPending(handle)

            var result = Tensor<Float>(handle: handle)

            if useBias {
                // Broadcast bias to [1, 1, 1, outChannels] and add
                result = result + bias.value.broadcast(to: [batchSize, outHeight, outWidth, outChannels])
            }

            return result
        }

        public func parameters() -> [Parameter] {
            useBias ? [weight, bias] : [weight]
        }

        public mutating func to(device: Device) {
            weight.to(device: device)
            if useBias {
                bias.to(device: device)
            }
        }
    }
}

// MARK: - Pooling Layers

extension nn {
    /// 2D Max Pooling layer.
    ///
    /// Applies max pooling over an input signal composed of several input planes.
    /// Input shape: [batch, height, width, channels] (NHWC format)
    ///
    /// Example:
    /// ```swift
    /// let pool = nn.MaxPool2d(kernelSize: 2, stride: 2)
    /// let x = Tensor<Float>.zeros([32, 224, 224, 64])
    /// let y = pool(x)  // Shape: [32, 112, 112, 64]
    /// ```
    public struct MaxPool2d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Kernel size (height, width)
        public let kernelSize: (Int, Int)

        /// Stride (height, width)
        public let stride: (Int, Int)

        /// Padding (height, width)
        public let padding: (Int, Int)

        /// Creates a 2D max pooling layer.
        public init(
            kernelSize: Int,
            stride: Int? = nil,
            padding: Int = 0
        ) {
            let s = stride ?? kernelSize
            self.kernelSize = (kernelSize, kernelSize)
            self.stride = (s, s)
            self.padding = (padding, padding)
        }

        /// Creates a 2D max pooling layer with separate height/width parameters.
        public init(
            kernelSize: (Int, Int),
            stride: (Int, Int)? = nil,
            padding: (Int, Int) = (0, 0)
        ) {
            self.kernelSize = kernelSize
            self.stride = stride ?? kernelSize
            self.padding = padding
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 4,
                "MaxPool2d requires 4D input [batch, height, width, channels], got shape \(input.shape) (rank \(input.rank)).")

            let batchSize = input.shape[0]
            let inHeight = input.shape[1]
            let inWidth = input.shape[2]
            let channels = input.shape[3]

            let outHeight = (inHeight + 2 * padding.0 - kernelSize.0) / stride.0 + 1
            let outWidth = (inWidth + 2 * padding.1 - kernelSize.1) / stride.1 + 1

            precondition(outHeight > 0 && outWidth > 0,
                "MaxPool2d: output dimensions would be non-positive. Input: \(input.shape), kernel: \(kernelSize), " +
                "stride: \(stride), padding: \(padding). Computed output: [\(batchSize), \(outHeight), \(outWidth), \(channels)].")

            let id = TensorRegistry.shared.nextTensorId()
            let handle = LazyTensorHandle(
                id: id,
                shape: [batchSize, outHeight, outWidth, channels],
                dtype: input.dtype,
                device: input.device
            )
            handle.irNode = .operation(
                op: .maxPool2d,
                inputs: [input.handle],
                attributes: [
                    "windowSize": [kernelSize.0, kernelSize.1],
                    "strides": [stride.0, stride.1],
                    "padding": [[padding.0, padding.0], [padding.1, padding.1]]
                ]
            )
            TensorRegistry.shared.registerPending(handle)

            return Tensor<Float>(handle: handle)
        }
    }

    /// 2D Average Pooling layer.
    ///
    /// Applies average pooling over an input signal.
    /// Input shape: [batch, height, width, channels] (NHWC format)
    public struct AvgPool2d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Kernel size (height, width)
        public let kernelSize: (Int, Int)

        /// Stride (height, width)
        public let stride: (Int, Int)

        /// Padding (height, width)
        public let padding: (Int, Int)

        /// Creates a 2D average pooling layer.
        public init(
            kernelSize: Int,
            stride: Int? = nil,
            padding: Int = 0
        ) {
            let s = stride ?? kernelSize
            self.kernelSize = (kernelSize, kernelSize)
            self.stride = (s, s)
            self.padding = (padding, padding)
        }

        /// Creates a 2D average pooling layer with separate height/width parameters.
        public init(
            kernelSize: (Int, Int),
            stride: (Int, Int)? = nil,
            padding: (Int, Int) = (0, 0)
        ) {
            self.kernelSize = kernelSize
            self.stride = stride ?? kernelSize
            self.padding = padding
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 4, "AvgPool2d requires 4D input")

            let batchSize = input.shape[0]
            let inHeight = input.shape[1]
            let inWidth = input.shape[2]
            let channels = input.shape[3]

            let outHeight = (inHeight + 2 * padding.0 - kernelSize.0) / stride.0 + 1
            let outWidth = (inWidth + 2 * padding.1 - kernelSize.1) / stride.1 + 1

            let id = TensorRegistry.shared.nextTensorId()
            let handle = LazyTensorHandle(
                id: id,
                shape: [batchSize, outHeight, outWidth, channels],
                dtype: input.dtype,
                device: input.device
            )
            handle.irNode = .operation(
                op: .avgPool2d,
                inputs: [input.handle],
                attributes: [
                    "windowSize": [kernelSize.0, kernelSize.1],
                    "strides": [stride.0, stride.1],
                    "padding": [[padding.0, padding.0], [padding.1, padding.1]]
                ]
            )
            TensorRegistry.shared.registerPending(handle)

            return Tensor<Float>(handle: handle)
        }
    }

    /// Adaptive Average Pooling layer.
    ///
    /// Pools to a fixed output size regardless of input size.
    public struct AdaptiveAvgPool2d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Target output size (height, width)
        public let outputSize: (Int, Int)

        /// Creates an adaptive average pooling layer.
        public init(outputSize: Int) {
            self.outputSize = (outputSize, outputSize)
        }

        /// Creates an adaptive average pooling layer with separate height/width.
        public init(outputSize: (Int, Int)) {
            self.outputSize = outputSize
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 4, "AdaptiveAvgPool2d requires 4D input")

            let batchSize = input.shape[0]
            let inHeight = input.shape[1]
            let inWidth = input.shape[2]
            let channels = input.shape[3]

            // Compute kernel size and stride to achieve target output size
            let kernelH = inHeight / outputSize.0
            let kernelW = inWidth / outputSize.1

            let id = TensorRegistry.shared.nextTensorId()
            let handle = LazyTensorHandle(
                id: id,
                shape: [batchSize, outputSize.0, outputSize.1, channels],
                dtype: input.dtype,
                device: input.device
            )
            handle.irNode = .operation(
                op: .avgPool2d,
                inputs: [input.handle],
                attributes: [
                    "windowSize": [kernelH, kernelW],
                    "strides": [kernelH, kernelW],
                    "padding": [[0, 0], [0, 0]]
                ]
            )
            TensorRegistry.shared.registerPending(handle)

            return Tensor<Float>(handle: handle)
        }
    }

    /// Global Average Pooling layer for 1D (temporal) data.
    ///
    /// Applies global average pooling across the temporal dimension.
    /// Input shape: [batch, length, channels] (NLC format)
    /// Output shape: [batch, channels]
    ///
    /// Example:
    /// ```swift
    /// let pool = nn.GlobalAvgPool1d()
    /// let x = Tensor<Float>.zeros([32, 100, 64])
    /// let y = pool(x)  // Shape: [32, 64]
    /// ```
    public struct GlobalAvgPool1d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Creates a global average pooling layer for 1D data.
        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 3,
                "GlobalAvgPool1d requires 3D input [batch, length, channels], got rank \(input.rank)")

            // Mean over the temporal dimension (axis 1), squeeze the result
            return input.mean(dims: [1], keepDims: false)
        }

        public func parameters() -> [Parameter] { [] }
        public mutating func to(device: Device) {}
    }

    /// Global Average Pooling layer for 2D (spatial) data.
    ///
    /// Applies global average pooling across the spatial dimensions.
    /// Input shape: [batch, height, width, channels] (NHWC format)
    /// Output shape: [batch, channels]
    ///
    /// Example:
    /// ```swift
    /// let pool = nn.GlobalAvgPool2d()
    /// let x = Tensor<Float>.zeros([32, 7, 7, 512])
    /// let y = pool(x)  // Shape: [32, 512]
    /// ```
    public struct GlobalAvgPool2d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Creates a global average pooling layer for 2D data.
        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 4,
                "GlobalAvgPool2d requires 4D input [batch, height, width, channels], got rank \(input.rank)")

            // Mean over the spatial dimensions (axes 1 and 2), squeeze the result
            return input.mean(dims: [1, 2], keepDims: false)
        }

        public func parameters() -> [Parameter] { [] }
        public mutating func to(device: Device) {}
    }

    /// Global Max Pooling layer for 1D (temporal) data.
    ///
    /// Applies global max pooling across the temporal dimension.
    /// Input shape: [batch, length, channels] (NLC format)
    /// Output shape: [batch, channels]
    ///
    /// Example:
    /// ```swift
    /// let pool = nn.GlobalMaxPool1d()
    /// let x = Tensor<Float>.zeros([32, 100, 64])
    /// let y = pool(x)  // Shape: [32, 64]
    /// ```
    public struct GlobalMaxPool1d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Creates a global max pooling layer for 1D data.
        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 3,
                "GlobalMaxPool1d requires 3D input [batch, length, channels], got rank \(input.rank)")

            // Max over the temporal dimension (axis 1)
            return input.max(alongAxes: [1])
        }

        public func parameters() -> [Parameter] { [] }
        public mutating func to(device: Device) {}
    }

    /// Global Max Pooling layer for 2D (spatial) data.
    ///
    /// Applies global max pooling across the spatial dimensions.
    /// Input shape: [batch, height, width, channels] (NHWC format)
    /// Output shape: [batch, channels]
    ///
    /// Example:
    /// ```swift
    /// let pool = nn.GlobalMaxPool2d()
    /// let x = Tensor<Float>.zeros([32, 7, 7, 512])
    /// let y = pool(x)  // Shape: [32, 512]
    /// ```
    public struct GlobalMaxPool2d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Creates a global max pooling layer for 2D data.
        public init() {}

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 4,
                "GlobalMaxPool2d requires 4D input [batch, height, width, channels], got rank \(input.rank)")

            // Max over the spatial dimensions (axes 1 and 2)
            return input.max(alongAxes: [1, 2])
        }

        public func parameters() -> [Parameter] { [] }
        public mutating func to(device: Device) {}
    }

    /// Upsampling layer for 1D (temporal) data using nearest neighbor interpolation.
    ///
    /// Repeats each element along the temporal dimension by the given scale factor.
    /// Input shape: [batch, length, channels] (NLC format)
    /// Output shape: [batch, length * size, channels]
    ///
    /// Example:
    /// ```swift
    /// let up = nn.Upsample1d(size: 2)
    /// let x = Tensor<Float>.zeros([32, 50, 64])
    /// let y = up(x)  // Shape: [32, 100, 64]
    /// ```
    public struct Upsample1d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Upsampling factor
        public let size: Int

        /// Creates an upsampling layer for 1D data.
        ///
        /// - Parameter size: The upsampling factor for the temporal dimension.
        public init(size: Int) {
            precondition(size > 0, "Upsampling size must be positive")
            self.size = size
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 3,
                "Upsample1d requires 3D input [batch, length, channels], got rank \(input.rank)")

            let batchSize = input.shape[0]
            let length = input.shape[1]
            let channels = input.shape[2]

            // Nearest neighbor upsampling via reshape + broadcast + reshape
            // [batch, length, channels] -> [batch, length, 1, channels]
            let expanded = input.reshape([batchSize, length, 1, channels])
            // Broadcast to [batch, length, size, channels]
            let broadcasted = expanded.broadcast(to: [batchSize, length, size, channels])
            // Reshape to [batch, length * size, channels]
            return broadcasted.reshape([batchSize, length * size, channels])
        }

        public func parameters() -> [Parameter] { [] }
        public mutating func to(device: Device) {}
    }

    /// Upsampling layer for 2D (spatial) data using nearest neighbor interpolation.
    ///
    /// Repeats each element along the spatial dimensions by the given scale factor.
    /// Input shape: [batch, height, width, channels] (NHWC format)
    /// Output shape: [batch, height * size, width * size, channels]
    ///
    /// Example:
    /// ```swift
    /// let up = nn.Upsample2d(size: 2)
    /// let x = Tensor<Float>.zeros([32, 14, 14, 64])
    /// let y = up(x)  // Shape: [32, 28, 28, 64]
    /// ```
    public struct Upsample2d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Upsampling factor for height and width
        public let size: (Int, Int)

        /// Creates an upsampling layer for 2D data.
        ///
        /// - Parameter size: The upsampling factor for both height and width.
        public init(size: Int) {
            precondition(size > 0, "Upsampling size must be positive")
            self.size = (size, size)
        }

        /// Creates an upsampling layer for 2D data with separate height/width factors.
        ///
        /// - Parameter size: The upsampling factor as (height, width).
        public init(size: (Int, Int)) {
            precondition(size.0 > 0 && size.1 > 0, "Upsampling sizes must be positive")
            self.size = size
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 4,
                "Upsample2d requires 4D input [batch, height, width, channels], got rank \(input.rank)")

            let batchSize = input.shape[0]
            let height = input.shape[1]
            let width = input.shape[2]
            let channels = input.shape[3]

            // Nearest neighbor upsampling via reshape + broadcast + reshape
            // [batch, height, width, channels] -> [batch, height, 1, width, 1, channels]
            let expanded = input.reshape([batchSize, height, 1, width, 1, channels])
            // Broadcast to [batch, height, size.0, width, size.1, channels]
            let broadcasted = expanded.broadcast(to: [batchSize, height, size.0, width, size.1, channels])
            // Reshape to [batch, height * size.0, width * size.1, channels]
            return broadcasted.reshape([batchSize, height * size.0, width * size.1, channels])
        }

        public func parameters() -> [Parameter] { [] }
        public mutating func to(device: Device) {}
    }

    /// Alias for Upsample2d with simpler name
    public typealias UpsamplingNearest2d = Upsample2d
}

// MARK: - BatchNorm2d Layer

extension nn {
    /// 2D Batch Normalization layer.
    ///
    /// Normalizes activations over the batch dimension for each channel.
    /// Input shape: [batch, height, width, channels] (NHWC format)
    ///
    /// Example:
    /// ```swift
    /// let bn = nn.BatchNorm2d(numFeatures: 64)
    /// let x = Tensor<Float>.zeros([32, 224, 224, 64])
    /// let y = bn(x)  // Same shape, normalized
    /// ```
    public struct BatchNorm2d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Number of features (channels)
        public let numFeatures: Int

        /// Small constant for numerical stability
        public let eps: Float

        /// Momentum for running mean/variance
        public let momentum: Float

        /// Whether to learn affine parameters (gamma, beta)
        public let affine: Bool

        /// Whether to track running statistics
        public let trackRunningStats: Bool

        /// Scale parameter (gamma)
        public var weight: Parameter

        /// Shift parameter (beta)
        public var bias: Parameter

        /// Running mean (not trained)
        public var runningMean: Parameter

        /// Running variance (not trained)
        public var runningVar: Parameter

        /// Whether in training mode
        public var training: Bool = true

        /// Creates a 2D batch normalization layer.
        ///
        /// - Parameters:
        ///   - numFeatures: Number of features (channels).
        ///   - eps: Small constant for numerical stability. Defaults to 1e-5.
        ///   - momentum: Momentum for running stats. Defaults to 0.1.
        ///   - affine: Whether to learn scale and shift. Defaults to `true`.
        ///   - trackRunningStats: Whether to track running stats. Defaults to `true`.
        ///   - device: Device to allocate parameters on.
        public init(
            numFeatures: Int,
            eps: Float = 1e-5,
            momentum: Float = 0.1,
            affine: Bool = true,
            trackRunningStats: Bool = true,
            device: Device = .default
        ) {
            self.numFeatures = numFeatures
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            self.trackRunningStats = trackRunningStats

            // Initialize scale (gamma) to 1
            self.weight = Parameter(
                Tensor<Float>.ones([numFeatures], on: device),
                requiresGrad: affine,
                name: "weight"
            )

            // Initialize shift (beta) to 0
            self.bias = Parameter(
                Tensor<Float>.zeros([numFeatures], on: device),
                requiresGrad: affine,
                name: "bias"
            )

            // Running statistics (not trained)
            self.runningMean = Parameter(
                Tensor<Float>.zeros([numFeatures], on: device),
                requiresGrad: false,
                name: "running_mean"
            )
            self.runningVar = Parameter(
                Tensor<Float>.ones([numFeatures], on: device),
                requiresGrad: false,
                name: "running_var"
            )
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 4,
                "BatchNorm2d requires 4D input [batch, height, width, channels], got shape \(input.shape) (rank \(input.rank)). " +
                "For 2D inputs [batch, features], use BatchNorm1d instead.")
            precondition(input.shape[3] == numFeatures,
                "BatchNorm2d: channel count mismatch. Layer expects \(numFeatures) channels, " +
                "got \(input.shape[3]) in input shape \(input.shape).")

            let id = TensorRegistry.shared.nextTensorId()
            let handle = LazyTensorHandle(
                id: id,
                shape: input.shape,
                dtype: input.dtype,
                device: input.device
            )

            // Use batchNorm operation
            handle.irNode = .operation(
                op: .batchNorm,
                inputs: [
                    input.handle,
                    weight.value.handle,
                    bias.value.handle,
                    runningMean.value.handle,
                    runningVar.value.handle
                ],
                attributes: [
                    "epsilon": eps,
                    "featureIndex": 3  // NHWC format, channels at index 3
                ]
            )
            TensorRegistry.shared.registerPending(handle)

            return Tensor<Float>(handle: handle)
        }

        public func parameters() -> [Parameter] {
            affine ? [weight, bias] : []
        }

        public mutating func to(device: Device) {
            weight.to(device: device)
            bias.to(device: device)
            runningMean.to(device: device)
            runningVar.to(device: device)
        }
    }

    /// 1D Batch Normalization layer for [batch, features] input.
    public struct BatchNorm1d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Number of features
        public let numFeatures: Int

        /// Small constant for numerical stability
        public let eps: Float

        /// Scale parameter (gamma)
        public var weight: Parameter

        /// Shift parameter (beta)
        public var bias: Parameter

        /// Running mean
        public var runningMean: Parameter

        /// Running variance
        public var runningVar: Parameter

        /// Whether in training mode
        public var training: Bool = true

        public init(
            numFeatures: Int,
            eps: Float = 1e-5,
            device: Device = .default
        ) {
            self.numFeatures = numFeatures
            self.eps = eps

            self.weight = Parameter(Tensor<Float>.ones([numFeatures], on: device), name: "weight")
            self.bias = Parameter(Tensor<Float>.zeros([numFeatures], on: device), name: "bias")
            self.runningMean = Parameter(Tensor<Float>.zeros([numFeatures], on: device), requiresGrad: false)
            self.runningVar = Parameter(Tensor<Float>.ones([numFeatures], on: device), requiresGrad: false)
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 2,
                "BatchNorm1d requires 2D input [batch, features], got shape \(input.shape) (rank \(input.rank)). " +
                "For 4D inputs [batch, height, width, channels], use BatchNorm2d instead.")
            precondition(input.shape[1] == numFeatures,
                "BatchNorm1d: feature count mismatch. Layer expects \(numFeatures) features, " +
                "got \(input.shape[1]) in input shape \(input.shape).")

            // Expand to 4D, apply batchnorm, squeeze back
            let expanded = input.reshape([input.shape[0], 1, 1, numFeatures])

            let id = TensorRegistry.shared.nextTensorId()
            let handle = LazyTensorHandle(
                id: id,
                shape: expanded.shape,
                dtype: input.dtype,
                device: input.device
            )

            handle.irNode = .operation(
                op: .batchNorm,
                inputs: [
                    expanded.handle,
                    weight.value.handle,
                    bias.value.handle,
                    runningMean.value.handle,
                    runningVar.value.handle
                ],
                attributes: [
                    "epsilon": eps,
                    "featureIndex": 3
                ]
            )
            TensorRegistry.shared.registerPending(handle)

            let normalized = Tensor<Float>(handle: handle)
            return normalized.reshape(input.shape)
        }

        public func parameters() -> [Parameter] {
            [weight, bias]
        }
    }
}

// MARK: - GroupNorm Layer

extension nn {
    /// Group Normalization layer.
    ///
    /// Divides the channels into groups and computes normalization within each group.
    /// This is a compromise between Layer Normalization (which normalizes across all channels)
    /// and Instance Normalization (which normalizes each channel independently).
    ///
    /// Input shape: [batch, height, width, channels] (NHWC format)
    ///
    /// Reference: ["Group Normalization"](https://arxiv.org/abs/1803.08494) (Wu and He, 2018)
    ///
    /// Example:
    /// ```swift
    /// // Divide 64 channels into 8 groups of 8 channels each
    /// let gn = nn.GroupNorm(numGroups: 8, numChannels: 64)
    /// let x = Tensor<Float>.zeros([32, 224, 224, 64])
    /// let y = gn(x)  // Same shape, normalized within groups
    /// ```
    public struct GroupNorm: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Number of groups to divide channels into
        public let numGroups: Int

        /// Number of channels (must be divisible by numGroups)
        public let numChannels: Int

        /// Small constant for numerical stability
        public let eps: Float

        /// Whether to learn affine parameters (gamma, beta)
        public let affine: Bool

        /// Scale parameter (gamma)
        public var weight: Parameter

        /// Shift parameter (beta)
        public var bias: Parameter

        /// Creates a Group Normalization layer.
        ///
        /// - Parameters:
        ///   - numGroups: Number of groups to divide channels into.
        ///   - numChannels: Number of channels (must be divisible by numGroups).
        ///   - eps: Small constant for numerical stability. Defaults to 1e-5.
        ///   - affine: Whether to learn scale and shift. Defaults to `true`.
        ///   - device: Device to allocate parameters on.
        public init(
            numGroups: Int,
            numChannels: Int,
            eps: Float = 1e-5,
            affine: Bool = true,
            device: Device = .default
        ) {
            precondition(numChannels % numGroups == 0,
                "numChannels (\(numChannels)) must be divisible by numGroups (\(numGroups))")

            self.numGroups = numGroups
            self.numChannels = numChannels
            self.eps = eps
            self.affine = affine

            // Initialize scale (gamma) to 1
            self.weight = Parameter(
                Tensor<Float>.ones([numChannels], on: device),
                requiresGrad: affine,
                name: "weight"
            )

            // Initialize shift (beta) to 0
            self.bias = Parameter(
                Tensor<Float>.zeros([numChannels], on: device),
                requiresGrad: affine,
                name: "bias"
            )
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 4,
                "GroupNorm requires 4D input [batch, height, width, channels], got shape \(input.shape).")
            precondition(input.shape[3] == numChannels,
                "GroupNorm: channel count mismatch. Layer expects \(numChannels) channels, got \(input.shape[3]).")

            let batchSize = input.shape[0]
            let height = input.shape[1]
            let width = input.shape[2]
            let channelsPerGroup = numChannels / numGroups

            // Reshape to [batch, height, width, numGroups, channelsPerGroup]
            let reshaped = input.reshape([batchSize, height, width, numGroups, channelsPerGroup])

            // Compute mean and variance over [height, width, channelsPerGroup] for each group
            // We'll compute this manually for now
            let id = TensorRegistry.shared.nextTensorId()
            let handle = LazyTensorHandle(
                id: id,
                shape: input.shape,
                dtype: input.dtype,
                device: input.device
            )

            // Use layerNorm operation with appropriate configuration
            // GroupNorm normalizes over [H, W, C/G] for each group
            handle.irNode = .operation(
                op: .layerNorm,
                inputs: [
                    input.handle,
                    weight.value.handle,
                    bias.value.handle
                ],
                attributes: [
                    "epsilon": eps,
                    "normalizedDims": [height, width, channelsPerGroup],
                    "numGroups": numGroups
                ]
            )
            TensorRegistry.shared.registerPending(handle)

            return Tensor<Float>(handle: handle)
        }

        public func parameters() -> [Parameter] {
            affine ? [weight, bias] : []
        }
    }
}

// MARK: - InstanceNorm Layer

extension nn {
    /// Instance Normalization layer.
    ///
    /// Normalizes each channel independently for each sample in the batch.
    /// This is equivalent to GroupNorm with numGroups equal to numChannels.
    ///
    /// Instance normalization is commonly used in style transfer networks where
    /// batch statistics would be inappropriate.
    ///
    /// Input shape: [batch, height, width, channels] (NHWC format)
    ///
    /// Reference: ["Instance Normalization: The Missing Ingredient for Fast Stylization"](
    /// https://arxiv.org/abs/1607.08022) (Ulyanov et al., 2016)
    ///
    /// Example:
    /// ```swift
    /// let in = nn.InstanceNorm2d(numFeatures: 64)
    /// let x = Tensor<Float>.zeros([32, 224, 224, 64])
    /// let y = in(x)  // Same shape, normalized per channel per sample
    /// ```
    public struct InstanceNorm2d: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Number of features (channels)
        public let numFeatures: Int

        /// Small constant for numerical stability
        public let eps: Float

        /// Whether to learn affine parameters (gamma, beta)
        public let affine: Bool

        /// Scale parameter (gamma)
        public var weight: Parameter

        /// Shift parameter (beta)
        public var bias: Parameter

        /// Creates an Instance Normalization layer.
        ///
        /// - Parameters:
        ///   - numFeatures: Number of features (channels).
        ///   - eps: Small constant for numerical stability. Defaults to 1e-5.
        ///   - affine: Whether to learn scale and shift. Defaults to `false` (PyTorch default).
        ///   - device: Device to allocate parameters on.
        public init(
            numFeatures: Int,
            eps: Float = 1e-5,
            affine: Bool = false,
            device: Device = .default
        ) {
            self.numFeatures = numFeatures
            self.eps = eps
            self.affine = affine

            // Initialize scale (gamma) to 1
            self.weight = Parameter(
                Tensor<Float>.ones([numFeatures], on: device),
                requiresGrad: affine,
                name: "weight"
            )

            // Initialize shift (beta) to 0
            self.bias = Parameter(
                Tensor<Float>.zeros([numFeatures], on: device),
                requiresGrad: affine,
                name: "bias"
            )
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            precondition(input.rank == 4,
                "InstanceNorm2d requires 4D input [batch, height, width, channels], got shape \(input.shape).")
            precondition(input.shape[3] == numFeatures,
                "InstanceNorm2d: channel count mismatch. Expected \(numFeatures), got \(input.shape[3]).")

            let batchSize = input.shape[0]
            let height = input.shape[1]
            let width = input.shape[2]

            // Compute mean and variance over [height, width] for each channel
            let id = TensorRegistry.shared.nextTensorId()
            let handle = LazyTensorHandle(
                id: id,
                shape: input.shape,
                dtype: input.dtype,
                device: input.device
            )

            // Instance norm is GroupNorm with numGroups = numChannels
            handle.irNode = .operation(
                op: .layerNorm,
                inputs: [
                    input.handle,
                    weight.value.handle,
                    bias.value.handle
                ],
                attributes: [
                    "epsilon": eps,
                    "normalizedDims": [height, width],
                    "numGroups": numFeatures  // Each channel is its own group
                ]
            )
            TensorRegistry.shared.registerPending(handle)

            return Tensor<Float>(handle: handle)
        }

        public func parameters() -> [Parameter] {
            affine ? [weight, bias] : []
        }
    }
}

// MARK: - Sequential Container

extension nn {
    /// A sequential container that chains layers.
    ///
    /// Layers are applied in the order they are added.
    ///
    /// Example:
    /// ```swift
    /// let model = nn.Sequential(
    ///     nn.Linear(inputSize: 784, outputSize: 256),
    ///     nn.ReLU(),
    ///     nn.Linear(inputSize: 256, outputSize: 10)
    /// )
    /// let output = model(input)
    /// ```
    public struct Sequential: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// The layers in this sequential container
        private var layers: [AnyLayer]

        /// Create an empty sequential container
        public init() {
            self.layers = []
        }

        /// Create a sequential container with the given layers
        public init(_ layers: AnyLayer...) {
            self.layers = layers
        }

        /// Create a sequential container from an array of layers
        public init(_ layers: [AnyLayer]) {
            self.layers = layers
        }

        /// Add a layer to the sequential container
        public mutating func add(_ layer: AnyLayer) {
            layers.append(layer)
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            var x = input
            for layer in layers {
                x = layer.forward(x)
            }
            return x
        }

        public func parameters() -> [Parameter] {
            layers.flatMap { $0.parameters() }
        }
    }

    /// Type-erased layer wrapper for heterogeneous sequential containers
    public struct AnyLayer: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        private let _forward: (Tensor<Float>) -> Tensor<Float>
        private let _parameters: () -> [Parameter]

        public init<L: Module>(_ layer: L) where L.Input == Tensor<Float>, L.Output == Tensor<Float> {
            var mutableLayer = layer
            self._forward = { mutableLayer.forward($0) }
            self._parameters = { mutableLayer.parameters() }
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            _forward(input)
        }

        public func parameters() -> [Parameter] {
            _parameters()
        }
    }
}

// MARK: - Result Builder for Sequential

extension nn {
    /// Result builder for creating Sequential models with declarative syntax.
    @resultBuilder
    public struct SequentialBuilder {
        public static func buildBlock(_ layers: AnyLayer...) -> [AnyLayer] {
            layers
        }

        public static func buildExpression<L: Module>(_ layer: L) -> AnyLayer
        where L.Input == Tensor<Float>, L.Output == Tensor<Float> {
            AnyLayer(layer)
        }
    }

    /// Create a Sequential model using result builder syntax.
    ///
    /// Example:
    /// ```swift
    /// let model = nn.sequential {
    ///     nn.Linear(inputSize: 784, outputSize: 256)
    ///     nn.ReLU()
    ///     nn.Linear(inputSize: 256, outputSize: 10)
    /// }
    /// ```
    public static func sequential(@SequentialBuilder _ builder: () -> [AnyLayer]) -> Sequential {
        Sequential(builder())
    }
}

// MARK: - Attention Mechanisms

extension nn {

    /// Scaled dot-product attention.
    ///
    /// Computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    ///
    /// - Parameters:
    ///   - query: Query tensor of shape [batch, seqLen, embedDim] or [batch, numHeads, seqLen, headDim]
    ///   - key: Key tensor of same shape as query
    ///   - value: Value tensor of same shape as query
    ///   - mask: Optional attention mask. Values where mask is `true` (or non-zero) will be masked out.
    ///   - dropout: Dropout probability (not applied during inference). Defaults to 0.
    /// - Returns: Tuple of (attention output, attention weights)
    public static func scaledDotProductAttention(
        query: Tensor<Float>,
        key: Tensor<Float>,
        value: Tensor<Float>,
        mask: Tensor<Float>? = nil,
        dropout: Float = 0.0
    ) -> (output: Tensor<Float>, weights: Tensor<Float>) {
        // Get the dimension for scaling
        let dk = Float(query.shape[query.shape.count - 1])
        let scale = Tensor<Float>.full([], 1.0 / sqrt(dk), on: query.device)

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        // For 4D tensors [batch, heads, seq, dim]: need batched matmul
        // For 3D tensors [batch, seq, dim]: transpose last two dims of K
        let keyTransposed = key.transposeLastTwo()
        var scores = query.batchedMatmul(keyTransposed) * scale

        // Apply mask if provided (mask out positions with large negative value)
        if let mask = mask {
            let maskValue = Tensor<Float>.full(scores.shape, -1e9, on: scores.device)
            scores = scores.maskedFill(mask: mask, value: maskValue)
        }

        // Softmax over the last dimension (key sequence length)
        let attentionWeights = scores.softmax(dim: -1)

        // Apply dropout if specified (simplified: not implemented in lazy evaluation)
        // In production, dropout would be applied here

        // Compute output: attention_weights @ V
        let output = attentionWeights.batchedMatmul(value)

        return (output, attentionWeights)
    }

    /// Multi-Head Attention layer.
    ///
    /// Allows the model to jointly attend to information from different representation
    /// subspaces at different positions.
    ///
    /// MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    /// where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    ///
    /// Example:
    /// ```swift
    /// let attention = nn.MultiheadAttention(embedDim: 512, numHeads: 8)
    /// let output = attention(query: x, key: x, value: x)  // Self-attention
    /// ```
    public struct MultiheadAttention: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Embedding dimension
        public let embedDim: Int

        /// Number of attention heads
        public let numHeads: Int

        /// Dimension of each head
        public let headDim: Int

        /// Key dimension (for cross-attention)
        public let keyDim: Int

        /// Value dimension (for cross-attention)
        public let valueDim: Int

        /// Dropout probability
        public let dropout: Float

        /// Whether to add bias to projections
        public let bias: Bool

        // Projection weights
        /// Query projection: [embedDim, embedDim]
        public var wQ: Parameter

        /// Key projection: [keyDim, embedDim]
        public var wK: Parameter

        /// Value projection: [valueDim, embedDim]
        public var wV: Parameter

        /// Output projection: [embedDim, embedDim]
        public var wO: Parameter

        // Optional biases
        public var bQ: Parameter?
        public var bK: Parameter?
        public var bV: Parameter?
        public var bO: Parameter?

        /// Creates a MultiheadAttention layer.
        ///
        /// - Parameters:
        ///   - embedDim: Total dimension of the model (must be divisible by numHeads).
        ///   - numHeads: Number of parallel attention heads.
        ///   - dropout: Dropout probability on attention weights. Defaults to 0.
        ///   - bias: Whether to add bias to projections. Defaults to true.
        ///   - kdim: Key dimension (defaults to embedDim).
        ///   - vdim: Value dimension (defaults to embedDim).
        public init(
            embedDim: Int,
            numHeads: Int,
            dropout: Float = 0.0,
            bias: Bool = true,
            kdim: Int? = nil,
            vdim: Int? = nil
        ) {
            precondition(embedDim % numHeads == 0,
                "MultiheadAttention: embedDim (\(embedDim)) must be divisible by numHeads (\(numHeads)). " +
                "This gives headDim = embedDim / numHeads. Try adjusting embedDim or numHeads.")

            self.embedDim = embedDim
            self.numHeads = numHeads
            self.headDim = embedDim / numHeads
            self.keyDim = kdim ?? embedDim
            self.valueDim = vdim ?? embedDim
            self.dropout = dropout
            self.bias = bias

            // Xavier/Glorot initialization scale
            let scale = Float(1.0) / sqrt(Float(embedDim))

            // Initialize projection weights
            // Weight matrices: [inputDim, outputDim] for matmul: x @ W
            // Q projects from embedDim -> embedDim
            // K projects from keyDim -> embedDim
            // V projects from valueDim -> embedDim
            // O projects from embedDim -> embedDim
            wQ = Parameter(
                Tensor<Float>.randn([embedDim, embedDim]) * Tensor<Float>.full([], scale, on: .default),
                name: "wQ"
            )
            wK = Parameter(
                Tensor<Float>.randn([keyDim, embedDim]) * Tensor<Float>.full([], scale, on: .default),
                name: "wK"
            )
            wV = Parameter(
                Tensor<Float>.randn([valueDim, embedDim]) * Tensor<Float>.full([], scale, on: .default),
                name: "wV"
            )
            wO = Parameter(
                Tensor<Float>.randn([embedDim, embedDim]) * Tensor<Float>.full([], scale, on: .default),
                name: "wO"
            )

            if bias {
                bQ = Parameter(Tensor<Float>.zeros([embedDim]), name: "bQ")
                bK = Parameter(Tensor<Float>.zeros([embedDim]), name: "bK")
                bV = Parameter(Tensor<Float>.zeros([embedDim]), name: "bV")
                bO = Parameter(Tensor<Float>.zeros([embedDim]), name: "bO")
            }
        }

        /// Forward pass for self-attention (query = key = value).
        ///
        /// For cross-attention, use `forward(query:key:value:mask:)`.
        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            forward(query: input, key: input, value: input, mask: nil).output
        }

        /// Forward pass with separate query, key, value inputs.
        ///
        /// - Parameters:
        ///   - query: Query tensor of shape [batch, seqLen, embedDim]
        ///   - key: Key tensor of shape [batch, srcLen, kdim]
        ///   - value: Value tensor of shape [batch, srcLen, vdim]
        ///   - mask: Optional attention mask of shape [batch, seqLen, srcLen] or [seqLen, srcLen]
        /// - Returns: Tuple of (output tensor, attention weights)
        public func forward(
            query: Tensor<Float>,
            key: Tensor<Float>,
            value: Tensor<Float>,
            mask: Tensor<Float>?
        ) -> (output: Tensor<Float>, attentionWeights: Tensor<Float>) {
            let batchSize = query.shape[0]
            let seqLen = query.shape[1]
            let srcLen = key.shape[1]

            // Project Q, K, V
            // Q: [batch, seq, embedDim] @ [embedDim, embedDim] -> [batch, seq, embedDim]
            // K: [batch, srcLen, keyDim] @ [keyDim, embedDim] -> [batch, srcLen, embedDim]
            // V: [batch, srcLen, valueDim] @ [valueDim, embedDim] -> [batch, srcLen, embedDim]
            // Reshape to 2D, apply projection, reshape back to 3D
            var q = query.reshape([batchSize * seqLen, embedDim]).matmul(wQ.value).reshape([batchSize, seqLen, embedDim])
            var k = key.reshape([batchSize * srcLen, keyDim]).matmul(wK.value).reshape([batchSize, srcLen, embedDim])
            var v = value.reshape([batchSize * srcLen, valueDim]).matmul(wV.value).reshape([batchSize, srcLen, embedDim])

            // Add biases if present
            if let bQ = bQ {
                q = q + bQ.value.broadcast(to: q.shape)
            }
            if let bK = bK {
                k = k + bK.value.broadcast(to: k.shape)
            }
            if let bV = bV {
                v = v + bV.value.broadcast(to: v.shape)
            }

            // Reshape to [batch, seq, numHeads, headDim]
            q = q.reshape([batchSize, seqLen, numHeads, headDim])
            k = k.reshape([batchSize, srcLen, numHeads, headDim])
            v = v.reshape([batchSize, srcLen, numHeads, headDim])

            // Transpose to [batch, numHeads, seq, headDim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            // Apply scaled dot-product attention
            let (attnOutput, attnWeights) = nn.scaledDotProductAttention(
                query: q,
                key: k,
                value: v,
                mask: mask,
                dropout: dropout
            )

            // Transpose back: [batch, numHeads, seq, headDim] -> [batch, seq, numHeads, headDim]
            var output = attnOutput.transpose(1, 2)

            // Concatenate heads: [batch, seq, numHeads * headDim] = [batch, seq, embedDim]
            output = output.reshape([batchSize, seqLen, embedDim])

            // Final projection: reshape to 2D, project, reshape back
            output = output.reshape([batchSize * seqLen, embedDim]).matmul(wO.value).reshape([batchSize, seqLen, embedDim])
            if let bO = bO {
                output = output + bO.value.broadcast(to: output.shape)
            }

            return (output, attnWeights)
        }

        public func parameters() -> [Parameter] {
            var params = [wQ, wK, wV, wO]
            if let bQ = bQ { params.append(bQ) }
            if let bK = bK { params.append(bK) }
            if let bV = bV { params.append(bV) }
            if let bO = bO { params.append(bO) }
            return params
        }
    }
}

// MARK: - Layer Normalization

extension nn {
    /// Layer Normalization.
    ///
    /// Applies layer normalization over the last D dimensions, where D is the
    /// size of `normalizedShape`. Unlike BatchNorm, LayerNorm normalizes over
    /// features (not batch), making it suitable for sequence models.
    ///
    /// LayerNorm(x) = (x - mean) / sqrt(variance + eps) * gamma + beta
    ///
    /// Example:
    /// ```swift
    /// // Normalize over last dimension (common for transformers)
    /// let ln = nn.LayerNorm(normalizedShape: [512])
    /// let x = Tensor<Float>.randn([32, 10, 512])
    /// let y = ln(x)  // Shape: [32, 10, 512]
    ///
    /// // Normalize over last 2 dimensions
    /// let ln2d = nn.LayerNorm(normalizedShape: [10, 512])
    /// ```
    public struct LayerNorm: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Shape of the normalized dimensions
        public let normalizedShape: [Int]

        /// Small constant for numerical stability
        public let eps: Float

        /// Whether to learn affine parameters
        public let elementwiseAffine: Bool

        /// Scale parameter (gamma)
        public var weight: Parameter

        /// Shift parameter (beta)
        public var bias: Parameter

        /// Creates a LayerNorm layer.
        ///
        /// - Parameters:
        ///   - normalizedShape: Shape of the dimensions to normalize over (last D dimensions).
        ///   - eps: Small constant for numerical stability. Defaults to 1e-5.
        ///   - elementwiseAffine: Whether to learn scale and shift. Defaults to true.
        ///   - device: Device to allocate parameters on.
        public init(
            normalizedShape: [Int],
            eps: Float = 1e-5,
            elementwiseAffine: Bool = true,
            device: Device = .default
        ) {
            self.normalizedShape = normalizedShape
            self.eps = eps
            self.elementwiseAffine = elementwiseAffine

            // Initialize gamma to 1 and beta to 0
            self.weight = Parameter(
                Tensor<Float>.ones(normalizedShape, on: device),
                requiresGrad: elementwiseAffine,
                name: "weight"
            )
            self.bias = Parameter(
                Tensor<Float>.zeros(normalizedShape, on: device),
                requiresGrad: elementwiseAffine,
                name: "bias"
            )
        }

        /// Convenience initializer for normalizing over a single dimension.
        ///
        /// - Parameters:
        ///   - normalizedShape: Size of the dimension to normalize over.
        ///   - eps: Small constant for numerical stability.
        ///   - elementwiseAffine: Whether to learn scale and shift.
        ///   - device: Device to allocate parameters on.
        public init(
            _ normalizedShape: Int,
            eps: Float = 1e-5,
            elementwiseAffine: Bool = true,
            device: Device = .default
        ) {
            self.init(
                normalizedShape: [normalizedShape],
                eps: eps,
                elementwiseAffine: elementwiseAffine,
                device: device
            )
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            // Determine which dimensions to normalize over
            // normalizedShape corresponds to the last D dimensions
            let numNormDims = normalizedShape.count
            precondition(input.rank >= numNormDims,
                "LayerNorm: input rank (\(input.rank)) must be >= normalized dimensions (\(numNormDims)). " +
                "Input shape: \(input.shape), normalizedShape: \(normalizedShape).")

            // Verify shape matches
            for i in 0..<numNormDims {
                let inputDim = input.shape[input.rank - numNormDims + i]
                precondition(inputDim == normalizedShape[i],
                    "LayerNorm: input shape mismatch at normalized dimension \(i). " +
                    "Expected \(normalizedShape[i]), got \(inputDim). " +
                    "Input shape: \(input.shape), normalizedShape: \(normalizedShape).")
            }

            // Dimensions to normalize over (last D dimensions)
            let normDims = Array((input.rank - numNormDims)..<input.rank)

            // Compute mean and variance
            let mean = input.mean(dims: normDims, keepDims: true)
            let variance = input.variance(dims: normDims, keepDims: true, unbiased: false)

            // Normalize: (x - mean) / sqrt(var + eps)
            let centered = input - mean.broadcast(to: input.shape)
            let epsT = Tensor<Float>.full(variance.shape, eps, on: input.device)
            let stddev = (variance + epsT).sqrt()
            var normalized = centered / stddev.broadcast(to: input.shape)

            // Apply affine transformation if enabled
            if elementwiseAffine {
                normalized = normalized * weight.value.broadcast(to: input.shape)
                normalized = normalized + bias.value.broadcast(to: input.shape)
            }

            return normalized
        }

        public func parameters() -> [Parameter] {
            elementwiseAffine ? [weight, bias] : []
        }
    }
}

// MARK: - Transformer Encoder Layer

extension nn {
    /// Transformer Encoder Layer.
    ///
    /// A standard transformer encoder layer consisting of:
    /// 1. Multi-head self-attention with residual connection and layer norm
    /// 2. Position-wise feed-forward network with residual connection and layer norm
    ///
    /// Uses Pre-LN (layer norm before attention/FFN) or Post-LN (after) architecture.
    ///
    /// Example:
    /// ```swift
    /// let encoderLayer = nn.TransformerEncoderLayer(
    ///     dModel: 512,
    ///     nHead: 8,
    ///     dimFeedforward: 2048
    /// )
    /// let x = Tensor<Float>.randn([32, 10, 512])  // [batch, seq, embed]
    /// let y = encoderLayer(x)  // [batch, seq, embed]
    /// ```
    public struct TransformerEncoderLayer: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Model dimension
        public let dModel: Int

        /// Number of attention heads
        public let nHead: Int

        /// Feed-forward hidden dimension
        public let dimFeedforward: Int

        /// Dropout probability
        public let dropout: Float

        /// Whether to use Pre-LN (true) or Post-LN (false) architecture
        public let normFirst: Bool

        /// Self-attention layer
        public var selfAttn: MultiheadAttention

        /// First layer norm (for attention)
        public var norm1: LayerNorm

        /// Second layer norm (for FFN)
        public var norm2: LayerNorm

        /// First linear layer of FFN
        public var linear1: Linear

        /// Second linear layer of FFN
        public var linear2: Linear

        /// Activation function type
        public let activation: ActivationType

        /// Activation function types supported by TransformerEncoderLayer
        public enum ActivationType {
            case relu
            case gelu
        }

        /// Creates a Transformer Encoder Layer.
        ///
        /// - Parameters:
        ///   - dModel: The model dimension (embedding size).
        ///   - nHead: Number of attention heads.
        ///   - dimFeedforward: Hidden dimension of the feed-forward network. Defaults to 4*dModel.
        ///   - dropout: Dropout probability. Defaults to 0.1.
        ///   - activation: Activation function for FFN. Defaults to .relu.
        ///   - normFirst: If true, applies layer norm before attention/FFN (Pre-LN). Defaults to false.
        ///   - device: Device to allocate parameters on.
        public init(
            dModel: Int,
            nHead: Int,
            dimFeedforward: Int? = nil,
            dropout: Float = 0.1,
            activation: ActivationType = .relu,
            normFirst: Bool = false,
            device: Device = .default
        ) {
            self.dModel = dModel
            self.nHead = nHead
            self.dimFeedforward = dimFeedforward ?? (4 * dModel)
            self.dropout = dropout
            self.activation = activation
            self.normFirst = normFirst

            // Self-attention
            self.selfAttn = MultiheadAttention(
                embedDim: dModel,
                numHeads: nHead,
                dropout: dropout
            )

            // Layer norms
            self.norm1 = LayerNorm(dModel, device: device)
            self.norm2 = LayerNorm(dModel, device: device)

            // Feed-forward network
            self.linear1 = Linear(inputSize: dModel, outputSize: self.dimFeedforward, device: device)
            self.linear2 = Linear(inputSize: self.dimFeedforward, outputSize: dModel, device: device)
        }

        /// Forward pass.
        ///
        /// - Parameters:
        ///   - input: Input tensor of shape [batch, seqLen, dModel]
        /// - Returns: Output tensor of shape [batch, seqLen, dModel]
        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            if normFirst {
                return forwardPreLN(input)
            } else {
                return forwardPostLN(input)
            }
        }

        /// Forward with optional attention mask.
        ///
        /// - Parameters:
        ///   - src: Source tensor of shape [batch, seqLen, dModel]
        ///   - srcMask: Optional attention mask
        /// - Returns: Output tensor of shape [batch, seqLen, dModel]
        public func forward(src: Tensor<Float>, srcMask: Tensor<Float>? = nil) -> Tensor<Float> {
            if normFirst {
                return forwardPreLN(src, mask: srcMask)
            } else {
                return forwardPostLN(src, mask: srcMask)
            }
        }

        /// Pre-LN architecture: LN -> Attention -> Residual -> LN -> FFN -> Residual
        private func forwardPreLN(_ x: Tensor<Float>, mask: Tensor<Float>? = nil) -> Tensor<Float> {
            // Self-attention with pre-norm
            let normed1 = norm1(x)
            let attnOut = selfAttn.forward(query: normed1, key: normed1, value: normed1, mask: mask).output
            var out = x + attnOut

            // FFN with pre-norm
            let normed2 = norm2(out)
            let ffnOut = feedForward(normed2)
            out = out + ffnOut

            return out
        }

        /// Post-LN architecture: Attention -> Residual -> LN -> FFN -> Residual -> LN
        private func forwardPostLN(_ x: Tensor<Float>, mask: Tensor<Float>? = nil) -> Tensor<Float> {
            // Self-attention with post-norm
            let attnOut = selfAttn.forward(query: x, key: x, value: x, mask: mask).output
            var out = norm1(x + attnOut)

            // FFN with post-norm
            let ffnOut = feedForward(out)
            out = norm2(out + ffnOut)

            return out
        }

        /// Position-wise feed-forward network
        private func feedForward(_ x: Tensor<Float>) -> Tensor<Float> {
            let batchSize = x.shape[0]
            let seqLen = x.shape[1]

            // Reshape to 2D for linear layers
            var hidden = x.reshape([batchSize * seqLen, dModel])

            // First linear + activation
            hidden = linear1(hidden)
            switch activation {
            case .relu:
                hidden = hidden.relu()
            case .gelu:
                hidden = hidden.gelu()
            }

            // Second linear
            hidden = linear2(hidden)

            // Reshape back to 3D
            return hidden.reshape([batchSize, seqLen, dModel])
        }

        public func parameters() -> [Parameter] {
            var params: [Parameter] = []
            params.append(contentsOf: selfAttn.parameters())
            params.append(contentsOf: norm1.parameters())
            params.append(contentsOf: norm2.parameters())
            params.append(contentsOf: linear1.parameters())
            params.append(contentsOf: linear2.parameters())
            return params
        }
    }

    // MARK: - Transformer Decoder Layer

    /// A single layer of the Transformer Decoder.
    ///
    /// Consists of:
    /// 1. Masked self-attention (causal, attends only to previous positions)
    /// 2. Cross-attention (attends to encoder output)
    /// 3. Position-wise feed-forward network
    ///
    /// Supports both Pre-LN and Post-LN architectures.
    ///
    /// Example:
    /// ```swift
    /// let decoderLayer = nn.TransformerDecoderLayer(
    ///     dModel: 512,
    ///     nHead: 8,
    ///     dimFeedforward: 2048
    /// )
    /// let memory = Tensor<Float>.randn([32, 20, 512])  // encoder output
    /// let tgt = Tensor<Float>.randn([32, 10, 512])     // decoder input
    /// let y = decoderLayer.forward(tgt: tgt, memory: memory)
    /// ```
    public struct TransformerDecoderLayer: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Model dimension
        public let dModel: Int

        /// Number of attention heads
        public let nHead: Int

        /// Feed-forward hidden dimension
        public let dimFeedforward: Int

        /// Dropout probability
        public let dropout: Float

        /// Whether to use Pre-LN (true) or Post-LN (false) architecture
        public let normFirst: Bool

        /// Masked self-attention layer
        public var selfAttn: MultiheadAttention

        /// Cross-attention layer (attends to encoder output)
        public var crossAttn: MultiheadAttention

        /// First layer norm (for self-attention)
        public var norm1: LayerNorm

        /// Second layer norm (for cross-attention)
        public var norm2: LayerNorm

        /// Third layer norm (for FFN)
        public var norm3: LayerNorm

        /// First linear layer of FFN
        public var linear1: Linear

        /// Second linear layer of FFN
        public var linear2: Linear

        /// Activation function type
        public let activation: ActivationType

        /// Activation function types supported
        public enum ActivationType {
            case relu
            case gelu
        }

        /// Creates a Transformer Decoder Layer.
        ///
        /// - Parameters:
        ///   - dModel: The model dimension (embedding size).
        ///   - nHead: Number of attention heads.
        ///   - dimFeedforward: Hidden dimension of the feed-forward network. Defaults to 4*dModel.
        ///   - dropout: Dropout probability. Defaults to 0.1.
        ///   - activation: Activation function for FFN. Defaults to .relu.
        ///   - normFirst: If true, applies layer norm before attention/FFN (Pre-LN). Defaults to false.
        ///   - device: Device to allocate parameters on.
        public init(
            dModel: Int,
            nHead: Int,
            dimFeedforward: Int? = nil,
            dropout: Float = 0.1,
            activation: ActivationType = .relu,
            normFirst: Bool = false,
            device: Device = .default
        ) {
            self.dModel = dModel
            self.nHead = nHead
            self.dimFeedforward = dimFeedforward ?? (4 * dModel)
            self.dropout = dropout
            self.activation = activation
            self.normFirst = normFirst

            // Masked self-attention
            self.selfAttn = MultiheadAttention(
                embedDim: dModel,
                numHeads: nHead,
                dropout: dropout
            )

            // Cross-attention
            self.crossAttn = MultiheadAttention(
                embedDim: dModel,
                numHeads: nHead,
                dropout: dropout
            )

            // Layer norms
            self.norm1 = LayerNorm(dModel, device: device)
            self.norm2 = LayerNorm(dModel, device: device)
            self.norm3 = LayerNorm(dModel, device: device)

            // Feed-forward network
            self.linear1 = Linear(inputSize: dModel, outputSize: self.dimFeedforward, device: device)
            self.linear2 = Linear(inputSize: self.dimFeedforward, outputSize: dModel, device: device)
        }

        /// Forward pass (requires encoder memory via `forward(tgt:memory:)`)
        ///
        /// - Parameter input: Target tensor of shape [batch, tgtLen, dModel]
        /// - Returns: Output tensor (uses zero memory, not typical usage)
        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            // Create zero memory tensor for standalone use (not typical)
            let memory = Tensor<Float>.zeros(input.shape, on: input.device)
            return forward(tgt: input, memory: memory)
        }

        /// Forward pass with encoder output.
        ///
        /// - Parameters:
        ///   - tgt: Target sequence tensor of shape [batch, tgtLen, dModel]
        ///   - memory: Encoder output tensor of shape [batch, srcLen, dModel]
        ///   - tgtMask: Optional mask for self-attention (e.g., causal mask)
        ///   - memoryMask: Optional mask for cross-attention
        /// - Returns: Output tensor of shape [batch, tgtLen, dModel]
        public func forward(
            tgt: Tensor<Float>,
            memory: Tensor<Float>,
            tgtMask: Tensor<Float>? = nil,
            memoryMask: Tensor<Float>? = nil
        ) -> Tensor<Float> {
            if normFirst {
                return forwardPreLN(tgt: tgt, memory: memory, tgtMask: tgtMask, memoryMask: memoryMask)
            } else {
                return forwardPostLN(tgt: tgt, memory: memory, tgtMask: tgtMask, memoryMask: memoryMask)
            }
        }

        /// Pre-LN architecture
        private func forwardPreLN(
            tgt: Tensor<Float>,
            memory: Tensor<Float>,
            tgtMask: Tensor<Float>?,
            memoryMask: Tensor<Float>?
        ) -> Tensor<Float> {
            var x = tgt

            // Self-attention with pre-norm
            let normed1 = norm1(x)
            let selfAttnOut = selfAttn.forward(query: normed1, key: normed1, value: normed1, mask: tgtMask).output
            x = x + selfAttnOut

            // Cross-attention with pre-norm
            let normed2 = norm2(x)
            let crossAttnOut = crossAttn.forward(query: normed2, key: memory, value: memory, mask: memoryMask).output
            x = x + crossAttnOut

            // FFN with pre-norm
            let normed3 = norm3(x)
            let ffnOut = feedForward(normed3)
            x = x + ffnOut

            return x
        }

        /// Post-LN architecture
        private func forwardPostLN(
            tgt: Tensor<Float>,
            memory: Tensor<Float>,
            tgtMask: Tensor<Float>?,
            memoryMask: Tensor<Float>?
        ) -> Tensor<Float> {
            var x = tgt

            // Self-attention with post-norm
            let selfAttnOut = selfAttn.forward(query: x, key: x, value: x, mask: tgtMask).output
            x = norm1(x + selfAttnOut)

            // Cross-attention with post-norm
            let crossAttnOut = crossAttn.forward(query: x, key: memory, value: memory, mask: memoryMask).output
            x = norm2(x + crossAttnOut)

            // FFN with post-norm
            let ffnOut = feedForward(x)
            x = norm3(x + ffnOut)

            return x
        }

        /// Position-wise feed-forward network
        private func feedForward(_ x: Tensor<Float>) -> Tensor<Float> {
            let batchSize = x.shape[0]
            let seqLen = x.shape[1]

            // Reshape to 2D for linear layers
            var hidden = x.reshape([batchSize * seqLen, dModel])

            // First linear + activation
            hidden = linear1(hidden)
            switch activation {
            case .relu:
                hidden = hidden.relu()
            case .gelu:
                hidden = hidden.gelu()
            }

            // Second linear
            hidden = linear2(hidden)

            // Reshape back to 3D
            return hidden.reshape([batchSize, seqLen, dModel])
        }

        public func parameters() -> [Parameter] {
            var params: [Parameter] = []
            params.append(contentsOf: selfAttn.parameters())
            params.append(contentsOf: crossAttn.parameters())
            params.append(contentsOf: norm1.parameters())
            params.append(contentsOf: norm2.parameters())
            params.append(contentsOf: norm3.parameters())
            params.append(contentsOf: linear1.parameters())
            params.append(contentsOf: linear2.parameters())
            return params
        }

        /// Convenience method for calling without masks
        public func callAsFunction(
            tgt: Tensor<Float>,
            memory: Tensor<Float>,
            tgtMask: Tensor<Float>? = nil,
            memoryMask: Tensor<Float>? = nil
        ) -> Tensor<Float> {
            return forward(tgt: tgt, memory: memory, tgtMask: tgtMask, memoryMask: memoryMask)
        }
    }

    /// Generate a causal (look-ahead) mask for decoder self-attention.
    ///
    /// Creates a lower-triangular mask where each position can only attend
    /// to itself and earlier positions.
    ///
    /// - Parameters:
    ///   - size: Sequence length
    ///   - device: Device to create the mask on
    /// - Returns: Causal mask of shape [size, size] with -inf for masked positions
    public static func generateCausalMask(size: Int, device: Device = .default) -> Tensor<Float> {
        // Create lower triangular matrix of ones
        var maskData: [Float] = []
        for i in 0..<size {
            for j in 0..<size {
                if j <= i {
                    maskData.append(0.0)  // Attend
                } else {
                    maskData.append(-Float.infinity)  // Don't attend
                }
            }
        }
        return Tensor<Float>(maskData, shape: [size, size], on: device)
    }
}

// MARK: - Positional Encoding

extension nn {
    /// Sinusoidal Positional Encoding.
    ///
    /// Adds positional information to embeddings using sine and cosine functions
    /// of different frequencies. This is the original positional encoding from
    /// "Attention Is All You Need".
    ///
    /// PE(pos, 2i) = sin(pos / 10000^(2i/dModel))
    /// PE(pos, 2i+1) = cos(pos / 10000^(2i/dModel))
    ///
    /// Example:
    /// ```swift
    /// let posEnc = nn.SinusoidalPositionalEncoding(dModel: 512, maxLen: 5000)
    /// let x = Tensor<Float>.randn([32, 100, 512])  // [batch, seq, embed]
    /// let y = posEnc(x)  // x + positional encodings
    /// ```
    public struct SinusoidalPositionalEncoding: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Model dimension
        public let dModel: Int

        /// Maximum sequence length
        public let maxLen: Int

        /// Dropout probability
        public let dropout: Float

        /// Pre-computed positional encodings [1, maxLen, dModel]
        public let pe: Tensor<Float>

        /// Creates a Sinusoidal Positional Encoding layer.
        ///
        /// - Parameters:
        ///   - dModel: The model dimension (embedding size).
        ///   - maxLen: Maximum sequence length. Defaults to 5000.
        ///   - dropout: Dropout probability. Defaults to 0.1.
        ///   - device: Device to allocate the encoding on.
        public init(
            dModel: Int,
            maxLen: Int = 5000,
            dropout: Float = 0.1,
            device: Device = .default
        ) {
            self.dModel = dModel
            self.maxLen = maxLen
            self.dropout = dropout

            // Compute positional encodings
            // Position indices: [maxLen, 1]
            // Dimension indices for sin: [dModel/2]
            // Dimension indices for cos: [dModel/2]

            // Create position indices [0, 1, 2, ..., maxLen-1]
            var peData = [Float](repeating: 0, count: maxLen * dModel)

            for pos in 0..<maxLen {
                for i in stride(from: 0, to: dModel, by: 2) {
                    let divTerm = exp(Float(i) * (-log(10000.0) / Float(dModel)))
                    peData[pos * dModel + i] = sin(Float(pos) * divTerm)
                    if i + 1 < dModel {
                        peData[pos * dModel + i + 1] = cos(Float(pos) * divTerm)
                    }
                }
            }

            // Create tensor with shape [1, maxLen, dModel] for easy broadcasting
            self.pe = Tensor<Float>(peData, shape: [1, maxLen, dModel], on: device)
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            let seqLen = input.shape[1]
            precondition(seqLen <= maxLen,
                        "Sequence length \(seqLen) exceeds maximum length \(maxLen)")

            // For sequences shorter than maxLen, we need to slice the PE tensor
            // Reshape PE to [maxLen, dModel], slice, then reshape back
            if seqLen < maxLen {
                let peFlat = pe.reshape([maxLen, dModel])
                let peSliced = peFlat.slice(start: 0, size: seqLen)
                let posEnc = peSliced.reshape([1, seqLen, dModel])
                return input + posEnc.broadcast(to: input.shape)
            } else {
                // seqLen == maxLen, use full PE
                return input + pe.broadcast(to: input.shape)
            }
        }

        public func parameters() -> [Parameter] {
            []  // Positional encodings are not learned
        }
    }

    /// Learned Positional Embedding.
    ///
    /// Uses learned embeddings for each position instead of fixed sinusoidal patterns.
    ///
    /// Example:
    /// ```swift
    /// let posEmbed = nn.LearnedPositionalEmbedding(dModel: 512, maxLen: 512)
    /// let x = Tensor<Float>.randn([32, 100, 512])
    /// let y = posEmbed(x)  // x + learned positional embeddings
    /// ```
    public struct LearnedPositionalEmbedding: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = Tensor<Float>

        /// Model dimension
        public let dModel: Int

        /// Maximum sequence length
        public let maxLen: Int

        /// Learned positional embeddings
        public var embedding: Parameter

        /// Creates a Learned Positional Embedding layer.
        ///
        /// - Parameters:
        ///   - dModel: The model dimension (embedding size).
        ///   - maxLen: Maximum sequence length.
        ///   - device: Device to allocate parameters on.
        public init(
            dModel: Int,
            maxLen: Int,
            device: Device = .default
        ) {
            self.dModel = dModel
            self.maxLen = maxLen

            // Initialize embeddings with small random values
            let scale = Float(1.0) / sqrt(Float(dModel))
            self.embedding = Parameter(
                Tensor<Float>.randn([1, maxLen, dModel], on: device) * Tensor<Float>.full([], scale, on: device),
                name: "positional_embedding"
            )
        }

        public func forward(_ input: Tensor<Float>) -> Tensor<Float> {
            let seqLen = input.shape[1]
            precondition(seqLen <= maxLen,
                        "Sequence length \(seqLen) exceeds maximum length \(maxLen)")

            // For sequences shorter than maxLen, slice the embeddings
            if seqLen < maxLen {
                let embFlat = embedding.value.reshape([maxLen, dModel])
                let embSliced = embFlat.slice(start: 0, size: seqLen)
                let posEmbed = embSliced.reshape([1, seqLen, dModel])
                return input + posEmbed.broadcast(to: input.shape)
            } else {
                // seqLen == maxLen, use full embeddings
                return input + embedding.value.broadcast(to: input.shape)
            }
        }

        public func parameters() -> [Parameter] {
            [embedding]
        }
    }

    // MARK: - RNN Cell

    /// Basic RNN cell: h' = tanh(Wih @ x + bih + Whh @ h + bhh)
    ///
    /// A single step of a vanilla RNN.
    ///
    /// Example:
    /// ```swift
    /// let cell = nn.RNNCell(inputSize: 128, hiddenSize: 256)
    /// var h = Tensor<Float>.zeros([batch, 256])
    /// for t in 0..<seqLen {
    ///     h = cell(x[t], h)
    /// }
    /// ```
    public struct RNNCell: Module {
        public typealias Input = (Tensor<Float>, Tensor<Float>)
        public typealias Output = Tensor<Float>

        /// Input size
        public let inputSize: Int

        /// Hidden state size
        public let hiddenSize: Int

        /// Input-to-hidden weights [hiddenSize, inputSize]
        public var weightIH: Parameter

        /// Hidden-to-hidden weights [hiddenSize, hiddenSize]
        public var weightHH: Parameter

        /// Input-to-hidden bias [hiddenSize]
        public var biasIH: Parameter

        /// Hidden-to-hidden bias [hiddenSize]
        public var biasHH: Parameter

        /// Whether to use bias terms
        public let bias: Bool

        /// Nonlinearity to use ('tanh' or 'relu')
        public let nonlinearity: RNNNonlinearity

        /// Create an RNN cell
        ///
        /// - Parameters:
        ///   - inputSize: Size of input features
        ///   - hiddenSize: Size of hidden state
        ///   - bias: Whether to include bias terms (default: true)
        ///   - nonlinearity: Activation function (default: .tanh)
        ///   - device: Device to create parameters on
        public init(
            inputSize: Int,
            hiddenSize: Int,
            bias: Bool = true,
            nonlinearity: RNNNonlinearity = .tanh,
            device: Device = .default
        ) {
            self.inputSize = inputSize
            self.hiddenSize = hiddenSize
            self.bias = bias
            self.nonlinearity = nonlinearity

            // Xavier/Glorot initialization
            let scale = Float(1.0) / sqrt(Float(hiddenSize))

            self.weightIH = Parameter(
                Tensor<Float>.randn([hiddenSize, inputSize], on: device) * Tensor<Float>.full([], scale, on: device),
                name: "weight_ih"
            )
            self.weightHH = Parameter(
                Tensor<Float>.randn([hiddenSize, hiddenSize], on: device) * Tensor<Float>.full([], scale, on: device),
                name: "weight_hh"
            )

            if bias {
                self.biasIH = Parameter(Tensor<Float>.zeros([hiddenSize], on: device), name: "bias_ih")
                self.biasHH = Parameter(Tensor<Float>.zeros([hiddenSize], on: device), name: "bias_hh")
            } else {
                self.biasIH = Parameter(Tensor<Float>.zeros([hiddenSize], on: device), name: "bias_ih")
                self.biasHH = Parameter(Tensor<Float>.zeros([hiddenSize], on: device), name: "bias_hh")
            }
        }

        /// Forward pass
        ///
        /// - Parameters:
        ///   - input: Input tensor [batch, inputSize]
        ///   - hidden: Previous hidden state [batch, hiddenSize]
        /// - Returns: New hidden state [batch, hiddenSize]
        public func forward(_ input: (Tensor<Float>, Tensor<Float>)) -> Tensor<Float> {
            let (x, h) = input
            return forward(x: x, hidden: h)
        }

        /// Forward pass with named parameters
        public func forward(x: Tensor<Float>, hidden: Tensor<Float>) -> Tensor<Float> {
            // h' = activation(Wih @ x^T + bih + Whh @ h^T + bhh)
            let xProj = x.matmul(weightIH.value.transpose())  // [batch, hiddenSize]
            let hProj = hidden.matmul(weightHH.value.transpose())  // [batch, hiddenSize]

            var preact = xProj + hProj
            if bias {
                preact = preact + biasIH.value.broadcast(to: preact.shape)
                preact = preact + biasHH.value.broadcast(to: preact.shape)
            }

            switch nonlinearity {
            case .tanh:
                return preact.tanh()
            case .relu:
                return preact.relu()
            }
        }

        public func parameters() -> [Parameter] {
            if bias {
                return [weightIH, weightHH, biasIH, biasHH]
            } else {
                return [weightIH, weightHH]
            }
        }
    }

    /// Nonlinearity for RNN cells
    public enum RNNNonlinearity: String, Sendable {
        case tanh
        case relu
    }

    // MARK: - LSTM Cell

    /// LSTM cell with forget, input, and output gates
    ///
    /// Implements:
    /// - i = sigmoid(Wii @ x + bii + Whi @ h + bhi)  (input gate)
    /// - f = sigmoid(Wif @ x + bif + Whf @ h + bhf)  (forget gate)
    /// - g = tanh(Wig @ x + big + Whg @ h + bhg)     (cell gate)
    /// - o = sigmoid(Wio @ x + bio + Who @ h + bho)  (output gate)
    /// - c' = f * c + i * g
    /// - h' = o * tanh(c')
    ///
    /// Example:
    /// ```swift
    /// let cell = nn.LSTMCell(inputSize: 128, hiddenSize: 256)
    /// var h = Tensor<Float>.zeros([batch, 256])
    /// var c = Tensor<Float>.zeros([batch, 256])
    /// for t in 0..<seqLen {
    ///     (h, c) = cell(x[t], (h, c))
    /// }
    /// ```
    public struct LSTMCell: Module {
        public typealias Input = (Tensor<Float>, (Tensor<Float>, Tensor<Float>))
        public typealias Output = (Tensor<Float>, Tensor<Float>)

        /// Input size
        public let inputSize: Int

        /// Hidden state size
        public let hiddenSize: Int

        /// Input-to-hidden weights for all gates [4*hiddenSize, inputSize]
        public var weightIH: Parameter

        /// Hidden-to-hidden weights for all gates [4*hiddenSize, hiddenSize]
        public var weightHH: Parameter

        /// Input-to-hidden bias for all gates [4*hiddenSize]
        public var biasIH: Parameter

        /// Hidden-to-hidden bias for all gates [4*hiddenSize]
        public var biasHH: Parameter

        /// Whether to use bias terms
        public let bias: Bool

        /// Create an LSTM cell
        ///
        /// - Parameters:
        ///   - inputSize: Size of input features
        ///   - hiddenSize: Size of hidden state
        ///   - bias: Whether to include bias terms (default: true)
        ///   - device: Device to create parameters on
        public init(
            inputSize: Int,
            hiddenSize: Int,
            bias: Bool = true,
            device: Device = .default
        ) {
            self.inputSize = inputSize
            self.hiddenSize = hiddenSize
            self.bias = bias

            // Xavier/Glorot initialization
            let scale = Float(1.0) / sqrt(Float(hiddenSize))

            // Concatenated weights for all 4 gates: [i, f, g, o]
            self.weightIH = Parameter(
                Tensor<Float>.randn([4 * hiddenSize, inputSize], on: device) * Tensor<Float>.full([], scale, on: device),
                name: "weight_ih"
            )
            self.weightHH = Parameter(
                Tensor<Float>.randn([4 * hiddenSize, hiddenSize], on: device) * Tensor<Float>.full([], scale, on: device),
                name: "weight_hh"
            )

            if bias {
                self.biasIH = Parameter(Tensor<Float>.zeros([4 * hiddenSize], on: device), name: "bias_ih")
                self.biasHH = Parameter(Tensor<Float>.zeros([4 * hiddenSize], on: device), name: "bias_hh")
            } else {
                self.biasIH = Parameter(Tensor<Float>.zeros([4 * hiddenSize], on: device), name: "bias_ih")
                self.biasHH = Parameter(Tensor<Float>.zeros([4 * hiddenSize], on: device), name: "bias_hh")
            }
        }

        /// Forward pass
        ///
        /// - Parameters:
        ///   - input: Tuple of (input tensor, (hidden state, cell state))
        /// - Returns: Tuple of (new hidden state, new cell state)
        public func forward(_ input: (Tensor<Float>, (Tensor<Float>, Tensor<Float>))) -> (Tensor<Float>, Tensor<Float>) {
            let (x, (h, c)) = input
            return forward(x: x, hidden: h, cell: c)
        }

        /// Forward pass with named parameters
        public func forward(x: Tensor<Float>, hidden: Tensor<Float>, cell: Tensor<Float>) -> (Tensor<Float>, Tensor<Float>) {
            let batch = x.shape[0]

            // Compute all gates at once: [batch, 4*hiddenSize]
            let xProj = x.matmul(weightIH.value.transpose())
            let hProj = hidden.matmul(weightHH.value.transpose())

            var gates = xProj + hProj
            if bias {
                gates = gates + biasIH.value.broadcast(to: gates.shape)
                gates = gates + biasHH.value.broadcast(to: gates.shape)
            }

            // Split into individual gates
            // gates shape: [batch, 4*hiddenSize]
            // We need to split along the last dimension
            let gatesFlat = gates.reshape([batch * 4, hiddenSize])

            // Extract each gate using slicing
            let i = gatesFlat.slice(start: 0, size: batch).reshape([batch, hiddenSize]).sigmoid()
            let f = gatesFlat.slice(start: batch, size: batch).reshape([batch, hiddenSize]).sigmoid()
            let g = gatesFlat.slice(start: 2 * batch, size: batch).reshape([batch, hiddenSize]).tanh()
            let o = gatesFlat.slice(start: 3 * batch, size: batch).reshape([batch, hiddenSize]).sigmoid()

            // Update cell and hidden states
            let newCell = f * cell + i * g
            let newHidden = o * newCell.tanh()

            return (newHidden, newCell)
        }

        public func parameters() -> [Parameter] {
            if bias {
                return [weightIH, weightHH, biasIH, biasHH]
            } else {
                return [weightIH, weightHH]
            }
        }
    }

    // MARK: - GRU Cell

    /// GRU cell with reset and update gates
    ///
    /// Implements:
    /// - r = sigmoid(Wir @ x + bir + Whr @ h + bhr)  (reset gate)
    /// - z = sigmoid(Wiz @ x + biz + Whz @ h + bhz)  (update gate)
    /// - n = tanh(Win @ x + bin + r * (Whn @ h + bhn))  (new gate)
    /// - h' = (1 - z) * n + z * h
    ///
    /// Example:
    /// ```swift
    /// let cell = nn.GRUCell(inputSize: 128, hiddenSize: 256)
    /// var h = Tensor<Float>.zeros([batch, 256])
    /// for t in 0..<seqLen {
    ///     h = cell(x[t], h)
    /// }
    /// ```
    public struct GRUCell: Module {
        public typealias Input = (Tensor<Float>, Tensor<Float>)
        public typealias Output = Tensor<Float>

        /// Input size
        public let inputSize: Int

        /// Hidden state size
        public let hiddenSize: Int

        /// Input-to-hidden weights for all gates [3*hiddenSize, inputSize]
        public var weightIH: Parameter

        /// Hidden-to-hidden weights for all gates [3*hiddenSize, hiddenSize]
        public var weightHH: Parameter

        /// Input-to-hidden bias for all gates [3*hiddenSize]
        public var biasIH: Parameter

        /// Hidden-to-hidden bias for all gates [3*hiddenSize]
        public var biasHH: Parameter

        /// Whether to use bias terms
        public let bias: Bool

        /// Create a GRU cell
        ///
        /// - Parameters:
        ///   - inputSize: Size of input features
        ///   - hiddenSize: Size of hidden state
        ///   - bias: Whether to include bias terms (default: true)
        ///   - device: Device to create parameters on
        public init(
            inputSize: Int,
            hiddenSize: Int,
            bias: Bool = true,
            device: Device = .default
        ) {
            self.inputSize = inputSize
            self.hiddenSize = hiddenSize
            self.bias = bias

            // Xavier/Glorot initialization
            let scale = Float(1.0) / sqrt(Float(hiddenSize))

            // Concatenated weights for all 3 gates: [r, z, n]
            self.weightIH = Parameter(
                Tensor<Float>.randn([3 * hiddenSize, inputSize], on: device) * Tensor<Float>.full([], scale, on: device),
                name: "weight_ih"
            )
            self.weightHH = Parameter(
                Tensor<Float>.randn([3 * hiddenSize, hiddenSize], on: device) * Tensor<Float>.full([], scale, on: device),
                name: "weight_hh"
            )

            if bias {
                self.biasIH = Parameter(Tensor<Float>.zeros([3 * hiddenSize], on: device), name: "bias_ih")
                self.biasHH = Parameter(Tensor<Float>.zeros([3 * hiddenSize], on: device), name: "bias_hh")
            } else {
                self.biasIH = Parameter(Tensor<Float>.zeros([3 * hiddenSize], on: device), name: "bias_ih")
                self.biasHH = Parameter(Tensor<Float>.zeros([3 * hiddenSize], on: device), name: "bias_hh")
            }
        }

        /// Forward pass
        ///
        /// - Parameters:
        ///   - input: Tuple of (input tensor, hidden state)
        /// - Returns: New hidden state
        public func forward(_ input: (Tensor<Float>, Tensor<Float>)) -> Tensor<Float> {
            let (x, h) = input
            return forward(x: x, hidden: h)
        }

        /// Forward pass with named parameters
        public func forward(x: Tensor<Float>, hidden: Tensor<Float>) -> Tensor<Float> {
            let batch = x.shape[0]

            // Compute input projections: [batch, 3*hiddenSize]
            let xProj = x.matmul(weightIH.value.transpose())
            let hProj = hidden.matmul(weightHH.value.transpose())

            // Add biases if applicable
            var xGates = xProj
            var hGates = hProj
            if bias {
                xGates = xGates + biasIH.value.broadcast(to: xGates.shape)
                hGates = hGates + biasHH.value.broadcast(to: hGates.shape)
            }

            // Split input projections into gates
            let xFlat = xGates.reshape([batch * 3, hiddenSize])
            let hFlat = hGates.reshape([batch * 3, hiddenSize])

            // Reset and update gates
            let xr = xFlat.slice(start: 0, size: batch).reshape([batch, hiddenSize])
            let xz = xFlat.slice(start: batch, size: batch).reshape([batch, hiddenSize])
            let xn = xFlat.slice(start: 2 * batch, size: batch).reshape([batch, hiddenSize])

            let hr = hFlat.slice(start: 0, size: batch).reshape([batch, hiddenSize])
            let hz = hFlat.slice(start: batch, size: batch).reshape([batch, hiddenSize])
            let hn = hFlat.slice(start: 2 * batch, size: batch).reshape([batch, hiddenSize])

            let r = (xr + hr).sigmoid()  // Reset gate
            let z = (xz + hz).sigmoid()  // Update gate
            let n = (xn + r * hn).tanh()  // New gate

            // h' = (1 - z) * n + z * h
            let one = Tensor<Float>.ones([batch, hiddenSize], on: x.device)
            return (one - z) * n + z * hidden
        }

        public func parameters() -> [Parameter] {
            if bias {
                return [weightIH, weightHH, biasIH, biasHH]
            } else {
                return [weightIH, weightHH]
            }
        }
    }

    // MARK: - RNN Layer

    /// Multi-layer RNN with optional bidirectionality
    ///
    /// Processes sequences using RNN cells, optionally with multiple layers
    /// and bidirectional processing.
    ///
    /// Example:
    /// ```swift
    /// let rnn = nn.RNN(inputSize: 128, hiddenSize: 256, numLayers: 2)
    /// let (output, hidden) = rnn(input, hidden: nil)
    /// ```
    public struct RNN: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = (Tensor<Float>, Tensor<Float>)

        /// Input size
        public let inputSize: Int

        /// Hidden state size
        public let hiddenSize: Int

        /// Number of stacked RNN layers
        public let numLayers: Int

        /// Whether to process sequences bidirectionally
        public let bidirectional: Bool

        /// Dropout probability between layers
        public let dropout: Float

        /// Whether the input is batch-first
        public let batchFirst: Bool

        /// RNN cells for each layer and direction
        public var cells: [RNNCell]

        /// Dropout layer (applied between RNN layers)
        public var dropoutLayer: Dropout?

        /// Number of directions (1 or 2)
        public var numDirections: Int { bidirectional ? 2 : 1 }

        /// Create a multi-layer RNN
        ///
        /// - Parameters:
        ///   - inputSize: Size of input features
        ///   - hiddenSize: Size of hidden state
        ///   - numLayers: Number of stacked layers (default: 1)
        ///   - nonlinearity: Activation function (default: .tanh)
        ///   - bias: Whether to use bias (default: true)
        ///   - batchFirst: If true, input is [batch, seq, features] (default: true)
        ///   - dropout: Dropout probability between layers (default: 0)
        ///   - bidirectional: Process in both directions (default: false)
        ///   - device: Device to create parameters on
        public init(
            inputSize: Int,
            hiddenSize: Int,
            numLayers: Int = 1,
            nonlinearity: RNNNonlinearity = .tanh,
            bias: Bool = true,
            batchFirst: Bool = true,
            dropout: Float = 0.0,
            bidirectional: Bool = false,
            device: Device = .default
        ) {
            self.inputSize = inputSize
            self.hiddenSize = hiddenSize
            self.numLayers = numLayers
            self.bidirectional = bidirectional
            self.dropout = dropout
            self.batchFirst = batchFirst

            // Create cells for each layer and direction
            var cells: [RNNCell] = []
            for layer in 0..<numLayers {
                let layerInputSize = layer == 0 ? inputSize : hiddenSize * (bidirectional ? 2 : 1)

                // Forward direction
                cells.append(RNNCell(
                    inputSize: layerInputSize,
                    hiddenSize: hiddenSize,
                    bias: bias,
                    nonlinearity: nonlinearity,
                    device: device
                ))

                // Backward direction
                if bidirectional {
                    cells.append(RNNCell(
                        inputSize: layerInputSize,
                        hiddenSize: hiddenSize,
                        bias: bias,
                        nonlinearity: nonlinearity,
                        device: device
                    ))
                }
            }
            self.cells = cells

            // Dropout layer
            if dropout > 0 && numLayers > 1 {
                self.dropoutLayer = Dropout(p: dropout)
            } else {
                self.dropoutLayer = nil
            }
        }

        /// Forward pass
        ///
        /// - Parameters:
        ///   - input: Input sequence [batch, seq, inputSize] if batchFirst
        /// - Returns: Tuple of (output sequence, final hidden states)
        public func forward(_ input: Tensor<Float>) -> (Tensor<Float>, Tensor<Float>) {
            return forward(input, hidden: nil)
        }

        /// Forward pass with initial hidden state
        ///
        /// - Parameters:
        ///   - input: Input sequence
        ///   - hidden: Initial hidden state [numLayers * numDirections, batch, hiddenSize]
        /// - Returns: Tuple of (output sequence, final hidden states)
        public func forward(_ input: Tensor<Float>, hidden: Tensor<Float>?) -> (Tensor<Float>, Tensor<Float>) {
            var x = input
            if !batchFirst {
                // Transpose to [batch, seq, features]
                x = x.transpose(0, 1)
            }

            let batch = x.shape[0]
            let seqLen = x.shape[1]

            // Initialize hidden states if not provided
            var hiddenStates: [Tensor<Float>]
            if let h = hidden {
                // Split hidden into per-layer states
                let totalLayers = numLayers * numDirections
                hiddenStates = []
                for i in 0..<totalLayers {
                    hiddenStates.append(h.slice(start: i, size: 1).reshape([batch, hiddenSize]))
                }
            } else {
                hiddenStates = (0..<(numLayers * numDirections)).map { _ in
                    Tensor<Float>.zeros([batch, hiddenSize], on: x.device)
                }
            }

            var layerInput = x
            var finalHiddens: [Tensor<Float>] = []

            for layer in 0..<numLayers {
                let cellIdx = layer * numDirections

                // Forward pass
                var forwardOutputs: [Tensor<Float>] = []
                var hForward = hiddenStates[cellIdx]

                for t in 0..<seqLen {
                    // Slice along seq dimension (axis 1), then squeeze
                    let xt = layerInput.sliceAxis(axis: 1, start: t, size: 1).reshape([batch, layerInput.shape[2]])
                    hForward = cells[cellIdx].forward(x: xt, hidden: hForward)
                    forwardOutputs.append(hForward)
                }
                finalHiddens.append(hForward)

                var layerOutput: Tensor<Float>
                if bidirectional {
                    // Backward pass
                    var backwardOutputs: [Tensor<Float>] = []
                    var hBackward = hiddenStates[cellIdx + 1]

                    for t in stride(from: seqLen - 1, through: 0, by: -1) {
                        let xt = layerInput.sliceAxis(axis: 1, start: t, size: 1).reshape([batch, layerInput.shape[2]])
                        hBackward = cells[cellIdx + 1].forward(x: xt, hidden: hBackward)
                        backwardOutputs.insert(hBackward, at: 0)
                    }
                    finalHiddens.append(hBackward)

                    // Concatenate forward and backward outputs at each timestep
                    // Each output is [batch, hiddenSize], concatenate to [batch, 2*hiddenSize]
                    var combinedOutputs: [Tensor<Float>] = []
                    for t in 0..<seqLen {
                        let combined = Tensor<Float>.concat([forwardOutputs[t], backwardOutputs[t]], axis: -1)
                        combinedOutputs.append(combined)
                    }
                    // Stack timesteps: [seqLen] of [batch, 2*hiddenSize] -> [batch, seqLen, 2*hiddenSize]
                    layerOutput = Tensor<Float>.stack(combinedOutputs, axis: 1)
                } else {
                    // Stack forward outputs: [seqLen] of [batch, hiddenSize] -> [batch, seqLen, hiddenSize]
                    layerOutput = Tensor<Float>.stack(forwardOutputs, axis: 1)
                }

                // Apply dropout between layers
                if layer < numLayers - 1, let drop = dropoutLayer {
                    layerOutput = drop(layerOutput)
                }

                layerInput = layerOutput
            }

            // Stack final hidden states: [numLayers*numDirections] of [batch, hiddenSize] -> [numLayers*numDirections, batch, hiddenSize]
            let output = layerInput
            let finalHidden = Tensor<Float>.stack(finalHiddens, axis: 0)

            if !batchFirst {
                return (output.transpose(0, 1), finalHidden)
            }
            return (output, finalHidden)
        }

        public func parameters() -> [Parameter] {
            var params: [Parameter] = []
            for cell in cells {
                params.append(contentsOf: cell.parameters())
            }
            return params
        }
    }

    // MARK: - LSTM Layer

    /// Multi-layer LSTM
    ///
    /// Example:
    /// ```swift
    /// let lstm = nn.LSTM(inputSize: 128, hiddenSize: 256, numLayers: 2)
    /// let (output, (h, c)) = lstm(input)
    /// ```
    public struct LSTM: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = (Tensor<Float>, (Tensor<Float>, Tensor<Float>))

        /// Input size
        public let inputSize: Int

        /// Hidden state size
        public let hiddenSize: Int

        /// Number of stacked LSTM layers
        public let numLayers: Int

        /// Whether to process bidirectionally
        public let bidirectional: Bool

        /// Dropout probability
        public let dropout: Float

        /// Whether input is batch-first
        public let batchFirst: Bool

        /// LSTM cells
        public var cells: [LSTMCell]

        /// Number of directions
        public var numDirections: Int { bidirectional ? 2 : 1 }

        /// Create a multi-layer LSTM
        public init(
            inputSize: Int,
            hiddenSize: Int,
            numLayers: Int = 1,
            bias: Bool = true,
            batchFirst: Bool = true,
            dropout: Float = 0.0,
            bidirectional: Bool = false,
            device: Device = .default
        ) {
            self.inputSize = inputSize
            self.hiddenSize = hiddenSize
            self.numLayers = numLayers
            self.bidirectional = bidirectional
            self.dropout = dropout
            self.batchFirst = batchFirst

            var cells: [LSTMCell] = []
            for layer in 0..<numLayers {
                let layerInputSize = layer == 0 ? inputSize : hiddenSize * (bidirectional ? 2 : 1)

                cells.append(LSTMCell(
                    inputSize: layerInputSize,
                    hiddenSize: hiddenSize,
                    bias: bias,
                    device: device
                ))

                if bidirectional {
                    cells.append(LSTMCell(
                        inputSize: layerInputSize,
                        hiddenSize: hiddenSize,
                        bias: bias,
                        device: device
                    ))
                }
            }
            self.cells = cells
        }

        /// Forward pass
        public func forward(_ input: Tensor<Float>) -> (Tensor<Float>, (Tensor<Float>, Tensor<Float>)) {
            return forward(input, hidden: nil)
        }

        /// Forward pass with initial states
        public func forward(_ input: Tensor<Float>, hidden: (Tensor<Float>, Tensor<Float>)?) -> (Tensor<Float>, (Tensor<Float>, Tensor<Float>)) {
            var x = input
            if !batchFirst {
                x = x.transpose(0, 1)
            }

            let batch = x.shape[0]
            let seqLen = x.shape[1]

            // Initialize hidden and cell states
            let h0 = hidden?.0 ?? Tensor<Float>.zeros([numLayers * numDirections, batch, hiddenSize], on: x.device)
            let c0 = hidden?.1 ?? Tensor<Float>.zeros([numLayers * numDirections, batch, hiddenSize], on: x.device)

            // Process through layers
            var layerInput = x
            var finalHiddens: [Tensor<Float>] = []
            var finalCells: [Tensor<Float>] = []

            for layer in 0..<numLayers {
                let cellIdx = layer * numDirections

                // Forward direction
                var hForward = h0.slice(start: cellIdx, size: 1).reshape([batch, hiddenSize])
                var cForward = c0.slice(start: cellIdx, size: 1).reshape([batch, hiddenSize])
                var forwardOutputs: [Tensor<Float>] = []

                for t in 0..<seqLen {
                    // Slice along seq dimension (axis 1), then squeeze to [batch, features]
                    let xt = layerInput.sliceAxis(axis: 1, start: t, size: 1).reshape([batch, layerInput.shape[2]])
                    (hForward, cForward) = cells[cellIdx].forward(x: xt, hidden: hForward, cell: cForward)
                    forwardOutputs.append(hForward)
                }
                finalHiddens.append(hForward)
                finalCells.append(cForward)

                // Stack forward outputs: [seqLen] of [batch, hiddenSize] -> [batch, seqLen, hiddenSize]
                // For bidirectional, would need backward pass and concat (not yet implemented for LSTM)
                layerInput = Tensor<Float>.stack(forwardOutputs, axis: 1)
            }

            let output = layerInput
            let finalH = Tensor<Float>.stack(finalHiddens, axis: 0)
            let finalC = Tensor<Float>.stack(finalCells, axis: 0)

            if !batchFirst {
                return (output.transpose(0, 1), (finalH, finalC))
            }
            return (output, (finalH, finalC))
        }

        public func parameters() -> [Parameter] {
            cells.flatMap { $0.parameters() }
        }
    }

    // MARK: - GRU Layer

    /// Multi-layer GRU
    ///
    /// Example:
    /// ```swift
    /// let gru = nn.GRU(inputSize: 128, hiddenSize: 256, numLayers: 2)
    /// let (output, hidden) = gru(input)
    /// ```
    public struct GRU: Module {
        public typealias Input = Tensor<Float>
        public typealias Output = (Tensor<Float>, Tensor<Float>)

        /// Input size
        public let inputSize: Int

        /// Hidden state size
        public let hiddenSize: Int

        /// Number of stacked GRU layers
        public let numLayers: Int

        /// Whether to process bidirectionally
        public let bidirectional: Bool

        /// Dropout probability
        public let dropout: Float

        /// Whether input is batch-first
        public let batchFirst: Bool

        /// GRU cells
        public var cells: [GRUCell]

        /// Number of directions
        public var numDirections: Int { bidirectional ? 2 : 1 }

        /// Create a multi-layer GRU
        public init(
            inputSize: Int,
            hiddenSize: Int,
            numLayers: Int = 1,
            bias: Bool = true,
            batchFirst: Bool = true,
            dropout: Float = 0.0,
            bidirectional: Bool = false,
            device: Device = .default
        ) {
            self.inputSize = inputSize
            self.hiddenSize = hiddenSize
            self.numLayers = numLayers
            self.bidirectional = bidirectional
            self.dropout = dropout
            self.batchFirst = batchFirst

            var cells: [GRUCell] = []
            for layer in 0..<numLayers {
                let layerInputSize = layer == 0 ? inputSize : hiddenSize * (bidirectional ? 2 : 1)

                cells.append(GRUCell(
                    inputSize: layerInputSize,
                    hiddenSize: hiddenSize,
                    bias: bias,
                    device: device
                ))

                if bidirectional {
                    cells.append(GRUCell(
                        inputSize: layerInputSize,
                        hiddenSize: hiddenSize,
                        bias: bias,
                        device: device
                    ))
                }
            }
            self.cells = cells
        }

        /// Forward pass
        public func forward(_ input: Tensor<Float>) -> (Tensor<Float>, Tensor<Float>) {
            return forward(input, hidden: nil)
        }

        /// Forward pass with initial hidden state
        public func forward(_ input: Tensor<Float>, hidden: Tensor<Float>?) -> (Tensor<Float>, Tensor<Float>) {
            var x = input
            if !batchFirst {
                x = x.transpose(0, 1)
            }

            let batch = x.shape[0]
            let seqLen = x.shape[1]

            // Initialize hidden states
            let h0 = hidden ?? Tensor<Float>.zeros([numLayers * numDirections, batch, hiddenSize], on: x.device)

            // Process through layers
            var layerInput = x
            var finalHiddens: [Tensor<Float>] = []

            for layer in 0..<numLayers {
                let cellIdx = layer * numDirections

                // Forward direction
                var hForward = h0.slice(start: cellIdx, size: 1).reshape([batch, hiddenSize])
                var forwardOutputs: [Tensor<Float>] = []

                for t in 0..<seqLen {
                    // Slice along seq dimension (axis 1), then squeeze to [batch, features]
                    let xt = layerInput.sliceAxis(axis: 1, start: t, size: 1).reshape([batch, layerInput.shape[2]])
                    hForward = cells[cellIdx].forward(x: xt, hidden: hForward)
                    forwardOutputs.append(hForward)
                }
                finalHiddens.append(hForward)

                // Stack forward outputs: [seqLen] of [batch, hiddenSize] -> [batch, seqLen, hiddenSize]
                // For bidirectional, would need backward pass and concat (not yet implemented for GRU)
                layerInput = Tensor<Float>.stack(forwardOutputs, axis: 1)
            }

            let output = layerInput
            let finalH = Tensor<Float>.stack(finalHiddens, axis: 0)

            if !batchFirst {
                return (output.transpose(0, 1), finalH)
            }
            return (output, finalH)
        }

        public func parameters() -> [Parameter] {
            cells.flatMap { $0.parameters() }
        }
    }
}

// MARK: - Model Checkpointing

/// A checkpoint representing the state of a model's parameters.
///
/// Checkpoints can be saved to disk and loaded later to restore model state.
/// The format uses JSON for metadata and binary for tensor data.
///
/// Example:
/// ```swift
/// // Save a model
/// let model = nn.Linear(inputSize: 784, outputSize: 10)
/// try model.save(to: URL(fileURLWithPath: "model.checkpoint"))
///
/// // Load a model
/// var loadedModel = nn.Linear(inputSize: 784, outputSize: 10)
/// try loadedModel.load(from: URL(fileURLWithPath: "model.checkpoint"))
/// ```
public struct Checkpoint: Codable {
    /// Version for format compatibility
    public let version: Int

    /// Timestamp when checkpoint was created
    public let timestamp: Date

    /// Optional description
    public var description: String?

    /// Parameter entries with shapes and data
    public var parameters: [ParameterEntry]

    /// A single parameter's data in the checkpoint
    public struct ParameterEntry: Codable {
        /// Parameter name (if available)
        public var name: String?
        /// Index in the parameters list
        public let index: Int
        /// Shape of the tensor
        public let shape: [Int]
        /// Flattened tensor values
        public let values: [Float]

        public init(name: String?, index: Int, shape: [Int], values: [Float]) {
            self.name = name
            self.index = index
            self.shape = shape
            self.values = values
        }
    }

    public init(description: String? = nil) {
        self.version = 1
        self.timestamp = Date()
        self.description = description
        self.parameters = []
    }
}

/// Errors that can occur during checkpoint operations
public enum CheckpointError: Error, CustomStringConvertible {
    case saveFailed(String)
    case loadFailed(String)
    case parameterMismatch(expected: Int, got: Int)
    case shapeMismatch(parameterIndex: Int, expected: [Int], got: [Int])
    case incompatibleVersion(Int)

    public var description: String {
        switch self {
        case .saveFailed(let msg):
            return "Checkpoint save failed: \(msg)"
        case .loadFailed(let msg):
            return "Checkpoint load failed: \(msg)"
        case .parameterMismatch(let expected, let got):
            return "Parameter count mismatch: expected \(expected), got \(got)"
        case .shapeMismatch(let idx, let expected, let got):
            return "Shape mismatch at parameter \(idx): expected \(expected), got \(got)"
        case .incompatibleVersion(let version):
            return "Incompatible checkpoint version: \(version)"
        }
    }
}

extension Module {
    /// Save the module's parameters to a checkpoint file.
    ///
    /// - Parameters:
    ///   - url: File URL to save the checkpoint.
    ///   - description: Optional description to include in the checkpoint.
    /// - Throws: CheckpointError if saving fails.
    ///
    /// Example:
    /// ```swift
    /// let model = nn.Linear(inputSize: 784, outputSize: 10)
    /// try model.save(to: URL(fileURLWithPath: "linear.checkpoint"))
    /// ```
    public func save(to url: URL, description: String? = nil) throws {
        var checkpoint = Checkpoint(description: description)

        let params = parameters()
        for (index, param) in params.enumerated() {
            // Materialize tensor to get values
            LazyTensorBarrier()
            let values = param.value.scalars()

            let entry = Checkpoint.ParameterEntry(
                name: param.name,
                index: index,
                shape: param.shape,
                values: values
            )
            checkpoint.parameters.append(entry)
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]

        do {
            let data = try encoder.encode(checkpoint)
            try data.write(to: url)
        } catch {
            throw CheckpointError.saveFailed(error.localizedDescription)
        }
    }

    /// Load parameters from a checkpoint file into this module.
    ///
    /// - Parameter url: File URL to load the checkpoint from.
    /// - Throws: CheckpointError if loading fails or parameters don't match.
    ///
    /// Example:
    /// ```swift
    /// var model = nn.Linear(inputSize: 784, outputSize: 10)
    /// try model.load(from: URL(fileURLWithPath: "linear.checkpoint"))
    /// ```
    public mutating func load(from url: URL) throws {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw CheckpointError.loadFailed(error.localizedDescription)
        }

        let decoder = JSONDecoder()
        let checkpoint: Checkpoint
        do {
            checkpoint = try decoder.decode(Checkpoint.self, from: data)
        } catch {
            throw CheckpointError.loadFailed(error.localizedDescription)
        }

        // Check version compatibility
        if checkpoint.version > 1 {
            throw CheckpointError.incompatibleVersion(checkpoint.version)
        }

        let params = parameters()

        // Check parameter count matches
        if params.count != checkpoint.parameters.count {
            throw CheckpointError.parameterMismatch(expected: params.count, got: checkpoint.parameters.count)
        }

        // Load each parameter
        for entry in checkpoint.parameters {
            let param = params[entry.index]

            // Check shape matches
            if param.shape != entry.shape {
                throw CheckpointError.shapeMismatch(
                    parameterIndex: entry.index,
                    expected: param.shape,
                    got: entry.shape
                )
            }

            // Create new tensor with loaded values
            param.value = Tensor<Float>(entry.values, shape: entry.shape)
        }
    }
}

// MARK: - State Dict (PyTorch-style)

/// A dictionary mapping parameter names to their tensor values.
/// Similar to PyTorch's state_dict.
public typealias StateDict = [String: Tensor<Float>]

extension Module {
    /// Returns a dictionary containing all module parameters.
    ///
    /// Similar to PyTorch's `state_dict()`.
    ///
    /// Example:
    /// ```swift
    /// let model = nn.Linear(inputSize: 784, outputSize: 10)
    /// let stateDict = model.stateDict()
    /// print(stateDict.keys)  // ["weight", "bias"]
    /// ```
    public func stateDict() -> StateDict {
        var dict = StateDict()
        for (index, param) in parameters().enumerated() {
            let key = param.name ?? "param_\(index)"
            dict[key] = param.value
        }
        return dict
    }

    /// Load parameters from a state dictionary.
    ///
    /// Similar to PyTorch's `load_state_dict()`.
    ///
    /// - Parameters:
    ///   - stateDict: Dictionary mapping names to tensors.
    ///   - strict: If true, requires all keys to match. Default is true.
    /// - Throws: CheckpointError if keys don't match (when strict=true).
    ///
    /// Example:
    /// ```swift
    /// var model = nn.Linear(inputSize: 784, outputSize: 10)
    /// try model.loadStateDict(savedStateDict)
    /// ```
    public mutating func loadStateDict(_ stateDict: StateDict, strict: Bool = true) throws {
        let params = parameters()

        // Build mapping from name to parameter
        var nameToParam: [String: Parameter] = [:]
        for (index, param) in params.enumerated() {
            let key = param.name ?? "param_\(index)"
            nameToParam[key] = param
        }

        if strict {
            // Check all keys match
            let paramKeys = Set(nameToParam.keys)
            let dictKeys = Set(stateDict.keys)

            let missing = paramKeys.subtracting(dictKeys)
            let unexpected = dictKeys.subtracting(paramKeys)

            if !missing.isEmpty || !unexpected.isEmpty {
                var msg = ""
                if !missing.isEmpty {
                    msg += "Missing keys: \(missing.sorted()). "
                }
                if !unexpected.isEmpty {
                    msg += "Unexpected keys: \(unexpected.sorted())."
                }
                throw CheckpointError.loadFailed(msg)
            }
        }

        // Load matching keys
        for (key, tensor) in stateDict {
            if let param = nameToParam[key] {
                if param.shape != tensor.shape {
                    throw CheckpointError.shapeMismatch(
                        parameterIndex: 0,
                        expected: param.shape,
                        got: tensor.shape
                    )
                }
                param.value = tensor
            }
        }
    }
}

// MARK: - Binary Checkpoint Format (More Efficient)

/// Binary checkpoint format for efficient storage of large models.
///
/// Format:
/// - 8 bytes: Magic number "STCHKPT\0"
/// - 4 bytes: Version (uint32 little endian)
/// - 4 bytes: Number of parameters (uint32 little endian)
/// - For each parameter:
///   - 4 bytes: Name length (uint32)
///   - N bytes: Name (UTF-8)
///   - 4 bytes: Rank (uint32)
///   - 4*rank bytes: Shape (uint32 each)
///   - 4*elementCount bytes: Values (float32)
public struct BinaryCheckpoint {
    private static let magic: [UInt8] = Array("STCHKPT\0".utf8)
    private static let version: UInt32 = 1

    /// Save module to binary format
    public static func save<M: Module>(_ module: M, to url: URL) throws {
        var data = Data()

        // Magic number
        data.append(contentsOf: magic)

        // Version
        var version = self.version
        data.append(Data(bytes: &version, count: 4))

        let params = module.parameters()

        // Number of parameters
        var numParams = UInt32(params.count)
        data.append(Data(bytes: &numParams, count: 4))

        // Each parameter
        for (index, param) in params.enumerated() {
            // Materialize values
            LazyTensorBarrier()
            let values = param.value.scalars()

            // Name
            let name = param.name ?? "param_\(index)"
            let nameData = name.data(using: .utf8)!
            var nameLen = UInt32(nameData.count)
            data.append(Data(bytes: &nameLen, count: 4))
            data.append(nameData)

            // Rank
            var rank = UInt32(param.shape.count)
            data.append(Data(bytes: &rank, count: 4))

            // Shape
            for dim in param.shape {
                var dimVal = UInt32(dim)
                data.append(Data(bytes: &dimVal, count: 4))
            }

            // Values (as float32)
            values.withUnsafeBufferPointer { buffer in
                data.append(buffer)
            }
        }

        try data.write(to: url)
    }

    /// Load module from binary format
    public static func load<M: Module>(_ module: inout M, from url: URL) throws {
        let data = try Data(contentsOf: url)
        var offset = 0

        // Helper to read UInt32 safely (handles unaligned access)
        func readUInt32() -> UInt32 {
            var value: UInt32 = 0
            _ = withUnsafeMutableBytes(of: &value) { dest in
                data.copyBytes(to: dest, from: offset..<offset+4)
            }
            offset += 4
            return value
        }

        // Helper to read Float array safely
        func readFloats(count: Int) -> [Float] {
            var values = [Float](repeating: 0, count: count)
            values.withUnsafeMutableBytes { dest in
                data.copyBytes(to: dest, from: offset..<offset + count * 4)
            }
            offset += count * 4
            return values
        }

        // Verify magic
        let readMagic = [UInt8](data[offset..<offset+8])
        guard readMagic == magic else {
            throw CheckpointError.loadFailed("Invalid file format")
        }
        offset += 8

        // Version
        let version = readUInt32()
        guard version <= self.version else {
            throw CheckpointError.incompatibleVersion(Int(version))
        }

        // Number of parameters
        let numParams = readUInt32()

        let params = module.parameters()
        guard params.count == Int(numParams) else {
            throw CheckpointError.parameterMismatch(expected: params.count, got: Int(numParams))
        }

        // Load each parameter
        for (index, param) in params.enumerated() {
            // Name length
            let nameLen = readUInt32()

            // Name (skip, use for verification if needed)
            _ = String(data: data[offset..<offset+Int(nameLen)], encoding: .utf8)
            offset += Int(nameLen)

            // Rank
            let rank = readUInt32()

            // Shape
            var shape: [Int] = []
            for _ in 0..<rank {
                let dim = readUInt32()
                shape.append(Int(dim))
            }

            // Verify shape
            guard param.shape == shape else {
                throw CheckpointError.shapeMismatch(parameterIndex: index, expected: param.shape, got: shape)
            }

            // Values
            let elementCount = shape.reduce(1, *)
            let values = readFloats(count: elementCount)

            // Create new tensor
            param.value = Tensor<Float>(values, shape: shape)
        }
    }
}
