# Magma API Reference

This document provides a comprehensive reference for Magma's neural network API, including layers, loss functions, optimizers, and initializers.

## Table of Contents

- [Tensor Creation](#tensor-creation)
- [Layers](#layers)
  - [Linear Layers](#linear-layers)
  - [Convolutional Layers](#convolutional-layers)
  - [Pooling Layers](#pooling-layers)
  - [Normalization Layers](#normalization-layers)
  - [Activation Layers](#activation-layers)
  - [Dropout Layers](#dropout-layers)
  - [Embedding Layers](#embedding-layers)
  - [Recurrent Layers](#recurrent-layers)
  - [Upsampling Layers](#upsampling-layers)
  - [Container Layers](#container-layers)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Initializers](#initializers)

---

## Tensor Creation

### Basic Tensor Construction

```swift
// From array with explicit shape
let tensor = Tensor<Float>([1, 2, 3, 4, 5, 6], shape: [2, 3], on: .default)

// Access properties
tensor.shape    // [2, 3]
tensor.dtype    // .float32
tensor.device   // Device
tensor.rank     // 2
```

### Factory Methods

```swift
// Zeros and Ones
let zeros = Tensor<Float>.zeros([32, 784])
let ones = Tensor<Float>.ones([32, 784])

// Filled with constant value
let filled = Tensor<Float>.full([32, 784], 0.5, on: .default)

// Random tensors
let uniform = Tensor<Float>.randomUniform([32, 784])
let normal = Tensor<Float>.randn([32, 784])
let truncatedNormal = Tensor<Float>.truncatedNormal([32, 784], mean: 0.0, stddev: 0.02)
```

### Type Aliases

```swift
typealias FloatTensor = Tensor<Float>
typealias DoubleTensor = Tensor<Double>
typealias Int32Tensor = Tensor<Int32>
typealias Int64Tensor = Tensor<Int64>
typealias BoolTensor = Tensor<Bool>
```

---

## Layers

All layers conform to the `Module` protocol:

```swift
public protocol Module: Differentiable {
    associatedtype Input
    associatedtype Output

    @differentiable(reverse)
    func callAsFunction(_ input: Input) -> Output

    func parameters() -> [Tensor<Float>]
}
```

### Linear Layers

#### nn.Linear

Fully connected layer with optional bias.

```swift
let linear = nn.Linear(inputSize: 784, outputSize: 256, bias: true)

// Usage
let input = Tensor<Float>.zeros([32, 784])
let output = linear(input)  // [32, 256]

// Properties
linear.weight.shape  // [256, 784]
linear.bias.shape    // [256]
linear.parameters().count  // 2 (or 1 if bias: false)
```

### Convolutional Layers

#### nn.Conv1d

1D convolution for sequential data (NLC format: batch, length, channels).

```swift
let conv1d = nn.Conv1d(
    inChannels: 3,
    outChannels: 64,
    kernelSize: 3,
    stride: 1,
    padding: 0,
    bias: true
)

// Usage
let input = Tensor<Float>.zeros([32, 100, 3])  // [batch, length, channels]
let output = conv1d(input)  // [32, 98, 64]

// Output length = (input + 2*padding - kernel) / stride + 1
```

#### nn.Conv2d

2D convolution for images (NHWC format: batch, height, width, channels).

```swift
let conv2d = nn.Conv2d(
    inChannels: 3,
    outChannels: 64,
    kernelSize: 3,        // or (3, 3) for asymmetric
    stride: 1,            // or (1, 1)
    padding: 0,           // or (0, 0)
    bias: true
)

// Usage
let input = Tensor<Float>.zeros([32, 224, 224, 3])  // [batch, H, W, channels]
let output = conv2d(input)  // [32, 222, 222, 64]
```

#### nn.ConvTranspose2d

Transposed 2D convolution (deconvolution) for upsampling.

```swift
let convT = nn.ConvTranspose2d(
    inChannels: 64,
    outChannels: 32,
    kernelSize: 4,        // or (4, 4)
    stride: 2,            // or (2, 2)
    padding: 1,           // or (1, 1)
    outputPadding: 0,     // or (0, 0)
    bias: true
)

// Usage: upsamples spatial dimensions
let input = Tensor<Float>.zeros([16, 14, 14, 64])
let output = convT(input)  // [16, 28, 28, 32]

// Output = (input - 1) * stride - 2 * padding + kernel + outputPadding
```

### Pooling Layers

#### nn.MaxPool2d / nn.AvgPool2d

Standard 2D pooling operations.

```swift
let maxPool = nn.MaxPool2d(kernelSize: 2, stride: 2)
let avgPool = nn.AvgPool2d(kernelSize: 2, stride: 2, padding: 0)

let input = Tensor<Float>.zeros([32, 28, 28, 64])
let output = maxPool(input)  // [32, 14, 14, 64]
```

#### nn.GlobalAvgPool1d / nn.GlobalAvgPool2d

Global average pooling that reduces spatial dimensions.

```swift
let globalAvg1d = nn.GlobalAvgPool1d()
let globalAvg2d = nn.GlobalAvgPool2d()

// 1D: [batch, length, channels] -> [batch, channels]
let input1d = Tensor<Float>.zeros([32, 100, 64])
let output1d = globalAvg1d(input1d)  // [32, 64]

// 2D: [batch, H, W, channels] -> [batch, channels]
let input2d = Tensor<Float>.zeros([32, 7, 7, 512])
let output2d = globalAvg2d(input2d)  // [32, 512]
```

#### nn.GlobalMaxPool1d / nn.GlobalMaxPool2d

Global max pooling.

```swift
let globalMax1d = nn.GlobalMaxPool1d()
let globalMax2d = nn.GlobalMaxPool2d()
```

### Normalization Layers

#### nn.BatchNorm2d

Batch normalization for 2D data.

```swift
let batchNorm = nn.BatchNorm2d(numFeatures: 64, eps: 1e-5, momentum: 0.1)

let input = Tensor<Float>.zeros([32, 14, 14, 64])
let output = batchNorm(input)  // [32, 14, 14, 64]
```

#### nn.LayerNorm

Layer normalization.

```swift
let layerNorm = nn.LayerNorm(normalizedShape: [256], eps: 1e-5)

let input = Tensor<Float>.zeros([32, 256])
let output = layerNorm(input)  // [32, 256]
```

#### nn.GroupNorm

Group normalization (divides channels into groups).

```swift
let groupNorm = nn.GroupNorm(
    numGroups: 8,
    numChannels: 64,
    eps: 1e-5,
    affine: true
)

let input = Tensor<Float>.zeros([32, 14, 14, 64])
let output = groupNorm(input)  // [32, 14, 14, 64]
```

#### nn.InstanceNorm2d

Instance normalization (normalizes each sample independently).

```swift
let instanceNorm = nn.InstanceNorm2d(
    numFeatures: 64,
    eps: 1e-5,
    affine: true
)

let input = Tensor<Float>.zeros([32, 14, 14, 64])
let output = instanceNorm(input)  // [32, 14, 14, 64]
```

### Activation Layers

#### Standard Activations

```swift
let relu = nn.ReLU()
let leakyRelu = nn.LeakyReLU(negativeSlope: 0.01)
let elu = nn.ELU(alpha: 1.0)
let sigmoid = nn.Sigmoid()
let tanh = nn.Tanh()
let gelu = nn.GELU()
let softmax = nn.Softmax(dim: -1)
```

#### Additional Activations (Ported from S4TF)

```swift
// SELU - Self-Normalizing Exponential Linear Unit
let selu = nn.SELU()

// Mish - Self-regularizing activation
let mish = nn.Mish()

// Softplus - Smooth approximation to ReLU
let softplus = nn.Softplus(beta: 1.0, threshold: 20.0)

// Softsign - Alternative to tanh
let softsign = nn.Softsign()

// PReLU - Parametric ReLU with learnable slope
let prelu = nn.PReLU(numParameters: 1, init: 0.25)
prelu.parameters().count  // 1 (learnable weight)
```

### Dropout Layers

#### nn.Dropout

Standard dropout for regularization.

```swift
let dropout = nn.Dropout(p: 0.5)

// During training, randomly zeros elements
// During inference, returns input unchanged
```

### Embedding Layers

#### nn.Embedding

Lookup table for discrete tokens.

```swift
let embedding = nn.Embedding(numEmbeddings: 10000, embeddingDim: 256)

// Usage with token indices
let indices = Tensor<Int32>([1, 5, 3, 8], shape: [4], on: .default)
let embeddings = embedding(indices)  // [4, 256]
```

### Recurrent Layers

#### nn.LSTM

Long Short-Term Memory layer.

```swift
let lstm = nn.LSTM(
    inputSize: 256,
    hiddenSize: 512,
    numLayers: 2,
    bidirectional: false,
    dropout: 0.1
)

let input = Tensor<Float>.zeros([32, 50, 256])  // [batch, seq, features]
let (output, (hn, cn)) = lstm(input)
// output: [32, 50, 512]
// hn, cn: hidden and cell states
```

### Upsampling Layers

#### nn.Upsample1d

Nearest-neighbor upsampling for 1D data.

```swift
let upsample1d = nn.Upsample1d(size: 2)  // 2x upsampling

let input = Tensor<Float>.zeros([32, 50, 64])
let output = upsample1d(input)  // [32, 100, 64]
```

#### nn.Upsample2d

Nearest-neighbor upsampling for 2D data.

```swift
let upsample2d = nn.Upsample2d(size: 2)  // 2x upsampling
// or asymmetric: nn.Upsample2d(size: (2, 3))

let input = Tensor<Float>.zeros([32, 14, 14, 64])
let output = upsample2d(input)  // [32, 28, 28, 64]
```

#### nn.UpsamplingNearest2d

Alias for `nn.Upsample2d` (PyTorch compatibility).

```swift
let upsampling: nn.UpsamplingNearest2d = nn.Upsample2d(size: 2)
```

### Container Layers

#### nn.Sequential

Chains layers in sequence.

```swift
let model = nn.Sequential {
    nn.Linear(784, 256)
    nn.ReLU()
    nn.Dropout(0.1)
    nn.Linear(256, 10)
}

let input = Tensor<Float>.zeros([32, 784])
let output = model(input)  // [32, 10]
```

---

## Loss Functions

All loss functions support three reduction modes:
- `.mean` - Average of all loss values (default for most)
- `.sum` - Sum of all loss values
- `.none` - No reduction, returns element-wise loss

### Regression Losses

```swift
// L1 Loss (Mean Absolute Error)
let loss = l1Loss(predicted: pred, expected: target, reduction: .mean)

// L2 Loss (Mean Squared Error)
let loss = l2Loss(predicted: pred, expected: target, reduction: .mean)

// Mean Absolute Error (convenience function)
let loss = meanAbsoluteError(predicted: pred, expected: target)

// Mean Squared Error (convenience function)
let loss = meanSquaredError(predicted: pred, expected: target)

// Huber Loss (less sensitive to outliers)
let loss = huberLoss(predicted: pred, expected: target, delta: 1.0, reduction: .mean)

// Log-Cosh Loss (smooth L1)
let loss = logCoshLoss(predicted: pred, expected: target, reduction: .mean)

// Poisson Loss
let loss = poissonLoss(predicted: pred, expected: target, reduction: .mean)
```

### Classification Losses

```swift
// Softmax Cross Entropy (multi-class classification)
let loss = softmaxCrossEntropy(logits: logits, probabilities: oneHotLabels, reduction: .mean)

// Sparse Softmax Cross Entropy (with integer labels)
let loss = softmaxCrossEntropyWithLabels(logits: logits, labels: [0, 1, 2], numClasses: 10)

// Sigmoid Cross Entropy (binary classification)
let loss = sigmoidCrossEntropy(logits: logits, labels: binaryLabels, reduction: .mean)

// Hinge Loss (SVM-style)
let loss = hingeLoss(predicted: pred, expected: target, reduction: .mean)

// Squared Hinge Loss
let loss = squaredHingeLoss(predicted: pred, expected: target, reduction: .mean)

// Categorical Hinge Loss
let loss = categoricalHingeLoss(predicted: pred, expected: oneHotTarget, reduction: .mean)
```

### Distribution Losses

```swift
// Kullback-Leibler Divergence
let loss = kullbackLeiblerDivergence(predicted: pred, expected: target, reduction: .sum)
```

### Metric Learning Losses

```swift
// Cosine Similarity
let similarity = cosineSimilarity(a, b, epsilon: 1e-8)

// Cosine Distance (1 - similarity)
let loss = cosineDistance(predicted: pred, expected: target, reduction: .mean)

// Contrastive Loss (Siamese networks)
let loss = contrastiveLoss(
    anchor: anchor,
    sample: sample,
    labels: labels,  // 1 for similar, 0 for dissimilar
    margin: 1.0,
    reduction: .mean
)

// Triplet Margin Loss
let loss = tripletMarginLoss(
    anchor: anchor,
    positive: positive,
    negative: negative,
    margin: 1.0,
    reduction: .mean
)
```

---

## Optimizers

All optimizers follow this pattern:

```swift
var optimizer = Optimizer(for: model, learningRate: 0.001)

// Training loop
for (data, labels) in dataLoader {
    let (loss, grads) = valueWithGradient(at: model) { m in
        let logits = m(data)
        return softmaxCrossEntropy(logits: logits, probabilities: labels)
    }
    optimizer.update(&model, along: grads)
}
```

### SGD

Stochastic Gradient Descent with optional momentum.

```swift
var sgd = SGD(
    for: model,
    learningRate: 0.01,
    momentum: 0.9,
    weightDecay: 0.0001,
    nesterov: true
)
```

### Adam

Adaptive Moment Estimation.

```swift
var adam = Adam(
    for: model,
    learningRate: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
    weightDecay: 0.0
)
```

### RMSProp

Root Mean Square Propagation.

```swift
var rmsprop = RMSProp(
    for: model,
    learningRate: 0.001,
    alpha: 0.99,       // Smoothing constant
    epsilon: 1e-8,
    weightDecay: 0.0,
    momentum: 0.0,
    centered: false
)
```

### AdaGrad

Adaptive Gradient Algorithm.

```swift
var adagrad = AdaGrad(
    for: model,
    learningRate: 0.01,
    epsilon: 1e-10,
    weightDecay: 0.0
)
```

### AdaDelta

Adaptive Delta (no learning rate required).

```swift
var adadelta = AdaDelta(
    for: model,
    rho: 0.9,
    epsilon: 1e-6,
    weightDecay: 0.0
)
```

---

## Initializers

All initializers are static methods on `Tensor<Float>`.

### Uniform and Normal Distributions

```swift
// Random uniform in [0, 1)
let t = Tensor<Float>.randomUniform([256, 128])

// Random uniform in custom range
let t = Tensor<Float>.randomUniform([256, 128], low: -1.0, high: 1.0)

// Standard normal (mean=0, std=1)
let t = Tensor<Float>.randn([256, 128])

// Truncated normal (values clipped to 2 standard deviations)
let t = Tensor<Float>.truncatedNormal([256, 128], mean: 0.0, stddev: 0.02)
```

### Xavier/Glorot Initialization

Designed for layers with tanh or sigmoid activation.

```swift
// Glorot Uniform: U[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
let t = Tensor<Float>.glorotUniform([256, 128])

// Glorot Normal: N(0, std) where std = sqrt(2 / (fan_in + fan_out))
let t = Tensor<Float>.glorotNormal([256, 128])
```

### He/Kaiming Initialization

Designed for layers with ReLU activation.

```swift
// He Uniform: U[-limit, limit] where limit = sqrt(6 / fan_in)
let t = Tensor<Float>.heUniform([256, 128])

// He Normal: N(0, std) where std = sqrt(2 / fan_in)
let t = Tensor<Float>.heNormal([256, 128])
```

### LeCun Initialization

Designed for SELU activation.

```swift
// LeCun Uniform: U[-limit, limit] where limit = sqrt(3 / fan_in)
let t = Tensor<Float>.lecunUniform([256, 128])

// LeCun Normal: N(0, std) where std = sqrt(1 / fan_in)
let t = Tensor<Float>.lecunNormal([256, 128])
```

### Orthogonal Initialization

For RNNs and maintaining gradient magnitude.

```swift
// Orthogonal matrix
let t = Tensor<Float>.orthogonal([256, 256])

// With custom gain (e.g., sqrt(2) for ReLU)
let t = Tensor<Float>.orthogonal([256, 256], gain: 1.41421)
```

### Constant Initialization

```swift
// All zeros
let t = Tensor<Float>.zeros([256, 128])

// All ones
let t = Tensor<Float>.ones([256, 128])

// Custom constant
let t = Tensor<Float>.full([256, 128], 0.01, on: .default)
```

### Fan Calculation

For convolutional layers, fan_in and fan_out are calculated as:
- `fan_in = kernel_h * kernel_w * in_channels`
- `fan_out = kernel_h * kernel_w * out_channels`

```swift
// Conv2d weight shape: [kernelH, kernelW, inChannels, outChannels]
let convWeight = Tensor<Float>.heUniform([3, 3, 64, 128])
// fan_in = 3 * 3 * 64 = 576
// fan_out = 3 * 3 * 128 = 1152
```

---

## Complete Example

```swift
import Magma

// Define model
let model = nn.Sequential {
    nn.Conv2d(inChannels: 3, outChannels: 32, kernelSize: 3, padding: 1)
    nn.BatchNorm2d(numFeatures: 32)
    nn.ReLU()
    nn.MaxPool2d(kernelSize: 2, stride: 2)

    nn.Conv2d(inChannels: 32, outChannels: 64, kernelSize: 3, padding: 1)
    nn.BatchNorm2d(numFeatures: 64)
    nn.ReLU()
    nn.GlobalAvgPool2d()

    nn.Linear(64, 10)
}

// Create optimizer
var optimizer = Adam(for: model, learningRate: 0.001)

// Training loop
for epoch in 1...10 {
    for (images, labels) in trainLoader {
        let (loss, grads) = valueWithGradient(at: model) { m in
            let logits = m(images)
            return softmaxCrossEntropy(logits: logits, probabilities: labels)
        }

        optimizer.update(&model, along: grads)
        LazyTensorBarrier()  // Compile and execute
    }
}
```

---

## API Compatibility

Magma aims for PyTorch API compatibility where possible:

| Magma | PyTorch |
|-------------|---------|
| `nn.Linear` | `torch.nn.Linear` |
| `nn.Conv1d` | `torch.nn.Conv1d` |
| `nn.Conv2d` | `torch.nn.Conv2d` |
| `nn.ConvTranspose2d` | `torch.nn.ConvTranspose2d` |
| `nn.BatchNorm2d` | `torch.nn.BatchNorm2d` |
| `nn.LayerNorm` | `torch.nn.LayerNorm` |
| `nn.GroupNorm` | `torch.nn.GroupNorm` |
| `nn.InstanceNorm2d` | `torch.nn.InstanceNorm2d` |
| `nn.Dropout` | `torch.nn.Dropout` |
| `nn.Embedding` | `torch.nn.Embedding` |
| `nn.LSTM` | `torch.nn.LSTM` |
| `nn.Sequential` | `torch.nn.Sequential` |
| `Adam` | `torch.optim.Adam` |
| `SGD` | `torch.optim.SGD` |
| `RMSProp` | `torch.optim.RMSprop` |
| `AdaGrad` | `torch.optim.Adagrad` |
| `AdaDelta` | `torch.optim.Adadelta` |

---

## Heritage

Many APIs are ported from Swift for TensorFlow (S4TF):
- Initializers (Glorot, He, LeCun, Orthogonal)
- Loss functions (l1Loss, l2Loss, huberLoss, etc.)
- Additional optimizers (RMSProp, AdaGrad, AdaDelta)
- Activation layers (SELU, Mish, Softplus, Softsign, PReLU)

See [LEGACY_MAPPING.md](LEGACY_MAPPING.md) for detailed mapping from S4TF to Magma.
