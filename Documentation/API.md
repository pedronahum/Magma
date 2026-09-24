# Magma API Reference

This is a reference for Magma's public API as of `0.1.0-alpha.1`. It covers
tensors, execution and errors, the two layer APIs, losses, optimizers,
initializers, checkpoints and device placement. The API is alpha and may change.
The doc comments in `Sources/` are the authoritative reference; this page
summarizes them.

## Table of Contents

- [Modules and Imports](#modules-and-imports)
- [Tensor Creation](#tensor-creation)
- [Execution Model and Error Handling](#execution-model-and-error-handling)
- [Reproducibility (Seeding)](#reproducibility-seeding)
- [Device Placement](#device-placement)
- [Two Layer APIs](#two-layer-apis)
- [Value-Semantic API](#value-semantic-api)
- [Reference-Semantic API (`nn`)](#reference-semantic-api-nn)
  - [The Module Protocol](#the-module-protocol)
  - [Linear Layers](#linear-layers)
  - [Convolutional Layers](#convolutional-layers)
  - [Pooling Layers](#pooling-layers)
  - [Normalization Layers](#normalization-layers)
  - [Activation Layers](#activation-layers)
  - [Dropout and Train/Eval Mode](#dropout-and-traineval-mode)
  - [Embedding Layers](#embedding-layers)
  - [Recurrent Layers](#recurrent-layers)
  - [Attention and Transformer Layers](#attention-and-transformer-layers)
  - [Upsampling Layers](#upsampling-layers)
  - [Container Layers](#container-layers)
  - [Training `nn` Modules](#training-nn-modules)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Initializers](#initializers)
- [Checkpoints](#checkpoints)
- [Complete Example](#complete-example)

---

## Modules and Imports

```swift
import Magma
```

`import Magma` is all you need. It re-exports the layers it is built on —
`LazyTensor` (`LazyTensorBarrier`, `LazyTensorBarrierThrowing`,
`MaterializationError`), `XLARuntime` (`Device`, `Backend`, `PJRTClient`),
`StableHLO` (`DType`, `DeviceMesh`) — and Swift's `_Differentiation` module
(`@differentiable`, `gradient(at:)`, `valueWithGradient(at:)`).

## Tensor Creation

### Basic Tensor Construction

```swift
// From an array with an explicit shape (the element count must match the shape)
let tensor = Tensor<Float>([1, 2, 3, 4, 5, 6], shape: [2, 3], on: .default)

// Properties
tensor.shape         // [2, 3]
tensor.rank          // 2
tensor.elementCount  // 6
tensor.dtype         // .float32
tensor.device        // CPU:0
```

`Tensor(_:shape:)` checks that the number of values equals the product of
`shape` and that no dimension is negative. The factories below reject negative
sizes too.

### Factory Methods

```swift
let zeros  = Tensor<Float>.zeros([32, 784])
let ones   = Tensor<Float>.ones([32, 784])
let filled = Tensor<Float>.full([32, 784], 0.5)
let range  = Tensor<Float>.arange(10)                          // [0, 1, ..., 9]

// Random tensors (generated on the host; seedable, see below)
let normal   = Tensor<Float>.randn([32, 784])
let uniform  = Tensor<Float>.uniform(low: -1, high: 1, shape: [32, 784])   // requires low < high
let uniform2 = Tensor<Float>.randomUniform([32, 784], lowerBound: 0, upperBound: 1)
let trunc    = Tensor<Float>.truncatedNormal([32, 784], mean: 0.0, stddev: 0.02)

// Device-side RNG (not seedable; avoids embedding large constants in the graph)
let big = Tensor<Float>.randn([4096, 4096], useDeviceRNG: true)   // same as randnDevice(_:)

// One-hot (indices are Float tensors)
let labels = Tensor<Float>([0, 2, 1], shape: [3])
let onehot = Tensor<Float>.oneHot(labels, numClasses: 3)         // [3, 3]
```

### Type Aliases

```swift
typealias FloatTensor = Tensor<Float>
typealias DoubleTensor = Tensor<Double>
typealias Int32Tensor = Tensor<Int32>
typealias Int64Tensor = Tensor<Int64>
typealias BoolTensor = Tensor<Bool>
```

Use `to(_:)` with a `DType` to convert a tensor to another element type, e.g.
`x.to(.bfloat16)`. `toReducedPrecision()` and `toFullPrecision()` convert to
bfloat16 and back.

---

## Execution Model and Error Handling

Tensor operations are **lazy**. Each one adds a node to a graph, and nothing
runs until a barrier compiles the graph to StableHLO and executes it on a PJRT
backend.

- Reading values (`scalars()`, `item()`) or calling `materialize()` marks that
  tensor and runs a barrier for it.
- `LazyTensorBarrier(on:)` runs only the tensors **marked** with
  `markForMaterialization()` since the last barrier on that device, plus
  everything they depend on. Creating a tensor does not mark it, so a barrier
  with nothing marked does nothing. The next barrier on the device consumes the
  marks, whether it succeeds or fails.

```swift

let y = x.matmul(w).relu()
let z = y.sum()
y.markForMaterialization()
z.markForMaterialization()
LazyTensorBarrier()        // one compiled program computes y and z
print(z.item())            // already materialized: just a copy to the host
```

In a loop that carries state across iterations, cut the graph once per
iteration. Mark the state and call `LazyTensorBarrier()`, or call
`.materialize()` on it. Otherwise the traced graph keeps growing.

### Errors

If a tensor cannot be computed, `scalars()`, `item()` and `materialize()` **stop
the program** with the underlying error. That happens, for example, when no PJRT
plugin is installed or the backend rejects the program. Before this alpha they
returned `[]`. Use the throwing counterparts to handle the error:

| Non-throwing | Throwing |
|--------------|----------|
| `scalars() -> [Scalar]` | `fetchScalars() throws -> [Scalar]` |
| `item() -> Scalar` | `fetchItem() throws -> Scalar` |
| `LazyTensorBarrier(on:)`: prints to stderr and records the error on the tensors | `LazyTensorBarrierThrowing(on:) throws` |

```swift
import Magma

do {
    let values = try logits.softmax(dim: -1).fetchScalars()
    print(values)
} catch let error as MaterializationError {
    switch error.stage {
    case .backendUnavailable: print("install a PJRT plugin: \(error)")
    case .compilation:        print("compile failed; MLIR at \(error.mlirDumpPath ?? "-")")
    default:                  print(error)
    }
}
```

`MaterializationError` has these properties:

- `stage`: one of `backendUnavailable`, `validation`, `compilation`,
  `inputTransfer`, `execution`, `outputTransfer`, `notMaterialized`
- `device`
- `underlying`: the runtime error, including PJRT's own message
- `mlirDumpPath`

The failing MLIR is written only when `MAGMA_DEBUG=1` (system temp directory) or
`MAGMA_DEBUG_DIR=<dir>` is set.

**Threads.** It is safe to read tensors from several threads or tasks. Barriers
are serialized process-wide, so graphs do not execute in parallel. If a batched
barrier fails because of another tensor, a read retries its own tensor alone, so
the error it reports is always about that tensor. `Parameter` values are not
synchronized, so a model must not be updated from several threads at once.

---

## Reproducibility (Seeding)

```swift
manualSeed(42)                                   // seed the global host stream
let layer = nn.Linear(inputSize: 4, outputSize: 2)   // same weights every run

// A private stream for one task (and the child tasks it creates); the global
// stream is left untouched, so this is reproducible even under concurrency.
let a = withManualSeed(7) { Tensor<Float>.randn([3]).scalars() }
let b = withManualSeed(7) { Tensor<Float>.randn([3]).scalars() }
// a == b
```

The seed controls every **host-side** random draw:

- `randn` and `uniform` / `randomUniform`
- `truncatedNormal` and the Glorot, He and LeCun initializers
- `nn.Dropout` masks, including the dropout inside attention, Transformer and
  recurrent layers
- shuffling in `DataLoader`, `SimpleBatchLoader` and the token batch iterators
- the random transforms

It does **not** control device-side RNG ops: `randnDevice`, `randDevice`,
`randn(_:useDeviceRNG: true)`, and the noise in `multinomial`.
`DistributedSampler` takes its own `seed:`. Inside a `withManualSeed` scope,
`manualSeed(_:)` reseeds that scope's stream.

---

## Device Placement

```swift

let gpu = Device(backend: .gpu, index: 0)

let x = Tensor<Float>.randn([8, 16], on: gpu)   // create on a device
let y = x.to(device: .default)                   // move (keeps the element type)

var model = nn.sequential {
    nn.Linear(inputSize: 16, outputSize: 32)
    nn.BatchNorm1d(numFeatures: 32)
}
model.to(device: gpu)          // moves every parameter *and* buffer (e.g. running stats)
```

The backend a graph runs on is chosen when it executes:

1. `MAGMA_DEFAULT_BACKEND` (`cpu`, `gpu`, `tpu`, `metal`), if that backend's
   plugin is available.
2. Otherwise the requested backend, if its plugin is available.
3. Otherwise the best available backend.

`Backend.availableBackends`, `Backend.bestAvailable` and `Backend.x.isAvailable`
report which backends are installed. `Backend.x.resolvedPluginPath` gives the
plugin file that would be loaded. See the README for the search order.

---

## Two Layer APIs

Magma has two layer APIs that share some names.

| | Value-semantic | Reference-semantic |
|--|----------------|--------------------|
| Names | `Linear`, `ReLU`, `Sigmoid`, `Conv2d`, `sequential { }` | `nn.Linear`, `nn.Conv2d`, …, `nn.sequential { }` |
| Weights | stored `Tensor<Float>` values | `Parameter` reference cells |
| `Linear.weight` shape | `[in, out]`, computes `x · W + b` | `[out, in]`, computes `x · Wᵀ + b` (PyTorch) |
| Gradients | whole-model autodiff: `modelGradient(of:input:target:lossFn:)` | `parameterGradients(of:loss:)` over parameter values |
| Optimizers | `Adam(learningRate:)`, `MomentumSGD`, `sgdUpdate` → `update(&model, gradient:)` | `optim.SGD/Adam/…(parameters:lr:)` → `step(_:)` |

Unqualified names always mean the value-semantic types.

---

## Value-Semantic API

A `Layer` is a `Differentiable` and `KeyPathIterable` value type. Its forward
pass is `@differentiable(reverse)`.

```swift
public protocol Layer: Differentiable, KeyPathIterable {
    @differentiable(reverse)
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float>
}
```

Available layers:

- `Linear(weight:bias:)`, with `weight` shaped `[in, out]` and `bias` shaped `[out]`
- `ReLU()` and `Sigmoid()`
- `Conv2d(weight:bias:strides:padding:)`, which takes NHWC input and an HWIO
  weight `[kH, kW, Cin, Cout]`

`sequential { ... }` chains layers into a nested `Sequential2` that is itself
`Differentiable`.

```swift
import Magma

func makeModel() -> some Layer {
    sequential {
        Linear(weight: Tensor<Float>.glorotUniform([4, 16]), bias: Tensor<Float>.zeros([16]))
        ReLU()
        Linear(weight: Tensor<Float>.glorotUniform([16, 1]), bias: Tensor<Float>.zeros([1]))
    }
}

let mse: @differentiable(reverse) (Tensor<Float>, Tensor<Float>) -> Tensor<Float> = { pred, target in
    let r = pred - target
    return (r * r).mean()
}

var model = makeModel()
var optimizer = Adam(learningRate: 0.01)   // beta1:, beta2:, epsilon: also available

for _ in 0..<100 {
    let grad = modelGradient(of: model, input: x, target: y, lossFn: mse)
    optimizer.update(&model, gradient: grad)
}
let (loss, _) = modelValueWithGradient(of: model, input: x, target: y, lossFn: mse)
print(loss.item())
```

`Adam`, `MomentumSGD(learningRate:momentum:)` and
`sgdUpdate(_:gradient:learningRate:)` find every `Tensor<Float>` in the model by
reflection. `Adam` and `MomentumSGD` materialize the updated weights on every
step, so the graph does not grow across iterations. When the model is held as
`some Layer`, differentiate it through `modelGradient` /
`modelValueWithGradient` rather than calling `gradient(at:)` directly. See
[KNOWN_COMPILER_ISSUES.md](KNOWN_COMPILER_ISSUES.md) for why.

---

## Reference-Semantic API (`nn`)

### The Module Protocol

```swift
public protocol Module {
    associatedtype Input
    associatedtype Output

    func forward(_ input: Input) -> Output
    func parameters() -> [Parameter]          // default: []
    func buffers() -> [Parameter]             // non-trained state, e.g. BatchNorm running stats; default: []
    mutating func to(device: Device)          // default: moves parameters() + buffers()
    mutating func setTraining(_ training: Bool)   // default: no-op
}

// Provided by an extension:
//   callAsFunction(_:)  -> forward(_:)
//   train() / eval()    -> setTraining(true / false)
```

`Module` is **not** `Differentiable`. A `Parameter` is a reference cell holding
`value: Tensor<Float>`, plus `requiresGrad` and `name`. `Parameter` is `Hashable`
by identity, so it can key a gradient dictionary. Copying a module shares its
`Parameter`s.

Most `nn` layers take `Tensor<Float>` input and output. Images use **NHWC**
layout: `[batch, height, width, channels]`.

### Linear Layers

#### nn.Linear

```swift
let linear = nn.Linear(inputSize: 784, outputSize: 256, bias: true)

let input = Tensor<Float>.zeros([32, 784])
let output = linear(input)          // [32, 256]; inputs of rank > 2 keep their leading dims

linear.weight.shape                 // [256, 784]  ([out, in])
linear.bias.shape                   // [256]
linear.parameters().count           // 2 (1 if bias: false)
```

### Convolutional Layers

#### nn.Conv1d

1D convolution in NLC format: `[batch, length, channels]`. The weight is
`[kernelSize, inChannels, outChannels]`.

```swift
let conv1d = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3, stride: 1, padding: 0, bias: true)

let input = Tensor<Float>.zeros([32, 100, 3])
let output = conv1d(input)  // [32, 98, 64]
// Output length = (input + 2*padding - kernel) / stride + 1
```

#### nn.Conv2d

2D convolution in NHWC format. The weight is `[kH, kW, inChannels, outChannels]`.

```swift
let conv2d = nn.Conv2d(
    inChannels: 3,
    outChannels: 64,
    kernelSize: 3,        // or (3, 3)
    stride: 1,            // or (1, 1)
    padding: 0,           // or (0, 0)
    bias: true
)

let input = Tensor<Float>.zeros([32, 224, 224, 3])
let output = conv2d(input)  // [32, 222, 222, 64]
```

The functional forms are `Tensor.conv2d(_:strides:padding:)` and
`Tensor.convTranspose2d(_:strides:padding:outputPadding:)`. Both are
differentiable.

#### nn.ConvTranspose2d

Transposed 2D convolution, used for upsampling. The weight is
`[kH, kW, outChannels, inChannels]`. `outputPadding` must be smaller than
`stride`.

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

let input = Tensor<Float>.zeros([16, 14, 14, 64])
let output = convT(input)  // [16, 28, 28, 32]
// Output = (input - 1) * stride - 2 * padding + kernel + outputPadding
```

### Pooling Layers

```swift
let maxPool = nn.MaxPool2d(kernelSize: 2, stride: 2)          // stride defaults to kernelSize
let avgPool = nn.AvgPool2d(kernelSize: 2, stride: 2, padding: 0)
let adaptive = nn.AdaptiveAvgPool2d(outputSize: 1)            // or (h, w)

let input = Tensor<Float>.zeros([32, 28, 28, 64])
let output = maxPool(input)  // [32, 14, 14, 64]

// Global pooling: [batch, length, C] -> [batch, C] and [batch, H, W, C] -> [batch, C]
let globalAvg1d = nn.GlobalAvgPool1d()
let globalAvg2d = nn.GlobalAvgPool2d()
let globalMax1d = nn.GlobalMaxPool1d()
let globalMax2d = nn.GlobalMaxPool2d()
```

### Normalization Layers

```swift
// Batch norm (NHWC for 2d; [batch, features] for 1d). Running stats are buffers().
let bn2d = nn.BatchNorm2d(numFeatures: 64, eps: 1e-5, momentum: 0.1)
let bn1d = nn.BatchNorm1d(numFeatures: 256)

// Layer norm over the trailing dimensions
let layerNorm = nn.LayerNorm(normalizedShape: [256], eps: 1e-5)   // or nn.LayerNorm(256)

// Group norm: channels split into groups (numChannels % numGroups == 0)
let groupNorm = nn.GroupNorm(numGroups: 8, numChannels: 64, eps: 1e-5, affine: true)

// Instance norm (affine defaults to false, as in PyTorch)
let instanceNorm = nn.InstanceNorm2d(numFeatures: 64, eps: 1e-5, affine: true)

let x = Tensor<Float>.zeros([32, 14, 14, 64])
let y = groupNorm(x)  // [32, 14, 14, 64]
```

BatchNorm uses batch statistics in training mode and running statistics after
`eval()`.

### Activation Layers

```swift
let relu = nn.ReLU()
let leakyRelu = nn.LeakyReLU(negativeSlope: 0.01)
let elu = nn.ELU(alpha: 1.0)
let silu = nn.SiLU()
let hardtanh = nn.Hardtanh(minVal: -1, maxVal: 1)
let sigmoid = nn.Sigmoid()
let tanh = nn.Tanh()
let gelu = nn.GELU()

// Ported from S4TF
let selu = nn.SELU()
let mish = nn.Mish()
let softplus = nn.Softplus(beta: 1.0, threshold: 20.0)
let softsign = nn.Softsign()
let prelu = nn.PReLU(numParameters: 1, init: 0.25)   // learnable slope: 1 parameter
```

There is no `nn.Softmax` layer. Use `x.softmax(dim: -1)` /
`x.logSoftmax(dim: -1)` or `nn.functional.softmax(x, dim: -1)`. The activations
are also available as `Tensor` methods: `relu()`, `sigmoid()`, `tanh()`,
`gelu()`, `silu()`, `elu(alpha:)`, `leakyRelu(negativeSlope:)`, `selu()`,
`mish()`, and so on.

### Dropout and Train/Eval Mode

```swift
var dropout = nn.Dropout(p: 0.5)     // p must be in [0, 1)
let y = dropout(x)                   // training (default): zeros ~p of the values, scales the rest by 1/(1-p)
dropout.eval()
let z = dropout(x)                   // inference: returns x unchanged
```

Layers with a train/eval mode start in **training** mode, as in PyTorch:

- `nn.Dropout`
- `nn.BatchNorm1d/2d`
- `nn.MultiheadAttention`
- `nn.TransformerEncoderLayer` and `nn.TransformerDecoderLayer`
- `nn.SinusoidalPositionalEncoding`
- `nn.RNN`, `nn.LSTM`, `nn.GRU`

The Transformer layers and `SinusoidalPositionalEncoding` default to
`dropout: 0.1`, so **call `eval()` for deterministic inference**. Containers
such as `nn.Sequential` pass `train()` / `eval()` on to their children.
`nn.scaledDotProductAttention` has no mode: it applies dropout whenever
`dropout > 0`.

### Embedding Layers

```swift
let embedding = nn.Embedding(numEmbeddings: 10000, embeddingDim: 256)

// Indices are passed as a Float tensor (Input == Tensor<Float>)
let indices = Tensor<Float>([1, 5, 3, 8], shape: [4])
let embeddings = embedding(indices)  // [4, 256]; [batch, seq] -> [batch, seq, 256]
```

`nn.LearnedPositionalEmbedding` and `nn.SinusoidalPositionalEncoding(dModel:maxLen:dropout:)`
provide positional information for sequence models.

### Recurrent Layers

```swift
let lstm = nn.LSTM(
    inputSize: 256,
    hiddenSize: 512,
    numLayers: 2,
    bias: true,
    batchFirst: true,     // input [batch, seq, features]
    dropout: 0.1,         // between layers, training mode only
    bidirectional: false
)

let input = Tensor<Float>.zeros([32, 50, 256])
let (output, (hn, cn)) = lstm(input)
// output: [32, 50, 512]  ([32, 50, 1024] when bidirectional)
// hn, cn: final states for each layer and direction
```

`nn.GRU` and `nn.RNN` (with `nonlinearity: .tanh` or `.relu`) take the same
arguments and return `(output, hn)`. Pass an initial state with
`forward(_:hidden:)`. The single-step cells are `nn.RNNCell`, `nn.LSTMCell` and
`nn.GRUCell`. For example, `forward(x:hidden:cell:)` on `LSTMCell` returns
`(h, c)`.

### Attention and Transformer Layers

```swift
let mha = nn.MultiheadAttention(embedDim: 256, numHeads: 8, dropout: 0.1)
let selfAttended = mha(x)                                            // [batch, seq, 256]
let (out, weights) = mha.forward(query: q, key: k, value: v, mask: nil)

var encoder = nn.TransformerEncoderLayer(dModel: 256, nHead: 8)     // dropout: 0.1 by default
encoder.eval()
let encoded = encoder.forward(src: x, srcMask: nil)

var decoder = nn.TransformerDecoderLayer(dModel: 256, nHead: 8)
decoder.eval()
let causal = nn.generateCausalMask(size: 50)
let decoded = decoder.forward(tgt: t, memory: encoded, tgtMask: causal, memoryMask: nil)

let (attn, probs) = nn.scaledDotProductAttention(query: q, key: k, value: v, mask: nil, dropout: 0)
```

### Upsampling Layers

```swift
let upsample1d = nn.Upsample1d(size: 2)        // factor 2: [32, 50, 64] -> [32, 100, 64]
let upsample2d = nn.Upsample2d(size: 2)        // or nn.Upsample2d(size: (2, 3))
let alias: nn.UpsamplingNearest2d = nn.Upsample2d(size: 2)   // PyTorch-compatible name
```

### Container Layers

#### nn.Sequential

```swift
var model = nn.sequential {
    nn.Linear(inputSize: 784, outputSize: 256)
    nn.ReLU()
    nn.Dropout(p: 0.1)
    nn.Linear(inputSize: 256, outputSize: 10)
}

let output = model(Tensor<Float>.zeros([32, 784]))  // [32, 10]
model.eval()

// Equivalent without the builder:
var model2 = nn.Sequential(nn.AnyLayer(nn.Linear(inputSize: 784, outputSize: 10)))
model2.add(nn.AnyLayer(nn.ReLU()))
```

`nn.Sequential` takes layers whose input and output are `Tensor<Float>`. It
passes `parameters()`, `buffers()`, `setTraining(_:)` and `to(device:)` on to
its children.

### Training `nn` Modules

An `nn.Module` is not `Differentiable`, so you cannot take `gradient(at: model)`.
Use `parameterGradients(of:loss:)` instead. The closure receives the parameters'
**values** in order and must compute the loss from those values. The result is
keyed by `Parameter`, and `step(_:)` applies it by identity.

```swift
import Magma

let fc = nn.Linear(inputSize: 2, outputSize: 1)         // parameters(): [weight, bias]
var opt = optim.Adam(parameters: fc.parameters(), lr: 0.1)
let n = Tensor<Float>.full([], Float(batch))

for _ in 0..<100 {
    let (loss, grads) = parameterGradients(of: fc.parameters()) { p in
        let pred = x.matmul(p[0].transpose()) + p[1].broadcast(to: [batch, 1])
        let r = pred - y
        return (r * r).sum() / n
    }
    opt.step(grads)                                     // [Parameter: Tensor] overload

    // Cut the lazy graph: the optimizer updates `param.value` lazily.
    for p in fc.parameters() { p.value.markForMaterialization() }
    LazyTensorBarrier()
}
```

If the closure calls `fc(x)` instead of using `p`, its gradients are zero. The
bridge gives zero gradients to any parameter the loss does not use, and sums the
gradients of a parameter that appears more than once (tied weights). To
differentiate a whole model without this bookkeeping, use the
[value-semantic API](#value-semantic-api).

---

## Loss Functions

`nn.functional` holds PyTorch-style losses and activations:

```swift
nn.functional.mse(pred, target)
nn.functional.crossEntropy(logits, targets)   // targets: class indices [batch], or one-hot / probabilities [batch, C]
nn.functional.nllLoss(logProbs, targets)
nn.functional.binaryCrossEntropy(probs, target)
nn.functional.binaryCrossEntropyWithLogits(logits, target)  // numerically stable; @differentiable(reverse, wrt: logits)
nn.functional.softmax(x, dim: -1)
nn.functional.logSoftmax(x, dim: -1)
```

The losses ported from S4TF are free functions. Most take a
`reduction: LossReduction` argument (`.mean`, `.sum`, `.none`), and the **default
differs per function**. `l1Loss`, `l2Loss`, `huberLoss` and
`kullbackLeiblerDivergence` default to `.sum`; most others default to `.mean`.

```swift
// Regression
l1Loss(predicted: pred, expected: target, reduction: .mean)
l2Loss(predicted: pred, expected: target, reduction: .mean)
meanAbsoluteError(predicted: pred, expected: target)
meanSquaredError(predicted: pred, expected: target)
meanSquaredLogarithmicError(predicted: pred, expected: target)
meanAbsolutePercentageError(predicted: pred, expected: target)
huberLoss(predicted: pred, expected: target, delta: 1.0, reduction: .mean)
logCoshLoss(predicted: pred, expected: target)
poissonLoss(predicted: pred, expected: target)

// Classification
softmaxCrossEntropy(logits: logits, probabilities: oneHotLabels)
softmaxCrossEntropyWithLabels(logits: logits, labels: [0, 1, 2], numClasses: 10)  // labels: [Int]
sigmoidCrossEntropy(logits: logits, labels: binaryLabels)
hingeLoss(predicted: pred, expected: target)
squaredHingeLoss(predicted: pred, expected: target)
categoricalHingeLoss(predicted: pred, expected: oneHotTarget)

// Distributions
kullbackLeiblerDivergence(predicted: pred, expected: target)

// Metric learning
cosineSimilarity(a, b, epsilon: 1e-8)
cosineDistance(predicted: pred, expected: target)
contrastiveLoss(anchor: anchor, sample: sample, labels: labels, margin: 1.0)  // labels: 1 similar, 0 dissimilar
tripletMarginLoss(anchor: anchor, positive: positive, negative: negative, margin: 1.0)
```

> **Differentiating a loss.** Every loss above, and the `nn.functional`
> losses and activations, can be used inside a closure you differentiate (with
> `gradient(at:)`, `modelGradient` or `parameterGradients`). Only the
> predictions/logits are differentiated; targets and labels are treated as
> constants.

---

## Optimizers

### Reference-semantic optimizers (`optim`)

All of them conform to `Optimizer`, which requires `parameters`,
`learningRate`, `step(_ gradients: [Tensor<Float>])` (positional) and
`resetState()`. The protocol extension adds the **identity-keyed**
`step(_ gradients: [Parameter: Tensor<Float>])`, which is the recommended way
to apply gradients from `parameterGradients`. `zeroGrad()` is deprecated: it
resets the optimizer state and does not clear gradients.

```swift
let params = model.parameters()

var sgd      = optim.SGD(parameters: params, lr: 0.01, momentum: 0.9, weightDecay: 1e-4, nesterov: true)
var adam     = optim.Adam(parameters: params, lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0)
var adamw    = optim.AdamW(parameters: params, lr: 0.001, weightDecay: 0.01)   // alias of optim.Adam (decoupled decay)
var rmsprop  = optim.RMSProp(parameters: params, lr: 0.01, rho: 0.99, eps: 1e-8, weightDecay: 0, momentum: 0, centered: false)
var adagrad  = optim.AdaGrad(parameters: params, lr: 0.01, initialAccumulatorValue: 0, eps: 1e-10, weightDecay: 0)
var adadelta = optim.AdaDelta(parameters: params, lr: 1.0, rho: 0.9, eps: 1e-6, weightDecay: 0)
// optim.AdamWGroups: per-group hyperparameters via ParameterGroup

adam.step(grads)              // grads: [Parameter: Tensor<Float>]
adam.learningRate = 1e-4
```

Gradient utilities:

- `optim.clipGradNorm(_:maxNorm:)` returns `(clipped, totalNorm)`.
- `optim.clipGradValue`
- `optim.GradientAccumulator(accumulationSteps:)`

### Learning-rate schedulers

Schedulers conform to `LRScheduler`, which provides `step()`, `currentLR` and
`reset()`. The constructors validate their arguments.

```swift
var sched = optim.StepLR(baseLR: 0.1, stepSize: 10, gamma: 0.1)      // stepSize > 0
// optim.ExponentialLR(baseLR:gamma:)
// optim.CosineAnnealingLR(baseLR:totalEpochs:minLR:)                // totalEpochs > 0
// optim.WarmupLR(baseLR:warmupSteps:)                                // warmupSteps >= 0
// optim.WarmupCosineScheduler(baseLR:warmupSteps:totalSteps:minLR:) // totalSteps > warmupSteps

sched.step()
opt.learningRate = sched.currentLR
```

### Value-semantic optimizers

`Adam(learningRate:beta1:beta2:epsilon:)`, `MomentumSGD(learningRate:momentum:)`
and `sgdUpdate(_:gradient:learningRate:)` update a `Differentiable &
KeyPathIterable` model from its `TangentVector`. Call them as
`optimizer.update(&model, gradient: grad)`. See
[Value-Semantic API](#value-semantic-api).

---

## Initializers

All initializers are static methods on `Tensor<Float>` and take an optional
`on: device`. They draw from the seedable host stream.

```swift
// Uniform in [lowerBound, upperBound)  (defaults 0 and 1)
Tensor<Float>.randomUniform([256, 128])
Tensor<Float>.randomUniform([256, 128], lowerBound: -1.0, upperBound: 1.0)

// Normal / truncated normal (resampled beyond 2 standard deviations)
Tensor<Float>.randn([256, 128])
Tensor<Float>.truncatedNormal([256, 128], mean: 0.0, stddev: 0.02)

// Glorot/Xavier: limit = sqrt(6 / (fan_in + fan_out)), std = sqrt(2 / (fan_in + fan_out))
Tensor<Float>.glorotUniform([256, 128])
Tensor<Float>.glorotNormal([256, 128])

// He/Kaiming: limit = sqrt(6 / fan_in), std = sqrt(2 / fan_in)
Tensor<Float>.heUniform([256, 128])
Tensor<Float>.heNormal([256, 128])

// LeCun: limit = sqrt(3 / fan_in), std = sqrt(1 / fan_in)
Tensor<Float>.lecunUniform([256, 128])
Tensor<Float>.lecunNormal([256, 128])

// Orthogonal (at least 2 dimensions), optional gain
Tensor<Float>.orthogonal([256, 256], gain: 1.41421)

// Constants
Tensor<Float>.zeros([256, 128])
Tensor<Float>.ones([256, 128])
Tensor<Float>.full([256, 128], 0.01)
```

**Fan calculation.** For a convolution weight
`[kernelH, kernelW, inChannels, outChannels]`:

- `fan_in = kernelH * kernelW * inChannels`
- `fan_out = kernelH * kernelW * outChannels`

```swift
let convWeight = Tensor<Float>.heUniform([3, 3, 64, 128])   // fan_in = 576, fan_out = 1152
```

---

## Checkpoints

Checkpoints store an `nn` module's `parameters()` and, from format version 2,
its `buffers()`, such as BatchNorm running statistics. Version-1 files still
load; they leave the buffers untouched. A load validates the whole file before
assigning anything. A truncated or corrupt file, or a shape mismatch, throws
`CheckpointError` and leaves the module unchanged. Loaded tensors stay on the
device of the parameter they replace.

```swift
import Foundation
import Magma

var model = nn.sequential {
    nn.Linear(inputSize: 784, outputSize: 128)
    nn.BatchNorm1d(numFeatures: 128)
    nn.Linear(inputSize: 128, outputSize: 10)
}

// JSON (human-readable)
try model.save(to: URL(fileURLWithPath: "model.json"), description: "epoch 5")
try model.load(from: URL(fileURLWithPath: "model.json"))

// Binary (compact)
try BinaryCheckpoint.save(model, to: URL(fileURLWithPath: "model.bin"))
try BinaryCheckpoint.load(&model, from: URL(fileURLWithPath: "model.bin"))

// PyTorch-style state dict
let state: StateDict = model.stateDict()          // [String: Tensor<Float>]
try model.loadStateDict(state, strict: true)
```

`TrainingState` bundles the model state with the step, epoch and loss, and has
its own `save(to:)` and `load(from:)`.

---

## Complete Example

This example trains a small classifier end to end with the value-semantic API,
then evaluates it without recording gradients. It uses a hand-written
cross-entropy built from differentiable primitives.

```swift
import Magma

manualSeed(0)

// Toy data: 256 samples, 20 features, 3 classes (replace with a real loader).
let batch = 256, features = 20, classes = 3
let x = Tensor<Float>.randn([batch, features])
let labelValues = (0..<batch).map { Float($0 % classes) }
let y = Tensor<Float>.oneHot(Tensor<Float>(labelValues, shape: [batch]), numClasses: classes)

func makeClassifier() -> some Layer {
    sequential {
        Linear(weight: Tensor<Float>.heUniform([features, 64]), bias: Tensor<Float>.zeros([64]))
        ReLU()
        Linear(weight: Tensor<Float>.glorotUniform([64, classes]), bias: Tensor<Float>.zeros([classes]))
    }
}

let n = Tensor<Float>.full([], Float(batch))
let eps = Tensor<Float>.full([], 1e-7)
let crossEntropy: @differentiable(reverse) (Tensor<Float>, Tensor<Float>) -> Tensor<Float> = { logits, target in
    -(target * (logits.softmax(dim: 1) + eps).log()).sum() / n
}

var model = makeClassifier()
var optimizer = Adam(learningRate: 0.01)

for epoch in 1...50 {
    let (loss, grad) = modelValueWithGradient(of: model, input: x, target: y, lossFn: crossEntropy)
    optimizer.update(&model, gradient: grad)
    if epoch % 10 == 0 {
        do {
            print("epoch \(epoch): loss \(try loss.fetchItem())")
        } catch {
            print("execution failed: \(error)")   // e.g. no PJRT plugin installed
            break
        }
    }
}
```

For an `nn.*` model, use the loop in [Training `nn` Modules](#training-nn-modules).

---

## API Compatibility

Magma aims for PyTorch API compatibility where possible:

| Magma | PyTorch |
|-------------|---------|
| `nn.Linear` | `torch.nn.Linear` |
| `nn.Conv1d` | `torch.nn.Conv1d` |
| `nn.Conv2d` | `torch.nn.Conv2d` (Magma uses NHWC) |
| `nn.ConvTranspose2d` | `torch.nn.ConvTranspose2d` |
| `nn.BatchNorm2d` | `torch.nn.BatchNorm2d` |
| `nn.LayerNorm` | `torch.nn.LayerNorm` |
| `nn.GroupNorm` | `torch.nn.GroupNorm` |
| `nn.InstanceNorm2d` | `torch.nn.InstanceNorm2d` |
| `nn.Dropout` | `torch.nn.Dropout` |
| `nn.Embedding` | `torch.nn.Embedding` (Float indices) |
| `nn.LSTM` / `nn.GRU` | `torch.nn.LSTM` / `torch.nn.GRU` |
| `nn.MultiheadAttention` | `torch.nn.MultiheadAttention` |
| `nn.TransformerEncoderLayer` | `torch.nn.TransformerEncoderLayer` |
| `nn.Sequential` / `nn.sequential { }` | `torch.nn.Sequential` |
| `optim.SGD` | `torch.optim.SGD` |
| `optim.Adam` / `optim.AdamW` | `torch.optim.AdamW` (decoupled weight decay) |
| `optim.RMSProp` | `torch.optim.RMSprop` |
| `optim.AdaGrad` | `torch.optim.Adagrad` |
| `optim.AdaDelta` | `torch.optim.Adadelta` |
| `manualSeed(_:)` | `torch.manual_seed` |
| `module.eval()` / `module.train()` | `module.eval()` / `module.train()` |

---

## Heritage

Many APIs are ported from Swift for TensorFlow (S4TF):
- Initializers (Glorot, He, LeCun, Orthogonal)
- Loss functions (l1Loss, l2Loss, huberLoss, etc.)
- Additional optimizers (RMSProp, AdaGrad, AdaDelta)
- Activation layers (SELU, Mish, Softplus, Softsign, PReLU)

See [LEGACY_MAPPING.md](LEGACY_MAPPING.md) for detailed mapping from S4TF to Magma.
