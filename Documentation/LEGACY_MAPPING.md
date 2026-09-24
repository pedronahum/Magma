# Legacy Code Mapping

This document maps what Magma inherited from its predecessors, Swift for
TensorFlow (S4TF), TaylorTorch and SwiftIR, to where that code or design lives in
Magma today.

> **Historical note.** This file started as a migration plan. The plan assumed a
> `Legacy/TaylorTorch` and `Legacy/SwiftIR` checkout inside the repository and a
> `Sources/Torch/...` layout. Neither exists: the legacy repositories were never
> vendored, and the user-facing layer is `Sources/Core` (module `Magma`). The
> migration is complete. The sections below describe the current state, and the
> plan is summarized at the end for reference only.

## Current Layout (for orientation)

```
Sources/
├── CXLARuntime/   # C wrapper over the PJRT C API (plugin dlopen'd at runtime)
├── XLARuntime/    # XLARuntime.swift (PJRT client/buffer/executable), MetalHLORuntime.swift
├── StableHLO/     # Builder/ (MLIRBuilder, Value), Types/ (DType, TensorType), Sharding/ (Shardy)
├── LazyTensor/    # LazyTensor.swift, StableHLOEmitter.swift, Optimization/
└── Core/          # module `Magma`: Tensor, autodiff, nn.*, optim.*, value layers, data, ...
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full file tree.

---

## From SwiftIR

[SwiftIR](https://github.com/pedronahum/SwiftIR) provided the MLIR/XLA
infrastructure.

| SwiftIR component | Where it lives in Magma | Notes |
|-------------------|-------------------------|-------|
| PJRT C API header + simplified C wrapper | `Sources/CXLARuntime/` (`pjrt_c_api.h`, `PJRTSimpleWrapper.c/.h`, `PJRTProtoHelper.cpp/.h`) | The wrapper loads the plugin with `dlopen`. Extended with multi-device execute and SPMD/Shardy compile options |
| Swift PJRT client / buffer / executable layer (`SwiftIRXLA`) | `Sources/XLARuntime/XLARuntime.swift` | Magma's counterpart: `PJRTClient`, `PJRTDevice`, `PJRTBuffer` and `PJRTExecutable` in one file, plus multi-device buffer distribution |
| `JTracingContext` / `JTracer` (graph building, op semantics) | `Sources/StableHLO/Builder/MLIRBuilder.swift` | Reimplemented as a pure-Swift text builder (no C++ MLIR bindings) |
| `JTracerValue` | `Sources/StableHLO/Builder/Value.swift` | SSA `Value` |
| `jWhileLoop` / `jCond` | `MLIRBuilder.whileLoop` / `MLIRBuilder.cond`, plus `scan`/`scanXLA` in `Sources/Core/Scan.swift` | Emit `stablehlo.while` / `stablehlo.if` |
| `SwiftIRShardingLite` (`DeviceMesh`, `TensorSharding`) | `Sources/StableHLO/Sharding/DeviceMesh.swift`, `TensorSharding.swift` | Ported and adapted. Sub-axis (`AxisRef`/`SubAxisInfo`) support was not ported |

**Not reused from SwiftIR**: the C++ MLIR bindings (Magma generates MLIR text in
Swift), SwiftIRJupyter, the benchmark infrastructure, the native Shardy C API
(`SdyCAPIWrapper`) and the standalone `sdy_opt` runner. Magma enables the Shardy
partitioner inside XLA through compile options instead. See
[MULTI_DEVICE_ASSESSMENT.md §9](MULTI_DEVICE_ASSESSMENT.md). `jVmap`-style
automatic batching has no equivalent yet.

---

## From TaylorTorch

TaylorTorch contributed the **PyTorch-style API design**: a `Module` protocol with
`nn.*` layers, `optim.*` optimizers, and a result-builder `Sequential`. Magma
reimplemented this design on lazy tensors (LibTorch, ATen and the `TorchCpp`
bindings were not reused).

| TaylorTorch concept | Where it lives in Magma |
|---------------------|-------------------------|
| `Tensor` API surface | `Sources/Core/Tensor.swift` (backed by `LazyTensorHandle`, not a LibTorch handle) |
| `Module` protocol, layers (`Linear`, `Conv2d`, `BatchNorm`, `Dropout`, attention, ...) | `Sources/Core/Module.swift` (`nn.Linear`, `nn.Conv2d`, `nn.BatchNorm2d`, `nn.Dropout`, `nn.MultiheadAttention`, ...) |
| `Sequential` with a result builder | `nn.Sequential` / `nn.sequential { ... }` in `Sources/Core/Module.swift` |
| Optimizers | `Sources/Core/Optimizer.swift` (`optim.SGD`, `optim.Adam`, ...) |
| Training-loop structure (MNIST, etc.), used as a design reference | `Examples/MNIST/` |

---

## Ported from Swift for TensorFlow (S4TF)

The following components were ported from the
[S4TF swift-apis](https://github.com/tensorflow/swift-apis) repository, or modeled
on it.

### Design

| S4TF | Magma | Notes |
|------|-------|-------|
| x10 lazy tensors + `LazyTensorBarrier()` | `Sources/LazyTensor/` | Traced into StableHLO and run through PJRT |
| `Layer` protocol (value-semantic, `Differentiable`) | `Layer` in `Sources/Core/ValueLayers.swift` | Coexists with the PyTorch-style `nn.Module` API |
| `KeyPathIterable` | `Sources/Core/KeyPathIterable.swift` | Backed by reflection, not compiler synthesis |
| Optimizers that update from `TangentVector` | `Adam`, `MomentumSGD`, `sgdUpdate` in `Sources/Core/TangentOptimizer.swift` | Generic over any `Differentiable & KeyPathIterable` model |

### Initializers (`Sources/Core/Initializers.swift`)

| S4TF | Magma | Notes |
|------|-------------|-------|
| `glorotUniform(forShape:)` | `Tensor<Float>.glorotUniform(_:)` | Xavier uniform init |
| `glorotNormal(forShape:)` | `Tensor<Float>.glorotNormal(_:)` | Xavier normal init |
| `heUniform(forShape:)` | `Tensor<Float>.heUniform(_:)` | Kaiming uniform init |
| `heNormal(forShape:)` | `Tensor<Float>.heNormal(_:)` | Kaiming normal init |
| `lecunUniform(forShape:)` | `Tensor<Float>.lecunUniform(_:)` | LeCun uniform init |
| `lecunNormal(forShape:)` | `Tensor<Float>.lecunNormal(_:)` | LeCun normal init |
| `truncatedNormal(forShape:)` | `Tensor<Float>.truncatedNormal(_:)` | Truncated normal init |
| `orthogonal(forShape:)` | `Tensor<Float>.orthogonal(_:)` | Orthogonal init |

### Loss Functions (`Sources/Core/Loss.swift`)

| S4TF | Magma | Notes |
|------|-------------|-------|
| `l1Loss(predicted:expected:)` | `l1Loss(predicted:expected:reduction:)` | L1/MAE loss |
| `l2Loss(predicted:expected:)` | `l2Loss(predicted:expected:reduction:)` | L2/MSE loss |
| `meanAbsoluteError(predicted:expected:)` | `meanAbsoluteError(predicted:expected:)` | MAE |
| `meanSquaredError(predicted:expected:)` | `meanSquaredError(predicted:expected:)` | MSE |
| `hingeLoss(predicted:expected:)` | `hingeLoss(predicted:expected:reduction:)` | SVM-style |
| `squaredHingeLoss(predicted:expected:)` | `squaredHingeLoss(predicted:expected:reduction:)` | Squared hinge |
| `categoricalHingeLoss(predicted:expected:)` | `categoricalHingeLoss(predicted:expected:reduction:)` | Multi-class hinge |
| `huberLoss(predicted:expected:delta:)` | `huberLoss(predicted:expected:delta:reduction:)` | Robust to outliers |
| `logCoshLoss(predicted:expected:)` | `logCoshLoss(predicted:expected:reduction:)` | Smooth L1 |
| `poissonLoss(predicted:expected:)` | `poissonLoss(predicted:expected:reduction:)` | Poisson NLL |
| `kullbackLeiblerDivergence(predicted:expected:)` | `kullbackLeiblerDivergence(predicted:expected:reduction:)` | KL divergence |
| `softmaxCrossEntropy(logits:labels:)` | `softmaxCrossEntropy(logits:probabilities:reduction:)` | Multi-class CE |
| `sigmoidCrossEntropy(logits:labels:)` | `sigmoidCrossEntropy(logits:labels:reduction:)` | Binary CE |
| N/A | `cosineDistance(predicted:expected:reduction:)` | Cosine distance |
| N/A | `contrastiveLoss(anchor:sample:labels:margin:reduction:)` | Siamese networks |
| N/A | `tripletMarginLoss(anchor:positive:negative:margin:reduction:)` | Metric learning |

### Optimizers (`Sources/Core/Optimizer.swift`)

| S4TF | Magma | Notes |
|------|-------------|-------|
| `SGD` | `optim.SGD` | With momentum, weight decay, nesterov |
| `Adam` | `optim.Adam` | Adaptive moment estimation (`optim.AdamW` is an alias) |
| `RMSProp` | `optim.RMSProp` | Root mean square propagation |
| `AdaGrad` | `optim.AdaGrad` | Adaptive gradients |
| `AdaDelta` | `optim.AdaDelta` | No learning rate needed |

The unqualified `Adam` in `TangentOptimizer.swift` is a different type: the
value-semantic optimizer that updates a model from its `TangentVector`.

### Layers (`Sources/Core/Module.swift`)

| S4TF | Magma | Notes |
|------|-------------|-------|
| `Dense` | `nn.Linear` | Fully connected layer (value-semantic: `Linear`) |
| `Conv1D` | `nn.Conv1d` | 1D convolution |
| `Conv2D` | `nn.Conv2d` | 2D convolution (value-semantic: `Conv2d`) |
| `TransposedConv2D` | `nn.ConvTranspose2d` | Transposed 2D convolution |
| `MaxPool2D` | `nn.MaxPool2d` | Max pooling |
| `AvgPool2D` | `nn.AvgPool2d` | Average pooling |
| `GlobalAveragePooling1D` | `nn.GlobalAvgPool1d` | Global average pooling 1D |
| `GlobalAveragePooling2D` | `nn.GlobalAvgPool2d` | Global average pooling 2D |
| `GlobalMaxPooling1D` | `nn.GlobalMaxPool1d` | Global max pooling 1D |
| `GlobalMaxPooling2D` | `nn.GlobalMaxPool2d` | Global max pooling 2D |
| `UpSampling1D` | `nn.Upsample1d` | Nearest neighbor upsampling 1D |
| `UpSampling2D` | `nn.Upsample2d` | Nearest neighbor upsampling 2D |
| `BatchNorm` | `nn.BatchNorm2d` | Batch normalization |
| `LayerNorm` | `nn.LayerNorm` | Layer normalization |
| `GroupNorm` | `nn.GroupNorm` | Group normalization |
| N/A | `nn.InstanceNorm2d` | Instance normalization |
| `Dropout` | `nn.Dropout` | Dropout regularization |
| `Embedding` | `nn.Embedding` | Embedding lookup |
| N/A | `nn.SELU` | Self-normalizing ELU |
| N/A | `nn.Mish` | Mish activation |
| N/A | `nn.Softplus` | Softplus activation |
| N/A | `nn.Softsign` | Softsign activation |
| `PReLU` (in Activation.swift) | `nn.PReLU` | Parametric ReLU |

### Key Differences

1. **Tensor Format**: S4TF used NHWC (TensorFlow style). Magma's convolution and
   pooling layers also use NHWC.

2. **Reduction Parameter**: Magma's loss functions take an explicit `reduction`
   parameter (`.mean`, `.sum`, `.none`), as PyTorch does.

3. **Layer protocols**: S4TF had one `Layer` protocol. Magma has two:
   - `Layer` is value-semantic and S4TF-like (`ValueLayers.swift`).
   - `nn.Module` is reference-semantic and PyTorch-like (`Module.swift`).

4. **Device Handling**: tensor creation takes an explicit `on device:` parameter.

5. **Lazy Execution**: execution is lazy, with `LazyTensorBarrier()`, as in the S4TF
   x10 backend.

---

## Operation Mapping (Magma → StableHLO)

The `StableHLOEmitter` and `MLIRBuilder` lower Magma's tensor ops as follows:

| Magma op | StableHLO | Notes |
|----------|-----------|-------|
| `matmul` | `stablehlo.dot` | `batchedMatmul` uses `stablehlo.dot_general` |
| `+`, `-`, `*`, `/` | `stablehlo.add` / `subtract` / `multiply` / `divide` | Broadcasting via `broadcast_in_dim` |
| `relu()` | `stablehlo.maximum(x, 0)` | Composite |
| `sigmoid()` | `1 / (1 + exp(-x))` | Composite |
| `softmax()` | `exp(x - max) / sum(exp(x - max))` | Max-shifted for stability |
| `gelu()` | tanh approximation | Composite |
| `sum()` | `stablehlo.reduce` | With an add reducer |
| `mean()` | `reduce_sum / count` | Composite |
| `conv2d` | `stablehlo.convolution` | NHWC |
| `maxPool2d` / `avgPool2d` | `stablehlo.reduce_window` | |
| while loops / `cond` | `stablehlo.while` / `stablehlo.if` | |
| `crossReplicaSum` / `crossReplicaMean` | `stablehlo.all_reduce` | Multi-device (DDP) |

---

## API Examples

### PyTorch-style (`nn.*`)

```swift
// PyTorch
# model = nn.Sequential(
#     nn.Linear(784, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10)
# )

// Magma (reference-semantic)
let model = nn.sequential {
    nn.Linear(inputSize: 784, outputSize: 256)
    nn.ReLU()
    nn.Linear(inputSize: 256, outputSize: 10)
}
```

### S4TF-style (value-semantic)

```swift
// A typed, Differentiable model built from value layers
func makeMLP() -> some Layer {
    sequential {
        Linear(weight: w1, bias: b1)
        ReLU()
        Linear(weight: w2, bias: b2)
    }
}

var model = makeMLP()
var optimizer = Adam(learningRate: 0.01)
let grad = modelGradient(of: model, input: x, target: y, lossFn: mse)
optimizer.update(&model, gradient: grad)

// S4TF style (kept)
LazyTensorBarrier()
```

---

## Historical Migration Plan (complete, for reference only)

The original plan had five phases. All of them are done, though the file layout
differs from what the plan proposed:

1. **Setup**: the plan was to clone TaylorTorch and SwiftIR into `Legacy/`. This
   was never vendored into this repository.
2. **XLARuntime (from SwiftIR)**: done. The PJRT headers and wrapper are in
   `Sources/CXLARuntime`, and the Swift types are consolidated in
   `Sources/XLARuntime/XLARuntime.swift`.
3. **StableHLO (new, referencing SwiftIR)**: done. See `DType.swift`,
   `TensorType.swift`, `Value.swift`, `MLIRBuilder.swift` (including convolution,
   pooling via `reduce_window`, control flow and collectives) and `Sharding/`.
4. **LazyTensor (new, inspired by x10)**: done. The handles, IR, graph, barrier and
   compilation cache are in `LazyTensor.swift` rather than separate files.
5. **User layer (from TaylorTorch)**: done, as `Sources/Core` (module `Magma`)
   rather than `Sources/Torch`.
