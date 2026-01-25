# Legacy Code Mapping

This document maps components from TaylorTorch and SwiftIR to their roles in Magma.

## Repository Structure

```
Magma/
├── Legacy/
│   ├── TaylorTorch/          # Clone of https://github.com/pedronahum/TaylorTorch
│   └── SwiftIR/              # Clone of https://github.com/pedronahum/SwiftIR
└── Sources/
    └── ...                   # New implementation
```

---

## From TaylorTorch

### To Reuse (Copy & Adapt)

| TaylorTorch File | Magma Target | Changes Needed |
|------------------|-------------------|----------------|
| `Sources/Torch/Core/Tensor.swift` | `Sources/Torch/Tensor/Tensor.swift` | Replace LibTorch handle with LazyTensorHandle |
| `Sources/Torch/Modules/Layers/Linear.swift` | `Sources/Torch/NN/Layers/Linear.swift` | Update Tensor references |
| `Sources/Torch/Modules/Layers/Conv2d.swift` | `Sources/Torch/NN/Layers/Conv2d.swift` | Update Tensor references |
| `Sources/Torch/Modules/Layers/BatchNorm.swift` | `Sources/Torch/NN/Layers/BatchNorm.swift` | Update Tensor references |
| `Sources/Torch/Modules/Layers/Dropout.swift` | `Sources/Torch/NN/Layers/Dropout.swift` | Update for functional PRNG |
| `Sources/Torch/Modules/Layers/Attention.swift` | `Sources/Torch/NN/Layers/Attention.swift` | Update Tensor references |
| `Sources/Torch/Modules/Sequential.swift` | `Sources/Torch/NN/Sequential.swift` | Keep result builder |
| `Sources/Torch/Optimizers/*.swift` | `Sources/Torch/Optim/*.swift` | Update for new Tensor |

### To Reference (Study Design)

| TaylorTorch Component | What to Learn |
|-----------------------|---------------|
| `Module` protocol | How to structure `@differentiable` layers |
| `TangentVector` implementations | Custom tangent vectors for complex layers |
| Graph neural network layers | Advanced layer patterns |
| Examples (MNIST, etc.) | Training loop structure |

### Not to Reuse

| Component | Reason |
|-----------|--------|
| LibTorch C++ bindings | We're using XLA instead |
| `TorchCpp` module | Not needed |
| ATen tensor operations | Replaced by StableHLO |

---

## From SwiftIR

### To Reuse (Copy & Adapt)

| SwiftIR File | Magma Target | Changes Needed |
|--------------|-------------------|----------------|
| `Sources/SwiftIRXLA/PJRTClient.swift` | `Sources/XLARuntime/PJRTClient.swift` | Clean up API |
| `Sources/SwiftIRXLA/PJRTBuffer.swift` | `Sources/XLARuntime/PJRTBuffer.swift` | Simplify |
| `Sources/SwiftIRXLA/PJRTExecutable.swift` | `Sources/XLARuntime/PJRTExecutable.swift` | Simplify |
| PJRT C headers | `Sources/CXLARuntime/` | Copy directly |

### To Reference (Study Design)

| SwiftIR Component | What to Learn |
|-------------------|---------------|
| `JTracingContext` | How to build graphs |
| `JTracer` operations | StableHLO op semantics |
| Shape inference | How to compute output shapes |
| `jWhileLoop` | Control flow compilation |
| `jVmap` | Batching implementation |
| Profiler integration | TensorBoard support |

### To Port (Reimplementing)

| SwiftIR Component | Magma Equivalent | Notes |
|-------------------|------------------------|-------|
| `JTracerValue` | `Value` in StableHLO | Pure Swift, simpler |
| `JTracingContext.buildModule()` | `MLIRBuilder.build()` | String-based MLIR |
| Op implementations | StableHLO ops | Map 1:1 |

### Not to Reuse

| Component | Reason |
|-----------|--------|
| C++ MLIR bindings | Using pure Swift MLIR generation |
| SwiftIRJupyter | Simplified into StableHLO layer |
| Benchmark infrastructure | Will rebuild |

---

## Operation Mapping

### TaylorTorch → StableHLO

| TaylorTorch | StableHLO | Notes |
|-------------|-----------|-------|
| `Tensor.matmul` | `stablehlo.dot` | Direct mapping |
| `Tensor + Tensor` | `stablehlo.add` | Direct |
| `Tensor * Tensor` | `stablehlo.multiply` | Direct |
| `Tensor.relu()` | `stablehlo.maximum(x, 0)` | Composite |
| `Tensor.sigmoid()` | `1 / (1 + exp(-x))` | Composite |
| `Tensor.softmax()` | `exp(x) / sum(exp(x))` | Composite |
| `Tensor.sum()` | `stablehlo.reduce` | With add reducer |
| `Tensor.mean()` | `reduce_sum / count` | Composite |
| `Tensor.conv2d()` | `stablehlo.convolution` | Complex attributes |
| `Tensor.batchNorm()` | Multiple ops | Decomposed |

### SwiftIR → StableHLO

| SwiftIR | StableHLO Layer | Notes |
|---------|-----------------|-------|
| `JTracer +` | `MLIRBuilder.add()` | Same semantics |
| `SwiftIR.matmul()` | `MLIRBuilder.dot()` | Same |
| `SwiftIR.relu()` | `MLIRBuilder.relu()` | Same |
| `jWhileLoop` | `stablehlo.while` | Control flow |
| `jCond` | `stablehlo.if` | Control flow |
| `jVmap` | Manual batching | Future work |

---

## API Compatibility Goals

### PyTorch Compatibility

Target API should feel familiar to PyTorch users:

```swift
// PyTorch
# model = nn.Sequential(
#     nn.Linear(784, 256),
#     nn.ReLU(),
#     nn.Linear(256, 10)
# )

// Magma
let model = nn.Sequential {
    nn.Linear(784, 256)
    nn.ReLU()
    nn.Linear(256, 10)
}
```

### S4TF Compatibility

Honor S4TF patterns where they make sense:

```swift
// S4TF style (keep)
let (loss, grads) = valueWithGradient(at: model) { m in
    m(input).mean()
}

// S4TF style (keep)
LazyTensorBarrier()
```

---

## Migration Checklist

### Phase 1: Setup
- [ ] Clone TaylorTorch into `Legacy/TaylorTorch/`
- [ ] Clone SwiftIR into `Legacy/SwiftIR/`
- [ ] Document key files in each repo
- [ ] Identify test cases to port

### Phase 2: XLARuntime (from SwiftIR)
- [ ] Copy PJRT C headers
- [ ] Port `PJRTClient.swift`
- [ ] Port `PJRTBuffer.swift`
- [ ] Port `PJRTExecutable.swift`
- [ ] Write integration tests

### Phase 3: StableHLO (new, referencing SwiftIR)
- [x] Create `DType.swift`
- [x] Create `TensorType.swift`
- [x] Create `Value.swift`
- [x] Create `MLIRBuilder.swift`
- [ ] Add remaining ops (conv, pooling, etc.)
- [x] Write pure-Swift tests

### Phase 4: LazyTensor (new, inspired by x10)
- [ ] Create `LazyTensorHandle.swift`
- [ ] Create `IRNode.swift`
- [ ] Create `IRGraph.swift`
- [ ] Create `LazyTensorBarrier.swift`
- [ ] Create `CompilationCache.swift`
- [ ] Write tests with XLA

### Phase 5: Torch (from TaylorTorch)
- [ ] Port `Tensor.swift` (major rewrite)
- [ ] Port `Layer.swift` protocol
- [ ] Port `Linear.swift`
- [ ] Port `Conv2d.swift`
- [ ] Port `Sequential.swift`
- [ ] Port optimizers
- [ ] Write end-to-end tests

---

## Ported from Swift for TensorFlow (S4TF)

The following components were ported from the [S4TF swift-apis](https://github.com/tensorflow/swift-apis) repository.

### Initializers

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

### Loss Functions

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

### Optimizers

| S4TF | Magma | Notes |
|------|-------------|-------|
| `SGD` | `SGD` | With momentum, weight decay, nesterov |
| `Adam` | `Adam` | Adaptive moment estimation |
| `RMSProp` | `RMSProp` | Root mean square propagation |
| `AdaGrad` | `AdaGrad` | Adaptive gradients |
| `AdaDelta` | `AdaDelta` | No learning rate needed |

### Layers

| S4TF | Magma | Notes |
|------|-------------|-------|
| `Dense` | `nn.Linear` | Fully connected layer |
| `Conv1D` | `nn.Conv1d` | 1D convolution |
| `Conv2D` | `nn.Conv2d` | 2D convolution |
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

1. **Tensor Format**: S4TF used NHWC (TensorFlow style), Magma also uses NHWC for compatibility with XLA.

2. **Reduction Parameter**: Magma loss functions have an explicit `reduction` parameter (`.mean`, `.sum`, `.none`), similar to PyTorch.

3. **Module Protocol**: S4TF used `Layer` protocol, Magma uses `Module` protocol for consistency with PyTorch naming.

4. **Device Handling**: Magma uses explicit `on device:` parameter for tensor creation.

5. **Lazy Execution**: Magma uses lazy tensor execution with `LazyTensorBarrier()` similar to S4TF x10 backend.

---

## Testing Strategy

### Unit Tests (No XLA)
- StableHLO MLIR generation
- Shape inference
- Type checking

### Integration Tests (With XLA)
- Compile and execute simple graphs
- Verify numerical correctness
- Memory management

### End-to-End Tests (Full Stack)
- Train MNIST
- Compare against PyTorch outputs
- Performance benchmarks
