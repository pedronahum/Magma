# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-24

### Initial Release

This is the first public release of Magma, a deep learning framework for Swift powered by XLA/StableHLO.

### Features

#### Core Tensor Operations
- `Tensor<Scalar>` generic type with full differentiable support
- Creation: `zeros`, `ones`, `full`, `randn`, `randnDevice`, `uniform`, `arange`
- Arithmetic: `+`, `-`, `*`, `/` with NumPy-style broadcasting
- Matrix: `matmul`, `batchedMatmul`, `transpose`, `transpose(dim1, dim2)`
- Reductions: `sum`, `mean`, `max`, `min`, `variance`
- Shape: `reshape`, `squeeze`, `unsqueeze`, `broadcast`, `expand`
- Slicing: `slice`, `sliceAxis`, advanced slicing with negative indices and step
- Indexing: `tensor[i]`, `tensor[i, j]`, `row(index)`, `gather`, `scatter`
- Comparison: `lessThan`, `greaterThan`, `equalTo`, operators (.<, .>, .==, .!=)
- One-hot encoding: `Tensor.oneHot(indices, numClasses:)`

#### Neural Network Layers (`nn` namespace)
- **Core**: `Linear`, `Embedding`, `Flatten`, `Sequential`
- **Convolution**: `Conv1d`, `Conv2d`, `ConvTranspose2d`
- **Normalization**: `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`
- **Pooling**: `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`
- **Regularization**: `Dropout` with proper random masking
- **Recurrent**: `RNNCell`, `LSTMCell`, `GRUCell`, `RNN`, `LSTM`, `GRU` (with bidirectional support)
- **Attention**: `MultiheadAttention`, `scaledDotProductAttention`
- **Transformer**: `TransformerEncoderLayer`, `TransformerDecoderLayer`
- **Positional**: `SinusoidalPositionalEncoding`, `LearnedPositionalEmbedding`

#### Activation Functions
- `relu`, `sigmoid`, `tanh`, `gelu`, `softmax`, `logSoftmax`
- `leakyRelu`, `silu`, `elu`, `hardtanh`
- Advanced: `selu`, `mish`, `softplus`, `prelu`

#### Loss Functions (`nn.functional`)
- `crossEntropy` - with proper one-hot encoding
- `nllLoss` - negative log likelihood
- `mse` - mean squared error
- `binaryCrossEntropy`, `binaryCrossEntropyWithLogits`
- Additional: `huberLoss`, `hingeLoss`, `l1Loss`, `l2Loss`

#### Optimizers (`optim` namespace)
- `SGD` with momentum, weight decay, and Nesterov acceleration
- `Adam` with configurable betas and epsilon
- `AdamW` with decoupled weight decay

#### Learning Rate Schedulers
- `StepLR`, `ExponentialLR`, `CosineAnnealingLR`
- `WarmupLR`, `WarmupCosineScheduler`

#### Data Loading
- `Dataset` protocol
- `TensorDataset` and `SimpleBatchLoader`
- `MNIST` dataset with automatic download and parsing
- PyTorch-compatible transforms: `Normalize`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `CenterCrop`, `RandomCrop`, `Pad`, `Grayscale`, `Compose`, `Lambda`

#### Device Support
- **CPU**: Full support via PJRT CPU plugin
- **TPU**: Full support on Google Cloud TPU VMs
- **GPU**: Planned for future release
- `Tensor.to(device:)` for device transfer
- `Module.to(device:)` for moving models between devices

#### Control Flow
- `select` / `where_` for conditional operations
- `while_loop` and `cond` for dynamic control flow
- `scan` for fixed-iteration loops with autodiff support

#### Mixed Precision
- `toReducedPrecision()` (bfloat16)
- `toFullPrecision()`
- `to(dtype:)` for type conversion
- `MixedPrecision.autocast`

#### Model Checkpointing
- JSON format for human-readable checkpoints
- Binary format for efficient storage
- PyTorch-compatible `state_dict` API

#### Developer Tools
- `Profiler` for timing and performance analysis
- `Benchmark` for measuring execution time with statistics
- `FLOPSEstimator` for performance calculations
- `GradientChecker` for validating autodiff against numerical gradients

#### Error Handling
- `TensorError` enum with detailed, actionable messages
- `TensorDebug` utilities for shape verification
- `TensorAssert` helpers for common validations

### Architecture

Magma uses a 5-layer architecture:

```
Layer 4: Core (Magma)     - PyTorch-compatible API
Layer 3: LazyTensor       - x10-style lazy execution
Layer 2: StableHLO        - Pure Swift MLIR generation
Layer 1: XLARuntime       - PJRT C API wrapper
Layer 0: CXLARuntime      - C bindings to PJRT
```

### Requirements

- Swift 6.0+ with autodiff support
- XLA PJRT runtime for execution
- Linux (x86_64) or macOS 14+

### Known Limitations

- GPU/CUDA support not yet integrated (planned for v0.2.0)
- `scanXLA` only supports single-tensor states (workarounds documented)
- Some advanced gather operations use one-hot multiplication approach

### Documentation

- [Architecture Overview](Documentation/ARCHITECTURE.md)
- [API Reference](Documentation/API.md)
- [TPU Deployment Guide](Documentation/TPU_DEPLOYMENT.md)
- [Contributing Guide](Documentation/CONTRIBUTING.md)
- [Roadmap](Documentation/ROADMAP.md)

### Examples

- **MNIST**: Complete handwritten digit classification training
- **BuildingSimulation**: Differentiable physics simulation port
- **Benchmarks**: Performance measurement suite

### Credits

Magma builds on the foundations of:
- [Swift for TensorFlow (S4TF)](https://github.com/tensorflow/swift-apis)
- TaylorTorch - PyTorch-style API design for Swift
- SwiftIR - MLIR/XLA infrastructure for Swift
