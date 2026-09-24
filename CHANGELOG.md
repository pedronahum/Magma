# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha.1] - 2026-09-24

First public pre-release of Magma, deep learning for Swift on XLA/StableHLO with
Swift-native autodiff. This is an **alpha**. The API is not stable and will
change before 0.1.0. Only the CPU backend is routinely tested. Read
[Known limitations](#known-limitations) before you rely on it.

### Platforms and backends

- **Toolchain:** Swift 6.0 or later from [swift.org](https://swift.org/download/).
  The full test suite passes on 6.0.3 and 6.3.3. Xcode's toolchain does not
  work because it does not ship the `_Differentiation` module.
- **Operating systems:** Linux x86_64 and aarch64 (verified on aarch64, NVIDIA
  GB10). macOS 15+ builds in CI, but that job is marked experimental.
- **Backends:** a PJRT plugin is `dlopen`ed at runtime. You don't need any build
  flags.

  | Backend | Status |
  |---------|--------|
  | CPU (PJRT CPU plugin) | Supported. The whole test suite runs on it. |
  | CUDA GPU (PJRT CUDA plugin) | Experimental. Verified on one device (NVIDIA GB10). |
  | Metal (MetalHLO, macOS) | Experimental and opt-in (`MAGMA_ENABLE_METAL=1`). |
  | TPU (`libtpu`) | Untested. The code path exists, but the maintainers have not run it on a TPU. |

- **Distributed:** data parallel (DDP) and Shardy/SPMD sharding have only been
  verified on emulated CPU devices, where each result is checked against a
  single-device reference. Real multi-GPU and multi-TPU runs are untested.

### Added

#### Tensors and execution
- `Tensor<Scalar>` with x10-style lazy tracing. Graphs compile to StableHLO and
  run through PJRT. Execution happens at barriers: `LazyTensorBarrier(on:)`,
  `LazyTensorBarrierThrowing(on:)`, `Tensor.markForMaterialization()`,
  `materialize()`, and the reads `scalars()` / `item()`.
- Creation: `init(_:shape:on:)`, `zeros`, `ones`, `full`, `arange`, `randn`,
  `uniform(low:high:shape:on:)`, `randomUniform`, `truncatedNormal`,
  `oneHot(_:numClasses:)`, and the device-side RNG ops `randnDevice` / `randDevice`.
- Arithmetic with NumPy-style broadcasting. Also `matmul`, `batchedMatmul`,
  `transpose()`, `transpose(_:_:)`, `transposeLastTwo()`, `conv2d`, `convTranspose2d`.
- Reductions: `sum`, `mean`, `max`, `sum(dims:keepDims:)`, `mean(dims:keepDims:)`,
  `max(dims:keepDims:)`, `variance(dims:…)`, `sum/max/min(alongAxes:)`.
- Shape and indexing: `reshape`, `squeeze()`, `squeeze(dim:)`,
  `expandingShape(at:)`, `broadcast(to:)`, `expand(_:)`, `slice`,
  `slice(axis:start:stop:step:)`, `sliceAxis`, `row(_:)`, subscripts,
  `gather(indices:axis:)`, `scatter`.
- Comparisons: `lessThan`, `greaterThan`, `equalTo` (and the `OrEqual` variants),
  plus the operators `.<`, `.>`, `.<=`, `.>=`, `.==`, `.!=`. Selection with `where_`.
- Activations as methods: `relu`, `sigmoid`, `tanh`, `gelu`, `silu`, `elu`,
  `leakyRelu`, `hardtanh`, `selu`, `mish`, `softplus`, `softmax(dim:)`,
  `logSoftmax(dim:)`.
- Mixed precision: `toReducedPrecision()` (bfloat16), `toFullPrecision()`,
  `to(_: DType)`, `MixedPrecision.autocast(inputs:computation:)`.
- Loops: `scan`, `scanWithOutputs`, and `scanXLA` / `scanXLATensor`, which lower to
  an XLA while loop.

#### Autodiff and training (value-semantic API)
- `Layer` protocol (`Differentiable` + `KeyPathIterable`) with `Linear`, `ReLU`,
  `Sigmoid`, `Conv2d`, and the `sequential { … }` result builder (`Sequential2`).
- `modelGradient(of:input:target:lossFn:)` and `modelValueWithGradient(...)`,
  which also work on models held as `some Layer`.
- Reflection-based optimizers `Adam(learningRate:…)`, `MomentumSGD`, and
  `sgdUpdate`. `optimizer.update(&model, gradient:)` updates every tensor in the
  model.

#### Layers and training (reference-semantic `nn` / `optim` API)
- The `Module` protocol has these requirements: `forward(_:)`, `parameters()`,
  `buffers()`, `to(device:)`, `setTraining(_:)`. It adds `train()`, `eval()`,
  and `callAsFunction`. Parameters are `Parameter` reference cells.
- Layers:
  - Core: `nn.Linear`, `nn.Embedding`, `nn.Flatten`.
  - Containers: `nn.Sequential`, `nn.sequential { }`, `nn.AnyLayer`.
  - Convolution: `nn.Conv1d`, `nn.Conv2d`, `nn.ConvTranspose2d`.
  - Pooling: `nn.MaxPool2d`, `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d`,
    `nn.GlobalAvgPool1d/2d`, `nn.GlobalMaxPool1d/2d`.
  - Upsampling: `nn.Upsample1d/2d`.
  - Normalization: `nn.BatchNorm1d/2d`, `nn.LayerNorm`, `nn.GroupNorm`,
    `nn.InstanceNorm2d`.
  - Regularization: `nn.Dropout`.
  - Recurrent: `nn.RNNCell`, `nn.LSTMCell`, `nn.GRUCell`, and `nn.RNN`,
    `nn.LSTM`, `nn.GRU` (multi-layer and bidirectional).
  - Attention: `nn.MultiheadAttention`, `nn.scaledDotProductAttention`.
  - Transformer: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`,
    `nn.generateCausalMask`.
  - Positional: `nn.SinusoidalPositionalEncoding`, `nn.LearnedPositionalEmbedding`.
- Activation layers: `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.GELU`,
  `nn.LeakyReLU`, `nn.SiLU`, `nn.ELU`, `nn.Hardtanh`, `nn.SELU`, `nn.Mish`,
  `nn.Softplus`, `nn.Softsign`, `nn.PReLU`. PReLU exists only as a layer.
- `parameterGradients(of:loss:)` is the autodiff bridge. It returns gradients
  keyed by `Parameter` identity, which `optimizer.step(_:)` applies.
- Optimizers:
  - `optim.SGD` (momentum, weight decay, Nesterov)
  - `optim.Adam` (weight decay is decoupled; `optim.AdamW` is an alias)
  - `optim.AdamWGroups`, `optim.RMSProp`, `optim.AdaGrad`, `optim.AdaDelta`
  - Gradient helpers: `optim.clipGradNorm`, `optim.clipGradValue`,
    `optim.GradientAccumulator`
- LR schedulers: `optim.StepLR`, `optim.ExponentialLR`, `optim.CosineAnnealingLR`,
  `optim.WarmupLR`, `optim.WarmupCosineScheduler`.
- Functional losses and activations in `nn.functional`: `mse`, `crossEntropy`,
  `nllLoss`, `binaryCrossEntropy`, `binaryCrossEntropyWithLogits`, `relu`,
  `sigmoid`, `tanh`, `gelu`, `softmax`, `logSoftmax`.
- Losses ported from S4TF, as free functions: `l1Loss`, `l2Loss`,
  `meanAbsoluteError`, `meanSquaredError`, `meanSquaredLogarithmicError`,
  `meanAbsolutePercentageError`, `huberLoss`, `logCoshLoss`, `poissonLoss`,
  `hingeLoss`, `squaredHingeLoss`, `categoricalHingeLoss`,
  `kullbackLeiblerDivergence`, `softmaxCrossEntropy`,
  `softmaxCrossEntropyWithLabels`, `sigmoidCrossEntropy`, `cosineSimilarity`,
  `cosineDistance`, `contrastiveLoss`, `tripletMarginLoss`.
- Initializers: `glorotUniform/Normal`, `heUniform/Normal`, `lecunUniform/Normal`,
  `orthogonal`, `truncatedNormal`, `randomUniform`.
- Checkpoints:
  - JSON: `Module.save(to:description:)` and `load(from:)`
  - Binary: `BinaryCheckpoint.save/load`
  - `stateDict()` and `loadStateDict(_:strict:)`
  - Training state: `TrainingState` stores the model state plus step, epoch,
    and loss metadata

#### Data
- `Dataset`, `TensorDataset`, `DataLoader`, `SimpleBatchLoader`,
  `DistributedSampler` (with `.multiHost`), and `MemoryMappedTokenDataset`.
- `MNIST` downloads and caches the dataset. The cache lives in
  `~/.magma/data/mnist`, or in `$MAGMA_DATA_DIR/mnist` when that variable is set.
- Transforms in `transforms`: `Compose`, `Normalize`, `Denormalize`,
  `RandomHorizontalFlip`, `RandomVerticalFlip`, `CenterCrop`, `RandomCrop`, `Pad`,
  `Grayscale`, `RandomGrayscale`, `RandomInvert`, `RandomApply`, `RandomChoice`,
  `Lambda`, `Identity`.

#### Reproducibility, errors, tooling
- `manualSeed(_:)` and `withManualSeed(_:_:)` seed every host-side random draw:
  `randn` / `uniform`, the initializers, dropout masks, data shuffles, and the
  random transforms. Device-side RNG ops are not seeded.
- `MaterializationError` records the stage that failed (`backendUnavailable`,
  `validation`, `compilation`, `inputTransfer`, `execution`, `outputTransfer`,
  `notMaterialized`). The throwing reads `fetchScalars()` and `fetchItem()` and
  `LazyTensorBarrierThrowing(on:)` report it.
- `MAGMA_DEBUG=1` or `MAGMA_DEBUG_DIR=<dir>` writes the failing MLIR to disk when
  compilation fails. Diagnostics go to stderr.
- Gradient checking: `gradcheck`, `gradcheckPasses`, `assertGradcheck`,
  `numericalGradient`, `numericalJacobian`.
- Profiling: `Profiler`, `Benchmark`, `MemoryProfiler`, `FLOPSEstimator`.
- Validation helpers: `TensorError`, `TensorDebug`, `TensorAssert`.

#### Distributed
- `Tensor.crossReplicaMean(groups:)`, `dataParallelSGDStep`, and
  `executeGraphReplicated` for DDP.
- `DeviceMesh`, `TensorSharding`, `Tensor.sharded(on:_:)`, and
  `executeGraphSharded` for Shardy/SPMD.

#### Plugin discovery and environment
- The plugin search order is:
  1. The per-backend override (`MAGMA_PJRT_PLUGIN_CPU`, `_GPU`, `_TPU`), which
     is a full file path.
  2. `$MAGMA_XLA_PATH`, where each of these names is accepted:
     `pjrt_c_api_<backend>_plugin.<ext>`, `libpjrt_c_api_<backend>_plugin.<ext>`,
     `xla_cuda_plugin.so` (GPU), and `libtpu` (TPU).
  3. For TPU, `TPU_LIBRARY_PATH` and the standard `libtpu.so` locations.
  4. The system directories.
- `Backend.resolvedPluginPath` reports which plugin file will be loaded. If no
  plugin is found, the error lists every path that was searched.
- Other variables: `MAGMA_DEFAULT_BACKEND`, `MAGMA_ALLOW_CONCURRENT_ACCEL_CLIENTS`,
  and `MAGMA_TEST_BACKEND` (tests only).

#### Project
- Examples you run with `swift run <Target>`: `ValueLayersExample`,
  `MNISTExample`, `BuildingSimulation`, `Benchmarks`, and `MetalExample` (only
  with `MAGMA_ENABLE_METAL=1`).
- About 1150 tests using swift-testing. CI runs Linux on Swift 6.0 and 6.3, and
  macOS as an experimental job.
- Added `NOTICE`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue and PR templates,
  and a rewritten `CONTRIBUTING.md`.

### Changed

- **Build:** no build flags are needed. `MAGMA_ENABLE_XLA` has been removed
  because it had no effect. Metal/MetalHLO is opt-in with `MAGMA_ENABLE_METAL=1`
  and an optional `MAGMA_METALHLO_PATH`. The minimum is now Swift tools 6.0 and
  macOS 15. Examples are no longer exported as products.
- **Reads fail loudly:** `scalars()`, `item()`, and `materialize()` used to
  return `[]` or do nothing when execution failed. They now stop the program
  with the real `MaterializationError`. To handle the error yourself, use
  `fetchScalars()`, `fetchItem()`, or `LazyTensorBarrierThrowing(on:)`.
- **Barrier semantics:** `LazyTensorBarrier()` executes only tensors marked with
  `markForMaterialization()` since the last barrier on that device, plus
  whatever they depend on. Reads mark their tensor automatically. A bare
  barrier with nothing marked does nothing.
- **Thread safety:** tensor reads are thread-safe. Barriers are serialized
  process-wide. If a batched barrier fails, a read retries its own tensor alone,
  so it only reports its own error.
- **Transformer dropout:** `nn.TransformerEncoderLayer`,
  `nn.TransformerDecoderLayer`, and `nn.SinusoidalPositionalEncoding` default to
  `dropout: 0.1` and training mode, as PyTorch does. Their outputs are random
  until you call `eval()`.
- **Dropout is applied:** `nn.MultiheadAttention`, `nn.LSTM`, `nn.GRU`, and
  `nn.RNN` now apply their configured dropout in training mode, and `eval()`
  turns it off. `nn.scaledDotProductAttention` has no mode, so it applies
  dropout whenever `dropout > 0`. Dropout rates outside `[0, 1)` are rejected.
- **Device moves:** `Module.to(device:)` moves every parameter and buffer by
  default, including through `nn.Sequential`. `Tensor.to(device:)` keeps the
  element type.
- **Checkpoints:** JSON and binary checkpoints are now version 2 and also store
  buffers, such as BatchNorm running statistics. Version-1 files still load.
  Corrupt or truncated files throw `CheckpointError`, and a failed load leaves
  the model unchanged.
- **Losses:**
  - `nn.functional.crossEntropy` also accepts one-hot or probability targets
    shaped `[batch, numClasses]`.
  - `binaryCrossEntropyWithLogits` is numerically stable and
    `@differentiable(reverse, wrt: logits)`.
- **Host RNG:** `randomUniform` and the uniform initializers now draw on the
  host so they can be seeded. `Tensor.uniform` requires `low < high`.
- **Compile caches:** the executable, MLIR, and trace caches are LRU-bounded.
  The `CompilationCache` counters are read-only.
- **Runtime objects:** `PJRTBuffer` and `PJRTExecutable` keep their client
  alive (`PJRTExecutable.client` is now a strong `let`).

### Fixed

- **Runtime:**
  - Executing a program with more than 16 outputs overflowed a buffer.
  - PJRT error messages are now included in thrown errors.
  - Host element sizes are validated in `createBuffer` and `toHost`.
  - Unsupported element types are rejected instead of misread.
  - Plugin loading is serialized.
  - The `PJRT_Executable` handle obtained after compiling is now destroyed
    instead of leaking.
- **Lazy execution:**
  - Batched matmul (rank ≥ 3) now compiles. Before, attention and Transformer
    layers could not execute.
  - Promoted constants keep their declared dtype, so Int32, Double, and Bool
    graphs compute correctly.
  - Executable and trace-cache keys cover everything the program depends on,
    including dtypes, attributes, constant bit patterns, and loop bodies.
  - CSE and algebraic simplification no longer change results. This covers
    merging different constants, broadcasting `x + 0`, `inf * 0`, and dangling
    reshapes.
- **Layers:**
  - `nn.Conv1d`, `nn.ConvTranspose2d`, `nn.GroupNorm`, and `nn.InstanceNorm2d`
    now lower to supported ops and execute. Before, they crashed the emitter.
  - Bidirectional `nn.LSTM` and `nn.GRU` are implemented; they used to run
    forward only.
  - `nn.LSTMCell` and `nn.GRUCell` mixed gates across samples when the batch
    size was above 1.
- **Autodiff and ops:**
  - The scatter lowering compiles.
  - The scatter gradient zeroes overwritten slots.
  - The gather gradient works for any axis and rank.
  - Broadcast pullbacks accept a rank-0 zero cotangent.
- **Validation:**
  - `Tensor(_:shape:)` and the shape factories validate element counts and
    reject negative dimensions.
  - The LR schedulers validate their arguments.
- **Training bridge:** `parameterGradients(of:loss:)` sums the gradients of
  tied parameters and returns zeros for unused ones.
- **Data:** the MNIST loader throws `MNISTError` on malformed files instead of
  trapping. It re-downloads a corrupt cache once, and downloads are atomic.
- **Examples:**
  - They check for a backend before running and exit non-zero on failure.
  - `BuildingSimulation` times materialized results.
  - `MNISTExample` is an honest smoke test.

### Known limitations

- **Two APIs share names.** Unqualified `Linear`, `ReLU`, `Sigmoid`, `Conv2d`,
  `Adam`, and `sequential` are the value-semantic types. `nn.*` and `optim.*` are
  the reference-semantic ones. Their conventions differ: value-semantic
  `Linear.weight` is `[in, out]` (`x · W + b`), while `nn.Linear.weight` is
  `[out, in]` (`x · Wᵀ + b`). Optimizers differ too: `Adam(learningRate:)` with
  `update(&model, gradient:)`, versus `optim.Adam(parameters:lr:)` with
  `step(_:)`.
- **Execution is serialized.** Barriers take a process-wide lock, so
  concurrent tasks don't execute graphs in parallel.
- **Constant precision:** the IR stores constants as `Float`, so large `Double`
  and `Int64` constants lose precision. Constant folding computes in `Float`.
  Algebraic simplification still rewrites `exp(log x)` and `log(exp x)` to `x`.
- **Distributed runners** (`executeGraphReplicated`, `executeGraphSharded`)
  handle `Float32` only.
- **Metal is less hardened than PJRT.** The Metal barrier does not record errors
  on individual tensors, and `MetalCompilationCache` is unbounded. Enabling
  Metal adds MetalHLO as an unversioned (`branch: "main"`) dependency.
- `scanXLA` supports only a single-tensor state.
- **Hardware coverage:** multi-GPU, TPU, and multi-host execution are untested.
  CUDA has only been verified on a single NVIDIA GB10.
- **Tests:** with a plugin installed, run `swift test --no-parallel`. Suites
  share process-wide state, and each CUDA client reserves most of device
  memory, so parallel suites can exhaust it. One process can load only one PJRT
  plugin, so GPU suites (`MAGMA_TEST_BACKEND=gpu`) need a separate invocation.
- **Device RNG** (`randnDevice`, `randDevice`, `randn(_:useDeviceRNG: true)`) is
  not covered by `manualSeed`.

[Unreleased]: https://github.com/pedronahum/Magma/compare/v0.1.0-alpha.1...HEAD
[0.1.0-alpha.1]: https://github.com/pedronahum/Magma/releases/tag/v0.1.0-alpha.1
