# Magma Roadmap

## Project Phases

```
Phase 0: Setup & Foundation        [COMPLETE] ✅
    ↓
Phase 1: Core Infrastructure       [COMPLETE] ✅
    ↓
Phase 2: Basic Neural Networks     [COMPLETE] ✅
    ↓
Phase 3: Training Infrastructure   [COMPLETE] ✅
    ↓
Phase 4: Advanced Features         [COMPLETE] ✅
    ↓
Phase 4.5: Distributed Training    [SINGLE-HOST DONE] ✅ (DDP + Shardy/SPMD, emulated CPU devices)
    ↓
Phase 5: Production Readiness      [IN PROGRESS] 🚧 ← preparing 0.1.0-alpha.1
    ↓
Phase 6: Ecosystem                 [Ongoing]
```

**Current Status**: preparing the first alpha release, **0.1.0-alpha.1**. The test
suite has roughly 1000 tests. The suites that need a PJRT plugin run on the CPU
plugin by default (`MAGMA_TEST_BACKEND` selects the backend) and are skipped when
no plugin is installed.

What works:
- **Models**: real MNIST data loading with autodiff training, Transformer
  encoder/decoder, RNN/LSTM/GRU.
- **Tensor ops**: advanced slicing, comparison ops, loops (`scan`/`scanXLA`),
  proper broadcasting, device transfer, one-hot encoding.
- **Training and tooling**: model checkpointing, PyTorch-compatible transforms,
  gradient checking, numerical stability tests, profiling and benchmarking
  utilities, mixed precision (bfloat16), error handling.
- **Two layer APIs**: value-semantic (`sequential`, `Linear`, `Adam`,
  `modelGradient`) and reference-semantic (`nn.*`, `optim.*`,
  `parameterGradients`).
- **Multi-device**: DDP and Shardy/SPMD, tested on emulated CPU devices.

Backends:
- **CPU**: via the PJRT plugin.
- **CUDA GPU**: single-device execution verified on an NVIDIA GB10.
- **Metal**: opt-in via MetalHLO (`MAGMA_ENABLE_METAL=1`, macOS).
- **Untested**: real multi-GPU and TPU.

---

## Phase 0: Setup & Foundation ✅

**Goal**: Establish monorepo structure with legacy code accessible

### Checklist

- [x] **Repository Setup**
  - [x] Create GitHub repository
  - [ ] Set up branch protection rules
  - [x] Configure CI/CD (GitHub Actions)
  - [x] Add LICENSE (Apache 2.0)
  - [x] Add CONTRIBUTING.md

- [x] **Legacy Integration**
  - [ ] ~~Clone TaylorTorch / SwiftIR into `Legacy/`~~ (never vendored into this repository; see [LEGACY_MAPPING.md](LEGACY_MAPPING.md))
  - [x] Document what to reuse from each
  - [x] Create mapping: TaylorTorch API → new implementation
  - [x] Create mapping: SwiftIR internals → new layers

- [x] **Package Structure**
  - [x] Create Package.swift with all targets
  - [x] Set up module structure (5 layers)
  - [x] Load the PJRT plugin at runtime via `dlopen` (`MAGMA_XLA_PATH`), so no XLA libraries are needed at build time
  - [x] Add development container (devcontainer.json)

- [x] **Documentation**
  - [x] ARCHITECTURE.md
  - [x] ROADMAP.md (this file)
  - [x] CONTRIBUTING.md
  - [x] API.md

### Exit Criteria
- [x] `swift build` succeeds (stub modules)
- [x] CI workflow configured (`.github/workflows/ci.yml`)
- [x] Initial tests passing
- [x] MNISTExample runs (placeholder demonstrating target API)

---

## Phase 1: Core Infrastructure ✅

**Goal**: Working lazy tensor execution from Swift to XLA

### Week 1-2: Bottom Layers (0-1)

#### Layer 0: CXLARuntime
- [x] Copy PJRT C headers from SwiftIR
- [x] Create module map for Swift import
- [x] Test basic C function access from Swift

#### Layer 1: XLARuntime
- [x] `PJRTClient` - device discovery, compilation
- [x] `PJRTDevice` - device abstraction
- [x] `PJRTBuffer` - data transfer to/from device
- [x] `PJRTExecutable` - compiled program execution
- [x] CPU plugin working
- [x] Tests: compile simple MLIR, execute, verify results

### Week 2-3: Middle Layers (2-3)

#### Layer 2: StableHLO
- [x] `TensorType`, `DType` types
- [x] `MLIRBuilder` - core builder class
- [x] `Value` - SSA value reference
- [x] Basic ops: `add`, `subtract`, `multiply`, `divide`
- [x] Matrix ops: `dot`, `dot_general` (batched), `transpose`, `reshape`
- [x] Activations: `maximum` (for relu), `tanh`, `exp`, `log`
- [x] Reductions: `reduce_sum`, `reduce_mean`, `reduce_max`
- [x] Shape inference for all ops
- [x] Tests: generate MLIR, verify syntax (no XLA needed!)

#### Layer 3: LazyTensor
- [x] `LazyTensorHandle` - graph node reference
- [x] `IRNode` - operation representation
- [x] `IRGraph` - full graph with topological sort
- [x] `Device` - Swift device abstraction (defined in XLARuntime)
- [x] `LazyTensorBarrier()` - trigger compilation/execution
- [x] `StableHLOEmitter` - IRGraph → MLIR
- [x] `CompilationCache` - hash-based caching
- [x] `TensorRegistry` - track pending tensors
- [x] Tests: build graph, emit MLIR, execute via XLARuntime

### Week 3-4: User Layer (4) - Basics

#### Layer 4: Core, module `Magma` (Tensor Only)
- [x] `Tensor<Scalar>` struct
- [x] Creation: `init`, `zeros`, `ones`, `full`, `randn`
- [x] Arithmetic: `+`, `-`, `*`, `/`
- [x] Matrix: `matmul`, `batchedMatmul`, `transpose`, `transpose(dim1, dim2)`
- [x] Reductions: `sum`, `mean`, `max`, `min`
- [x] Shape: `reshape`, `squeeze`, `unsqueeze`, `broadcast`
- [x] Slicing: `slice`, `pad`
- [x] Device: `Tensor.to(device:)` - device transfer for tensors and modules
- [x] `@differentiable` conformance
- [x] Tests: end-to-end tensor operations

### Exit Criteria
- [x] Can execute: `let z = (x * y + z).sum(); print(z.item())`
- [x] Compilation caching works (second run faster)
- [x] Basic autodiff works: `gradient(at: x) { $0.sum() }`
- [x] All layer tests pass
- [x] Documentation updated

---

## Phase 2: Basic Neural Networks ✅

**Goal**: Train a simple model (MLP on MNIST)

### Week 5-6: Layer API ✅

#### nn.Module
- [x] `Module` protocol definition
- [x] `Parameter` wrapper type
- [x] Autodiff via `parameterGradients(of:loss:)` (an `nn.Module` is not itself `Differentiable`; the value-semantic `Layer` protocol is)
- [x] `parameters()` - collect all parameters
- [x] `to(device:)` - move model to device

#### Core Layers
- [x] `nn.Linear` (dense/fully-connected)
- [x] `nn.Conv2d` (2D convolution)
- [x] `nn.BatchNorm2d`
- [x] `nn.BatchNorm1d`
- [x] `nn.Dropout`
- [x] `nn.Flatten`
- [x] `nn.Sequential` (with result builder)
- [x] `nn.MaxPool2d`
- [x] `nn.AvgPool2d`
- [x] `nn.AdaptiveAvgPool2d`

#### Activations (nn.functional & layers)
- [x] `relu` (and `nn.ReLU`)
- [x] `sigmoid` (and `nn.Sigmoid`)
- [x] `tanh` (and `nn.Tanh`)
- [x] `gelu` (and `nn.GELU`)
- [x] `softmax`, `logSoftmax`
- [x] `leakyRelu` (and `nn.LeakyReLU`)
- [x] `silu` (and `nn.SiLU`)
- [x] `elu` (and `nn.ELU`)
- [x] `hardtanh` (and `nn.Hardtanh`)

### Week 7-8: Loss & Training Basics

#### Loss Functions
- [x] `nn.functional.crossEntropy`
- [x] `nn.functional.mse`
- [x] `nn.functional.binaryCrossEntropy`
- [x] `nn.functional.binaryCrossEntropyWithLogits`
- [x] `nn.functional.nllLoss`

#### Autodiff Integration
- [x] `Tensor` conforms to `Differentiable`
- [x] VJPs for arithmetic operations (+, -, *, /)
- [x] VJPs for activations (relu, sigmoid, tanh, gelu, exp, log)
- [x] VJPs for matrix ops (matmul, transpose, reshape, broadcast)
- [x] VJPs for reductions (sum, mean)
- [x] VJPs for softmax
- [x] `gradient(at:of:)` function
- [x] `valueWithGradient(at:of:)` function

### Exit Criteria
- [x] MLP model compiles and runs
- [x] Forward + backward with autodiff works
- [x] `model.parameters()` returns all weights

---

## Phase 3: Training Infrastructure ✅

**Goal**: Full training loop with optimizer and data loading

### Week 9-10: Optimizers ✅

#### Optimizer Protocol
- [x] `Optimizer` protocol definition
- [x] `step(_:)` method (array form, and `[Parameter: Tensor]` matched by identity)
- [x] `resetState()` method (`zeroGrad()` is a deprecated alias)
- [x] `learningRate` property (get/set)

#### Implementations
- [x] `optim.SGD` (with momentum, weight decay, Nesterov)
- [x] `optim.Adam` (with weight decay = AdamW)
- [x] `optim.AdamW` (alias for Adam with decoupled weight decay)

#### Learning Rate Schedulers
- [x] `LRScheduler` protocol
- [x] `optim.StepLR` - decay by gamma every N steps
- [x] `optim.ExponentialLR` - exponential decay
- [x] `optim.CosineAnnealingLR` - cosine annealing
- [x] `optim.WarmupLR` - linear warmup then constant
- [x] `optim.WarmupCosineScheduler` - warmup + cosine decay (transformer-style)

### Week 10-11: Data Loading ✅

#### Dataset & DataLoader
- [x] `Dataset` protocol
- [x] `TensorDataset` - simple tensor-based dataset
- [x] `SimpleBatchLoader` with batching and iteration
- [x] Batch iteration via `Sequence` protocol
- [x] Shuffling (SimpleBatchLoader with `shuffle: true`)
- [x] Drop last incomplete batch (`dropLast: true`)
- [x] Basic transforms (PyTorch-compatible `transforms` module)

#### Tensor Slicing
- [x] `slice(start:size:)` for batch extraction
- [x] `pad` operation

#### MNIST Example
- [x] Full training example in `Examples/MNIST/main.swift`
- [x] Actual MNIST data loading (downloads from Google Storage, parses IDX format)
- [x] Forward-backward training with real autodiff gradients

### Exit Criteria
- [x] Optimizer updates parameters correctly
- [x] LR schedulers work as expected
- [x] Data loading with batching works
- [x] MNIST MLP training with real data (downloads from Google Storage, parses IDX format)
- [x] Checkpointing works (save/load model) - JSON and binary formats implemented

---

## Phase 4: Advanced Features ✅ [COMPLETE]

**Goal**: Support complex architectures (Transformers, RNNs)

### Week 12-13: Advanced Layers

#### Attention ✅
- [x] `nn.scaledDotProductAttention` - scaled dot-product attention function
- [x] `nn.MultiheadAttention` - full multi-head attention layer
- [x] Attention mask support (via `maskedFill`)
- [x] Cross-attention support (different Q/K/V dimensions)
- [x] Configurable bias, dropout, head dimensions

#### Supporting Tensor Operations ✅
- [x] `transpose(dim1, dim2)` - transpose specific dimensions
- [x] `transposeLastTwo()` - transpose last two dims
- [x] `batchedMatmul()` - batched matrix multiplication (3D, 4D)
- [x] `maskedFill(mask:value:)` - conditional filling

#### Recurrent ✅
- [x] `nn.RNNCell` - basic RNN cell
- [x] `nn.LSTMCell` - LSTM cell with gates
- [x] `nn.GRUCell` - GRU cell with reset/update gates
- [x] `nn.RNN` - multi-layer RNN with bidirectional support
- [x] `nn.LSTM` - multi-layer LSTM with bidirectional support
- [x] `nn.GRU` - multi-layer GRU with bidirectional support

#### Transformer Building Blocks ✅
- [x] `nn.MultiheadAttention` - multi-head attention with Q/K/V projections
- [x] `nn.LayerNorm` - layer normalization with affine parameters
- [x] `nn.TransformerEncoderLayer` - complete encoder layer (Pre-LN and Post-LN)
- [x] `nn.TransformerDecoderLayer` - decoder with self-attention, cross-attention, FFN (Pre-LN and Post-LN)
- [x] `nn.SinusoidalPositionalEncoding` - fixed sinusoidal positional encoding
- [x] `nn.LearnedPositionalEmbedding` - learned positional embeddings
- [x] `nn.generateCausalMask` - helper for decoder causal masking

### Week 13-14: Advanced Ops

#### Math Operations ✅
- [x] `abs` - absolute value (differentiable)
- [x] `pow` - power
- [x] `clamp` - clamp to range
- [x] `sqrt` - square root

#### Indexing ✅
- [x] Basic slicing (`slice`, `sliceAxis`)
- [x] Advanced slicing (negative indices, step) - `slice(axis:start:stop:step:)`
- [x] `gather`, `scatter`
- [x] Boolean masking - `where_`, `maskedSelect`, comparison operators (.<, .>, .==, etc.)
- [x] **Element indexing** - `tensor[index]` subscript for 1D tensors (differentiable)
- [x] **Row indexing** - `tensor.row(index)` for 2D tensors (differentiable)
- [x] **2D subscript** - `tensor[row, col]` for 2D element access (differentiable)

#### Tensor Manipulation ✅
- [x] `Tensor.concat` - concatenate tensors along existing axis
- [x] `Tensor.stack` - stack tensors along new axis

#### Broadcasting ✅
- [x] Automatic broadcasting
- [x] `broadcast(to:)`
- [x] `expand`

#### Control Flow ✅
- [x] `select` (conditional select / where)
- [x] `stablehlo.while` loops, traced via `scanXLA` / `scanXLATensor`
- [ ] `cond` for branching: available in `MLIRBuilder` (`stablehlo.if`), but there is no Tensor-level API yet
- [x] `scan` for fixed-iteration loops with autodiff support

### Week 14-15: Multi-Device

#### TPU Support ✅ (untested on real TPU hardware)
- [x] `Backend.tpu` with automatic plugin detection
- [x] `Backend.isAvailable` - check if backend plugin exists
- [x] `Backend.bestAvailable` - auto-select best backend (TPU > Metal > GPU > CPU)
- [x] `TPUEnvironment` - detect TPU VM, topology, chip count
- [x] Cloud TPU VM deployment documentation

#### Device Management
- [x] TPU support (via libtpu.so on Cloud TPU VMs); untested on real TPU hardware
- [x] GPU support (CUDA plugin) — single-device execution verified (NVIDIA GB10)
- [x] Multi-device data parallel (DDP), tested on emulated CPU devices; real multi-GPU untested (see [Phase 4.5](#phase-45-distributed-training-shardyspmd-integration--single-host-implemented))
- [ ] `DistributedDataParallel` wrapper type (DDP is provided by `executeGraphReplicated` / `dataParallelSGDStep`)

#### Mixed Precision ✅
- [x] `toReducedPrecision` (bfloat16)
- [x] `toFullPrecision`
- [x] `to(dtype:)` - general type conversion
- [x] `MixedPrecision.autocast` - automatic precision management
- [x] `MixedPrecision.isRecommended(on:)` - device-specific recommendations

### Exit Criteria
- [x] MultiheadAttention works ✅
- [x] TransformerEncoderLayer works ✅
- [x] TransformerDecoderLayer works ✅
- [x] RNN/LSTM/GRU works with bidirectional support ✅
- [x] GPU execution works ✅ (single-device CUDA, verified on NVIDIA GB10)
- [x] Mixed precision training works ✅

---

## Phase 4.5: Distributed Training (Shardy/SPMD Integration) ✅ [single-host implemented]

**Goal**: Enable distributed training across multiple devices (TPUs/GPUs) using OpenXLA's Shardy tensor partitioning system

**Status**: single-host multi-device training is implemented. This covers
data-parallel (DDP) training, Shardy/SPMD sharding and tensor parallelism through
the Shardy partitioner. All of it is tested on **emulated CPU devices** (the XLA
CPU plugin exposing N virtual devices), and each result is checked against a
single-device reference. On CUDA, only single-GPU Shardy compile and execute is
verified. **Real multi-GPU and TPU runs are untested.** Multi-host support is
configuration only (topology and data sharding); the PJRT coordination service is
not implemented. Full design and status:
[MULTI_DEVICE_ASSESSMENT.md](MULTI_DEVICE_ASSESSMENT.md).

The integration differs from the original plan below in two main ways. Shardy
runs **inside XLA**, enabled by the `use_shardy_partitioner` compile option, so no
`sdy_opt` or `libsdy_capi` is needed. DDP is a pair of runners and helpers rather
than a `DistributedDataParallel` wrapper type.

### Background & Motivation

#### What is Shardy?

[Shardy](https://openxla.org/shardy/overview) is an MLIR-based tensor partitioning system from OpenXLA that provides:

- **Automatic sharding propagation**: Compiler determines optimal tensor partitioning based on user hints + cost models
- **Axis-based representation**: More predictable and debuggable than previous GSPMD approaches
- **Novel reshape handling**: Reduces communication overhead that plagued earlier distributed systems
- **SPMD partitioner**: Transforms single-device programs into partitioned multi-device programs with automatic collectives

#### Why Shardy for Magma?

| Framework | Distributed Approach |
|-----------|---------------------|
| PyTorch | Manual DDP/FSDP, increasingly SPMD via TorchTPU |
| JAX | GSPMD/Shardy native |
| TensorFlow | GSPMD via DTensor |
| **Magma** | Shardy-based SPMD |

Swift's compile-time type safety + Shardy's automatic sharding = safer distributed training than Python alternatives.

#### TorchTPU Context

Google's [TorchTPU](https://github.com/pytorch/xla) initiative (with Meta collaboration, announced December 2025) aims to make TPUs feel native to PyTorch. Key developments:

- **RFC #9684** (October 2025): Proposed more native PyTorch/TPU integration
- **Meta partnership**: Active collaboration on PyTorch/XLA, potential TPU cloud usage from 2026
- **TPU v7 (Ironwood)**: 10x performance over v5p, GA November 2025

Magma can leverage the same Shardy infrastructure that powers TorchTPU's SPMD capabilities.

#### Legacy Foundation

The sharding types were adapted from `SwiftIRShardingLite` in
[pedronahum/SwiftIR](https://github.com/pedronahum/SwiftIR). That code is not
vendored in this repository. SwiftIR emitted and propagated `sdy` annotations but
never partitioned or executed them. Partitioning, collectives, multi-device
execution and gradient sync were new work in Magma (see
[MULTI_DEVICE_ASSESSMENT.md §9](MULTI_DEVICE_ASSESSMENT.md)).

---

### Sharding Foundation ✅

#### Device Mesh Types (`Sources/StableHLO/Sharding/DeviceMesh.swift`)

- [x] `MeshAxis` - Named axis with size (e.g., `MeshAxis(name: "x", size: 4)`)
- [x] `DeviceMesh` - Multi-dimensional device topology
  - [x] `DeviceMesh.linear(name:axisName:size:)` - 1D mesh
  - [x] `DeviceMesh.grid(name:rows:cols:rowAxis:colAxis:)` - 2D mesh (axis names default to `x`/`y`)
  - [x] `DeviceMesh.cube(name:x:y:z:...)` - 3D mesh
  - [x] `deviceCount` - Total devices in mesh
  - [x] `axis(named:)` - Get axis by name
  - [x] `mlirText` - Generate the `sdy.mesh` text
  - [ ] ~~`toAttribute(context:)`~~ - not needed (text-based builder)

```swift
let mesh = DeviceMesh.grid(name: "tpu_mesh", rows: 2, cols: 4)  // 8 devices
print(mesh.deviceCount)  // 8

let mesh = DeviceMesh(name: "training_mesh", axes: [
    MeshAxis(name: "data", size: 4),   // Data parallelism
    MeshAxis(name: "model", size: 2)   // Tensor parallelism
])
```

#### Tensor Sharding Specification Types (`Sources/StableHLO/Sharding/TensorSharding.swift`)

- [ ] `AxisRef` / `SubAxisInfo` - sub-axis (hierarchical) sharding: not ported
- [x] `DimensionSharding` - Per-dimension sharding specification
  - [x] `DimensionSharding.sharded(on: "x")` - Shard along axis "x"
  - [x] `DimensionSharding.replicated` - Fully replicated (closed)
  - [x] `DimensionSharding.open(on:)` - Allow propagation to determine
  - [x] Priority support for propagation ordering
- [x] `TensorSharding` - Complete tensor sharding specification
  - [x] `TensorSharding(meshName:dimShardings:replicatedAxes:)` - Full specification
  - [x] `TensorSharding(meshName:axisNames:)` - Convenience with axis names
  - [x] `TensorSharding.replicated(meshName:rank:)` - Fully replicated
  - [x] `replicatedAxes`
  - [x] `validate(against:rank:)` - mesh/rank/axis validation
  - [ ] `unreducedAxes`

```swift
// Shard a 2D tensor: batch on "data", features replicated
let sharding = TensorSharding(
    meshName: "training_mesh",
    dimShardings: [.sharded(on: "data"), .replicated]
)

// Or using the convenience initializer
let sharding = TensorSharding(meshName: "training_mesh", axisNames: ["data", nil])
```

#### StableHLO Sharding Support (`MLIRBuilder`)

- [x] `sdy.mesh` generation - `declareMesh(_:)`
- [x] `sdy.sharding` attribute on arguments - `argument(_:sharding:)`
- [x] `sdy.sharding_constraint` for intermediate tensors - `shardingConstraint(_:_:)`
- [ ] `sdy.manual_computation` for user-defined partitioned regions

#### LazyTensor Sharding Integration

- [x] Sharding on graph nodes - `LazyTensorHandle.sharding` + `IRGraph.mesh`
- [x] Sharding validation - `IRGraph.validateShardings()`
- [x] `StableHLOEmitter` emits `sdy.mesh`, argument shardings and constraints
- [x] Propagation is left to Shardy inside XLA (no Swift-side propagation)

---

### SPMD User API

#### Core Sharding API

The original plan called for a PyTorch/XLA-style `markSharding` / `PartitionSpec`
API. The implemented equivalent is simpler:

- [x] `Tensor.sharded(on:_:)` / `Tensor.sharded(_:)` - annotate a tensor with a sharding
- [x] `Tensor.input(from:)` - a distributable `.data`-backed input
- [x] `Tensor.makeGraph(mesh:)` - extract the traced `IRGraph` carrying the mesh
- [x] `executeGraphSharded(_:numDevices:client:)` - compile with the Shardy partitioner and run across N devices
- [x] Sharding validation and error messages (`ShardingError`)
- [ ] `markSharding(mesh:partitionSpec:)`, `PartitionSpec`, `XLAShardedTensor`
- [ ] `Mesh.with(_:body:)` scoped mesh context

```swift
let x = Tensor<Float>.input(from: xBuf).sharded(on: "mesh", ["x", nil])  // row-shard
let y = x.matmul(w).sharded(on: "mesh", ["x", nil])
let outs = try executeGraphSharded(y.makeGraph(mesh: mesh), numDevices: 2, client: client)
```

#### Collective Operations

- [x] `allReduce` / `allReduceMean` - in-graph `stablehlo.all_reduce` (`Tensor.crossReplicaSum/Mean(groups:)`)
- [x] Partitioner-inserted collectives - Shardy inserts the collectives an SPMD program needs (e.g. an all-reduce for a contracting-dim-sharded matmul)
- [ ] Hand-written `allGather`, `reduceScatter`, `allToAll` ops
- [ ] Collective operation fusion

---

### Automatic Sharding Propagation

- [x] Shardy propagation and partitioning inside XLA (`PJRTClient.compile(_:numPartitions:useSPMDPartitioning:useShardyPartitioner:)`); CPU plugin and single-GPU CUDA plugin verified
- [ ] ~~`sdy_opt` integration via `SdyOptRunner`~~ - not needed; offline inspection only
- [ ] `ShardingPropagationConfig` (priorities, cost model, propagation modes)
- [ ] `model.autoShard(mesh:strategy:)` and predefined strategies
- [ ] `@Sharded` property wrapper / `Module.shardingSpec`

---

### Distributed Training Wrappers

#### Data Parallel (DDP)

- [x] Replicated execution - `executeGraphReplicated(_:numReplicas:distribution:client:)` with `.replicated` / `.perReplica` inputs
- [x] Gradient all-reduce - `grad.crossReplicaMean(groups:)`, `Optimizer.step(syncing:groups:)`
- [x] End-to-end step from a differentiable loss - `dataParallelSGDStep(...)`
- [x] DDP result == single-device reference (emulated CPU)
- [ ] `DistributedDataParallel` wrapper type
- [ ] Bucketed all-reduce, gradient compression

```swift
let synced = grad.crossReplicaMean(groups: [[0, 1]])   // average grads across replicas
let wNew   = w - synced * lr                           // identical update on every replica

let updated = try dataParallelSGDStep(
    w: w, lr: 0.1, numReplicas: 2, client: client,
    dataDistribution: [ObjectIdentifier(dataBuf): .perReplica(shards)]
) { w in loss(w) }
```

#### FullyShardedDataParallel (FSDP)

- [ ] `FullyShardedDataParallel` wrapper and sharding strategies (the Shardy collective-insertion mechanism it would build on is proven)

#### Tensor Parallelism

- [x] Tensor parallelism via Shardy - a contracting-dim-sharded matmul partitions correctly and matches the reference (`TensorParallelTests`)
- [ ] `nn.ColumnParallelLinear`, `nn.RowParallelLinear`, `nn.ParallelEmbedding`, `nn.ParallelMultiheadAttention`

---

### Multi-Host & Advanced Features

- [x] `MultiHostConfig` - process/device topology, global device indices
- [x] `DistributedSampler` - partition a dataset across replicas; `DistributedSampler.multiHost` for global sharding across processes
- [ ] PJRT coordination service (gRPC coordinator, KV rendezvous, NCCL id exchange) - needs a real cluster
- [ ] Pipeline parallelism
- [ ] Distributed (sharded) checkpointing

---

### Supported Parallelism Strategies Summary

| Strategy | Description | Status |
|----------|-------------|--------|
| **Data Parallel** | Replicate model, shard data batches | ✅ `executeGraphReplicated` + `crossReplicaMean` (emulated CPU) |
| **SPMD sharding** | Annotate shardings, Shardy partitions | ✅ `executeGraphSharded` (emulated CPU) |
| **Tensor Parallel** | Shard large weight matrices across devices | ✅ via Shardy (emulated CPU); no parallel-layer types |
| **FSDP / ZeRO** | Shard parameters, gradients, optimizer states | ⬜ Not implemented |
| **Pipeline Parallel** | Shard model layers across devices | ⬜ Not implemented |
| **Hybrid** | Combine data + tensor parallelism | ⬜ Not packaged (multi-axis meshes are supported by the types) |

---

### Exit Criteria

- [x] `DeviceMesh` and `TensorSharding` types working
- [x] Tensor sharding API functional (`Tensor.sharded(on:_:)`, not `markSharding()`)
- [x] Basic data parallelism running on multiple (emulated CPU) devices
- [x] DDP working (`executeGraphReplicated`, `dataParallelSGDStep`; no wrapper type)
- [ ] `FullyShardedDataParallel` wrapper working
- [x] Shardy propagation + partitioning integrated (inside XLA, not via `sdy_opt`)
- [ ] Multi-host training example (2+ hosts)
- [ ] Distributed checkpointing working
- [ ] Documentation: Tutorial on distributed training (design/status doc: [MULTI_DEVICE_ASSESSMENT.md](MULTI_DEVICE_ASSESSMENT.md))
- [x] Tests: distributed suites (DDP, SPMD, tensor parallel, collectives, sampler, multi-host config)
- [ ] Validation on real multi-GPU and TPU hardware

---

### Key Files

| File | Purpose |
|------|---------|
| `Sources/StableHLO/Sharding/DeviceMesh.swift` | Device mesh topology types |
| `Sources/StableHLO/Sharding/TensorSharding.swift` | Sharding specification types |
| `Sources/StableHLO/Builder/MLIRBuilder.swift` | `sdy` hooks, `all_reduce` |
| `Sources/LazyTensor/LazyTensor.swift` | Graph sharding, `executeGraphReplicated`, `executeGraphSharded` |
| `Sources/LazyTensor/StableHLOEmitter.swift` | Sharding and collective emission |
| `Sources/XLARuntime/XLARuntime.swift` | Multi-device execute, buffer distribution, SPMD/Shardy compile options |
| `Sources/CXLARuntime/PJRTProtoHelper.cpp` | `use_spmd_partitioning` / `use_shardy_partitioner` compile options |
| `Sources/Core/Distributed.swift` | Collectives on `Tensor`, `sharded`, `makeGraph`, `dataParallelSGDStep` |
| `Sources/Core/MultiHost.swift` | `MultiHostConfig`, `DistributedSampler.multiHost` |
| `Documentation/MULTI_DEVICE_ASSESSMENT.md` | Design and status |

---

### External Dependencies

| Dependency | Purpose | Source |
|------------|---------|--------|
| PJRT plugin built with Shardy | Propagation + partitioning at compile time | The XLA CPU/GPU plugin (see README for the tested pin) |
| PJRT multi-device | Multi-device execution | Already in XLA |

`libsdy_capi.so` and `sdy_opt` are **not** required.

---

### References

- [Shardy Overview - OpenXLA](https://openxla.org/shardy/overview)
- [Shardy GitHub](https://github.com/openxla/shardy)
- [Shardy Sharding Representation](https://openxla.org/shardy/sharding_representation)
- [PyTorch/XLA SPMD Guide](https://docs.pytorch.org/xla/master/spmd.html)
- [PyTorch/XLA SPMD Blog Post](https://pytorch.org/blog/pytorch-xla-spmd/)
- [GSPMD Paper](https://arxiv.org/abs/2105.04663)
- [TorchTPU Coverage](https://hyperframeresearch.com/2025/12/24/can-googles-torchtpu-eventually-bridge-nvidias-cuda-moat/)

---

## Phase 5: Production Readiness 🚧 [IN PROGRESS]

**Goal**: Stable, documented, performant

### Week 16-17: Performance

#### Graph Optimization Passes ✅
- [x] `PassManager` - Infrastructure for running optimization passes
- [x] `DeadCodeEliminationPass` - Remove unused operations
- [x] `CommonSubexpressionEliminationPass` - Eliminate redundant computations
- [x] `ConstantFoldingPass` - Evaluate constant expressions at compile time
- [x] `AlgebraicSimplificationPass` - Simplify patterns (x+0=x, x*1=x, -(-x)=x, etc.)
- [ ] `OperationFusionPass` - present but **disabled by default and performs no transformations** (`Sources/LazyTensor/Optimization/Passes/OperationFusion.swift`). Only the `FusionPattern` protocol and helpers remain (`Optimization/Fusion/FusionPattern.swift`). No patterns ship.

#### Fusion
Fusion is left to the backend compiler: Magma emits plain StableHLO, and XLA
(or MetalHLO) fuses operations during compilation. The earlier graph-level
fusion patterns (attention, LayerNorm, RMSNorm, MatMul+bias+activation, softmax,
GELU) are no longer in the tree. Graph-level fusion in Magma is listed under
[Beyond the alpha](#beyond-the-alpha).

#### Optimization
- [ ] Profile compilation times
- [ ] Profile execution times
- [ ] Identify and fix bottlenecks
- [ ] Benchmark vs PyTorch

#### Memory
- [ ] Memory profiling
- [ ] Fix any leaks
- [ ] Optimize buffer reuse

### Week 17-18: Polish

#### Error Handling ✅
- [x] Helpful error messages - `TensorError` enum with detailed descriptions
- [x] Shape mismatch debugging - `TensorDebug` utilities
- [x] Device mismatch debugging - `TensorError.deviceMismatch`
- [x] `TensorAssert` helpers - broadcastable, shapesEqual, validAxis, validMatmul
- [ ] Compilation error mapping

#### Documentation
- [ ] Complete API docs
- [ ] Tutorial: Getting Started
- [ ] Tutorial: Custom Layers
- [ ] Tutorial: Distributed Training
- [ ] Migration guide from PyTorch

#### Testing
- [x] Gradient checking for common ops ✅
  - `gradcheck` function comparing autodiff vs numerical gradients
  - `numericalGradient` and `numericalGradientForward` utilities
  - `numericalJacobian` for vector-valued functions
  - Gradient checking tests covering arithmetic, activations, matrix ops
- [x] Numerical stability tests ✅
  - Very large/small values handling
  - Near-zero division and underflow/overflow
  - NaN/Inf propagation tests
  - Softmax stability with extreme values
  - Gradient stability tests
- [x] Edge case tests ✅
  - Broadcasting edge cases
  - Reduction edge cases
  - Matrix operation edge cases
  - Type conversion precision tests
- [ ] Performance regression tests

#### Profiling & Benchmarking ✅
- [x] `Profiler` module with timing utilities
  - `Profiler.timed` - measure block execution time
  - `Profiler.timedWithBarrier` - timing with tensor materialization
  - `Profiler.step` / `Profiler.scope` - xprof-compatible trace markers
  - `Profiler.instant` - instant trace events
- [x] `Benchmark` utilities
  - `Benchmark.measure` - run multiple iterations with statistics
  - `Benchmark.measureWithBarrier` - benchmarking with tensor materialization
  - `Benchmark.compare` - compare multiple implementations
  - `BenchmarkStats` - mean, min, max, std dev, median, percentiles
- [x] `FLOPSEstimator` for performance analysis
  - `matmul` FLOPS calculation
  - `conv2d` FLOPS calculation
  - `gflops` - compute effective GFLOPS
- [x] `MemoryProfiler` for memory tracking
- [x] XLA tracing integration via PJRT TraceMe API

### Exit Criteria
- [ ] >90% test coverage
- [ ] All public APIs documented
- [ ] No known memory leaks
- [ ] Competitive performance benchmarks
- [ ] 0.1.0-alpha.1 release (in preparation)

---

## Phase 6: Ecosystem (Ongoing)

### Model Zoo
- [ ] ResNet
- [ ] VGG
- [ ] BERT
- [ ] GPT-2
- [ ] ViT

### Integrations
- [ ] Hugging Face weight loading
- [ ] ONNX export/import
- [ ] SafeTensors format
- [ ] Weights & Biases logging

### Platforms
- [x] Linux: CI builds on x86_64 (Swift 6.0 and 6.3); development and GPU verification on aarch64 (NVIDIA GB10)
- [ ] macOS (Apple Silicon): CI build job is experimental (`continue-on-error`)
- [x] GPU support (CUDA): single-device execution verified; multi-GPU untested
- [x] Metal backend: opt-in via [MetalHLO](https://github.com/pedronahum/MetalHLO) (`MAGMA_ENABLE_METAL=1`, macOS only)
- [ ] TPU support (Cloud TPU VMs): plugin detection and a deployment guide exist ([TPU_DEPLOYMENT.md](TPU_DEPLOYMENT.md)), but it is untested on real TPU hardware

### Community
- [ ] Discord/Slack
- [ ] GitHub Discussions
- [ ] Blog posts
- [ ] Conference talks

---

## Current Implementation Summary

### What's Working

| Category | Components |
|----------|------------|
| **Tensor** | Creation, arithmetic, matrix ops, reductions, activations, broadcasting, slicing (advanced with step/negative indices), subscript indexing (`tensor[i]`, `tensor[i,j]`), concat, stack |
| **Comparison** | lessThan, greaterThan, equalTo, operators (.<, .>, .<=, .>=, .==, .!=), where_, maskedSelect |
| **Autodiff** | Full VJP support for common ops, `gradient()`, `valueWithGradient()` |
| **Layers (`nn.*`)** | Linear, Embedding, Conv1d, Conv2d, ConvTranspose2d, BatchNorm1d/2d, LayerNorm, GroupNorm, InstanceNorm2d, Dropout, Flatten, Sequential, pooling, upsampling |
| **Value-semantic layers** | `Layer` protocol, `Linear`, `ReLU`, `Sigmoid`, `Conv2d`, typed `sequential { }`, `modelGradient` |
| **Activations** | ReLU, Sigmoid, Tanh, GELU, LeakyReLU, SiLU, ELU, Hardtanh, SELU, Mish, Softplus, Softsign, PReLU |
| **Attention** | ScaledDotProductAttention, MultiheadAttention |
| **Transformer** | TransformerEncoderLayer, TransformerDecoderLayer, SinusoidalPositionalEncoding, LearnedPositionalEmbedding, CausalMask |
| **Recurrent** | RNNCell, LSTMCell, GRUCell, RNN, LSTM, GRU (with multi-layer and bidirectional support) |
| **Optimizers** | `optim.SGD` (momentum, Nesterov), `optim.Adam`/`optim.AdamW`, `optim.AdamWGroups`, `optim.RMSProp`, `optim.AdaGrad`, `optim.AdaDelta`; value-semantic `Adam`, `MomentumSGD`, `sgdUpdate` |
| **Schedulers** | StepLR, ExponentialLR, CosineAnnealingLR, WarmupLR, WarmupCosine |
| **Data** | Dataset protocol, TensorDataset, DataLoader, SimpleBatchLoader (shuffle, dropLast), DistributedSampler, MNIST |
| **Transforms** | Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, RandomCrop, Pad, Grayscale, Lambda |
| **Backends** | CPU (via PJRT); GPU (CUDA, single-device verified); Metal (macOS, opt-in via MetalHLO); TPU (plugin detection, untested on hardware) |
| **Distributed** | DeviceMesh, TensorSharding, Shardy/SPMD (`executeGraphSharded`), DDP (`executeGraphReplicated`, `crossReplicaMean`, `dataParallelSGDStep`), tensor parallelism via Shardy. Tested on emulated CPU devices; real multi-GPU/TPU untested. No FSDP |
| **Checkpointing** | JSON format, Binary format, state_dict API |
| **Control Flow** | select, `stablehlo.while` via `scanXLA`, scan (with autodiff); `cond` only at the MLIRBuilder level |
| **Mixed Precision** | toReducedPrecision, toFullPrecision, to(dtype:), MixedPrecision.autocast |
| **Profiling** | Timing, Benchmarking, FLOPS estimation, Memory profiling, XLA tracing |
| **Error Handling** | TensorError types, TensorDebug utilities, TensorAssert helpers |
| **Testing Utils** | Gradient checking, Numerical stability tests |
| **Graph Optimization** | PassManager, DCE, CSE, ConstantFolding, AlgebraicSimplification. OperationFusion is disabled and performs no transformations; XLA does the fusion |

### Key Files

| File | Purpose |
|------|---------|
| `Sources/Core/Tensor.swift` | Tensor type, operations, mixed precision, `nn`/`optim` namespaces |
| `Sources/Core/Module.swift` | `Module` protocol and all `nn.*` layers |
| `Sources/Core/ValueLayers.swift`, `LayerGradient.swift`, `TangentOptimizer.swift` | Value-semantic `Layer` API, `modelGradient`, generic optimizers |
| `Sources/Core/NNGradient.swift` | `parameterGradients` autodiff bridge for `nn.*` |
| `Sources/Core/Autodiff.swift` | Differentiable conformance and VJPs |
| `Sources/Core/Optimizer.swift` | `optim.*` optimizers and LR schedulers |
| `Sources/Core/Data.swift` | Dataset, DataLoader, DistributedSampler, and tensor indexing |
| `Sources/Core/Scan.swift` | Loop operations (scan, scanXLA) with autodiff |
| `Sources/Core/Transforms.swift` | PyTorch-compatible data transforms |
| `Sources/Core/Profiling.swift` | Timing, benchmarking, FLOPS estimation |
| `Sources/Core/TensorError.swift` | Error types and debugging utilities |
| `Sources/Core/Distributed.swift`, `MultiHost.swift` | Distributed Tensor helpers, DDP step, multi-host config |
| `Sources/LazyTensor/` | Lazy execution engine, replicated/sharded runners |
| `Sources/LazyTensor/Optimization/` | Graph optimization passes (PassManager, DCE, CSE, constant folding, algebraic simplification) |
| `Sources/StableHLO/` | MLIR code generation, Shardy sharding types |
| `Sources/XLARuntime/` | XLA/PJRT integration, TPU detection, MetalHLO backend |
| `Sources/CXLARuntime/` | C wrapper over the PJRT C API (plugin loaded at runtime) |
| `Examples/MNIST/` | MNIST training example |
| `Examples/ValueLayers/` | Value-semantic MLP trained with `modelGradient` + `Adam` |
| `Examples/BuildingSimulation/` | Differentiable physics simulation (PyTorch port) |
| `Documentation/TPU_DEPLOYMENT.md` | TPU deployment guide |

### Examples

| Example | Description |
|---------|-------------|
| **MNIST** | Handwritten digit classification with MLP, real data loading and autodiff training |
| **ValueLayers** | The README Quick Example: a value-semantic MLP held as `some Layer`, trained with `modelGradient` and the generic `Adam` |
| **Metal** | Metal backend demo via MetalHLO (built only with `MAGMA_ENABLE_METAL=1` on macOS) |
| **BuildingSimulation** | Port of PyTorch building thermal simulation benchmark from [differentiable-swift-examples](https://github.com/PassiveLogic/differentiable-swift-examples). Demonstrates differentiable multi-timestep simulation with gradient computation through loops. |
| **Benchmarks** | Performance benchmarking suite for matrix operations. Measures GFLOPS for matmul at various sizes (256×256 to 4096×4096). Includes framework for element-wise, activation, reduction, and layer benchmarks. |
| **DistributedMNIST** | (Planned) Data parallel MNIST training across multiple devices |
| **LargeModelTraining** | (Planned) FSDP + tensor parallelism example for large transformer models |

---

## Milestones

| Milestone | Status | Key Deliverable |
|-----------|--------|-----------------|
| M0: Repo Setup | ✅ Complete | CI/CD working, can build |
| M1: First Execution | ✅ Complete | `x * y + z` runs on XLA |
| M2: First Model | ✅ Complete | MLP forward pass works |
| M3: Autodiff | ✅ Complete | Gradients computed correctly |
| M4: Optimizers | ✅ Complete | SGD, Adam working |
| M5: Attention | ✅ Complete | MultiheadAttention works |
| M6: Transformer | ✅ Complete | TransformerEncoderLayer + positional encoding |
| M7: Decoder | ✅ Complete | TransformerDecoderLayer with cross-attention |
| M8: RNN | ✅ Complete | RNN/LSTM/GRU with bidirectional support |
| M9: Slicing | ✅ Complete | Advanced slicing, comparison ops, boolean masking |
| M10: Element Indexing | ✅ Complete | Subscript indexing (`tensor[i]`, `tensor[i,j]`) with autodiff |
| M11: Building Simulation | ✅ Complete | PyTorch port of differentiable physics simulation |
| M12: Embedding Layer | ✅ Complete | nn.Embedding with gather/scatter, VJP for NLP models |
| M13: Model Checkpointing | ✅ Complete | JSON and binary checkpoint formats, state_dict support |
| M14: Mixed Precision | ✅ Complete | bfloat16 support, toReducedPrecision/toFullPrecision, autocast |
| M15: Error Handling | ✅ Complete | TensorError types, TensorDebug utilities, TensorAssert helpers |
| M16: Benchmarks | ✅ Complete | Examples/Benchmarks with matmul GFLOPS measurement (peak ~71 GFLOPS) |
| M17: Graph Optimization | ✅ Complete | PassManager, DCE, CSE, ConstantFolding, AlgebraicSimplification (fusion left to XLA) |
| M18: Sharding Foundation | ✅ Complete | DeviceMesh, TensorSharding types, StableHLO `sdy` support, graph sharding |
| M19: SPMD API | ✅ Complete (emulated CPU) | `Tensor.sharded(on:_:)`, `makeGraph(mesh:)`, `executeGraphSharded`, in-graph `all_reduce` |
| M20: Auto-Sharding | 🟡 Partial | Shardy propagation + partitioning inside XLA; no `autoShard()` API |
| M21: DDP | ✅ Complete (emulated CPU) | `executeGraphReplicated`, `crossReplicaMean`, `dataParallelSGDStep`, `Optimizer.step(syncing:)` |
| M22: FSDP | Not Started | FullyShardedDataParallel with ZeRO-style sharding |
| M23: Tensor Parallelism | 🟡 Partial | Tensor parallelism via Shardy verified; no Column/Row parallel layer types |
| M24: Multi-Host | 🟡 Partial | `MultiHostConfig` + `DistributedSampler.multiHost`; no coordination service or distributed checkpointing |
| M25: 0.1.0-alpha.1 | 🚧 Preparing | First alpha release |

---

## Next Steps

### Completed ✅
- ~~**Real MNIST Training**~~ - Downloads from Google Storage, parses IDX format, trains with real autodiff
- ~~**Embedding Layer**~~ - nn.Embedding with gather/scatter and backward pass support
- ~~**Model Checkpointing**~~ - JSON format, binary format, state_dict API (PyTorch-compatible)
- ~~**Control Flow in XLA**~~ - `stablehlo.while` loops (via `scanXLA`) and scan with autodiff support
- ~~**Distributed Training (Phase 4.5)**~~ - DDP and Shardy/SPMD, tested on emulated CPU devices
- ~~**Value-semantic layer API**~~ - `Layer`, `sequential { }`, `modelGradient`, generic `Adam`
- ~~**nn.* autodiff bridge**~~ - `parameterGradients` with identity-keyed `optimizer.step(_:)`
- ~~**Conv2D Layer**~~ - nn.Conv2d for image models

### In Progress 🚧
1. ~~**GPU Support**~~ ✅ - single-device CUDA execution verified (NVIDIA GB10)
2. ~~**Mixed Precision**~~ ✅ - bfloat16 training support (toReducedPrecision, toFullPrecision, MixedPrecision.autocast)
3. ~~**Performance Benchmarking**~~ ✅ - Examples/Benchmarks suite measuring matmul GFLOPS (peak ~71 GFLOPS on 4096×4096)
4. ~~**DataLoader Shuffling**~~ ✅ - Implemented in SimpleBatchLoader with dropLast support
5. ~~**Data Transforms**~~ ✅ - PyTorch-compatible transforms module (Normalize, Flip, Crop, Pad, Grayscale, Compose, Lambda)
6. ~~**Improved Error Messages**~~ ✅ - TensorError enum with detailed descriptions, TensorDebug utilities, TensorAssert helpers

### Up Next 📋
1. **0.1.0-alpha.1 release**: finish docs, tag, and publish.

### Beyond the alpha
- **Naming unification**: the value-semantic API (`Linear`, `Adam`, `sequential`)
  and the `nn.*`/`optim.*` API use overlapping names and different initializers.
  Unify or clearly separate them.
- **CI with a real CPU PJRT plugin**: CI builds everything and runs
  `StableHLOTests`, `LazyTensorTests` and `XLARuntimeTests` without a plugin. The
  plugin-backed suites are skipped there, and the Core suites (`MagmaTests`) are
  not run in CI.
- **Multi-GPU and TPU validation**: run the DDP, SPMD and tensor-parallel suites
  on a real multi-GPU host and a TPU board. So far they are only verified on
  emulated CPU devices, and on CUDA only single-GPU is verified.
- **Fusion**: `OperationFusionPass` is a disabled no-op, and all fusion is left to
  XLA/MetalHLO. Decide whether Magma needs graph-level fusion at all.
- Remaining distributed work: FSDP, a multi-host coordination service,
  distributed checkpointing. See [Phase 4.5](#phase-45-distributed-training-shardyspmd-integration--single-host-implemented).

---

## Success Metrics

### Correctness
- All ops match PyTorch output within tolerance
- Gradients verified against numerical differentiation
- No silent numerical issues

### Performance
- Within 2x of PyTorch on common models
- Compilation cache hit rate >95% during training
- Memory usage comparable to PyTorch

### Usability
- PyTorch users productive within 1 hour
- Error messages actionable
- Documentation complete

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Swift autodiff compiler bugs | High | Track issues, workarounds, contribute fixes |
| XLA API changes | Medium | Pin versions, abstract behind layer |
| Performance gaps vs PyTorch | Medium | Profile early, optimize hot paths |
| Dynamic shapes | Medium | Bucket shapes, interpreter fallback |
| Limited contributors | Medium | Good docs, welcoming community |
| Shardy API changes | Medium | Track OpenXLA releases, maintain compatibility layer |
| Multi-device debugging complexity | Medium | Comprehensive logging, device-specific error messages |
| Cross-host network latency | Medium | Optimize collective algorithms, overlap compute/comm |
| Sharding configuration complexity | Low | Provide high-level APIs (autoShard), good defaults |
