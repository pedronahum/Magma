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
Phase 4.5: Distributed Training    [Next] ← Shardy/SPMD Integration
    ↓
Phase 5: Production Readiness      [IN PROGRESS] 🚧
    ↓
Phase 6: Ecosystem                 [Ongoing]
```

**Current Status**: v0.1.0 Release Ready! 650+ tests passing, real MNIST data loader with autodiff training, full Transformer encoder/decoder, RNN/LSTM/GRU, advanced slicing, comparison ops, control flow, model checkpointing, PyTorch-compatible transforms, gradient checking utilities, numerical stability tests, XLA profiling/benchmarking utilities, mixed precision (bfloat16) support, comprehensive error handling, proper broadcasting, device transfer, one-hot encoding, and performance benchmarks example (peak ~71 GFLOPS on CPU)

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
  - [x] Clone TaylorTorch into `Legacy/TaylorTorch/`
  - [x] Clone SwiftIR into `Legacy/SwiftIR/`
  - [x] Document what to reuse from each
  - [x] Create mapping: TaylorTorch API → new implementation
  - [x] Create mapping: SwiftIR internals → new layers

- [x] **Package Structure**
  - [x] Create Package.swift with all targets
  - [x] Set up module structure (5 layers)
  - [x] Configure build settings for XLA (conditional via MAGMA_ENABLE_XLA)
  - [x] Add development container (devcontainer.json)

- [x] **Documentation**
  - [x] ARCHITECTURE.md
  - [x] ROADMAP.md (this file)
  - [x] CONTRIBUTING.md
  - [ ] API.md (stub)

### Exit Criteria
- [x] `swift build` succeeds (stub modules)
- [x] CI workflow configured (`.github/workflows/ci.yml`)
- [x] 49 tests passing
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
- [x] `TensorType`, `DType`, `Shape` types
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
- [x] `Device` - Swift device abstraction
- [x] `LazyTensorBarrier()` - trigger compilation/execution
- [x] `StableHLOEmitter` - IRGraph → MLIR
- [x] `CompilationCache` - hash-based caching
- [x] `TensorRegistry` - track pending tensors
- [x] Tests: build graph, emit MLIR, execute via XLARuntime

### Week 3-4: User Layer (4) - Basics

#### Layer 4: Torch (Tensor Only)
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
- [x] All layer tests pass (65+ tests)
- [x] Documentation updated

---

## Phase 2: Basic Neural Networks ✅

**Goal**: Train a simple model (MLP on MNIST)

### Week 5-6: Layer API ✅

#### nn.Module
- [x] `Module` protocol definition
- [x] `Parameter` wrapper type
- [x] `@differentiable` requirements
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
- [x] 145+ tests passing

---

## Phase 3: Training Infrastructure ✅

**Goal**: Full training loop with optimizer and data loading

### Week 9-10: Optimizers ✅

#### Optimizer Protocol
- [x] `Optimizer` protocol definition
- [x] `step(gradients:)` method
- [x] `zeroGrad()` method
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
- [x] 178+ tests passing
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
- [x] `while_loop` for dynamic iteration
- [x] `cond` for branching
- [x] `scan` for fixed-iteration loops with autodiff support

### Week 14-15: Multi-Device

#### TPU Support ✅
- [x] `Backend.tpu` with automatic plugin detection
- [x] `Backend.isAvailable` - check if backend plugin exists
- [x] `Backend.bestAvailable` - auto-select best backend (TPU > GPU > CPU)
- [x] `TPUEnvironment` - detect TPU VM, topology, chip count
- [x] Cloud TPU VM deployment documentation

#### Device Management (Partially Done)
- [x] TPU support (via libtpu.so on Cloud TPU VMs)
- [x] GPU support (CUDA plugin) — single-device execution verified (NVIDIA GB10)
- [ ] Multi-GPU data parallel
- [ ] `DistributedDataParallel` (basic)

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

## Phase 4.5: Distributed Training (Shardy/SPMD Integration)

**Goal**: Enable distributed training across multiple devices (TPUs/GPUs) using Google's Shardy tensor partitioning system

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
| **Magma** | Shardy-based SPMD (first Swift framework with this capability) |

Swift's compile-time type safety + Shardy's automatic sharding = safer distributed training than Python alternatives.

#### TorchTPU Context

Google's [TorchTPU](https://github.com/pytorch/xla) initiative (with Meta collaboration, announced December 2025) aims to make TPUs feel native to PyTorch. Key developments:

- **RFC #9684** (October 2025): Proposed more native PyTorch/TPU integration
- **Meta partnership**: Active collaboration on PyTorch/XLA, potential TPU cloud usage from 2026
- **TPU v7 (Ironwood)**: 10x performance over v5p, GA November 2025

Magma can leverage the same Shardy infrastructure that powers TorchTPU's SPMD capabilities.

#### Legacy Foundation

The `Legacy/SwiftIR/` codebase already contains substantial Shardy infrastructure:

| File | Description |
|------|-------------|
| `Sources/SwiftIRSharding/DeviceMesh.swift` | Device mesh topology types |
| `Sources/SwiftIRSharding/TensorSharding.swift` | Sharding specification types |
| `Sources/SwiftIRSharding/ShardingPipeline.swift` | Integration with sdy_opt |
| `Sources/SwiftIRSharding/SdyDialect.swift` | Shardy dialect bindings |
| `scripts/install-swiftir-ubuntu.sh` | Builds libsdy_capi.so, sdy_opt |

---

### Week 15-16: Sharding Foundation

#### Device Mesh Types

Port and adapt mesh types from SwiftIRSharding:

- [ ] `MeshAxis` - Named axis with size (e.g., `MeshAxis(name: "x", size: 4)`)
- [ ] `DeviceMesh` - Multi-dimensional device topology
  - [ ] `DeviceMesh.linear(name:axisName:size:)` - 1D mesh
  - [ ] `DeviceMesh.grid(name:rows:cols:)` - 2D mesh
  - [ ] `DeviceMesh.grid(name:rows:cols:rowAxis:colAxis:)` - 2D with custom axis names
  - [ ] `deviceCount` - Total devices in mesh
  - [ ] `axis(named:)` - Get axis by name
  - [ ] `toAttribute(context:)` - Convert to MLIR attribute
  - [ ] `mlirText` - Generate MLIR textual representation

```swift
// Target API
let mesh = DeviceMesh.grid(name: "tpu_mesh", rows: 2, cols: 4)  // 8 TPUs
print(mesh.deviceCount)  // 8

// Or with semantic names for different parallelism strategies
let mesh = DeviceMesh(name: "training_mesh", axes: [
    MeshAxis(name: "data", size: 4),   // Data parallelism
    MeshAxis(name: "model", size: 2)   // Tensor parallelism
])
```

#### Tensor Sharding Specification Types

- [ ] `AxisRef` - Reference to a mesh axis
  - [ ] Simple axis reference: `AxisRef("x")`
  - [ ] Sub-axis support for hierarchical partitioning: `AxisRef(name: "x", subAxisInfo: SubAxisInfo(preSize: 1, size: 2))`
- [ ] `SubAxisInfo` - Hierarchical axis splitting
- [ ] `DimensionSharding` - Per-dimension sharding specification
  - [ ] `DimensionSharding("x")` - Shard along axis "x"
  - [ ] `DimensionSharding.replicated` - Fully replicated (closed)
  - [ ] `DimensionSharding.open` - Allow propagation to determine
  - [ ] Priority support for propagation ordering
- [ ] `TensorSharding` - Complete tensor sharding specification
  - [ ] `TensorSharding(meshName:dimShardings:)` - Full specification
  - [ ] `TensorSharding(meshName:axisNames:)` - Convenience with axis names
  - [ ] `TensorSharding.replicated(meshName:rank:)` - Fully replicated
  - [ ] `TensorSharding.open(meshName:rank:)` - Open for propagation
  - [ ] `replicatedAxes` - Axes along which tensor is replicated
  - [ ] `unreducedAxes` - Axes not yet reduced (for reduce ops)

```swift
// Target API - Shard a 2D tensor: batch on "data", features replicated
let sharding = TensorSharding(
    meshName: "training_mesh",
    dimShardings: [
        DimensionSharding("data"),      // Batch dimension sharded
        .replicated                      // Feature dimension replicated
    ]
)

// Or using convenience initializer
let sharding = TensorSharding(meshName: "training_mesh", axisNames: ["data", nil])
```

#### StableHLO Sharding Support

Extend `Sources/StableHLO/` to emit sharding operations:

- [ ] `sdy.mesh` operation generation
- [ ] `sdy.sharding` attribute attachment
- [ ] `sdy.sharding_constraint` for intermediate tensors
- [ ] `sdy.manual_computation` for user-defined partitioned regions

```swift
// MLIRBuilder extensions
extension MLIRBuilder {
    func emitMesh(_ mesh: DeviceMesh)
    func addShardingConstraint(_ value: Value, sharding: TensorSharding)
    func emitShardingAttribute(_ sharding: TensorSharding) -> String
}
```

#### LazyTensor Sharding Integration

Extend IR graph to track sharding:

- [ ] Add `sharding: TensorSharding?` to `IRNode`
- [ ] Sharding propagation through graph operations
- [ ] Sharding validation (mesh compatibility, dimension matching)
- [ ] `StableHLOEmitter` updates for sharding emission

---

### Week 17-18: SPMD User API

#### Core Sharding API

PyTorch/XLA-inspired API for tensor sharding:

- [ ] `Tensor.markSharding(mesh:partitionSpec:)` - Annotate tensor with sharding
- [ ] `XLAShardedTensor` wrapper type (optional, for explicit tracking)
- [ ] `PartitionSpec` type for dimension-to-axis mapping
- [ ] Sharding validation and error messages

```swift
// Target API - inspired by PyTorch/XLA
let mesh = Mesh(shape: (2, 4), axisNames: ("data", "model"))

// Mark input sharding - batch dimension across 'data' axis
var inputs = inputs.markSharding(mesh: mesh, partitionSpec: ("data", nil))

// Mark weight sharding for tensor parallelism
var weights = model.linear.weight.markSharding(
    mesh: mesh,
    partitionSpec: (nil, "model")  // Shard output features across 'model' axis
)
```

#### Partition Spec

- [ ] `PartitionSpec` - Maps tensor dimensions to mesh axes
  - [ ] Tuple-based initialization: `("data", nil, "model")`
  - [ ] Named dimension support
  - [ ] Validation against mesh and tensor rank
- [ ] `nil` means replicated along that dimension
- [ ] String axis name means sharded along that axis

```swift
// PartitionSpec examples
let spec1 = PartitionSpec("data", nil)        // [sharded, replicated]
let spec2 = PartitionSpec("data", "model")    // [sharded, sharded]
let spec3 = PartitionSpec(nil, nil, nil)      // Fully replicated 3D tensor
```

#### Mesh Context Manager

- [ ] `Mesh.with(_:body:)` - Scoped mesh context
- [ ] Default mesh resolution
- [ ] Nested mesh support for complex topologies

```swift
// Target API
Mesh.with(mesh) {
    // All sharding operations in this scope use 'mesh'
    let shardedInputs = inputs.shard(along: 0, axis: "data")
    let output = model(shardedInputs)
}
```

#### Collective Operations

Automatic insertion of collective operations:

- [ ] `allReduce` - Sum/mean gradients across devices
- [ ] `allGather` - Gather sharded tensor to all devices
- [ ] `reduceScatter` - Reduce and scatter result
- [ ] `allToAll` - Redistribute tensor sharding
- [ ] Collective operation fusion for efficiency

---

### Week 19-20: Automatic Sharding Propagation

#### Shardy Integration

- [ ] `sdy_opt` binary integration via `SdyOptRunner`
- [ ] MLIR module preprocessing for Shardy
- [ ] Sharding propagation pass execution
- [ ] Post-propagation MLIR parsing

```swift
// SdyOptRunner - execute Shardy propagation
struct SdyOptRunner {
    static func propagate(mlir: String, passes: [String]) throws -> String
    static func runPipeline(_ pipeline: ShardingPipeline, on mlir: String) throws -> String
}
```

#### Propagation Configuration

- [ ] `ShardingPropagationConfig` - Control propagation behavior
  - [ ] User priorities (e.g., "do batch parallelism first, then ZeRO")
  - [ ] Op-based priorities (e.g., "element-wise ops first, then matmuls")
  - [ ] Cost model selection
  - [ ] Aggressive vs conservative propagation modes

```swift
// Target API
let config = ShardingPropagationConfig()
    .priority(.batchParallelism, then: .tensorParallelism)
    .costModel(.memory)  // Optimize for memory over compute
    .propagationMode(.aggressive)
```

#### Auto-Sharding API

High-level API for automatic model sharding:

- [ ] `model.autoShard(mesh:strategy:)` - Automatically shard entire model
- [ ] Predefined strategies: `.dataParallel`, `.tensorParallel`, `.fsdp`, `.hybrid`
- [ ] Custom strategy support via `ShardingStrategy` protocol

```swift
// Target API - simple data parallelism
let mesh = DeviceMesh.linear(name: "devices", size: 8)
let shardedModel = model.autoShard(mesh: mesh, strategy: .dataParallel)

// Hybrid parallelism for large models
let mesh = DeviceMesh.grid(name: "mesh", rows: 4, cols: 2)
let shardedModel = model.autoShard(mesh: mesh, strategy: .hybrid(
    dataAxis: "x",      // 4-way data parallel
    modelAxis: "y"      // 2-way tensor parallel
))
```

#### Module Sharding Annotations

- [ ] `@Sharded` property wrapper for Module parameters
- [ ] `Module.shardingSpec` - Declare per-parameter sharding
- [ ] Automatic gradient aggregation setup

```swift
// Target API
struct LargeTransformer: Module {
    // Shard embedding table across model axis
    @Sharded(axis: "model", dim: 0)
    var embedding: nn.Embedding

    // Shard attention weights
    @Sharded(axis: "model", dim: 1)
    var attention: nn.MultiheadAttention

    // Replicated small layers
    var layerNorm: nn.LayerNorm  // Automatically replicated
}
```

---

### Week 21-22: Distributed Training Wrappers

#### DistributedDataParallel (DDP)

Simple data parallelism wrapper:

- [ ] `DistributedDataParallel` wrapper struct
- [ ] Automatic input sharding across batch dimension
- [ ] Gradient all-reduce after backward pass
- [ ] Bucket gradient all-reduce for efficiency
- [ ] Gradient compression options (optional)

```swift
// Target API
let mesh = DeviceMesh.linear(name: "workers", size: 8)
let ddpModel = DistributedDataParallel(model, mesh: mesh)

for batch in dataLoader {
    let (loss, grads) = valueWithGradient(at: ddpModel.module) { m in
        let output = m(batch.inputs)
        return crossEntropy(output, batch.labels)
    }
    // Gradients automatically all-reduced across devices
    optimizer.step(grads)
}
```

#### FullyShardedDataParallel (FSDP)

Memory-efficient distributed training:

- [ ] `FullyShardedDataParallel` wrapper (inspired by PyTorch FSDP)
- [ ] Parameter sharding across devices
- [ ] All-gather parameters before forward
- [ ] Reduce-scatter gradients after backward
- [ ] Configurable sharding strategies:
  - [ ] `ShardingStrategy.fullShard` - Shard params, grads, and optimizer states
  - [ ] `ShardingStrategy.shardGradOp` - Shard grads and optimizer states only
  - [ ] `ShardingStrategy.noShard` - DDP-style, no parameter sharding
- [ ] Mixed precision integration
- [ ] Activation checkpointing integration

```swift
// Target API
let mesh = DeviceMesh.linear(name: "workers", size: 8)
let fsdpModel = FullyShardedDataParallel(
    model,
    mesh: mesh,
    shardingStrategy: .fullShard,
    mixedPrecision: .bfloat16
)

// Training loop - parameters automatically gathered/scattered
for batch in dataLoader {
    let (loss, grads) = valueWithGradient(at: fsdpModel) { m in
        m(batch.inputs).loss(batch.labels)
    }
    optimizer.step(grads)
}
```

#### Tensor Parallelism Utilities

For large model layers:

- [ ] `nn.ColumnParallelLinear` - Shard output features
- [ ] `nn.RowParallelLinear` - Shard input features
- [ ] `nn.ParallelEmbedding` - Shard embedding table
- [ ] `nn.ParallelMultiheadAttention` - Distributed attention
- [ ] Automatic all-reduce/all-gather insertion

```swift
// Target API for tensor-parallel layers
let mesh = DeviceMesh.linear(name: "model", size: 4)

// Column parallel: output features sharded across 4 devices
let columnLinear = nn.ColumnParallelLinear(
    inFeatures: 1024,
    outFeatures: 4096,
    mesh: mesh,
    axis: "model"
)

// Row parallel: input features sharded, output all-reduced
let rowLinear = nn.RowParallelLinear(
    inFeatures: 4096,
    outFeatures: 1024,
    mesh: mesh,
    axis: "model"
)
```

---

### Week 23-24: Multi-Host & Advanced Features

#### Multi-Host Training

Support for training across multiple machines:

- [ ] Multi-host mesh initialization
- [ ] Cross-host collective operations via PJRT
- [ ] Host-to-host communication setup
- [ ] Process group management
- [ ] Fault tolerance basics (checkpoint on failure)

```swift
// Target API
let topology = DistributedTopology.detect()  // Auto-detect hosts and devices
print("Hosts: \(topology.hostCount), Devices per host: \(topology.devicesPerHost)")

let mesh = DeviceMesh.fromTopology(
    topology,
    name: "global_mesh",
    hostAxis: "host",
    deviceAxis: "device"
)
```

#### Pipeline Parallelism (Basic)

For very deep models:

- [ ] `nn.PipelineStage` - Mark model stage boundaries
- [ ] `PipelineParallel` wrapper
- [ ] Micro-batch scheduling (GPipe-style)
- [ ] Inter-stage communication

```swift
// Target API - basic pipeline parallelism
let stages = [
    PipelineStage(model.embedding, device: 0),
    PipelineStage(model.encoderLayers[0..<6], device: 1),
    PipelineStage(model.encoderLayers[6..<12], device: 2),
    PipelineStage(model.outputHead, device: 3)
]

let pipelineModel = PipelineParallel(stages: stages, microBatches: 4)
```

#### Distributed Checkpointing

- [ ] Sharded checkpoint save/load
- [ ] Efficient parallel I/O
- [ ] Checkpoint format compatible with resharding
- [ ] Async checkpointing (non-blocking)

```swift
// Target API
// Save sharded checkpoint - each device saves its shard
try fsdpModel.saveShardedCheckpoint(to: "checkpoint/step_1000/")

// Load with potentially different sharding
let newMesh = DeviceMesh.linear(name: "new", size: 16)  // Different device count
let restored = try FullyShardedDataParallel.loadShardedCheckpoint(
    from: "checkpoint/step_1000/",
    mesh: newMesh
)
```

#### Distributed Data Loading

- [ ] `DistributedSampler` - Partition dataset across devices
- [ ] Automatic shard-aware batching
- [ ] Prefetching across shards

```swift
// Target API
let sampler = DistributedSampler(
    dataset: trainDataset,
    mesh: mesh,
    shuffle: true,
    seed: 42
)

let dataLoader = DataLoader(
    dataset: trainDataset,
    sampler: sampler,
    batchSize: 32  // Per-device batch size
)
```

---

### Supported Parallelism Strategies Summary

| Strategy | Description | Shardy Implementation |
|----------|-------------|----------------------|
| **Data Parallel** | Replicate model, shard data batches | Shard batch dimension across mesh axis |
| **Tensor Parallel** | Shard large weight matrices across devices | Shard weight dimensions, auto all-reduce |
| **FSDP** | Shard parameters, gradients, and optimizer states | Combined sharding with all-gather/reduce-scatter |
| **Pipeline Parallel** | Shard model layers across devices | `sdy.manual_computation` regions |
| **ZeRO Stage 1** | Shard optimizer states only | Optimizer state sharding |
| **ZeRO Stage 2** | Shard optimizer states + gradients | Gradient reduce-scatter |
| **ZeRO Stage 3** | Full parameter sharding (= FSDP) | Full sharding with all-gather |
| **Hybrid** | Combine data + tensor parallelism | Multi-axis mesh with combined specs |

---

### Exit Criteria

- [ ] `DeviceMesh` and `TensorSharding` types working
- [ ] `Tensor.markSharding()` API functional
- [ ] Basic data parallelism example running on multi-device
- [ ] `DistributedDataParallel` wrapper working
- [ ] `FullyShardedDataParallel` wrapper working
- [ ] Shardy propagation integrated via `sdy_opt`
- [ ] Multi-host training example (2+ hosts)
- [ ] Distributed checkpointing working
- [ ] Documentation: Tutorial on distributed training
- [ ] Tests: 50+ distributed training tests

---

### Key Files to Create

| File | Purpose |
|------|---------|
| `Sources/Torch/Distributed/DeviceMesh.swift` | Device mesh topology types |
| `Sources/Torch/Distributed/TensorSharding.swift` | Sharding specification types |
| `Sources/Torch/Distributed/PartitionSpec.swift` | Dimension-to-axis mapping |
| `Sources/Torch/Distributed/ShardingAPI.swift` | `markSharding`, `autoShard` APIs |
| `Sources/Torch/Distributed/Collectives.swift` | All-reduce, all-gather, etc. |
| `Sources/Torch/Distributed/DDP.swift` | DistributedDataParallel wrapper |
| `Sources/Torch/Distributed/FSDP.swift` | FullyShardedDataParallel wrapper |
| `Sources/Torch/Distributed/TensorParallel.swift` | Column/Row parallel layers |
| `Sources/Torch/Distributed/Pipeline.swift` | Pipeline parallelism |
| `Sources/Torch/Distributed/Checkpoint.swift` | Distributed checkpointing |
| `Sources/StableHLO/Sharding.swift` | Sharding MLIR generation |
| `Sources/LazyTensor/ShardedExecution.swift` | Multi-device execution |
| `Sources/XLARuntime/MultiDevice.swift` | PJRT multi-device support |
| `Sources/Shardy/SdyOptRunner.swift` | Shardy tool integration |
| `Sources/Shardy/ShardingPropagation.swift` | Propagation configuration |
| `Documentation/DISTRIBUTED_TRAINING.md` | Comprehensive guide |
| `Examples/DistributedMNIST/` | Data parallel MNIST example |
| `Examples/LargeModelTraining/` | FSDP + tensor parallel example |

---

### External Dependencies

| Dependency | Purpose | Source |
|------------|---------|--------|
| `libsdy_capi.so` | Shardy C API | Build from [openxla/shardy](https://github.com/openxla/shardy) |
| `sdy_opt` | Shardy optimization tool | Build from openxla/shardy |
| PJRT multi-device | Multi-device execution | Already in XLA |

Build script location: `Legacy/SwiftIR/scripts/install-swiftir-ubuntu.sh` (includes Shardy build)

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
- [x] `OperationFusionPass` - Pattern matching for fused operations

#### Fusion Patterns ✅
- [x] `ScaledDotProductAttentionPattern` - Fuse Q·K^T/√d · V attention
- [x] `LayerNormPattern` - Fuse layer normalization operations
- [x] `RMSNormPattern` - Fuse RMS normalization operations
- [x] `MatMulBiasActivationPattern` - Fuse matmul + bias + activation
- [x] `SoftmaxPattern` - Fuse exp-sum-div softmax pattern
- [x] `GeluPattern` - Fuse GELU activation pattern

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
- [x] Gradient checking for all ops ✅
  - `gradcheck` function comparing autodiff vs numerical gradients
  - `numericalGradient` and `numericalGradientForward` utilities
  - `numericalJacobian` for vector-valued functions
  - 34 gradient checking tests covering arithmetic, activations, matrix ops
- [x] Numerical stability tests ✅
  - Very large/small values handling
  - Near-zero division and underflow/overflow
  - NaN/Inf propagation tests
  - Softmax stability with extreme values
  - Gradient stability tests
  - 45+ numerical stability tests
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
- [ ] v0.1.0 release

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
- [ ] macOS (Apple Silicon)
- [x] Linux (x86_64)
- [x] TPU support (Cloud TPU VMs) - See [TPU_DEPLOYMENT.md](TPU_DEPLOYMENT.md)
- [ ] GPU support (CUDA)
- [ ] Metal backend (future)

### Community
- [ ] Discord/Slack
- [ ] GitHub Discussions
- [ ] Blog posts
- [ ] Conference talks

---

## Current Implementation Summary

### What's Working (650+ tests)

| Category | Components |
|----------|------------|
| **Tensor** | Creation, arithmetic, matrix ops, reductions, activations, broadcasting, slicing (advanced with step/negative indices), subscript indexing (`tensor[i]`, `tensor[i,j]`), concat, stack |
| **Comparison** | lessThan, greaterThan, equalTo, operators (.<, .>, .<=, .>=, .==, .!=), where_, maskedSelect |
| **Autodiff** | Full VJP support for common ops, `gradient()`, `valueWithGradient()` |
| **Layers** | Linear, Conv2d, BatchNorm, LayerNorm, Dropout, Flatten, Sequential, Pooling |
| **Activations** | ReLU, Sigmoid, Tanh, GELU, LeakyReLU, SiLU, ELU, Hardtanh |
| **Attention** | ScaledDotProductAttention, MultiheadAttention |
| **Transformer** | TransformerEncoderLayer, TransformerDecoderLayer, SinusoidalPositionalEncoding, LearnedPositionalEmbedding, CausalMask |
| **Recurrent** | RNNCell, LSTMCell, GRUCell, RNN, LSTM, GRU (with multi-layer and bidirectional support) |
| **Optimizers** | SGD (momentum, Nesterov), Adam/AdamW |
| **Schedulers** | StepLR, ExponentialLR, CosineAnnealingLR, WarmupLR, WarmupCosine |
| **Data** | Dataset protocol, SimpleBatchLoader (shuffle, dropLast) |
| **Transforms** | Compose, Normalize, RandomHorizontalFlip, RandomVerticalFlip, CenterCrop, RandomCrop, Pad, Grayscale, Lambda |
| **Backends** | CPU (via PJRT), TPU (Cloud TPU VMs), GPU (CUDA, single-device), Metal (macOS) |
| **Distributed** | Shardy integration planned (DeviceMesh, TensorSharding, SPMD, DDP, FSDP) |
| **Checkpointing** | JSON format, Binary format, state_dict API |
| **Control Flow** | select, while_loop, cond, scan (with autodiff) |
| **Mixed Precision** | toReducedPrecision, toFullPrecision, to(dtype:), MixedPrecision.autocast |
| **Profiling** | Timing, Benchmarking, FLOPS estimation, Memory profiling, XLA tracing |
| **Error Handling** | TensorError types, TensorDebug utilities, TensorAssert helpers |
| **Testing Utils** | Gradient checking, Numerical stability tests |
| **Graph Optimization** | PassManager, DCE, CSE, ConstantFolding, AlgebraicSimplification, OperationFusion |
| **Fusion Patterns** | ScaledDotProductAttention, LayerNorm, RMSNorm, MatMulBiasActivation, Softmax, Gelu |

### Key Files

| File | Purpose |
|------|---------|
| `Sources/Torch/Torch.swift` | Tensor type, operations, mixed precision |
| `Sources/Torch/Module.swift` | Module protocol and all layers |
| `Sources/Torch/Autodiff.swift` | Differentiable conformance and VJPs |
| `Sources/Torch/Optimizer.swift` | Optimizers and LR schedulers |
| `Sources/Torch/Data.swift` | Dataset, DataLoader, and tensor indexing |
| `Sources/Torch/Scan.swift` | Loop operations (scan, while_loop) with autodiff |
| `Sources/Torch/Transforms.swift` | PyTorch-compatible data transforms |
| `Sources/Torch/Profiling.swift` | Timing, benchmarking, FLOPS estimation |
| `Sources/Torch/TensorError.swift` | Error types and debugging utilities |
| `Sources/LazyTensor/` | Lazy execution engine |
| `Sources/LazyTensor/Optimization/` | Graph optimization passes (PassManager, DCE, CSE, fusion) |
| `Sources/StableHLO/` | MLIR code generation |
| `Sources/XLARuntime/` | XLA/PJRT integration, TPU detection |
| `Sources/Torch/Distributed/` | Distributed training (planned): DDP, FSDP, sharding |
| `Sources/Shardy/` | Shardy integration (planned): sdy_opt, propagation |
| `Examples/MNIST/` | MNIST training example |
| `Examples/BuildingSimulation/` | Differentiable physics simulation (PyTorch port) |
| `Documentation/TPU_DEPLOYMENT.md` | TPU deployment guide |

### Examples

| Example | Description |
|---------|-------------|
| **MNIST** | Handwritten digit classification with MLP, real data loading and autodiff training |
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
| M17: Graph Optimization | ✅ Complete | PassManager, DCE, CSE, ConstantFolding, AlgebraicSimplification, Fusion patterns |
| M18: Sharding Foundation | Not Started | DeviceMesh, TensorSharding types, StableHLO sharding support |
| M19: SPMD API | Not Started | `markSharding()`, `PartitionSpec`, collective operations |
| M20: Auto-Sharding | Not Started | Shardy propagation via sdy_opt, `autoShard()` API |
| M21: DDP | Not Started | DistributedDataParallel wrapper for multi-device training |
| M22: FSDP | Not Started | FullyShardedDataParallel with ZeRO-style sharding |
| M23: Tensor Parallelism | Not Started | Column/Row parallel layers for large models |
| M24: Multi-Host | Not Started | Cross-host training, distributed checkpointing |
| M25: v0.1.0 | 🚧 In Progress | Public release - blocking issues resolved |

---

## Next Steps

### Completed ✅
- ~~**Real MNIST Training**~~ - Downloads from Google Storage, parses IDX format, trains with real autodiff
- ~~**Embedding Layer**~~ - nn.Embedding with gather/scatter and backward pass support
- ~~**Model Checkpointing**~~ - JSON format, binary format, state_dict API (PyTorch-compatible)
- ~~**Control Flow in XLA**~~ - while_loop, cond, and scan with autodiff support
- ~~**Conv2D Layer**~~ - nn.Conv2d for image models

### In Progress 🚧
1. **GPU Support** - CUDA plugin integration for GPU execution
2. ~~**Mixed Precision**~~ ✅ - bfloat16 training support (toReducedPrecision, toFullPrecision, MixedPrecision.autocast)
3. ~~**Performance Benchmarking**~~ ✅ - Examples/Benchmarks suite measuring matmul GFLOPS (peak ~71 GFLOPS on 4096×4096)
4. ~~**DataLoader Shuffling**~~ ✅ - Implemented in SimpleBatchLoader with dropLast support
5. ~~**Data Transforms**~~ ✅ - PyTorch-compatible transforms module (Normalize, Flip, Crop, Pad, Grayscale, Compose, Lambda)
6. ~~**Improved Error Messages**~~ ✅ - TensorError enum with detailed descriptions, TensorDebug utilities, TensorAssert helpers

### Up Next 📋
1. **Distributed Training (Phase 4.5)** - Shardy/SPMD integration for multi-device training
   - Port DeviceMesh and TensorSharding from Legacy/SwiftIR
   - Implement `markSharding()` API (PyTorch/XLA-style)
   - Integrate Shardy propagation via sdy_opt
   - DistributedDataParallel wrapper
   - FullyShardedDataParallel (FSDP) wrapper
   - Tensor parallelism for large models
   - Multi-host training support
   - See [Phase 4.5](#phase-45-distributed-training-shardyspmd-integration) for full details

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
