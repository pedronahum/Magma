# Magma Architecture

## Overview

Magma is organized as a strictly layered Swift package. Each layer is its own
SwiftPM target with a single responsibility, and it only depends on layers below it
(see `Package.swift`).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LAYER 4: Core  (module `Magma`)                         │
│         Tensor, autodiff (VJPs), layers, optimizers, data, distributed      │
│   value-semantic API: sequential / Linear / Adam / modelGradient            │
│   reference-semantic API: nn.* / optim.* / parameterGradients               │
├─────────────────────────────────────────────────────────────────────────────┤
│                            LAYER 3: LazyTensor                              │
│              x10-style lazy tracing with explicit barriers                  │
│   LazyTensorHandle, IRNode, IRGraph, PassManager, StableHLOEmitter,         │
│   CompilationCache, LazyTensorBarrier, replicated/sharded graph runners     │
├─────────────────────────────────────────────────────────────────────────────┤
│                            LAYER 2: StableHLO                               │
│             Pure-Swift MLIR/StableHLO text generation (no deps)             │
│        MLIRBuilder, Value, TensorType, DType, Shardy sharding types         │
├─────────────────────────────────────────────────────────────────────────────┤
│                            LAYER 1: XLARuntime                              │
│                  Swift wrapper over PJRT: compile + execute                 │
│   PJRTClient, PJRTDevice, PJRTBuffer, PJRTExecutable, Backend, Device       │
│   + opt-in MetalHLO backend on macOS (MAGMA_ENABLE_METAL=1)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                            LAYER 0: CXLARuntime                             │
│             C wrapper over the PJRT C API (pjrt_c_api.h)                    │
│         PJRT plugin loaded at runtime with dlopen (no link-time XLA)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

Products: `Magma` (what users import), plus `LazyTensor`, `StableHLO` and
`XLARuntime` for advanced users. The Core layer lives in `Sources/Core` but its
module name is `Magma`.

## Design Principles

### 1. Strict Layering
- Each layer only imports layers below it
- No circular dependencies
- Clear API boundaries between layers

### 2. Testability at Every Layer
- The StableHLO layer is pure Swift, so it can be tested without XLA
- The LazyTensor tests build graphs and emit MLIR without executing them, so no
  PJRT plugin is needed
- The XLARuntime and Core suites execute on a real PJRT plugin. When no plugin is
  available they are skipped, not failed.

### 3. Replaceability
- The PJRT plugin is chosen at runtime (CPU, CUDA GPU or TPU), so the same binary
  runs on any backend that has a plugin
- MetalHLO is an alternative runtime for the same StableHLO text on macOS
- The StableHLO layer only produces MLIR text, so any consumer that accepts
  StableHLO can use it

### 4. Swift-Native Autodiff
- `Tensor` conforms to `Differentiable` (for floating-point scalars), and its ops
  have `@derivative` VJPs
- Gradients are lazy too: the pullback builds more graph nodes, which are compiled
  together with the forward pass at the next barrier
- No runtime tape and no Python interop

---

## Layer Details

### Layer 0: CXLARuntime

**Purpose**: A C wrapper over the PJRT C API that Swift can import easily.

**Contents**:
```
Sources/CXLARuntime/
├── include/
│   ├── pjrt_c_api.h            # Upstream PJRT C API header
│   ├── PJRTSimpleWrapper.h     # Swift-facing C API (PJRT_LoadPlugin, PJRT_CompileWrapper*, execute, buffers, ...)
│   └── PJRTProtoHelper.h
├── PJRTSimpleWrapper.c         # dlopen()s the plugin, resolves GetPjrtApi, wraps client/compile/execute/buffer calls
└── PJRTProtoHelper.cpp         # Hand-encodes CompileOptionsProto (replicas, partitions, SPMD/Shardy flags) without XLA's proto libs
```

**How the plugin is loaded**: `PJRT_LoadPlugin` calls `dlopen` on the plugin path
and resolves `GetPjrtApi`, so no XLA libraries are needed at build time. Only one
plugin can be loaded per process. A request for a different plugin fails loudly
instead of reusing the one already loaded.

**Dependencies**: none at build time (`libdl` on Linux). At run time it needs a
PJRT plugin (`pjrt_c_api_<backend>_plugin.so`/`.dylib`, or `libtpu.so`).

---

### Layer 1: XLARuntime

**Purpose**: A Swift wrapper around PJRT for device management, compilation and
execution. On macOS it also has an opt-in MetalHLO backend.

**Contents**:
```
Sources/XLARuntime/
├── XLARuntime.swift        # Backend, TPUEnvironment, Device, XLAError, ElementType,
│                           # PJRTClient, PJRTDevice, PJRTBuffer, PJRTExecutable
└── MetalHLORuntime.swift   # MetalBackend, MetalHLOClient/Buffer/Executable
                            # (compiled only when MetalHLO is available: macOS + MAGMA_ENABLE_METAL=1)
```

**Plugin discovery**: `Backend.pluginPath()` looks in `$MAGMA_XLA_PATH` first.
Otherwise it checks the standard system directories, and for TPU it checks
`TPU_LIBRARY_PATH` and then the usual `libtpu.so` locations. Accelerator clients
(GPU, TPU) reserve most of device memory, so creating a second live accelerator
client throws, unless `MAGMA_ALLOW_CONCURRENT_ACCEL_CLIENTS=1` is set.

**Key Types** (abridged from `XLARuntime.swift`):
```swift
public enum Backend: String, Sendable {
    case cpu, gpu, tpu
    case metal  // Metal GPU via MetalHLO (macOS only)
    public var isAvailable: Bool
    public static var availableBackends: [Backend]
    public static var bestAvailable: Backend
}

public struct Device: Hashable, Sendable {
    public let backend: Backend
    public let index: Int
    public static let `default`: Device   // .cpu, index 0
}

public final class PJRTClient: @unchecked Sendable {
    public let backend: Backend
    public private(set) var devices: [PJRTDevice]
    public var deviceCount: Int { get }

    // cpuDeviceCount asks the CPU plugin for N virtual devices (multi-device tests)
    public static func create(backend: Backend = .cpu, cpuDeviceCount: Int? = nil) throws -> PJRTClient
    public func createBuffer<T>(_ data: [T], shape: [Int], elementType: ElementType,
                                device: PJRTDevice? = nil) throws -> PJRTBuffer
    public func compile(_ mlir: String) throws -> PJRTExecutable
    public func compile(_ mlir: String, numReplicas: Int = 1, numPartitions: Int = 1,
                        useSPMDPartitioning: Bool = true,
                        useShardyPartitioner: Bool = false) throws -> PJRTExecutable

    // Buffer distribution for multi-device execution
    public func replicate<T>(_ hostData: [T], shape: [Int], elementType: ElementType, count: Int) throws -> [PJRTBuffer]
    public func scatterAlongAxis0<T>(_ hostData: [T], shape: [Int], elementType: ElementType, count: Int) throws -> [PJRTBuffer]
    public func gatherAlongAxis0(_ shards: [PJRTBuffer]) throws -> (data: [Float], shape: [Int])
}

public final class PJRTBuffer: @unchecked Sendable {
    public let shape: [Int]
    public let elementType: ElementType
    public let device: PJRTDevice
    public func toHost<T>(_ type: T.Type) throws -> [T]
    public func toFloatArray() throws -> [Float]
}

public final class PJRTExecutable: @unchecked Sendable {
    public func execute(_ inputs: [PJRTBuffer]) throws -> [PJRTBuffer]
    public func executeMultiDevice(inputsPerDevice: [[PJRTBuffer]]) throws -> [[PJRTBuffer]]
}
```

**Dependencies**: CXLARuntime (and MetalHLO when `MAGMA_ENABLE_METAL=1` on macOS)

---

### Layer 2: StableHLO

**Purpose**: Pure-Swift generation of StableHLO MLIR text, including Shardy
(`sdy`) sharding annotations.

**Contents**:
```
Sources/StableHLO/
├── Types/
│   ├── DType.swift            # DType (f32, f16, bf16, ints, bool, ...), TensorScalar protocols
│   └── TensorType.swift       # Shape + dtype, precomputed MLIR type string
├── Builder/
│   ├── MLIRBuilder.swift      # Text builder: all ops, control flow, collectives, sdy hooks, build()
│   └── Value.swift            # SSA Value, Argument, CompareDirection
└── Sharding/
    ├── DeviceMesh.swift       # MeshAxis, DeviceMesh (.linear/.grid/.cube) -> `sdy.mesh`
    └── TensorSharding.swift   # DimensionSharding, TensorSharding -> `#sdy.sharding<...>`, validation
```

**Key Types** (abridged):
```swift
public final class MLIRBuilder: @unchecked Sendable {
    public init()

    // Function arguments, optionally with a Shardy sharding
    public func argument(_ type: TensorType, sharding: TensorSharding? = nil) -> Value

    // Operations (a few of many)
    public func add(_ lhs: Value, _ rhs: Value) -> Value
    public func dot(_ lhs: Value, _ rhs: Value) -> Value
    public func relu(_ input: Value) -> Value
    public func convolution(...) -> Value
    public func whileLoop(...) -> [Value]
    public func allReduce(_ input: Value, reduction: String, replicaGroups: [[Int]]) -> Value

    // Shardy hooks (emit nothing unless used, so single-device output is unchanged)
    public func declareMesh(_ mesh: DeviceMesh)
    public func shardingConstraint(_ input: Value, _ sharding: TensorSharding) -> Value

    // Build the final module text
    public func build(name: String, outputs: [Value]) -> String
}

public struct Value: Hashable, Sendable {
    public let id: Int
    public let type: TensorType
}

public struct TensorType: Hashable, Sendable {
    public let shape: [Int]
    public let dtype: DType
    public let mlirType: String   // e.g. "tensor<32x784xf32>"
}
```

**Dependencies**: None (pure Swift)

**Example Output** (illustrative, abridged; `build()` wraps the ops in
`module @name { func.func @main(...) }`):
```mlir
module @matmul_relu {
  func.func @main(%arg0: tensor<32x784xf32>, %arg1: tensor<784x256xf32>, %arg2: tensor<256xf32>) -> (tensor<32x256xf32>) {
    %0 = stablehlo.dot %arg0, %arg1 : (tensor<32x784xf32>, tensor<784x256xf32>) -> tensor<32x256xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<32x256xf32>
    %2 = stablehlo.add %0, %1 : tensor<32x256xf32>
    %3 = stablehlo.constant dense<0.0> : tensor<32x256xf32>
    %4 = stablehlo.maximum %2, %3 : tensor<32x256xf32>
    return %4 : tensor<32x256xf32>
  }
}
```

---

### Layer 3: LazyTensor

**Purpose**: x10-style lazy evaluation. It traces operations into a graph, runs
graph-level optimization passes, emits StableHLO, caches compiled executables and
executes them at a barrier. It also contains the multi-device runners.

**Contents**:
```
Sources/LazyTensor/
├── LazyTensor.swift                 # LazyTensorHandle, IRNode, OpKind, IRGraph, ExecutionContext,
│                                    # TensorRegistry, constant promotion, CompilationCache,
│                                    # getGlobalClient, LazyTensorBarrier, executeGraph,
│                                    # executeGraphReplicated, executeGraphSharded,
│                                    # WhileLoopTracer, PrintMetrics (+ Metal barrier/cache on macOS)
├── StableHLOEmitter.swift           # IRGraph -> StableHLO text (incl. sdy.mesh / shardings),
│                                    # topological ordering, graph hashing
└── Optimization/
    ├── PassManager.swift            # OptimizationPass protocol, PassManager, PassMetrics
    ├── Fusion/
    │   └── FusionPattern.swift      # FusionPattern protocol + matching helpers (no patterns shipped)
    └── Passes/
        ├── DeadCodeElimination.swift
        ├── CommonSubexpressionElimination.swift
        ├── ConstantFolding.swift
        ├── AlgebraicSimplification.swift   # x+0=x, x*1=x, -(-x)=x, ...
        └── OperationFusion.swift           # disabled by default; performs no transformations
```

**About fusion**: `OperationFusionPass` is registered but off by default
(`enabledByDefault = false`). It ships with no patterns, so it returns the graph
unchanged. Magma emits plain StableHLO and relies on XLA (or MetalHLO) to fuse
operations when it compiles. The passes that run by default are DCE, CSE,
constant folding and algebraic simplification. Set `MAGMA_NO_OPT=1` to skip them.

**Key Types** (abridged):
```swift
public final class LazyTensorHandle: @unchecked Sendable {
    public let id: UInt64
    public let shape: [Int]
    public let dtype: DType
    public let device: Device
    public var irNode: IRNode?
    public var materializedBuffer: PJRTBuffer?
    public var isMaterialized: Bool { get }
    public var sharding: TensorSharding?      // Shardy annotation (SPMD)
}

public indirect enum IRNode: @unchecked Sendable {
    case data(PJRTBuffer)
    case constant(values: [Float], shape: [Int])
    case operation(op: OpKind, inputs: [LazyTensorHandle], attributes: [String: Any])
    case whileLoopTraced(iterations: Int, initialValues: [LazyTensorHandle],
                         bodyInputs: [LazyTensorHandle], bodyOutputs: [LazyTensorHandle],
                         bodyNodes: [LazyTensorHandle])
    // + case metalData(MetalHLOBuffer) when MetalHLO is available
}

public enum OpKind: String, Sendable {
    case add, subtract, multiply, divide, maximum, minimum, power
    case negate, abs, exponential, log, sqrt, rsqrt, sine, cosine, tanh, floor, ceil
    case convert
    case matmul, batchedMatmul, transpose, reshape, broadcast
    case reduceSum, reduceMax, reduceMin, reduceMean
    case relu, sigmoid, softmax, gelu, leakyRelu, elu, silu, clamp
    case conv1d, conv2d, convTranspose2d, maxPool2d, avgPool2d
    case batchNorm, layerNorm
    case equal, less, greater, select
    case slice, pad, gather, scatter, concatenate
    case whileLoop, cond
    case rngUniform, rngNormal
    case allReduce, allReduceMean              // cross-replica collectives
}

public final class IRGraph: @unchecked Sendable {
    public var nodes: [LazyTensorHandle]
    public var outputs: [LazyTensorHandle]
    public var mesh: DeviceMesh?               // emitted as sdy.mesh when set
    public func validateShardings() throws
    public func emitStableHLO(name: String) -> String
}

public protocol OptimizationPass {
    var name: String { get }
    var dependencies: [String] { get }
    var enabledByDefault: Bool { get }
    func run(on graph: IRGraph) -> IRGraph
}

public final class PassManager: @unchecked Sendable {
    public static let shared: PassManager      // DCE, CSE, ConstantFolding, AlgebraicSimplification (+ fusion, disabled)
    public func register(_ pass: OptimizationPass)
    public func enable(_ passName: String)
    public func disable(_ passName: String)
}

// Single-device: compile and execute everything pending for a device
public func LazyTensorBarrier(on device: Device = .default)
public func executeGraph(_ graph: IRGraph, on device: Device = .default) throws -> [PJRTBuffer]

// Multi-device
public func executeGraphReplicated(_ graph: IRGraph, numReplicas: Int,
    distribution: [ObjectIdentifier: ReplicaInputDistribution] = [:],
    client: PJRTClient) throws -> [[PJRTBuffer]]          // data parallel (DDP)
public func executeGraphSharded(_ graph: IRGraph, numDevices: Int,
    client: PJRTClient) throws -> [[PJRTBuffer]]          // SPMD via the Shardy partitioner

public func PrintMetrics()
```

**Backend selection**: `getGlobalClient(backend:)` caches one client per resolved
backend. `resolveExecutionBackend` applies the `MAGMA_DEFAULT_BACKEND` override
first. If the requested backend has no plugin, it falls back to
`Backend.bestAvailable`.

**Dependencies**: StableHLO, XLARuntime

---

### Layer 4: Core (module `Magma`)

**Purpose**: The user-facing API: `Tensor`, autodiff, layers, optimizers, losses,
data loading, checkpointing, profiling and distributed helpers.

**Contents**:
```
Sources/Core/
├── Tensor.swift             # Tensor<Scalar>, creation, arithmetic, matmul, reductions, activations,
│                            # comparisons, MixedPrecision, `nn`/`optim` namespaces, nn.functional
├── Autodiff.swift           # Differentiable conformance + VJPs
├── Convolution.swift        # Differentiable conv2d op
├── Scan.swift               # scan / scanWithOutputs / scanXLA (loops with autodiff)
├── Module.swift             # Module protocol, Parameter, all nn.* layers, nn.Sequential / nn.sequential,
│                            # attention, transformer, RNN/LSTM/GRU, checkpointing, stateDict
├── ValueLayers.swift        # Layer protocol, Linear, ReLU, Sigmoid, Conv2d, Sequential2, sequential { }
├── LayerGradient.swift      # modelGradient / modelValueWithGradient (value-semantic API)
├── NNGradient.swift         # parameterGradients (bridge for the nn.* API)
├── KeyPathIterable.swift    # Reflection-backed KeyPathIterable
├── TangentOptimizer.swift   # sgdUpdate, MomentumSGD, Adam (generic over Differentiable & KeyPathIterable)
├── Optimizer.swift          # Optimizer protocol, optim.SGD/Adam/AdamW/AdamWGroups/RMSProp/AdaGrad/AdaDelta,
│                            # LR schedulers, gradient clipping/accumulation, TrainingState
├── Loss.swift               # S4TF-ported losses (l1Loss, huberLoss, softmaxCrossEntropy, ...)
├── Initializers.swift       # glorot/he/lecun/orthogonal/truncatedNormal
├── Data.swift               # Dataset, TensorDataset, DataLoader, SimpleBatchLoader, DistributedSampler,
│                            # indexing/slicing helpers, MNIST, MemoryMappedTokenDataset
├── Transforms.swift         # PyTorch-style `transforms`
├── Distributed.swift        # crossReplicaSum/Mean, Tensor.input(from:), makeGraph(mesh:), sharded(on:_:),
│                            # dataParallelSGDStep, Optimizer.step(syncing:groups:)
├── MultiHost.swift          # MultiHostConfig, DistributedSampler.multiHost
├── GradientChecking.swift   # gradcheck, numerical gradients/Jacobians
├── Profiling.swift          # Profiler, Benchmark, MemoryProfiler, FLOPSEstimator
└── TensorError.swift        # TensorError, TensorDebug, TensorAssert
```

**Two layer APIs.** Core has two ways to build models. They live side by side:

| | Value-semantic ("Design A") | Reference-semantic (PyTorch-style) |
|---|---|---|
| Protocol | `Layer: Differentiable, KeyPathIterable` | `Module` (not `Differentiable`) |
| Weights | stored `Tensor`s | `Parameter` reference cells |
| Layers | `Linear`, `ReLU`, `Sigmoid`, `Conv2d`, `Sequential2` | `nn.Linear`, `nn.Conv2d`, `nn.LSTM`, `nn.TransformerEncoderLayer`, ... |
| Composition | `sequential { ... }` (typed, differentiable) | `nn.Sequential(...)` / `nn.sequential { ... }` (type-erased `[AnyLayer]`) |
| Gradients | `modelGradient(of:input:target:lossFn:)` or `gradient(at: model)` | `parameterGradients(of:loss:)` returns `[Parameter: Tensor]` |
| Optimizers | `Adam`, `MomentumSGD`, `sgdUpdate` (update via `TangentVector`) | `optim.SGD`, `optim.Adam`, ... (`step(_:)` by parameter identity) |

Unqualified names (`Linear`, `Adam`, `sequential`) refer to the value-semantic
types. The reference-semantic ones are always qualified with `nn.` or `optim.`.

**Key Types** (abridged):
```swift
public struct Tensor<Scalar: TensorScalar>: Sendable {
    internal var handle: LazyTensorHandle

    public var shape: [Int] { get }
    public var rank: Int { get }
    public var dtype: DType { get }
    public var device: Device { get }

    public init(_ data: [Scalar], shape: [Int], on device: Device = .default)
    public static func zeros(_ shape: [Int], on device: Device = .default) -> Tensor
    public static func ones(_ shape: [Int], on device: Device = .default) -> Tensor
    public static func randn(...) -> Tensor

    public func scalars() -> [Scalar]      // materializes (implicit barrier)
    public func item() -> Scalar
    public func to(device targetDevice: Device) -> Tensor
}
extension Tensor: Differentiable where Scalar: TensorScalar & BinaryFloatingPoint {
    public typealias TangentVector = Tensor<Scalar>
}

// Reference-semantic API
public protocol Module {
    associatedtype Input
    associatedtype Output
    func forward(_ input: Input) -> Output
    func parameters() -> [Parameter]
    func buffers() -> [Parameter]
    mutating func to(device: Device)
    mutating func setTraining(_ training: Bool)
}   // callAsFunction, train(), eval() come from an extension

public final class Parameter: @unchecked Sendable, Hashable {   // identity-based equality
    public var value: Tensor<Float>
    public var requiresGrad: Bool
}

public protocol Optimizer {
    mutating func step(_ gradients: [Tensor<Float>])
    mutating func resetState()
    // + step(_ gradients: [Parameter: Tensor<Float>]) in an extension
}

// Value-semantic API
public protocol Layer: Differentiable, KeyPathIterable {
    @differentiable(reverse)
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float>
}
public func sequential<Body: Layer>(@LayerBuilder _ content: () -> Body) -> Body
public func modelGradient<M: Layer>(of model: M, input: Tensor<Float>, target: Tensor<Float>,
    lossFn: @differentiable(reverse) (Tensor<Float>, Tensor<Float>) -> Tensor<Float>) -> M.TangentVector

// Namespaces
public enum nn { public enum functional { ... } }   // nn.Linear(inputSize:outputSize:), nn.functional.relu, ...
public enum optim { }                               // optim.SGD, optim.Adam, ...
```

**Dependencies**: LazyTensor (Core also imports StableHLO and XLARuntime, which it reaches through LazyTensor)

---

## Data Flow Example

When a user writes:
```swift
let y = (x.matmul(w) + b).relu()
LazyTensorBarrier()
```

1. **Core layer**: every op creates a new `Tensor` whose `LazyTensorHandle` holds an
   `IRNode.operation`. Nothing runs yet.
2. **LazyTensor layer**: at the barrier, the pending handles are collected into an
   `IRGraph`, the enabled optimization passes run, and `StableHLOEmitter` produces
   MLIR text. Graph hashes and cached executables let a repeated step skip
   recompiling.
3. **StableHLO layer**: the emitter uses `MLIRBuilder`, which returns the
   `module @... { func.func @main ... }` text.
4. **XLARuntime layer**: `PJRTClient.compile` and `PJRTExecutable.execute` run the
   program on the plugin's device. XLA does its own fusion at this step.
5. **Results**: output handles get their `materializedBuffer` set and become
   `.data` nodes, so later graphs use them as inputs.

---

## Autodiff Integration

Swift's autodiff works through all layers:

```swift
// Value-semantic model
let grad = modelGradient(of: model, input: x, target: y, lossFn: mse)
optimizer.update(&model, gradient: grad)

// Or directly on tensors
let g = gradient(at: w) { w in (x.matmul(w) - y).sum() }
```

1. The Swift compiler generates pullbacks from the `@derivative` VJPs (mostly in `Autodiff.swift`)
2. Pullback calls create more lazy operations
3. The forward and backward passes are traced into one graph
4. XLA compiles and fuses everything together
5. One execution computes both the loss and the gradients

Because tensors are lazy and XLA compiles the whole traced step, the gradient
computation is optimized together with the forward pass.

Compiler caveat: calling `gradient(at:)` directly on a value of opaque type
(`some Layer`) crashes the compiler. `modelGradient` avoids the crash. See
[KNOWN_COMPILER_ISSUES.md](KNOWN_COMPILER_ISSUES.md).

---

## File Organization Conventions

- Files are grouped by topic rather than one type per file. For example,
  `Module.swift` holds the `Module` protocol and all `nn.*` layers, and
  `Optimizer.swift` holds all `optim.*` optimizers and schedulers.
- `nn`, `optim`, `transforms` and `data` are caseless enums used as namespaces.
  They are extended from the files that add to them.
- Tests are named `<Topic>Tests.swift` (e.g. `OptimizerTests.swift`) under
  `Tests/<Target>Tests/`. The Core tests live in `Tests/CoreTests` (target
  `MagmaTests`).

---

## Testing Strategy

| Layer | Test target | PJRT plugin required? |
|-------|-------------|-----------------------|
| StableHLO | `StableHLOTests` (MLIR text generation, sharding types) | No |
| LazyTensor | `LazyTensorTests` (graph building, emitter, optimization passes) | No |
| XLARuntime | `XLARuntimeTests` (compile/execute, multi-device, Shardy, GPU guard, Metal) | Yes for most suites (skipped when absent) |
| Core | `MagmaTests` (end-to-end tensors, autodiff, layers, training, DDP/SPMD) | Yes (skipped when absent) |

Suites that need a plugin are gated with `.enabled(if:)` on `PluginAvailability`.
`MAGMA_TEST_BACKEND` (`cpu`, the default, or `gpu`) selects the backend. Only one
plugin can be loaded per process, so the CPU and GPU suites run in separate
invocations. Multi-device suites use the CPU plugin's emulated devices
(`PJRTClient.create(backend: .cpu, cpuDeviceCount: N)`). CI builds everything and
runs the tests that need no plugin.
