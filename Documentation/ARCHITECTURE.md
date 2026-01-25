# Magma Architecture

## Overview

Magma is organized as a strictly layered monorepo. Each layer has a single responsibility and only depends on layers below it.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LAYER 4: Core                                  │
│                     PyTorch-compatible user-facing API                      │
│         nn.Module, nn.Linear, optim.Adam, Tensor operations                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                            LAYER 3: LazyTensor                              │
│              x10-style lazy execution with explicit barriers                │
│      LazyTensorHandle, IRNode, IRGraph, LazyTensorBarrier, Cache           │
├─────────────────────────────────────────────────────────────────────────────┤
│                            LAYER 2: StableHLO                               │
│                Pure Swift MLIR/StableHLO IR generation                      │
│           MLIRBuilder, Op definitions, Shape inference                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                            LAYER 1: XLARuntime                              │
│                      PJRT client and execution layer                        │
│          PJRTClient, PJRTBuffer, PJRTExecutable, Device plugins            │
├─────────────────────────────────────────────────────────────────────────────┤
│                            LAYER 0: CXLARuntime                             │
│                         C bindings to PJRT API                              │
│                         pjrt_c_api.h wrappers                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Strict Layering
- Each layer only imports layers below it
- No circular dependencies
- Clear API boundaries between layers

### 2. Testability at Every Layer
- StableHLO layer is pure Swift - testable without XLA installed
- LazyTensor layer can mock XLARuntime for unit tests
- Torch layer tests use real execution for integration tests

### 3. Replaceability
- XLARuntime could be swapped for Metal, Vulkan, etc.
- StableHLO could target different IRs
- Torch API could be replaced with different frontend

### 4. Swift-Native Autodiff
- Use `@differentiable` attribute throughout
- Gradients are also lazy - traced and compiled with forward pass
- No runtime tape, no Python interop needed

---

## Layer Details

### Layer 0: CXLARuntime

**Purpose**: Minimal C bindings to PJRT API

**Contents**:
```
Sources/CXLARuntime/
├── include/
│   └── pjrt_c_api.h          # PJRT C API header
└── shim.c                     # Minimal Swift-compatible wrappers
```

**Key Types**:
- Raw C function pointers to PJRT API
- Opaque handle types

**Dependencies**: libpjrt (system library)

---

### Layer 1: XLARuntime

**Purpose**: Swift wrapper around PJRT for compilation and execution

**Contents**:
```
Sources/XLARuntime/
├── PJRTClient.swift          # Main client for device management
├── PJRTDevice.swift          # Device abstraction
├── PJRTBuffer.swift          # On-device data buffers
├── PJRTExecutable.swift      # Compiled XLA program
├── PJRTError.swift           # Error handling
└── Plugins/
    ├── CPUPlugin.swift       # CPU backend
    ├── GPUPlugin.swift       # CUDA backend
    └── TPUPlugin.swift       # TPU backend
```

**Key Types**:
```swift
public final class PJRTClient {
    public static func create(backend: Backend = .cpu) throws -> PJRTClient
    public func compile(_ mlir: String) throws -> PJRTExecutable
    public func devices() -> [PJRTDevice]
}

public final class PJRTBuffer {
    public static func create<T>(_ data: [T], shape: [Int], on device: PJRTDevice) throws -> PJRTBuffer
    public func toArray<T>() -> [T]
}

public final class PJRTExecutable {
    public func execute(_ inputs: [PJRTBuffer]) throws -> [PJRTBuffer]
}
```

**Dependencies**: CXLARuntime

---

### Layer 2: StableHLO

**Purpose**: Pure Swift generation of StableHLO MLIR

**Contents**:
```
Sources/StableHLO/
├── Types/
│   ├── TensorType.swift      # Tensor type representation
│   ├── DType.swift           # Data types (f32, f16, bf16, etc.)
│   └── Shape.swift           # Shape utilities
├── Builder/
│   ├── MLIRBuilder.swift     # Main builder class
│   ├── Value.swift           # SSA value reference
│   └── Block.swift           # Basic block representation
├── Ops/
│   ├── ElementwiseOps.swift  # add, mul, div, exp, log, etc.
│   ├── MatrixOps.swift       # dot, transpose, reshape
│   ├── ConvolutionOps.swift  # conv, pooling
│   ├── ReductionOps.swift    # sum, mean, max, min
│   ├── ControlFlowOps.swift  # while, cond, scan
│   └── CompareOps.swift      # eq, lt, gt, etc.
├── ShapeInference.swift      # Infer output shapes
└── Verification.swift        # Validate generated IR
```

**Key Types**:
```swift
public final class MLIRBuilder {
    public init()
    
    // Create operations
    public func add(_ lhs: Value, _ rhs: Value) -> Value
    public func matmul(_ lhs: Value, _ rhs: Value) -> Value
    public func relu(_ input: Value) -> Value
    // ... all StableHLO ops
    
    // Build final module
    public func buildModule(name: String, inputs: [TensorType], outputs: [Value]) -> String
}

public struct Value {
    let id: Int
    let type: TensorType
}

public struct TensorType {
    let shape: [Int]
    let dtype: DType
    
    var mlirType: String { "tensor<\(shape.map(String.init).joined(separator: "x"))x\(dtype.mlirName)>" }
}
```

**Dependencies**: None (pure Swift!)

**Example Output**:
```mlir
module @matmul_relu {
  func.func @main(%arg0: tensor<32x784xf32>, %arg1: tensor<784x256xf32>, %arg2: tensor<256xf32>) -> tensor<32x256xf32> {
    %0 = stablehlo.dot %arg0, %arg1 : (tensor<32x784xf32>, tensor<784x256xf32>) -> tensor<32x256xf32>
    %1 = stablehlo.add %0, %arg2 : tensor<32x256xf32>
    %2 = stablehlo.constant dense<0.0> : tensor<32x256xf32>
    %3 = stablehlo.maximum %1, %2 : tensor<32x256xf32>
    return %3 : tensor<32x256xf32>
  }
}
```

---

### Layer 3: LazyTensor

**Purpose**: x10-style lazy evaluation with graph tracing and compilation caching

**Contents**:
```
Sources/LazyTensor/
├── Core/
│   ├── LazyTensorHandle.swift    # Reference to graph node
│   ├── IRNode.swift              # Operation graph node
│   ├── IRGraph.swift             # Full computation graph
│   └── Device.swift              # Device abstraction
├── Execution/
│   ├── LazyTensorBarrier.swift   # Trigger compilation/execution
│   ├── CompilationCache.swift    # Cache compiled executables
│   ├── GraphHash.swift           # Hash graphs for cache lookup
│   └── StableHLOEmitter.swift    # IRGraph → StableHLO
├── Registry/
│   ├── TensorRegistry.swift      # Track live tensors
│   └── WeakRef.swift             # Weak reference wrapper
└── Debug/
    ├── IRPrinter.swift           # Print IR for debugging
    └── Metrics.swift             # Compilation metrics
```

**Key Types**:
```swift
public final class LazyTensorHandle {
    let id: UInt64
    let shape: [Int]
    let dtype: DType
    let device: Device
    
    var irNode: IRNode?
    var materializedBuffer: PJRTBuffer?
    
    var isMaterialized: Bool { materializedBuffer != nil }
}

public indirect enum IRNode {
    case data(PJRTBuffer)
    case constant(values: [Float], shape: [Int])
    case operation(op: OpKind, inputs: [LazyTensorHandle], attributes: [String: Any])
}

public enum OpKind: String {
    case add, sub, mul, div, neg
    case matmul, transpose, reshape, broadcast
    case conv2d, maxPool2d, avgPool2d
    case relu, sigmoid, tanh, gelu, softmax
    case sum, mean, max, min
    case batchNorm, layerNorm
    // ... complete op set
}

/// Trigger compilation and execution
public func LazyTensorBarrier(on device: Device = .default)

/// Print compilation metrics
public func PrintMetrics()
```

**Dependencies**: StableHLO, XLARuntime

---

### Layer 4: Core

**Purpose**: PyTorch-compatible user-facing API

**Contents**:
```
Sources/Core/
├── Tensor/
│   ├── Tensor.swift              # Main tensor type
│   ├── TensorCreate.swift        # zeros, ones, randn, etc.
│   ├── TensorArithmetic.swift    # +, -, *, /, @
│   ├── TensorComparison.swift    # ==, <, >, etc.
│   ├── TensorReduction.swift     # sum, mean, max, etc.
│   ├── TensorShape.swift         # reshape, transpose, etc.
│   ├── TensorIndex.swift         # Subscript, slicing
│   └── TensorDifferentiable.swift # @differentiable conformance
├── NN/
│   ├── Module.swift              # Base protocol
│   ├── Sequential.swift          # Sequential container
│   ├── Parameter.swift           # Trainable parameter
│   ├── Layers/
│   │   ├── Linear.swift
│   │   ├── Conv1d.swift
│   │   ├── Conv2d.swift
│   │   ├── BatchNorm.swift
│   │   ├── LayerNorm.swift
│   │   ├── Dropout.swift
│   │   ├── Embedding.swift
│   │   ├── RNN.swift
│   │   ├── LSTM.swift
│   │   ├── Attention.swift
│   │   └── Transformer.swift
│   └── Functional/
│       ├── Activations.swift     # relu, gelu, sigmoid, etc.
│       ├── Loss.swift            # crossEntropy, mse, etc.
│       ├── Pooling.swift         # maxPool, avgPool
│       └── Normalization.swift   # batchNorm, layerNorm
├── Optim/
│   ├── Optimizer.swift           # Base protocol
│   ├── SGD.swift
│   ├── Adam.swift
│   ├── AdamW.swift
│   └── LRScheduler.swift
├── Data/
│   ├── Dataset.swift
│   ├── DataLoader.swift
│   └── Transforms.swift
└── Utils/
    ├── Checkpoint.swift          # Save/load models
    ├── Summary.swift             # Model summary
    └── Seed.swift                # Random seed control
```

**Key Types**:
```swift
public struct Tensor<Scalar: TensorScalar>: Differentiable {
    var handle: LazyTensorHandle
    
    // Properties
    public var shape: [Int]
    public var dtype: DType
    public var device: Device
    
    // Creation
    public init(_ data: [Scalar], shape: [Int], on device: Device = .default)
    public static func zeros(_ shape: [Int], on device: Device = .default) -> Tensor
    public static func randn(_ shape: [Int], on device: Device = .default) -> Tensor
    
    // Materialization
    public func scalars() -> [Scalar]
    public func item() -> Scalar  // For scalar tensors
}

public protocol Module: Differentiable {
    associatedtype Input
    associatedtype Output
    
    @differentiable(reverse)
    func callAsFunction(_ input: Input) -> Output
}

public struct nn {
    public struct Linear: Module { ... }
    public struct Conv2d: Module { ... }
    public struct Sequential<Layer: Module>: Module { ... }
    
    public struct functional {
        public static func relu<T>(_ x: Tensor<T>) -> Tensor<T>
        public static func crossEntropy<T>(_ input: Tensor<T>, _ target: Tensor<Int>) -> Tensor<T>
    }
}

public struct optim {
    public class Adam<Model: Differentiable>: Optimizer { ... }
    public class SGD<Model: Differentiable>: Optimizer { ... }
}
```

**Dependencies**: LazyTensor

---

## Data Flow Example

When a user writes:
```swift
let y = relu(matmul(x, w) + b)
LazyTensorBarrier()
```

1. **Core layer**: Creates `Tensor` with `LazyTensorHandle` for each operation
2. **LazyTensor layer**: Builds `IRGraph` from handles
3. **StableHLO layer**: Converts `IRGraph` to MLIR text
4. **XLARuntime layer**: Compiles MLIR, executes on device
5. **Back up**: Results populate `LazyTensorHandle.materializedBuffer`

---

## Autodiff Integration

Swift's autodiff works through all layers:

```swift
// User code
let (loss, grads) = valueWithGradient(at: model) { m in
    m(input).mean()
}
```

1. Swift compiler generates pullback functions
2. Pullback calls create MORE lazy operations
3. Forward and backward are traced into ONE graph
4. XLA compiles and fuses everything together
5. Single execution computes loss AND gradients

This is why lazy tensors + XLA + Swift autodiff is powerful: **gradients are fused with forward pass**.

---

## File Naming Conventions

- Types: `PascalCase.swift` (e.g., `LazyTensorHandle.swift`)
- Extensions: `Type+Feature.swift` (e.g., `Tensor+Arithmetic.swift`)
- Protocols: `Protocol.swift` (e.g., `Module.swift`)
- Tests: `TypeTests.swift` (e.g., `TensorTests.swift`)

---

## Testing Strategy

| Layer | Test Type | XLA Required? |
|-------|-----------|---------------|
| StableHLO | Unit tests | No |
| LazyTensor | Unit + mock tests | No |
| XLARuntime | Integration tests | Yes |
| Core | End-to-end tests | Yes |

This allows rapid iteration on upper layers without slow XLA compilation.
