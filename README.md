<p align="center">
  <img src="assets/Magma.jpeg" alt="Magma" width="600">
</p>

<h1 align="center">Magma</h1>

<p align="center">
  <strong>Deep learning for Swift, powered by XLA</strong>
</p>

<p align="center">
  <a href="https://swift.org"><img src="https://img.shields.io/badge/Swift-6.0+-orange.svg" alt="Swift"></a>
  <a href="https://openxla.org"><img src="https://img.shields.io/badge/Backend-XLA%2FStableHLO-blue.svg" alt="XLA"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

---

## The Story of Magma

> **Magma is rock made fluid by intense internal heat.**

* 🪨 **The Rock:** The foundation is **XLA and StableHLO**—rigid, unbreakable, and high-performance.
* 🔥 **The Heat:** **Swift's native differentiation** provides the internal energy. Unlike Python libraries that need an external heat source (a "tape" or tracer) to melt the code, Swift generates gradients intrinsically. This internal heat turns rigid static code into a malleable, trainable medium.
* 🌊 **The Flow:** You shape this molten code with a **PyTorch-compatible API** that feels natural and fluid.
* 💎 **The Result:** When you are ready to execute, the magma cools instantly—compiling back into solid, optimized machine code for your hardware.

---

## Quick Example

```swift
import Magma

// Define a model using familiar PyTorch-style API
let model = nn.Sequential {
    nn.Linear(784, 256)
    nn.ReLU()
    nn.Dropout(0.1)
    nn.Linear(256, 10)
}

// Training with Swift's native autodiff
for (images, labels) in dataLoader {
    let (loss, grads) = valueWithGradient(at: model) { m in
        let logits = m(images)
        return softmaxCrossEntropy(logits: logits, probabilities: labels)
    }
    optimizer.update(&model, along: grads)
    Magma.barrier()  // Compile & execute
}
```

## Key Features

- **PyTorch-compatible API**: Familiar `nn.Module`, `nn.Linear`, `optim.Adam`, etc.
- **XLA backend**: Automatic operation fusion, hardware portability (CPU/GPU/TPU)
- **Metal backend**: Native macOS GPU acceleration via [MetalHLO](https://github.com/pedronahum/MetalHLO)
- **Swift-native autodiff**: First-class `@differentiable` support, not a bolted-on tape
- **Lazy execution**: x10-style tracing with explicit barriers for optimal compilation
- **Graph optimization**: DCE, CSE, constant folding, algebraic simplification, operation fusion
- **Pure Swift StableHLO**: IR generation with no C dependencies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Magma          PyTorch-compatible API (nn, optim, etc.)    │
├─────────────────────────────────────────────────────────────┤
│  LazyTensor     x10-style tracing, optimization, caching    │
│                 DCE, CSE, constant folding, op fusion       │
├─────────────────────────────────────────────────────────────┤
│  StableHLO      Pure Swift MLIR generation                  │
├─────────────────────────────────────────────────────────────┤
│  XLARuntime     PJRT execution    │  MetalHLO (macOS)       │
│                 (CPU, GPU, TPU)   │  Metal GPU via MPSGraph │
└─────────────────────────────────────────────────────────────┘
```

## Project Status

🚧 **Active Development** - See [ROADMAP.md](Documentation/ROADMAP.md) for current progress.

### Supported Backends (v0.1.0)

| Backend | Status | Notes |
|---------|--------|-------|
| **CPU** | ✅ Supported | Full functionality via PJRT CPU plugin |
| **TPU** | ✅ Supported | On Google Cloud TPU VMs with `libtpu.so` |
| **Metal** | ✅ Supported | macOS GPU via [MetalHLO](https://github.com/pedronahum/MetalHLO) |
| **GPU (CUDA)** | ✅ Experimental | Single-device execution verified via the CUDA PJRT plugin |

> **Note:** Single-device CUDA GPU execution is working — the CUDA PJRT plugin
> loads, compiles StableHLO, and executes (buffer transfers, elementwise ops, and
> cuBLAS GEMM verified on an NVIDIA GB10). It requires the CUDA PJRT plugin
> (`pjrt_c_api_gpu_plugin.so`, e.g. JAX's `xla_cuda_plugin.so`) on `MAGMA_XLA_PATH`.
> Multi-GPU / sharding is **not** yet available (single device only). CPU and TPU
> backends remain the most thoroughly tested paths for v0.1.0.

### Metal Backend Benchmarking

🔬 **Performance benchmarking is underway** comparing Magma's Metal backend against [MLX](https://github.com/ml-explore/mlx). The benchmark suite covers core operations (matmul, softmax, GELU, LayerNorm) and transformer patterns (FFN, attention). Results and optimizations will be published as development progresses.

## Heritage

Magma builds on the foundations of:
- [Swift for TensorFlow (S4TF)](https://github.com/tensorflow/swift-apis) - Original lazy tensor design, initializers, loss functions, and layer patterns
- [TaylorTorch](Legacy/TaylorTorch/) - PyTorch-style API design for Swift
- [SwiftIR](Legacy/SwiftIR/) - MLIR/XLA infrastructure for Swift

Many components are ported from S4TF including:
- Parameter initializers (Glorot, He, LeCun, Orthogonal)
- Loss functions (L1, L2, Hinge, Huber, Cross-entropy, etc.)
- Additional optimizers (RMSProp, AdaGrad, AdaDelta)
- Activation layers (SELU, Mish, Softplus, PReLU)

See [LEGACY_MAPPING.md](Documentation/LEGACY_MAPPING.md) for detailed mapping.

## Requirements

### Swift 6.0+

Magma requires Swift 6.0 or later with autodiff support. Get it from [swift.org](https://swift.org/download/).

### XLA/PJRT Runtime (Required for Execution)

> ⚠️ **Important**: Magma requires the XLA PJRT runtime to execute computations. Without it, you can build and develop but not run models.

Magma uses [OpenXLA's PJRT](https://openxla.org/xla) (Portable JAX Runtime) for hardware acceleration. You need the PJRT plugin library for your target platform:

| Platform | Library | Source |
|----------|---------|--------|
| CPU | `libpjrt_c_api_cpu_plugin.so` | Build from [OpenXLA/XLA](https://github.com/openxla/xla) |
| CUDA GPU | `libpjrt_c_api_gpu_plugin.so` | Build from OpenXLA/XLA |
| TPU | `libtpu.so` | Available on GCP TPU VMs |

**Option 1: Build XLA from source**

> **Tested Versions**:
> - **CPU** — plugin built from XLA commit `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69` (jaxlib 0.10.1, PJRT C-API 0.108); this is the recommended pin and runs the full test suite.
> - **GPU (CUDA)** — verified with JAX's bundled CUDA plugin (`xla_cuda_plugin.so`, CUDA 13 build) on an NVIDIA GB10. Matching the CPU pin's PJRT C-API version is recommended to avoid ABI skew.
>
> The framework was originally validated against XLA commit `bb760b047bdbfeff962f0366ad5cc782c98657e0` (jaxlib 0.9.0); newer pins compatible with the PJRT C-API above also work.

```bash
# Clone XLA
git clone https://github.com/openxla/xla.git
cd xla

# Checkout the tested version (recommended)
git checkout bb760b047bdbfeff962f0366ad5cc782c98657e0

# Build PJRT CPU plugin (requires Bazel)
# On macOS:
bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin.dylib
# On Linux:
bazel build //xla/pjrt/c:pjrt_c_api_cpu_plugin.so

# Copy to your preferred location
# macOS:
cp bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.dylib /opt/xla/lib/
# Linux:
cp bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so /opt/xla/lib/
```

**Option 2: Use prebuilt binaries** (if available)
Check the [releases page](https://github.com/openxla/xla/releases) or use JAX's bundled libraries.

### Environment Variables

Set these before building/running with XLA:

```bash
# Path to directory containing PJRT plugin libraries
export MAGMA_XLA_PATH=/opt/xla/lib

# Enable XLA linking (required for execution)
export MAGMA_ENABLE_XLA=1
```

Optional overrides:

```bash
# Force a specific backend regardless of the requested one (cpu/gpu/tpu/metal),
# when that plugin is available.
export MAGMA_DEFAULT_BACKEND=gpu

# Override the safety guard that refuses a second concurrent accelerator
# (GPU/TPU) client. Each accelerator client reserves most of device memory, so
# on a unified-memory machine a second client can exhaust memory and freeze the
# box; the guard prevents that. Set to 1 only if you know the machine has room.
export MAGMA_ALLOW_CONCURRENT_ACCEL_CLIENTS=1
```

## Quick Start

### Development Mode (No XLA)

You can build and test the pure Swift components without XLA:

```bash
# Clone
git clone https://github.com/pedronahum/Magma.git
cd Magma

# Build (stub mode - no execution)
swift build

# Run pure Swift tests (StableHLO, shape inference, etc.)
swift test --filter StableHLOTests
swift test --filter LazyTensorTests
```

### Full Mode (With XLA)

To run actual computations, you need XLA installed:

```bash
# Set environment
export MAGMA_XLA_PATH=/opt/xla/lib
export MAGMA_ENABLE_XLA=1

# Build with XLA
swift build

# Run all tests. Prefer a CPU PJRT plugin ($MAGMA_XLA_PATH/pjrt_c_api_cpu_plugin.so):
# it runs the whole suite in seconds with no accelerator memory reserved. Keep
# --no-parallel — the GPU smoke suites each spin up a PJRT client, and on a
# unified-memory box (e.g. NVIDIA GB10) each CUDA client reserves ~75-80% of
# memory, so a parallel run can OOM-freeze the machine. Note: one process can
# only load one plugin, so run the GPU-only suites in a separate invocation.
swift test --no-parallel --filter MagmaTests   # CoreTests, runs on CPU

# Run MNIST example
swift run MNISTExample
```

### TPU Deployment

See [TPU_DEPLOYMENT.md](Documentation/TPU_DEPLOYMENT.md) for running on Google Cloud TPUs.

## Documentation

- [Architecture Overview](Documentation/ARCHITECTURE.md)
- [Roadmap & Phases](Documentation/ROADMAP.md)
- [API Reference](Documentation/API.md)
- [Contributing](Documentation/CONTRIBUTING.md)

## License

Apache 2.0 - See [LICENSE](LICENSE)
