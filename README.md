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
- **Pure Swift StableHLO**: IR generation with no C dependencies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Magma          PyTorch-compatible API (nn, optim, etc.)    │
├─────────────────────────────────────────────────────────────┤
│  LazyTensor     x10-style tracing, barriers, caching        │
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
| **Metal** | ✅ Supported | macOS GPU via [MetalHLO](https://github.com/pedronahum/MetalHLO) - ~66 GFLOPS on M1 |
| **GPU** | 🚧 Planned | CUDA support planned for future release |

> **Note:** GPU/CUDA support requires the PJRT GPU plugin which is not yet fully integrated. CPU and TPU backends are production-ready for v0.1.0. Metal backend achieves ~66 GFLOPS for 512x512 matmul after warmup (~4ms/iteration).

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

> **Tested Version**: Magma has been tested with XLA commit `bb760b047bdbfeff962f0366ad5cc782c98657e0` (compatible with jaxlib 0.9.0). Using this specific version is recommended for compatibility.

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

# Run all tests
swift test

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
