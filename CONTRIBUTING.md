# Contributing to Magma

Thank you for your interest in contributing to Magma! This document explains
how to set up a development environment, run the tests, and submit changes.

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful,
inclusive and constructive. Report security problems privately as described in
[SECURITY.md](SECURITY.md), not in public issues.

## Getting Started

### Prerequisites

- **Swift 6.0 or later from [swift.org](https://swift.org/install)** (CI tests
  6.0 and 6.3). Xcode's bundled toolchain does **not** include the
  `_Differentiation` module Magma needs; on macOS install a swift.org toolchain,
  e.g. with [swiftly](https://www.swift.org/swiftly/).
- **Linux** (x86_64 or aarch64, e.g. Ubuntu 22.04+) or **macOS 15+**.
- **A PJRT plugin** to execute anything (see the README's *Requirements*). You
  can build and run the plugin-free tests without one.

### Development Setup

```bash
git clone https://github.com/pedronahum/Magma.git
cd Magma

# Build everything, including examples and tests
swift build --build-tests
```

The PJRT plugin is loaded at runtime, so no build flags are needed. To execute
models and the full test suite, point `MAGMA_XLA_PATH` at the directory that
holds the plugin:

```bash
export MAGMA_XLA_PATH=/opt/xla/lib   # contains pjrt_c_api_cpu_plugin.so
```

The Metal backend (macOS) is opt-in: set `MAGMA_ENABLE_METAL=1` before building
to pull in [MetalHLO](https://github.com/pedronahum/MetalHLO), and optionally
`MAGMA_METALHLO_PATH=/path/to/MetalHLO` to use a local checkout.

### Using the Dev Container

For a consistent Linux environment, use the provided dev container
(Swift 6.3, `MAGMA_XLA_PATH=/opt/xla/lib`):

```bash
# With VS Code: "Dev Containers: Reopen in Container"
# Or with Docker directly
docker build -t magma-dev .devcontainer/
docker run -it -v "$(pwd)":/workspace magma-dev
```

Mount or copy a PJRT plugin into `/opt/xla/lib` to run the full test suite.

## Running the Tests

| Test target | Needs a PJRT plugin | What it covers |
|-------------|---------------------|----------------|
| `StableHLOTests` | No | Pure-Swift MLIR/StableHLO generation, Shardy types |
| `LazyTensorTests` | No | Graph building, optimization passes, emitter |
| `XLARuntimeTests` | Partly (plugin suites are skipped without one) | PJRT client, buffers, compilation, multi-device |
| `MagmaTests` (`Tests/CoreTests`) | Yes | Tensors, autodiff, layers, optimizers, data, distributed |

```bash
# No plugin needed (this is what CI runs)
swift test --filter 'StableHLOTests|LazyTensorTests|XLARuntimeTests'

# Full suite on the CPU plugin (~1000 tests, under a minute)
MAGMA_XLA_PATH=/opt/xla/lib swift test --no-parallel \
    --filter 'MagmaTests|StableHLOTests|LazyTensorTests|XLARuntimeTests'

# GPU-only suites: a separate invocation, because one process can load only
# one PJRT plugin
MAGMA_XLA_PATH=/opt/xla/lib MAGMA_TEST_BACKEND=gpu swift test --no-parallel \
    --filter 'GPU|OutputDType|PluginMismatch'

# A single suite or test
swift test --filter 'MLIRBuilderTests'
```

Always pass `--no-parallel` when a plugin is present. A CUDA client reserves
most of device memory (~75-80% of the unified pool on an NVIDIA GB10), so
parallel suites that each create one can exhaust memory and freeze the machine.
`MAGMA_TEST_BACKEND` (`cpu` by default, or `gpu`) selects which backend's suites
run; suites for the other backend are skipped without creating a client.

### Writing Tests

Tests use [swift-testing](https://github.com/swiftlang/swift-testing):

```swift
import Testing
@testable import Magma

@Suite("My Feature")
struct MyFeatureTests {
    @Test("relu zeroes negative inputs")
    func reluZeroesNegatives() {
        let x = Tensor<Float>([-1, 0, 2], shape: [3])
        #expect(x.relu().scalars() == [0, 0, 2])
    }
}
```

- Check **values**, not only shapes: a test that asserts only `output.shape`
  passes even when the op cannot execute.
- Suites that need a real backend should gate on availability so they are
  skipped, not failed, without a plugin:
  `@Suite("…", .enabled(if: PluginAvailability.cpu, "CPU PJRT plugin not available"))`.
- For gradients, compare against a numerical reference with `gradcheck` /
  `GradientChecker` where practical.

## Project Structure

```
Magma/
├── Sources/
│   ├── CXLARuntime/     # Layer 0: C wrapper over the PJRT C API (dlopen'd plugin)
│   ├── XLARuntime/      # Layer 1: Swift PJRT client, buffers, executables; MetalHLO
│   ├── StableHLO/       # Layer 2: pure-Swift MLIR/StableHLO builder + Shardy types
│   ├── LazyTensor/      # Layer 3: x10-style tracing, graph passes, emitter, runners
│   └── Core/            # Layer 4: the `Magma` module — Tensor, autodiff, nn, optim, data
├── Tests/
│   ├── StableHLOTests/
│   ├── LazyTensorTests/
│   ├── XLARuntimeTests/
│   └── CoreTests/       # test target `MagmaTests`
├── Examples/            # ValueLayers, MNIST, BuildingSimulation, Benchmarks, Metal
├── Benchmarks/          # separate macOS/Metal benchmark packages
└── Documentation/       # architecture, API, roadmap, deployment, compiler issues
```

### Architecture Principles

1. **Strict layering**: each layer depends only on the layers below it.
2. **Pure-Swift StableHLO**: the StableHLO layer has no dependencies.
3. **Testability**: graph construction and MLIR generation are testable without XLA.
4. **Swift-native autodiff**: differentiable APIs use `@differentiable(reverse)`.

See [ARCHITECTURE.md](Documentation/ARCHITECTURE.md) for details.

## How to Contribute

### Reporting Issues

Use the issue templates. Include your Swift version, OS and architecture, the
backend and PJRT plugin (and its XLA commit), a minimal reproduction, and the
full error output.

### Submitting Pull Requests

1. Fork the repository and create a branch from `main`.
2. Make focused changes that follow the coding style below.
3. Add tests for new behavior and bug fixes.
4. Run the plugin-free tests, and the full CPU suite if you have a plugin.
5. Open a PR with a clear description. CI must pass.

Keep one feature or fix per PR, and update doc comments, the README and the
CHANGELOG (`[Unreleased]` section) when behavior or public API changes.

## Coding Style

- Follow the [Swift API Design Guidelines](https://www.swift.org/documentation/api-design-guidelines/).
- Prefer `let` over `var`, and value types unless reference semantics are the point.
- Document every public declaration with `///` comments, including preconditions.
- Fail loudly on invalid input (a precondition with an actionable message or a
  thrown error), never by silently returning a wrong or empty result.
- Match the style of the surrounding code.

## Areas for Contribution

See the [roadmap](Documentation/ROADMAP.md) and open issues. Good starting points:

- Missing StableHLO operations and better error messages
- Value-level tests for layers and losses
- Documentation examples
- Verification on hardware we have not tested (multi-GPU, TPU)

## License

By contributing, you agree that your contributions will be licensed under the
[Apache 2.0 License](LICENSE).
