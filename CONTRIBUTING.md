# Contributing to Magma

Thank you for your interest in contributing to Magma! This document provides guidelines and information for contributors.

## Code of Conduct

Be respectful, inclusive, and constructive. We're building something together.

## Getting Started

### Prerequisites

- **Swift 5.9+** (Swift 6.0 recommended)
- **macOS 14+** or **Linux** (Ubuntu 22.04+)
- **XLA/PJRT** libraries (for full functionality)

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/pedronahum/Magma.git
   cd Magma
   ```

2. Build the project:
   ```bash
   swift build
   ```

3. Run tests:
   ```bash
   # StableHLO tests (no XLA required)
   swift test --filter StableHLOTests

   # All tests (requires XLA)
   swift test
   ```

### Using the Dev Container

For a consistent development environment, use the provided dev container:

```bash
# With VS Code
code --folder-uri vscode-remote://dev-container+$(pwd)/.devcontainer

# Or with Docker directly
docker build -t magma-dev .devcontainer/
docker run -it -v $(pwd):/workspace magma-dev
```

## Project Structure

```
Magma/
├── Sources/
│   ├── CXLARuntime/     # Layer 0: C bindings to PJRT
│   ├── XLARuntime/      # Layer 1: Swift PJRT wrapper
│   ├── StableHLO/       # Layer 2: Pure Swift MLIR generation
│   ├── LazyTensor/      # Layer 3: Lazy execution engine
│   └── Torch/           # Layer 4: PyTorch-compatible API
├── Tests/
│   ├── StableHLOTests/  # No XLA required
│   ├── LazyTensorTests/ # Mocked tests
│   ├── XLARuntimeTests/ # Requires XLA
│   └── TorchTests/      # End-to-end tests
├── Examples/
│   └── MNIST/           # Example training script
├── Legacy/
│   ├── TaylorTorch/     # Reference: PyTorch API design
│   └── SwiftIR/         # Reference: XLA infrastructure
└── Documentation/
    ├── ARCHITECTURE.md
    ├── ROADMAP.md
    └── LEGACY_MAPPING.md
```

## Architecture Principles

1. **Strict Layering**: Each layer only depends on layers below it
2. **Pure Swift StableHLO**: Layer 2 has zero dependencies
3. **Testability**: Most code testable without XLA installed
4. **Swift-Native Autodiff**: Use `@differentiable` throughout

See [ARCHITECTURE.md](Documentation/ARCHITECTURE.md) for details.

## How to Contribute

### Reporting Issues

- Check existing issues first
- Include Swift version, OS, and XLA version
- Provide minimal reproduction steps
- Include error messages and stack traces

### Submitting Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the coding style
4. **Add tests** for new functionality
5. **Run the test suite**:
   ```bash
   swift test
   ```
6. **Submit a PR** with a clear description

### Pull Request Guidelines

- One feature/fix per PR
- Keep changes focused and minimal
- Update documentation if needed
- Add tests for new code
- Ensure CI passes

## Coding Style

### Swift Conventions

- Use Swift's standard naming conventions
- Prefer `let` over `var`
- Use explicit types for public APIs
- Document public interfaces with `///` comments

### File Organization

- One type per file (generally)
- Extensions: `Type+Feature.swift`
- Tests: `TypeTests.swift`

### Example

```swift
/// A tensor value in the computation graph.
public struct Value: Hashable, Sendable {
    /// Unique identifier for this value.
    public let id: Int

    /// The tensor type of this value.
    public let type: TensorType

    /// Creates a new value with the given ID and type.
    public init(id: Int, type: TensorType) {
        self.id = id
        self.type = type
    }
}
```

## Testing

### Test Categories

| Test Target | XLA Required | Purpose |
|-------------|--------------|---------|
| `StableHLOTests` | No | Pure Swift MLIR generation |
| `LazyTensorTests` | No | Graph building with mocks |
| `XLARuntimeTests` | Yes | PJRT integration |
| `TorchTests` | Yes | End-to-end training |

### Running Specific Tests

```bash
# Run only StableHLO tests (fast, no XLA)
swift test --filter StableHLOTests

# Run a specific test
swift test --filter MLIRBuilderTests/testAddOperation
```

### Writing Tests

```swift
import XCTest
@testable import StableHLO

final class MyFeatureTests: XCTestCase {
    func testBasicFunctionality() {
        // Arrange
        let builder = MLIRBuilder()

        // Act
        let result = builder.someOperation()

        // Assert
        XCTAssertEqual(result, expectedValue)
    }
}
```

## Areas for Contribution

Check [ROADMAP.md](Documentation/ROADMAP.md) for current priorities. High-impact areas:

### Good First Issues
- Add missing StableHLO operations
- Improve error messages
- Add documentation examples
- Write additional tests

### Intermediate
- Implement new `nn.Module` layers
- Add optimizer implementations
- Improve shape inference

### Advanced
- Control flow operations (`while`, `cond`)
- GPU plugin support
- Performance optimization

## Communication

- **Issues**: Bug reports and feature requests
- **Pull Requests**: Code contributions
- **Discussions**: Questions and ideas

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

Thank you for helping make Magma better!
