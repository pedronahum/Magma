# Benchmarks

Experimental benchmark and demo packages for Magma's Metal backend. They are
**not part of the Magma package** or its supported surface. Each directory is
a separate Swift package that depends on the Magma checkout two levels up
(`../..`).

| Package | Executable | What it does |
|---------|------------|--------------|
| `GPT2Shakespeare/` | `GPT2Shakespeare` | Character-level GPT-2 training on Tiny Shakespeare (see its [README](GPT2Shakespeare/README.md)) |
| `MagmaBenchmark/` | `MagmaBenchmark` | Per-operation overhead of Magma compared with calling MetalHLO directly |
| `MLXComparison/` | `MagmaMLXBenchmark` | The same operations timed in Magma and in [MLX](https://github.com/ml-explore/mlx-swift) |

## Requirements

- macOS 15 or later on Apple Silicon. None of these packages run on Linux.
  Without MetalHLO they print `ERROR: MetalHLO not available` and exit 1.
- A Swift 6.0+ toolchain from swift.org, the same one Magma needs.
- Magma's Metal backend must be enabled when you build. MetalHLO is opt-in in
  Magma's `Package.swift`, so set `MAGMA_ENABLE_METAL=1`. To build against a
  local MetalHLO checkout rather than the GitHub one, also set
  `MAGMA_METALHLO_PATH=/path/to/MetalHLO`.
- `MLXComparison` also fetches `mlx-swift`.

## Running

From a package directory:

```bash
cd Benchmarks/MagmaBenchmark
MAGMA_ENABLE_METAL=1 swift run -c release MagmaBenchmark --quick

cd Benchmarks/MLXComparison
MAGMA_ENABLE_METAL=1 swift run -c release MagmaMLXBenchmark --quick
```

`--quick` (or `-q`) runs a smaller set of sizes. For GPT2Shakespeare, see its
README: it needs a data-preparation step first.

The numbers depend on your machine, macOS version and MetalHLO revision.
Nothing here has been published as a reference result.
