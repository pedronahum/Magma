# Releasing Magma

This file holds the checklist for cutting a release and the record of what was
reviewed for the first alpha, **0.1.0-alpha.1**.

## Release steps

1. **Update the changelog.** Move entries from `[Unreleased]` into a new
   version section with today's date, and update the link references at the
   bottom of `CHANGELOG.md`.
2. **Verify locally** (CPU plugin in `$MAGMA_XLA_PATH`):
   ```bash
   swift build --build-tests
   MAGMA_XLA_PATH=/nonexistent swift test          # no plugin: backend suites skip
   swift test --no-parallel                        # full suite on the CPU plugin
   MAGMA_TEST_BACKEND=gpu swift test --no-parallel \
       --filter 'GPU|OutputDType|PluginMismatch'   # if a CUDA plugin is available
   swift run ValueLayersExample && swift run MNISTExample
   ```
   Repeat the build and full suite with the oldest supported toolchain (Swift 6.0).
3. **Check CI is green** on `main` (Linux Swift 6.0 and 6.3; macOS is
   experimental).
4. **Tag and push:**
   ```bash
   git tag -a v0.1.0-alpha.1 -m "Magma 0.1.0-alpha.1"
   git push origin main v0.1.0-alpha.1
   ```
   SwiftPM consumers depend on the tag without the `v` prefix
   (`exact: "0.1.0-alpha.1"`), which SwiftPM resolves from `v0.1.0-alpha.1`.
5. **Publish a GitHub release** from the tag. Mark it as a pre-release and paste
   the changelog section, including *Known limitations*.
6. **Smoke-test as a consumer:** create a new package that depends on
   `.package(url: "https://github.com/pedronahum/Magma.git", exact: "0.1.0-alpha.1")`,
   `import Magma`, and run a small training loop.

## 0.1.0-alpha.1 readiness review

The repository was audited for release readiness across packaging, CI,
documentation, public API, runtime robustness, examples, and tests. Each fix was
reviewed independently and is covered by tests that check values, not only
shapes.

### Done

**Packaging and CI**
- [x] Clean `swift build --build-tests` works (test targets declare their real dependencies)
- [x] swift-tools-version 6.0. Minimum Swift 6.0 verified with the full suite on 6.0.3 and 6.3.3
- [x] No unversioned dependencies by default: MetalHLO is opt-in (`MAGMA_ENABLE_METAL=1`)
- [x] `-ldl` always linked on Linux. Dead build settings and env vars removed
- [x] Examples are no longer exported as products
- [x] `import Magma` re-exports the lower layers and `_Differentiation`
- [x] CI: Linux Swift 6.0 and 6.3 build every target and run every test without a plugin. macOS runs on a swift.org toolchain (experimental)
- [x] Dev container on Swift 6.3, `MAGMA_XLA_PATH`

**Runtime (C wrapper and XLARuntime)**
- [x] Buffer overflow when a program has more than 16 outputs
- [x] PJRT error messages kept and included in errors
- [x] Stale output-count cache after an executable is destroyed
- [x] Element-size validation in `toHost` and `createBuffer` (fixes a heap over-read)
- [x] Buffers and executables keep their client alive (fixes a use-after-free)
- [x] Unsupported element types rejected
- [x] Plugin discovery: per-backend overrides, lib-prefixed and JAX names, `libtpu` fallback, and an actionable error when no plugin is found
- [x] Plugin loading and execution counters are thread-safe

**Execution (LazyTensor)**
- [x] Materialization errors surfaced (`MaterializationError`, `fetchScalars`, `LazyTensorBarrierThrowing`) instead of `[]`
- [x] Diagnostics go to stderr. MLIR dumps only with `MAGMA_DEBUG` / `MAGMA_DEBUG_DIR`
- [x] Promoted constants keep their dtype (Int32, Double, Bool, and half-precision graphs)
- [x] Compilation and trace caches are LRU-bounded (PJRT and Metal)
- [x] Cache-key collisions fixed (attributes, constant values, dtypes, loop bodies, ±0)
- [x] Batched matmul (rank ≥ 3) compiles, so attention and Transformers now execute
- [x] Fixes to CSE and algebraic simplification that changed results
- [x] Thread-safe reads: barriers serialized, and a failed shared batch retries a tensor on its own

**nn / Core**
- [x] Conv1d, ConvTranspose2d, GroupNorm, and InstanceNorm2d execute
- [x] `Module.to(device:)` moves all parameters and buffers
- [x] Dropout honored (attention, Transformer, positional encoding, LSTM/GRU/RNN), and training flags propagate
- [x] Bidirectional LSTM/GRU implemented. LSTMCell/GRUCell gate split fixed for batch > 1
- [x] Checkpoints (v2) include buffers. Corrupt files throw, and v1 still loads
- [x] `manualSeed` / `withManualSeed` for host-side randomness
- [x] `Tensor(_:shape:)` validates counts. Scatter/gather gradients fixed
- [x] MNIST loader throws on bad files and recovers its cache
- [x] `crossEntropy` accepts one-hot targets. BCE-with-logits is stable
- [x] LR scheduler validation. `parameterGradients` handles tied and unused parameters
- [x] Activations, reductions, and all losses are differentiable from user code. `selu`, `pow`, `softplus`, and `elu` fixes

**Examples, docs, and repository**
- [x] Examples check for a backend first and exit non-zero on failure. BuildingSimulation times executed work. MNIST reports test accuracy
- [x] README, CHANGELOG, CONTRIBUTING, API, ARCHITECTURE, ROADMAP, LEGACY_MAPPING, and the TPU guide match the code. Code samples were compiled in a consumer package
- [x] NOTICE (OpenXLA, S4TF, SwiftIR attribution), SECURITY.md, CODE_OF_CONDUCT.md, issue and PR templates, dependabot
- [x] README image reduced from 2.4 MB to 48 KB

### Before tagging (maintainer)

- [ ] Push `main` and confirm the new CI workflow is green on GitHub
- [ ] Optionally, build on a Mac with `MAGMA_ENABLE_METAL=1` (the Metal changes were only type-checked on Linux)
- [ ] Tag, publish the pre-release, and run the consumer smoke test (steps 4–6 above)

### Deferred beyond the alpha

These are listed under *Known limitations* in the changelog:
- Unify or namespace the value-semantic (`Linear`, `Adam`) and reference-semantic (`nn.*`, `optim.*`) APIs
- Tag MetalHLO so the Metal backend can be enabled from a tagged Magma
- Run a CI job with a real CPU PJRT plugin (needs a hosted, pinned plugin build)
- Validate on real multi-GPU, TPU, and multi-host hardware
- Allow barriers to run concurrently (they are serialized today)
- Device-side dropout masks, so training with dropout can use the fast trace cache
- Keep `Double`/`Int64` constants at full precision in the IR. Add non-f32 distributed runners
