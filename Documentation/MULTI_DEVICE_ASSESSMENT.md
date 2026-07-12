# Multi-Device (Multi-GPU / Multi-TPU) Support — Assessment

**Status:** assessment only — nothing here is implemented yet.
**Scope:** what it would take to run Magma across more than one accelerator,
covering both multi-GPU (CUDA) and multi-TPU, single-host and multi-host.

---

## 1. Executive summary

Magma today is **strictly single-device at every layer**. Going multi-device is
not a localized change; it touches the FFI shim, compile options, the runtime
client, the lazy-tensor graph, autodiff, the optimizer, and data loading. The
good news is that PJRT (the backend Magma already uses) has first-class
multi-device support — the contract exists, it simply isn't wired through.

Three decisions frame the whole effort:

1. **Single-controller vs multi-controller.** One process driving N devices via a
   single client (JAX/TPU style) vs one process per device that rendezvous
   (PyTorch DDP style). These need different plumbing and interact differently
   with Magma's current *single global client* design and the new
   *concurrent-accelerator-client guard*.
2. **Partitioning paradigm.** Compiler-partitioned SPMD (GSPMD/Shardy — annotate
   shardings, let XLA insert collectives) vs manual collectives (write
   `all_reduce`/`all_gather` yourself and build DDP/FSDP by hand).
3. **Single-host vs multi-host.** A single box with N GPUs / one TPU board is far
   less work than a TPU pod or multi-node GPU cluster, which additionally needs a
   distributed coordination service, global device numbering, and rendezvous.

**Headline caveats**

- The ROADMAP's "Legacy Foundation" (`SwiftIRSharding`) **is not in this
  repository**. It exists in `pedronahum/SwiftIR`, but — verified — it is a
  *sharding-annotation + propagation* foundation, **not** a distributed-execution
  one: it emits `sdy` annotations (reusable, pure-Swift in the `…Lite` variant)
  and runs propagation via a standalone `sdy_opt`, but never partitions, inserts
  collectives, or executes multi-device. Partitioning, collectives, multi-device
  execution, and grad sync are new work regardless. Treat the ROADMAP timeline as
  optimistic. (Details in §3.9.)
- **This machine (single GB10) cannot test real multi-GPU.** However the XLA CPU
  plugin can emulate N devices, so the *logic* (mesh, sharding, collectives, grad
  sync) is developable and testable on CPU with zero OOM risk (see §6).

---

## 2. Current state — where single-device is baked in

| Layer | File | Single-device assumption |
|-------|------|--------------------------|
| FFI execute | `Sources/CXLARuntime/PJRTSimpleWrapper.c:1071,1169` | `num_devices = 1` hardcoded in every execute path; one `argument_lists` / `output_lists`; global `g_output_lists_array` sized for one device |
| FFI compile | `PJRTSimpleWrapper.c:945` | `PJRT_CreateCompileOptionsWithOptLevel(1, 1, …)` — one replica, one partition, no `device_assignment`, no `use_spmd_partitioning` |
| Compile-options proto | `Sources/CXLARuntime/PJRTProtoHelper.cpp` | Only emits `num_replicas`/`num_partitions`; no device-assignment or SPMD fields |
| Client | `Sources/XLARuntime/XLARuntime.swift:487` | Enumerates *all* addressable devices but exposes only `defaultDevice = devices.first` |
| Buffer placement | `XLARuntime.swift:492` | `createBuffer` targets a single device; no scatter/replicate; no device→device transfer |
| Execute (Swift) | `XLARuntime.swift:671` | Uses `devices.first`; returns a flat `[PJRTBuffer]` (no per-device output lists) |
| Global client | `LazyTensor.swift` | One cached client per backend; fine for single-controller, but there is no notion of a device set or mesh |
| Graph IR | `LazyTensor` | No sharding annotations on nodes; trace cache keyed without sharding/device set |
| Tensor / Device | `Sources/Core/Tensor.swift` | `Device(backend, index)` has an index but no mesh, no sharding spec, no placement/replication API |
| Autodiff | `Sources/Core/Autodiff.swift` | No gradient all-reduce / no cross-replica reduction hook |
| Optimizer | `Sources/Core/Optimizer.swift` | Updates local params only; no sharded state, no grad sync |
| Data | `Sources/Core/Data.swift` | No distributed sampler / per-replica sharding of batches |
| Multi-host | `XLARuntime.swift:186–200` | `TPUEnvironment.isMultiHost` / `numChips` are *detection stubs only* — nothing consumes them; no distributed init |
| Safety guard | `XLARuntime.swift` (new) | The concurrent-accelerator-client guard forbids a **2nd client** — correct for single-controller (1 client, N devices), but multi-controller (1 client per process) would need the opt-out and per-process reasoning |

**What already helps:** PJRT enumerates every addressable device
(`PJRT_GetAddressableDevices`), the CUDA plugin already reserves a
`CollectiveBFCAllocator` (seen in logs), and PJRT's execute contract is natively
multi-device:

```c
// pjrt_c_api.h — PJRT_LoadedExecutable_Execute_Args
PJRT_Buffer* const* const* argument_lists;  // [num_devices][num_args]
size_t num_devices;
PJRT_Buffer** const* output_lists;          // [num_devices][num_outputs]
PJRT_Event** device_complete_events;        // [num_devices]
PJRT_Device* execute_device;                // or pin to one device
```

So the backend can do it; the work is exposing it.

---

## 3. What multi-device requires — by capability

### 3.1 Device & topology model
- `MeshAxis(name, size)` and `DeviceMesh` (1-D and N-D), logical→physical device
  assignment, `deviceCount`, `axis(named:)`.
- A `Device`/placement extension so a tensor can be "on a mesh" and replicated or
  sharded, not just on `index`.

### 3.2 Data distribution (buffers)
- Per-device buffer creation and a host→N-device **scatter** (shard along a dim)
  and **replicate** (same bytes to all), plus N-device→host **gather**.
- Optionally device↔device transfer (`PJRT_Buffer` copy-to-device) for
  re-sharding.
- Rework `PJRTExecutable.execute` to accept/return `[[PJRTBuffer]]`
  (per-device), and rework the FFI to pass `num_devices = N` with the nested
  `argument_lists`/`output_lists` (careful caller-owned allocation/free).

### 3.3 Program partitioning — Shardy is the chosen paradigm

The intended approach is **Shardy** (OpenXLA's MLIR-based partitioner — the
successor to GSPMD, https://github.com/openxla/shardy), which the legacy SwiftIR
codebase already used. See §3.9 for the concrete Shardy integration architecture.
In short: annotate the StableHLO Magma already emits with `sdy` shardings, enable
the Shardy partitioner in the compile options, and let XLA run propagation +
partitioning and insert the collectives.

A **manual-collectives** path (writing `all_reduce`/`all_gather` yourself for a
plain DDP loop) is still worth landing *first* as an MVP — it's the smallest
end-to-end slice that proves multi-device buffers + a real collective +
multi-device execute + grad sync, and de-risks the FFI rework before the
partitioner work. It is a stepping stone, not the destination.

### 3.4 Collectives (needed by both, explicitly by B)
- StableHLO: `stablehlo.all_reduce` (with a reduction region), `all_gather`,
  `reduce_scatter`, `all_to_all`, `collective_permute`; attributes
  `replica_groups`, `channel_handle`, cross-replica vs cross-partition.
- Runtime: NCCL (GPU) / ICI (TPU) provided by the XLA plugin, but NCCL
  initialization and the collective allocator must be configured; multi-host
  additionally needs the NCCL unique-id exchanged via the coordinator.

### 3.5 Execution
- Single-controller SPMD: compile with a device assignment, execute once across
  all addressable devices (`num_devices = N`).
- Manual/DDP: same executable launched per replica with per-device inputs.
- Either way: FFI + Swift `execute` must handle N input lists, N output lists,
  and N completion events.

### 3.6 Multi-host (only if targeting pods / multi-node)
- PJRT **distributed init**: a coordination service (gRPC) with
  `process_index`/`process_count`, global device-id assignment, and a KV store
  for rendezvous; GPU also needs the NCCL unique-id broadcast.
- Client creation must pass these options (the wrapper currently creates the
  client with *no* options).
- This is a large, self-contained workstream on top of single-host.

### 3.7 Autodiff & training integration
- **DDP:** replicate the model, all-reduce (average) gradients after backward and
  before the optimizer step — a hook in `valueWithGradient` consumers / the
  training loop.
- **FSDP:** shard parameters and optimizer state across the mesh; all-gather
  shards for forward/backward, reduce-scatter grads. Requires optimizer +
  parameter-store changes.

### 3.8 Data loading
- `DistributedSampler`: shard the dataset by `process_index` / replica, consistent
  seeding, per-replica batch sizing.

### 3.9 Shardy integration architecture (the chosen path)

**What Shardy is.** An MLIR `sdy` dialect + compiler pipeline that (1) *propagates*
user-specified shardings through the program, then (2) *partitions* it into an
SPMD program, inserting the necessary collectives. It works across TPU/GPU/CPU
(collectives lower to ICI/NCCL); the CPU backend can emulate N devices for tests.

**The `sdy` dialect maps almost 1:1 onto the ROADMAP's planned Swift types** —
which confirms the API design was modeling Shardy directly:

| Swift type (ROADMAP / SwiftIR) | `sdy` construct | Expresses |
|--------------------------------|-----------------|-----------|
| `DeviceMesh` / `MeshAxis` | `sdy.mesh` op, `MeshAttr` | Named device topology with named axes + sizes |
| `TensorSharding` | `TensorShardingAttr` | Full per-tensor sharding (dim shardings, replicated axes, unreduced axes) |
| `DimensionSharding` | `DimensionShardingAttr` | Which axes shard a dimension; open/closed; priority |
| `AxisRef` | `AxisRefAttr` | Reference to a full axis or a split sub-axis |
| `SubAxisInfo` | `SubAxisInfoAttr` | Hierarchical axis splitting (pre-size, size) |
| sharding constraint on an intermediate | `sdy.sharding_constraint` op | Pin an intermediate tensor's sharding |
| manual/explicit-collective region | `sdy.manual_computation` op | Device-local code with explicit collectives |
| re-shard between layouts | `sdy.reshard` op | Change a tensor's sharding (may add collectives) |

Shardy also defines collective ops (`sdy.all_reduce`, `all_gather`, `all_slice`,
`all_to_all`, `collective_permute`) that the partitioner emits — Magma does not
hand-write these on the Shardy path.

**Two integration models — pick one:**

- **A) In-XLA (recommended for Magma).** Emit `sdy.mesh` + `sdy.sharding`
  attributes into the StableHLO module Magma already produces, enable the Shardy
  partitioner via a compile flag (`use_shardy_partitioner`, alongside
  `use_spmd_partitioning` + `num_partitions = N` + a `device_assignment`), and let
  PJRT's XLA run propagation + partitioning at compile time. **No extra native
  dependency**, and the compile flag can reuse the *existing* `env_option_overrides`
  proto machinery in `PJRTProtoHelper.cpp` (it already writes an XLA flag override
  for `xla_backend_optimization_level` — a Shardy flag is the same shape). Magma's
  job shrinks to: build correct `sdy` textual attributes + set the flags + supply
  the device assignment.

- **B) Standalone `sdy_opt` (offline propagation / inspection only).** This was
  SwiftIR's *actual* approach: `SdyOptRunner.swift` shells out (Foundation
  `Process`) to a Bazel-built `sdy_opt` binary running
  `--sdy-add-data-flow-edges --sdy-propagation`. It is useful to *inspect what
  shardings propagation infers*, but note what SwiftIR did **not** do: it ran
  **propagation only** — no export/SPMD-partitioning pass, no collective
  insertion, and **no handoff to PJRT for execution** (`ShardingPipeline.swift`
  literally "for now, return the prepared module"; its `SdyCAPIWrapper` C-API was
  incomplete). So `sdy_opt` is an optional offline validation tool, **not** the
  execution path, and it needs a native `sdy_opt` artifact built and
  version-matched to the plugin's MLIR.

Because Magma already compiles StableHLO straight through PJRT, **model A is both
lower-effort and the path that actually reaches execution** — it is precisely the
loop SwiftIR left open (SwiftIR emitted + propagated shardings but never
partitioned or ran them).

**What is genuinely reusable from SwiftIR** (repo `pedronahum/SwiftIR`, verified):

- **`SwiftIRShardingLite/` — pure-Swift `sdy` annotation emission, no native dep.**
  `StableHLOSharding.swift` attaches `{sdy.sharding = …}` to ops and emits `sdy.`
  constraint/reshard ops via plain string generation; `TensorSharding.swift`
  supplies the `mlirAttributeText`; `DeviceMesh.swift` produces `sdy.mesh @name =
  <[…]>` text. This is the real, borrowable value and fits Magma's *pure-Swift
  StableHLO* generator directly.
- **Avoid** the full `SwiftIRSharding/` `DeviceMesh` here — it binds
  `SdyCAPIWrapper`/`MLIRCoreWrapper` (native MLIR C-API), the heavier dependency
  Magma doesn't need for model A.

**Caveats.**
- None of this Shardy code is vendored in *this* repo — port the `…Lite` types
  from `pedronahum/SwiftIR` or rebuild them from the §3.9 mapping table.
- SwiftIR proves annotation + propagation only; **partitioning, collective
  insertion, multi-device execution, and gradient sync are all still new work**
  regardless of how much is borrowed. Do not treat SwiftIR as a distributed-
  execution foundation — it is a sharding-annotation foundation.
- Confirm the target XLA plugin was built with Shardy enabled and whether
  `use_shardy_partitioner` is default-on at that pin.

---

## 4. Recommendation

Adopt **SPMD-via-Shardy as the strategic direction**, using the **in-XLA
integration model** (§3.9-A): emit `sdy` annotations into Magma's StableHLO and
enable the Shardy partitioner through compile flags — no extra native dependency,
and the flag reuses the existing `env_option_overrides` proto path. This matches
XLA/JAX/TorchTPU, minimizes hand-written communication, and is what the ROADMAP
and SwiftIR already targeted.

**Land a single-host data-parallel (DDP) MVP first** via manual collectives. DDP
is the smallest end-to-end slice that exercises the *entire* new stack
(multi-device buffers → a real collective → multi-device execute → grad sync in
the training loop) and is independently useful. It also de-risks the FFI/execute
rework before the harder partitioner work. Then bring up Shardy for
tensor-parallel/FSDP.

Target **single-controller** for single-host (one client, N devices) — it fits the
current global-client design and the accelerator guard. Reserve **multi-controller**
for the multi-host phase, where it's unavoidable.

---

## 5. Phased plan & rough effort

Effort: **S** ≈ days, **M** ≈ 1–2 weeks, **L** ≈ multiple weeks (one engineer).

| Phase | Deliverable | Effort |
|-------|-------------|--------|
| **0. Foundations** | Surface all devices on `PJRTClient`; multi-device execute in the FFI (`num_devices=N`, nested arg/output lists, events) and Swift `execute` returning `[[PJRTBuffer]]`; buffer scatter/replicate/gather | **L** |
| **1. DDP MVP (single host)** | `all_reduce` StableHLO op + runtime; `DeviceMesh` (1-D); replicate params; gradient all-reduce hook; `DistributedSampler`; CPU-emulated multi-device tests | **M–L** |
| **2. SPMD / Shardy** | Port the pure-Swift sdy-annotation types from `SwiftIRShardingLite` (`DeviceMesh`, `TensorSharding`/`DimensionSharding`/`AxisRef`/`SubAxisInfo`, `StableHLOSharding` — §3.9); emit `sdy.mesh` + `{sdy.sharding=…}` (+ `sdy.sharding_constraint`) into Magma's StableHLO; add `use_shardy_partitioner` + `use_spmd_partitioning` + `num_partitions` + `device_assignment` to compile options (reusing the `env_option_overrides` path); track sharding on graph nodes. **New vs SwiftIR:** the partition→collectives→multi-device-execute closure (via PJRT in-XLA) that SwiftIR never did | **L** |
| **3. FSDP / tensor parallel** | Sharded parameter & optimizer state, all-gather/reduce-scatter integration, sharded checkpointing | **L** |
| **4. Multi-host** | PJRT distributed init / coordination service, global device ids, NCCL id exchange, multi-host data sharding, launch tooling | **L** |

Phases 0–1 are the critical path and prove the architecture. 2–4 are largely
independent follow-ons.

---

## 6. Testing strategy (important on this hardware)

- **Real multi-GPU is not testable on the single-GPU GB10.** Do not attempt to
  fake it by spinning up multiple CUDA clients — that is exactly the OOM-freeze
  the accelerator guard now prevents.
- **Emulate multiple devices on the CPU plugin.** XLA's CPU backend can expose N
  virtual devices (e.g. `xla_force_host_platform_device_count` / a PJRT CPU
  `create_options` device count). This lets mesh, sharding, collectives, and DDP
  grad-sync be developed and tested **on CPU with 8+ virtual devices, no GPU, no
  OOM risk, ~100× faster**. The wrapper currently creates the client with *no*
  options, so enabling this is itself Phase-0 work.
- Reserve real multi-GPU / multi-TPU validation for a genuinely multi-device host
  (a multi-GPU node or a TPU board/pod). Keep those runs `--no-parallel` and
  single-controller so one client owns all devices.

---

## 7. Risks & gotchas

- **Missing legacy foundation.** The ROADMAP's port-from-`SwiftIRSharding` premise
  is invalid here; budget for greenfield sharding types.
- **One plugin per process.** Fine for single-controller (one client, N devices).
  Multi-controller (one process per GPU) means one client each — compatible with
  the plugin guard, but the *single global client* cache in `LazyTensor` assumes
  one client and would need per-process reasoning.
- **Accelerator guard vs multi-controller.** The new guard forbids a 2nd *client*
  in a process; single-controller multi-device never hits it, but any design that
  wants two accelerator clients per process must use
  `MAGMA_ALLOW_CONCURRENT_ACCEL_CLIENTS=1` deliberately.
- **FFI memory management.** Multi-device execute uses caller-owned nested pointer
  arrays (`output_lists[N][*]`); the current fixed-global-buffer approach needs a
  careful rework to avoid leaks/races across N devices.
- **Determinism & performance.** Collective ordering, NCCL init cost, and the
  interaction of the collective allocator with the per-client memory fraction all
  affect correctness and speed; add collective-numerics tests early.
- **Sharding propagation subtlety.** GSPMD/Shardy can silently pick surprising
  shardings; sharding-constraint tests and explicit annotations matter.

---

## 8. Concrete first step

If this moves forward, the highest-leverage first PR is **Phase 0 device
enumeration + CPU multi-device emulation**: expose `client.devices` publicly, add
a client-create option to request N CPU devices, and add a test that enumerates 8
virtual CPU devices. That unlocks test-driven development of everything above with
no GPU and no OOM exposure — before any FFI execute rework.

---

## 9. Porting map: `SwiftIRShardingLite` → Magma (Phase 2 detail)

Verified against `pedronahum/SwiftIR` and Magma's `Sources/StableHLO/Builder/MLIRBuilder.swift`.
Magma's `MLIRBuilder` is a text builder — it accumulates op strings in
`operations: [String]` and `build(name:outputs:)` (MLIRBuilder.swift:1162) wraps
them in `module @name { func.func @main(args) -> (types) { … } }`. That is
structurally identical to SwiftIR's `ShardedModuleBuilder`, so the `sdy`
vocabulary ports cleanly and the only new work is three small hooks.

### 9.1 Port verbatim — pure-Swift data types (Foundation only, no native deps)

| SwiftIR file → type | Emits | Port target |
|---------------------|-------|-------------|
| `SwiftIRShardingLite/DeviceMesh.swift` → `MeshAxis`, `DeviceMesh` (`.linear`/`.grid`/`.cube`, `deviceCount`, `mlirText`) | `sdy.mesh @mesh = <["x"=4, "y"=2]>` | `Sources/StableHLO/Sharding/DeviceMesh.swift` ✅ done |
| `SwiftIRShardingLite/TensorSharding.swift` → `DimensionSharding` (`.replicated`, `.sharded(on:)`, `.open(on:)`), `TensorSharding` (`.replicated(meshName:rank:)`, `mlirAttributeText`, `validate(against:rank:)`) | `#sdy.sharding<@mesh, [{"x"}, {}]>` | `Sources/StableHLO/Sharding/TensorSharding.swift` ✅ done |

> **Placement note:** these live in the **StableHLO module**, not a standalone
> `Sources/Sharding/` target — co-located with their consumer `MLIRBuilder` and
> already reachable up the stack (LazyTensor → Core depend on StableHLO), so they
> are integrated rather than an isolated target. `DeviceMesh` is already wired
> into `MLIRBuilder.declareMesh(_:)` / `build()` (emits `sdy.mesh` in the header);
> `TensorSharding` is validated against `DeviceMesh` and its `MLIRBuilder`
> consumer (argument shardings + `sdy.sharding_constraint`) is task **#17**.

Both are `Equatable`/`Hashable`, `import Foundation` only.
**Caveat:** the Lite `DimensionSharding` uses `axes: [String]` and drops the full
variant's `AxisRef`/`SubAxisInfo`, so hierarchical *sub-axis* splitting isn't
expressible. Fine for a data/tensor-parallel MVP; add `AxisRef`/`SubAxisInfo`
later if sub-axis sharding is needed.

### 9.2 Do NOT port

- **`StableHLOSharding.swift` / `ShardedModuleBuilder`** — SwiftIR's own text
  builder. Magma has `MLIRBuilder`; lift only the *techniques* (§9.3), not a
  second builder.
- **`SdyOptRunner.swift`** — optional offline propagation/inspection tool, not the
  execution path.
- **`SdyCAPIWrapper.c` / full `SwiftIRSharding/DeviceMesh`** — native MLIR C-API,
  incomplete; unnecessary for the in-XLA model.

### 9.3 Add to `MLIRBuilder` — three hooks (the real new work)

1. **Mesh op in the module header.** `build(name:outputs:)` (line 1162): accept a
   `mesh: DeviceMesh?` and inject `mesh.mlirText` between `module @name {` and
   `func.func`.
2. **Argument shardings.** `argument(_:)` (line 44): add an optional
   `TensorSharding?`; in `build()`, append `{sdy.sharding = <attr>}` to each arg
   in `argDefs` (line 1163).
3. **Result / intermediate shardings.** Add a `shardingConstraint(_:_:)` method
   that appends `%r = sdy.sharding_constraint %v <#sdy.sharding<…>> : type` via the
   existing `addRawOperation` (line 82) — no need to touch every op emitter.

### 9.4 One compile-options change (reuses existing machinery)

Set `use_shardy_partitioner` + `use_spmd_partitioning` + `num_partitions = N` + a
`device_assignment` in `PJRTProtoHelper.cpp`. The flags go through the **same
`env_option_overrides` map** the file already writes for
`xla_backend_optimization_level` — an extension of existing code, not new proto
plumbing.

**Net:** the `sdy` vocabulary (~2 pure-Swift files) ports directly; the new work
is (9.3) the `MLIRBuilder` hooks, (9.4) the compile flags, and — the part SwiftIR
never reached — letting PJRT partition and execute across devices.
