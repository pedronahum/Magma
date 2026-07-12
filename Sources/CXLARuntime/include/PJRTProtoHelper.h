//===-- PJRTProtoHelper.h - PJRT Protobuf Helper ------------*- C++ -*-===//
//
// SwiftIR - Phase 11B: PJRT Integration
// C helper functions for creating XLA protobuf messages
//
//===------------------------------------------------------------------===//

#ifndef PJRT_PROTO_HELPER_H
#define PJRT_PROTO_HELPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Create a serialized CompileOptionsProto with an optimization level.
///
/// num_replicas / num_partitions describe the device partitioning; Magma is
/// currently single-device and the sole caller passes (1, 1). Multi-device
/// SPMD is future work (see Documentation/ROADMAP.md).
///
/// xla_opt_level: XLA backend optimization level (0-3)
///   0 = No optimization (fastest compile, slowest execution)
///   1 = Basic optimization
///   2 = Standard optimization (recommended, GPU requires >= 2)
///   3 = Maximum optimization (slowest compile, best execution)
///  -1 = Use XLA's default optimization level
/// Returns a malloc'd buffer containing the serialized protobuf.
/// Caller must free the returned buffer with PJRT_FreeCompileOptions.
char* PJRT_CreateCompileOptionsWithOptLevel(
    int64_t num_replicas,
    int64_t num_partitions,
    int32_t xla_opt_level,
    size_t* out_size
);

/// Create a serialized CompileOptionsProto with SPMD / Shardy options.
///
/// use_spmd_partitioning enables XLA's SPMD partitioner (ExecutableBuildOptions
/// field 6); use_shardy_partitioner runs the Shardy propagation + partitioning
/// pipeline (field 19). Both are bools passed as int (0/1) and are omitted from
/// the proto when 0, so passing (…, 0, 0, …) yields the same bytes as
/// PJRT_CreateCompileOptionsWithOptLevel.
///
/// num_partitions > 1 requires a client with that many devices. A device
/// assignment is not set here — XLA uses its default (iota) assignment.
char* PJRT_CreateCompileOptionsSPMD(
    int64_t num_replicas,
    int64_t num_partitions,
    int32_t xla_opt_level,
    int use_spmd_partitioning,
    int use_shardy_partitioner,
    size_t* out_size
);

/// Free a buffer allocated by PJRT_CreateCompileOptions
void PJRT_FreeCompileOptions(char* buffer);

#ifdef __cplusplus
}
#endif

#endif // PJRT_PROTO_HELPER_H
