// Magma - Shardy compile-options tests
// Validate the SPMD / Shardy compile path (PJRTClient.compile(_:numPartitions:…))
// end to end on CPU: the CompileOptionsProto we encode is accepted by the plugin,
// the plugin actually supports the Shardy partitioner, and a single-partition
// program still executes correctly. Multi-device partitioning is exercised later
// (P2 SPMD test) once CPU multi-device emulation exists.
//
// Runs on the CPU plugin (no accelerator, no OOM). Gated on CPU availability.

import Testing
@testable import XLARuntime

@Suite("Shardy Compile Tests", .serialized)
struct ShardyCompileTests {
    static let cpuClient: PJRTClient? = try? PJRTClient.create(backend: .cpu)

    /// A plain module compiled with SPMD + Shardy enabled must still compile and
    /// execute correctly on a single device — this proves the proto encoding is
    /// valid and the plugin accepts `use_shardy_partitioner`.
    @Test("compile with SPMD + Shardy enabled, single partition, executes")
    func shardyEnabledExecutes() throws {
        let client = try #require(Self.cpuClient, "CPU PJRT plugin not available")
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let exe = try client.compile(
            mlir, numPartitions: 1, useSPMDPartitioning: true, useShardyPartitioner: true)

        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let y = try client.createBuffer([10, 20, 30, 40] as [Float], shape: [4], elementType: .float32)
        let outs = try exe.execute([x, y])
        let result = try outs[0].toFloatArray()
        #expect(result == [11, 22, 33, 44])
    }

    /// A module carrying the `sdy` annotations our MLIRBuilder emits (mesh + an
    /// argument sharding) must survive the Shardy partitioner. Uses a size-1 mesh
    /// so it maps onto the single CPU device (num_partitions = 1).
    @Test("module with sdy.mesh + sdy.sharding compiles under Shardy")
    func sdyAnnotatedCompiles() throws {
        let client = try #require(Self.cpuClient, "CPU PJRT plugin not available")
        let mlir = """
        module @m {
          sdy.mesh @mesh = <["x"=1]>
          func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<4xf32> {
            %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let exe = try client.compile(mlir, numPartitions: 1, useShardyPartitioner: true)

        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let outs = try exe.execute([x])
        #expect(try outs[0].toFloatArray() == [2, 4, 6, 8])
    }

    /// The default single-device compile path is unchanged.
    @Test("plain compile still works alongside the SPMD path")
    func plainCompileUnaffected() throws {
        let client = try #require(Self.cpuClient, "CPU PJRT plugin not available")
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let exe = try client.compile(mlir)   // original API
        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let outs = try exe.execute([x])
        #expect(try outs[0].toFloatArray() == [2, 4, 6, 8])
    }
}
