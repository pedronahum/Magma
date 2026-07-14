// Magma - Shardy on GPU (single-device) tests
// Confirm the CUDA PJRT plugin was built with Shardy: it must accept the
// use_shardy_partitioner compile flag and the sdy.mesh / #sdy.sharding annotation
// format our emitter produces, and still execute correctly on one GPU.
//
// This machine has a single GPU, so real MULTI-GPU SPMD is NOT testable here
// (num_partitions must be 1). This validates only that the GPU plugin's Shardy
// pipeline is present and accepts our IR — the backend-portable half of the SPMD
// path. Gated on GPU availability; run in its own invocation (--no-parallel).

import Testing
@testable import XLARuntime

@Suite("Shardy GPU Tests", .serialized)
struct ShardyGPUTests {
    static let gpuAvailable: Bool = { (try? PJRTClient.create(backend: .gpu)) != nil }()

    @Test("CUDA plugin accepts use_shardy_partitioner and executes")
    func gpuAcceptsShardyFlag() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .gpu)
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
        let out = try exe.execute([x, y])[0].toFloatArray()
        #expect(out == [11, 22, 33, 44])
    }

    @Test("CUDA plugin's Shardy accepts the sdy.mesh + #sdy.sharding annotations")
    func gpuAcceptsSdyAnnotations() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .gpu)
        // Size-1 mesh so it maps onto the single GPU (num_partitions = 1).
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
        let out = try exe.execute([x])[0].toFloatArray()
        #expect(out == [2, 4, 6, 8])
    }
}
