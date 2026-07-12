// Magma - SPMD (Shardy) end-to-end tests
// The full SPMD closure that SwiftIR never reached: annotate a graph with sdy
// shardings, let the Shardy partitioner split it across a mesh, execute the
// partitioned program across emulated devices with each device holding its shard,
// gather the sharded outputs, and match a single-device reference.
//
// Model: y = x @ w with x row-sharded across a 2-device mesh and w replicated
// (data-parallel matmul — each device computes its rows). w = identity so y = x,
// making the gathered result trivially checkable. CPU-only, no OOM.

import Testing
@testable import LazyTensor
@testable import StableHLO
@testable import XLARuntime

@Suite("SPMD End-to-End Tests", .serialized)
struct SPMDEndToEndTests {
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    private func handle(_ shape: [Int]) -> LazyTensorHandle {
        LazyTensorHandle(id: TensorRegistry.shared.nextTensorId(),
                         shape: shape, dtype: .float32, device: .default)
    }

    // 8x4 input (rows 1..32) and a 4x4 identity weight, so x @ w == x.
    private let rows = 8, cols = 4
    private var xFull: [Float] { (1...32).map(Float.init) }
    private var identity: [Float] {
        var m = [Float](repeating: 0, count: 16)
        for i in 0..<4 { m[i * 4 + i] = 1 }
        return m
    }

    /// Build y = x @ w. When `sharded`, annotate x and y as row-sharded over a
    /// 2-device mesh axis "x" and give the graph that mesh.
    private func matmulGraph(client: PJRTClient, sharded: Bool)
        throws -> (IRGraph, xBuf: PJRTBuffer, wBuf: PJRTBuffer)
    {
        let dev0 = client.devices[0]
        let xBuf = try client.createBuffer(xFull, shape: [rows, cols], elementType: .float32, device: dev0)
        let wBuf = try client.createBuffer(identity, shape: [cols, cols], elementType: .float32, device: dev0)

        let graph = IRGraph()
        let x = handle([rows, cols]); x.irNode = .data(xBuf)
        let w = handle([cols, cols]); w.irNode = .data(wBuf)
        let y = handle([rows, cols]); y.irNode = .operation(op: .matmul, inputs: [x, w], attributes: [:])

        if sharded {
            graph.mesh = .linear(name: "mesh", axisName: "x", size: 2)
            // Row-shard x and y; w stays replicated (default, no annotation).
            x.sharding = TensorSharding(meshName: "mesh", dimShardings: [.sharded(on: "x"), .replicated])
            y.sharding = TensorSharding(meshName: "mesh", dimShardings: [.sharded(on: "x"), .replicated])
        }
        graph.addOutput(y)
        return (graph, xBuf, wBuf)
    }

    @Test("row-sharded matmul via Shardy matches the single-device reference")
    func shardedMatmulMatchesReference() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)

        // --- Single-device reference. ---
        let (refGraph, refX, refW) = try matmulGraph(client: client, sharded: false)
        let refExe = try client.compile(refGraph.emitStableHLO(name: "ref"))
        let yRef = try refExe.execute([refX, refW])[0].toFloatArray()
        #expect(yRef == xFull)   // x @ I == x

        // --- SPMD: Shardy partitions the annotated graph across 2 devices. ---
        let (spmdGraph, _, _) = try matmulGraph(client: client, sharded: true)
        try spmdGraph.validateShardings()
        let mlir = spmdGraph.emitStableHLO(name: "spmd")
        #expect(mlir.contains("sdy.mesh @mesh"))
        #expect(mlir.contains("#sdy.sharding<@mesh, [{\"x\"}, {}]>"))

        let spmdExe = try client.compile(
            mlir, numPartitions: 2, useSPMDPartitioning: true, useShardyPartitioner: true)

        // Scatter: device d gets its 4 rows of x (local shard) and full w.
        let halfRows = rows / 2                     // 4 rows per device
        let sliceElems = halfRows * cols            // 16 floats
        var perDevice: [[PJRTBuffer]] = []
        for d in 0..<2 {
            let xSlice = Array(xFull[(d * sliceElems)..<((d + 1) * sliceElems)])
            let xShard = try client.createBuffer(xSlice, shape: [halfRows, cols],
                                                 elementType: .float32, device: client.devices[d])
            let wRep = try client.createBuffer(identity, shape: [cols, cols],
                                               elementType: .float32, device: client.devices[d])
            perDevice.append([xShard, wRep])
        }

        let outs = try spmdExe.executeMultiDevice(inputsPerDevice: perDevice)
        #expect(outs.count == 2)

        // Gather: concatenate the per-device row shards back to the full output.
        let yGathered = try outs.flatMap { try $0[0].toFloatArray() }
        #expect(yGathered.count == rows * cols)
        #expect(yGathered == yRef)   // SPMD result == single-device reference
    }
}
