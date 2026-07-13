// Magma - Tensor-parallel (Shardy) tests
// Tensor parallelism: shard a matmul's CONTRACTING dimension across a mesh. Each
// device computes a partial product; Shardy inserts an all_reduce so every device
// ends with the full result. Unlike row-sharding (#20, no communication), this
// exercises Shardy's collective insertion. Verified == single-device reference.
// CPU-emulated, no OOM.
//
// FSDP builds on the same partitioner (sharded params all-gathered on use, grads
// reduce-scattered); this covers the core collective-insertion mechanism.

import Testing
@testable import LazyTensor
@testable import StableHLO
@testable import XLARuntime

@Suite("Tensor Parallel Tests", .serialized)
struct TensorParallelTests {
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    private func handle(_ shape: [Int]) -> LazyTensorHandle {
        LazyTensorHandle(id: TensorRegistry.shared.nextTensorId(),
                         shape: shape, dtype: .float32, device: .default)
    }

    // x[2,4] @ w[4,3] = y[2,3], with the shared K=4 dimension the contraction.
    private let xFull: [Float] = [1, 2, 3, 4, 5, 6, 7, 8]              // 2x4
    private let wFull: [Float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  // 4x3

    private func matmulGraph(client: PJRTClient, sharded: Bool)
        throws -> (IRGraph, PJRTBuffer, PJRTBuffer)
    {
        let dev0 = client.devices[0]
        let xBuf = try client.createBuffer(xFull, shape: [2, 4], elementType: .float32, device: dev0)
        let wBuf = try client.createBuffer(wFull, shape: [4, 3], elementType: .float32, device: dev0)

        let graph = IRGraph()
        let x = handle([2, 4]); x.irNode = .data(xBuf)
        let w = handle([4, 3]); w.irNode = .data(wBuf)
        let y = handle([2, 3]); y.irNode = .operation(op: .matmul, inputs: [x, w], attributes: [:])

        if sharded {
            graph.mesh = .linear(name: "mesh", axisName: "k", size: 2)
            // Shard the contracting dim: x on its columns (dim 1), w on its rows
            // (dim 0). y is left for Shardy to propagate (it inserts the all_reduce).
            x.sharding = TensorSharding(meshName: "mesh", dimShardings: [.replicated, .sharded(on: "k")])
            w.sharding = TensorSharding(meshName: "mesh", dimShardings: [.sharded(on: "k"), .replicated])
        }
        graph.addOutput(y)
        return (graph, xBuf, wBuf)
    }

    @Test("contracting-dim-sharded matmul (Shardy all_reduce) matches reference")
    func contractingDimSharded() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)

        // --- Single-device reference. ---
        let (refGraph, refX, refW) = try matmulGraph(client: client, sharded: false)
        let yRef = try client.compile(refGraph.emitStableHLO(name: "ref")).execute([refX, refW])[0].toFloatArray()

        // --- Tensor-parallel via Shardy. ---
        let (tpGraph, _, _) = try matmulGraph(client: client, sharded: true)
        try tpGraph.validateShardings()
        let exe = try client.compile(
            tpGraph.emitStableHLO(name: "tp"),
            numPartitions: 2, useSPMDPartitioning: true, useShardyPartitioner: true)

        // Each device holds its K-slice: x columns [2k:2k+2] (a [2,2]) and w rows
        // [2k:2k+2] (a [2,3]). x is row-major [2,4], so slice columns by hand.
        func xColumns(_ k: Int) -> [Float] {
            var out: [Float] = []
            for row in 0..<2 { for c in (2 * k)..<(2 * k + 2) { out.append(xFull[row * 4 + c]) } }
            return out
        }
        var perDevice: [[PJRTBuffer]] = []
        for k in 0..<2 {
            let xShard = try client.createBuffer(xColumns(k), shape: [2, 2],
                                                 elementType: .float32, device: client.devices[k])
            let wShard = try client.createBuffer(Array(wFull[(k * 6)..<(k * 6 + 6)]), shape: [2, 3],
                                                 elementType: .float32, device: client.devices[k])
            perDevice.append([xShard, wShard])
        }

        let outs = try exe.executeMultiDevice(inputsPerDevice: perDevice)
        #expect(outs.count == 2)
        // After the all_reduce, every device holds the full [2,3] result.
        for devOut in outs {
            #expect(try devOut[0].toFloatArray() == yRef)
        }
        // Sanity: y[0][0] = 1*1 + 2*4 + 3*7 + 4*10 = 70.
        #expect(yRef.first == 70)
    }
}
