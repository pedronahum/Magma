// Magma - End-to-end DDP tests
// The defining DDP correctness property: training with N replicas, each on a data
// shard, averaging gradients across replicas (all_reduce mean), produces the SAME
// parameter update as single-device training on the full batch. Built as a graph
// (reduce-mean over the local batch -> cross-replica mean -> SGD step) and run
// through the high-level replicated barrier. CPU-only, no OOM.

import Testing
@testable import LazyTensor
@testable import XLARuntime

@Suite("DDP End-to-End Tests", .serialized)
struct DDPEndToEndTests {
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    private func handle(_ shape: [Int]) -> LazyTensorHandle {
        LazyTensorHandle(id: TensorRegistry.shared.nextTensorId(),
                         shape: shape, dtype: .float32, device: .default)
    }

    // Per-sample gradients for a 4-sample batch, D = 4. Full-batch mean gradient
    // is [7,8,9,10]; with w=[100,…] and lr=0.1 the step is [99.3,99.2,99.1,99.0].
    private let g1: [Float] = [1, 2, 3, 4]
    private let g2: [Float] = [5, 6, 7, 8]
    private let g3: [Float] = [9, 10, 11, 12]
    private let g4: [Float] = [13, 14, 15, 16]
    private let w0: [Float] = [100, 100, 100, 100]
    private let expectedW: [Float] = [99.3, 99.2, 99.1, 99.0]

    /// One optimizer step over a batch of per-sample gradients `gShape`:
    /// mean over the local batch, optionally averaged across replicas, then
    /// `w - lr * grad`. Returns the graph and the gradient input buffer.
    private func stepGraph(
        client: PJRTClient, gShape: [Int], replicaGroups: [[Int]]?
    ) throws -> (IRGraph, PJRTBuffer, PJRTBuffer) {
        let gBuf = try client.createBuffer([Float](repeating: 0, count: gShape.reduce(1, *)),
                                           shape: gShape, elementType: .float32, device: client.devices[0])
        let wBuf = try client.createBuffer(w0, shape: [4], elementType: .float32, device: client.devices[0])

        let graph = IRGraph()
        let g = handle(gShape); g.irNode = .data(gBuf)
        let localMean = handle([4])
        localMean.irNode = .operation(op: .reduceMean, inputs: [g],
                                      attributes: ["axes": [0], "keepDims": false])

        // Cross-replica average only in the distributed case.
        let grad: LazyTensorHandle
        if let groups = replicaGroups {
            grad = handle([4])
            grad.irNode = .operation(op: .allReduceMean, inputs: [localMean],
                                     attributes: ["replicaGroups": groups])
        } else {
            grad = localMean
        }

        let w = handle([4]); w.irNode = .data(wBuf)
        let lr = handle([4]); lr.irNode = .constant(values: [0.1], shape: [4])
        let step = handle([4]); step.irNode = .operation(op: .multiply, inputs: [grad, lr], attributes: [:])
        let wNew = handle([4]); wNew.irNode = .operation(op: .subtract, inputs: [w, step], attributes: [:])
        graph.addOutput(wNew)
        return (graph, gBuf, wBuf)
    }

    @Test("DDP (2 replicas, sharded batch) matches single-device full-batch step")
    func ddpMatchesSingleDevice() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)

        // --- Single-device reference: full 4-sample batch, one replica. ---
        let (singleGraph, singleG, _) = try stepGraph(client: client, gShape: [4, 4], replicaGroups: nil)
        let singleOut = try executeGraphReplicated(
            singleGraph, numReplicas: 1,
            distribution: [ObjectIdentifier(singleG): .perReplica([g1 + g2 + g3 + g4])],
            client: client)
        let wSingle = try singleOut[0][0].toFloatArray()

        // --- DDP: 2 replicas, 2 samples each, gradients averaged across replicas. ---
        let (ddpGraph, ddpG, _) = try stepGraph(client: client, gShape: [2, 4], replicaGroups: [[0, 1]])
        let ddpOut = try executeGraphReplicated(
            ddpGraph, numReplicas: 2,
            distribution: [ObjectIdentifier(ddpG): .perReplica([g1 + g2, g3 + g4])],
            client: client)

        #expect(ddpOut.count == 2)
        let wDdp0 = try ddpOut[0][0].toFloatArray()
        let wDdp1 = try ddpOut[1][0].toFloatArray()

        func close(_ a: [Float], _ b: [Float]) -> Bool {
            a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) < 1e-4 }
        }
        // Single-device step is correct...
        #expect(close(wSingle, expectedW))
        // ...both replicas agree (params stay in sync)...
        #expect(close(wDdp0, wDdp1))
        // ...and DDP equals the single-device full-batch update.
        #expect(close(wDdp0, wSingle))
    }
}
