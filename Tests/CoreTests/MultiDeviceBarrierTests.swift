// Magma - High-level multi-device barrier tests
// executeGraphReplicated compiles a traced IRGraph for N replicas and runs it
// across N emulated devices, distributing each .data input (replicated param vs
// per-replica data). This is the high-level barrier DDP builds on. CPU-only.

import Testing
@testable import LazyTensor
@testable import XLARuntime

@Suite("High-Level Multi-Device Barrier Tests", .serialized)
struct MultiDeviceBarrierTests {
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    private func handle(_ shape: [Int]) -> LazyTensorHandle {
        LazyTensorHandle(id: TensorRegistry.shared.nextTensorId(),
                         shape: shape, dtype: .float32, device: .default)
    }

    @Test("replicated graph: per-replica data + replicated param")
    func replicatedGraph() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        let dev0 = client.devices[0]

        // Build y = x + w. x is per-replica data (value irrelevant here); w is a
        // replicated parameter (its baked value is copied to every replica).
        let xBuf = try client.createBuffer([0, 0, 0, 0] as [Float], shape: [4],
                                           elementType: .float32, device: dev0)
        let wBuf = try client.createBuffer([100, 100, 100, 100] as [Float], shape: [4],
                                           elementType: .float32, device: dev0)
        let graph = IRGraph()
        let x = handle([4]); x.irNode = .data(xBuf)
        let w = handle([4]); w.irNode = .data(wBuf)
        let y = handle([4]); y.irNode = .operation(op: .add, inputs: [x, w], attributes: [:])
        graph.addOutput(y)

        let outs = try executeGraphReplicated(
            graph, numReplicas: 2,
            distribution: [ObjectIdentifier(xBuf): .perReplica([[1, 2, 3, 4], [10, 20, 30, 40]])],
            client: client)

        #expect(outs.count == 2)
        // Replica 0: [1,2,3,4] + [100,...]; replica 1: [10,20,30,40] + [100,...].
        #expect(try outs[0][0].toFloatArray() == [101, 102, 103, 104])
        #expect(try outs[1][0].toFloatArray() == [110, 120, 130, 140])
    }

    @Test("param not listed in distribution defaults to replicated")
    func defaultReplicated() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        let dev0 = client.devices[0]

        // y = x * w, both defaulting to replicated (empty distribution): both
        // replicas compute the same result.
        let xBuf = try client.createBuffer([2, 2, 2, 2] as [Float], shape: [4],
                                           elementType: .float32, device: dev0)
        let wBuf = try client.createBuffer([3, 3, 3, 3] as [Float], shape: [4],
                                           elementType: .float32, device: dev0)
        let graph = IRGraph()
        let x = handle([4]); x.irNode = .data(xBuf)
        let w = handle([4]); w.irNode = .data(wBuf)
        let y = handle([4]); y.irNode = .operation(op: .multiply, inputs: [x, w], attributes: [:])
        graph.addOutput(y)

        let outs = try executeGraphReplicated(graph, numReplicas: 2, client: client)
        #expect(outs.count == 2)
        for devOut in outs {
            #expect(try devOut[0].toFloatArray() == [6, 6, 6, 6])
        }
    }

    @Test("in-graph allReduceMean syncs across replicas (transparent DDP grad-sync)")
    func inGraphAllReduceMean() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        let dev0 = client.devices[0]

        // y = allReduceMean(g): each replica supplies its own g; the graph
        // averages across replicas, so both get the mean — the DDP grad-sync
        // pattern expressed through the high-level graph + barrier.
        let gBuf = try client.createBuffer([0, 0, 0, 0] as [Float], shape: [4],
                                           elementType: .float32, device: dev0)
        let graph = IRGraph()
        let g = handle([4]); g.irNode = .data(gBuf)
        let mean = handle([4])
        mean.irNode = .operation(op: .allReduceMean, inputs: [g],
                                 attributes: ["replicaGroups": [[0, 1]]])
        graph.addOutput(mean)

        let outs = try executeGraphReplicated(
            graph, numReplicas: 2,
            distribution: [ObjectIdentifier(gBuf): .perReplica([[2, 4, 6, 8], [10, 20, 30, 40]])],
            client: client)

        #expect(outs.count == 2)
        for devOut in outs {   // mean([2,4,6,8],[10,20,30,40]) = [6,12,18,24]
            #expect(try devOut[0].toFloatArray() == [6, 12, 18, 24])
        }
    }

    @Test("in-graph allReduce(add) sums across replicas")
    func inGraphAllReduceSum() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        let dev0 = client.devices[0]

        let gBuf = try client.createBuffer([0, 0, 0, 0] as [Float], shape: [4],
                                           elementType: .float32, device: dev0)
        let graph = IRGraph()
        let g = handle([4]); g.irNode = .data(gBuf)
        let summed = handle([4])
        summed.irNode = .operation(op: .allReduce, inputs: [g],
                                   attributes: ["reduction": "add", "replicaGroups": [[0, 1]]])
        graph.addOutput(summed)

        let outs = try executeGraphReplicated(
            graph, numReplicas: 2,
            distribution: [ObjectIdentifier(gBuf): .perReplica([[1, 2, 3, 4], [10, 20, 30, 40]])],
            client: client)
        for devOut in outs {
            #expect(try devOut[0].toFloatArray() == [11, 22, 33, 44])
        }
    }
}
