// Magma - High-level distributed Tensor op tests
// A DDP step written in ordinary Tensor code — grad.crossReplicaMean(...) and
// w - grad*lr — runs across replicas via the high-level barrier and keeps
// parameters in sync. This is the ergonomic layer over the multi-device stack.
// CPU-emulated, no OOM.

import Testing
@testable import Magma
@testable import LazyTensor
@testable import XLARuntime

@Suite("Distributed Tensor Ops Tests", .serialized)
struct DistributedTensorTests {
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    @Test("a DDP step written in Tensor code syncs across replicas")
    func tensorLevelDDPStep() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        let dev0 = client.devices[0]

        let gBuf = try client.createBuffer([Float](repeating: 0, count: 4), shape: [4],
                                           elementType: .float32, device: dev0)
        let wBuf = try client.createBuffer([100, 100, 100, 100] as [Float], shape: [4],
                                           elementType: .float32, device: dev0)

        // The whole DDP step, in ordinary Tensor code:
        let g = Tensor<Float>.input(from: gBuf)              // this replica's local gradient
        let w = Tensor<Float>.input(from: wBuf)              // replicated parameter
        let lr = Tensor<Float>([0.1, 0.1, 0.1, 0.1], shape: [4])
        let synced = g.crossReplicaMean(groups: [[0, 1]])   // average grads across replicas
        let wNew = w - synced * lr                          // SGD update

        let outs = try executeGraphReplicated(
            wNew.makeGraph(), numReplicas: 2,
            distribution: [ObjectIdentifier(gBuf): .perReplica([[2, 4, 6, 8], [10, 20, 30, 40]])],
            client: client)

        #expect(outs.count == 2)
        let r0 = try outs[0][0].toFloatArray()
        let r1 = try outs[1][0].toFloatArray()

        func close(_ a: [Float], _ b: [Float]) -> Bool {
            a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) < 1e-4 }
        }
        // mean([2,4,6,8],[10,20,30,40]) = [6,12,18,24]; w - 0.1*that = [99.4,98.8,98.2,97.6].
        let expected: [Float] = [99.4, 98.8, 98.2, 97.6]
        #expect(close(r0, r1))          // both replicas identical -> in sync
        #expect(close(r0, expected))
    }

    @Test("crossReplicaSum in Tensor code sums across replicas")
    func tensorCrossReplicaSum() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        let gBuf = try client.createBuffer([Float](repeating: 0, count: 4), shape: [4],
                                           elementType: .float32, device: client.devices[0])
        let g = Tensor<Float>.input(from: gBuf)
        let summed = g.crossReplicaSum(groups: [[0, 1]])

        let outs = try executeGraphReplicated(
            summed.makeGraph(), numReplicas: 2,
            distribution: [ObjectIdentifier(gBuf): .perReplica([[1, 2, 3, 4], [10, 20, 30, 40]])],
            client: client)
        for devOut in outs {
            #expect(try devOut[0].toFloatArray() == [11, 22, 33, 44])
        }
    }
}
