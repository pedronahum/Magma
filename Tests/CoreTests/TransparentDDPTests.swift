// Magma - Transparent DDP tests
// The whole DDP loop driven by autodiff: gradient(of: loss) -> average across
// replicas -> SGD update -> execute across devices, with the sync automatic.
// Verified equal to a single-device full-batch step. CPU-emulated, no OOM.

import Testing
import _Differentiation
@testable import Magma
@testable import LazyTensor
@testable import XLARuntime

@Suite("Transparent DDP Tests", .serialized)
struct TransparentDDPTests {
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    private func close(_ a: [Float], _ b: [Float]) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) < 1e-4 }
    }

    @Test("dataParallelSGDStep (autodiff) matches single-device, replicas in sync")
    func autodiffDDPStep() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        let dev0 = client.devices[0]

        // Distributable inputs: x = this replica's data, w = replicated parameter.
        let xBuf = try client.createBuffer([Float](repeating: 0, count: 4), shape: [4],
                                           elementType: .float32, device: dev0)
        let wBuf = try client.createBuffer([100, 100, 100, 100] as [Float], shape: [4],
                                           elementType: .float32, device: dev0)
        let x = Tensor<Float>.input(from: xBuf)
        let w = Tensor<Float>.input(from: wBuf)

        // loss(w) = sum(w * x)  =>  grad wrt w = x (this replica's local gradient).
        let updated = try dataParallelSGDStep(
            w: w, lr: 0.1, numReplicas: 2, client: client,
            dataDistribution: [ObjectIdentifier(xBuf): .perReplica([[1, 2, 3, 4], [10, 20, 30, 40]])]
        ) { w in (w * x).sum() }

        #expect(updated.count == 2)
        // synced grad = mean([1,2,3,4],[10,20,30,40]) = [5.5,11,16.5,22];
        // w - 0.1*that = [99.45, 98.9, 98.35, 97.8]. Same as a single-device
        // full-batch mean-loss step.
        let expected: [Float] = [99.45, 98.9, 98.35, 97.8]
        #expect(close(updated[0], updated[1]))   // replicas identical -> in sync
        #expect(close(updated[0], expected))
    }

    @Test("Optimizer.step(syncing:) averages gradients across replicas")
    func optimizerStepSyncing() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        let dev0 = client.devices[0]

        let gBuf = try client.createBuffer([Float](repeating: 0, count: 4), shape: [4],
                                           elementType: .float32, device: dev0)
        let wBuf = try client.createBuffer([100, 100, 100, 100] as [Float], shape: [4],
                                           elementType: .float32, device: dev0)
        let localGrad = Tensor<Float>.input(from: gBuf)

        // An optimizer over a distributable parameter; the DDP hook syncs grads.
        var opt = optim.SGD(parameters: [Parameter(Tensor<Float>.input(from: wBuf))], lr: 0.1)
        opt.step(syncing: [localGrad], groups: [[0, 1]])

        // Run the (now lazy, sync-baked) updated parameter across replicas.
        let updatedParam = opt.parameters[0].value
        let outs = try executeGraphReplicated(
            updatedParam.makeGraph(), numReplicas: 2,
            distribution: [ObjectIdentifier(gBuf): .perReplica([[2, 4, 6, 8], [10, 20, 30, 40]])],
            client: client)

        // mean grad = [6,12,18,24]; w - 0.1*that = [99.4, 98.8, 98.2, 97.6].
        let expected: [Float] = [99.4, 98.8, 98.2, 97.6]
        #expect(outs.count == 2)
        #expect(close(try outs[0][0].toFloatArray(), expected))
        #expect(close(try outs[1][0].toFloatArray(), expected))
    }
}
