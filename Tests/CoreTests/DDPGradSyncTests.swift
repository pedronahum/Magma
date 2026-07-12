// Magma - DDP gradient-sync tests
// The core DDP invariant: each replica computes its own local gradient, the
// gradients are averaged across replicas (all_reduce mean), and applying the
// averaged gradient leaves every replica's parameters identical — so replicated
// parameters stay in sync. Built with MLIRBuilder, run across emulated CPU
// devices. CPU-only, no OOM.

import Testing
@testable import StableHLO
@testable import XLARuntime

@Suite("DDP Gradient Sync Tests", .serialized)
struct DDPGradSyncTests {
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    /// One SGD step with a DDP-synced gradient, as a 2-replica graph.
    ///
    /// Inputs per replica: the (replicated) weights `w` and that replica's local
    /// gradient `g`. The graph averages `g` across replicas and returns
    /// `(w - lr * mean(g), mean(g))`.
    @Test("averaged gradient keeps replicas' parameters in sync")
    func gradSyncKeepsParamsIdentical() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)

        let b = MLIRBuilder()
        let w = b.argument(TensorType(shape: [4], dtype: .float32))
        let g = b.argument(TensorType(shape: [4], dtype: .float32))
        let synced = b.allReduceMean(g, replicaGroups: [[0, 1]])
        let lr = b.constant([0.1], shape: [4], dtype: .float32)
        let wNew = b.subtract(w, b.multiply(synced, lr))
        let mlir = b.build(name: "ddp_step", outputs: [wNew, synced])

        let exe = try client.compile(mlir, numReplicas: 2, useSPMDPartitioning: false)

        let d0 = try #require(client.device(at: 0))
        let d1 = try #require(client.device(at: 1))
        func buf(_ v: [Float], on dev: PJRTDevice) throws -> PJRTBuffer {
            try client.createBuffer(v, shape: [4], elementType: .float32, device: dev)
        }
        // Same weights on both replicas; different local gradients.
        let w0: [Float] = [1, 1, 1, 1]
        let inputs = [
            [try buf(w0, on: d0), try buf([2, 4, 6, 8], on: d0)],
            [try buf(w0, on: d1), try buf([10, 20, 30, 40], on: d1)],
        ]

        let outs = try exe.executeMultiDevice(inputsPerDevice: inputs)
        #expect(outs.count == 2)

        // Output order is [wNew, synced].
        let wNew0 = try outs[0][0].toFloatArray()
        let wNew1 = try outs[1][0].toFloatArray()
        let synced0 = try outs[0][1].toFloatArray()
        let synced1 = try outs[1][1].toFloatArray()

        // 1) Averaged gradient == elementwise mean of the local gradients.
        let expectedMean: [Float] = [6, 12, 18, 24]   // mean([2,4,6,8],[10,20,30,40])
        #expect(synced0 == expectedMean)
        #expect(synced1 == expectedMean)

        // 2) Updated weights are identical on both replicas (the DDP invariant).
        #expect(wNew0 == wNew1)

        // 3) And numerically correct: w - lr*mean = [1,1,1,1] - 0.1*[6,12,18,24].
        let expectedW: [Float] = [0.4, -0.2, -0.8, -1.4]
        for i in 0..<4 {
            #expect(abs(wNew0[i] - expectedW[i]) < 1e-4)
        }
    }

    /// Divergent gradients that average to zero leave the weights unchanged —
    /// another check that the sync is a true cross-replica reduction.
    @Test("opposite gradients average to zero, weights unchanged")
    func opposingGradientsCancel() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)

        let b = MLIRBuilder()
        let w = b.argument(TensorType(shape: [4], dtype: .float32))
        let g = b.argument(TensorType(shape: [4], dtype: .float32))
        let synced = b.allReduceMean(g, replicaGroups: [[0, 1]])
        let lr = b.constant([0.5], shape: [4], dtype: .float32)
        let wNew = b.subtract(w, b.multiply(synced, lr))
        let mlir = b.build(name: "ddp_step", outputs: [wNew])

        let exe = try client.compile(mlir, numReplicas: 2, useSPMDPartitioning: false)
        let d0 = try #require(client.device(at: 0))
        let d1 = try #require(client.device(at: 1))
        func buf(_ v: [Float], on dev: PJRTDevice) throws -> PJRTBuffer {
            try client.createBuffer(v, shape: [4], elementType: .float32, device: dev)
        }
        let w0: [Float] = [5, 6, 7, 8]
        let inputs = [
            [try buf(w0, on: d0), try buf([1, 2, 3, 4], on: d0)],
            [try buf(w0, on: d1), try buf([-1, -2, -3, -4], on: d1)],
        ]
        let outs = try exe.executeMultiDevice(inputsPerDevice: inputs)
        #expect(try outs[0][0].toFloatArray() == w0)
        #expect(try outs[1][0].toFloatArray() == w0)
    }
}
