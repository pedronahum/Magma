// Magma - Collective execution tests
// Drive a real cross-replica collective (all_reduce) end to end: build the module
// with MLIRBuilder, compile it as num_replicas = N, and execute across N emulated
// CPU devices, checking the reduced values. This is the first cross-device
// communication and the primitive DDP gradient sync is built on.
//
// Emitter (StableHLO) and runtime (XLARuntime) are exercised together so they
// cannot drift. CPU-only, no OOM.

import Testing
@testable import StableHLO
@testable import XLARuntime

@Suite("Collective Execution Tests", .serialized)
struct CollectiveExecutionTests {
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    /// Build `all_reduce(reduction)` over a `tensor<4xf32>` argument.
    private func allReduceModule(_ reduction: String, replicas: Int) -> String {
        let b = MLIRBuilder()
        let x = b.argument(TensorType(shape: [4], dtype: .float32))
        let group = Array(0..<replicas)
        let r = b.allReduce(x, reduction: reduction, replicaGroups: [group])
        return b.build(name: "ar", outputs: [r])
    }

    private func perDeviceInputs(
        _ client: PJRTClient, _ values: [[Float]]
    ) throws -> [[PJRTBuffer]] {
        try values.enumerated().map { (d, v) in
            let dev = try #require(client.device(at: d))
            return [try client.createBuffer(v, shape: [4], elementType: .float32, device: dev)]
        }
    }

    @Test("all_reduce(add) sums across 2 replicas")
    func allReduceSum() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)

        let exe = try client.compile(
            allReduceModule("add", replicas: 2), numReplicas: 2, useSPMDPartitioning: false)

        let inputs = try perDeviceInputs(client, [[1, 2, 3, 4], [10, 20, 30, 40]])
        let outs = try exe.executeMultiDevice(inputsPerDevice: inputs)

        // Every replica holds the sum [11, 22, 33, 44].
        #expect(outs.count == 2)
        for devOut in outs {
            #expect(try devOut[0].toFloatArray() == [11, 22, 33, 44])
        }
    }

    @Test("all_reduce(maximum) takes the elementwise max across replicas")
    func allReduceMax() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)

        let exe = try client.compile(
            allReduceModule("maximum", replicas: 2), numReplicas: 2, useSPMDPartitioning: false)

        let inputs = try perDeviceInputs(client, [[1, 20, 3, 40], [10, 2, 30, 4]])
        let outs = try exe.executeMultiDevice(inputsPerDevice: inputs)
        for devOut in outs {
            #expect(try devOut[0].toFloatArray() == [10, 20, 30, 40])
        }
    }

    @Test("mean = all_reduce(add) then scale by 1/N")
    func allReduceMean() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)

        // Build sum-all_reduce then multiply by 1/2.
        let b = MLIRBuilder()
        let x = b.argument(TensorType(shape: [4], dtype: .float32))
        let summed = b.allReduce(x, reduction: "add", replicaGroups: [[0, 1]])
        let half = b.constant([0.5], shape: [4], dtype: .float32)
        let mean = b.multiply(summed, half)
        let mlir = b.build(name: "ar_mean", outputs: [mean])

        let exe = try client.compile(mlir, numReplicas: 2, useSPMDPartitioning: false)
        let inputs = try perDeviceInputs(client, [[1, 2, 3, 4], [10, 20, 30, 40]])
        let outs = try exe.executeMultiDevice(inputsPerDevice: inputs)

        // mean of [1,2,3,4] and [10,20,30,40] = [5.5, 11, 16.5, 22]
        for devOut in outs {
            let r = try devOut[0].toFloatArray()
            let expected: [Float] = [5.5, 11, 16.5, 22]
            #expect(r.count == 4)
            for i in 0..<4 { #expect(abs(r[i] - expected[i]) < 1e-4) }
        }
    }
}
