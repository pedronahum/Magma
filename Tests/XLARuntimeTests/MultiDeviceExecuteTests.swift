// Magma - Multi-device execute tests
// Exercise PJRTExecutable.executeMultiDevice across several (emulated) CPU
// devices: an executable compiled for N devices runs on all N, each with its own
// inputs resident on that device, producing per-device outputs. No accelerator,
// no OOM.

import Testing
@testable import XLARuntime

@Suite("Multi-Device Execute Tests", .serialized)
struct MultiDeviceExecuteTests {
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    private static let addModule = """
    module @m {
      func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
        %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
        return %0 : tensor<4xf32>
      }
    }
    """

    @Test("replicated program runs across 2 devices with identical inputs")
    func replicatedAcrossTwoDevices() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        #expect(client.deviceCount == 2)

        let exe = try client.compile(Self.addModule, numPartitions: 2, useSPMDPartitioning: true)

        let d0 = try #require(client.device(at: 0))
        let d1 = try #require(client.device(at: 1))
        func buf(_ v: [Float], on dev: PJRTDevice) throws -> PJRTBuffer {
            try client.createBuffer(v, shape: [4], elementType: .float32, device: dev)
        }
        let inputs = [
            [try buf([1, 2, 3, 4], on: d0), try buf([10, 20, 30, 40], on: d0)],
            [try buf([1, 2, 3, 4], on: d1), try buf([10, 20, 30, 40], on: d1)],
        ]

        let outs = try exe.executeMultiDevice(inputsPerDevice: inputs)
        #expect(outs.count == 2)
        for devOut in outs {
            #expect(devOut.count == 1)
            #expect(try devOut[0].toFloatArray() == [11, 22, 33, 44])
        }
    }

    @Test("each device computes on its own inputs (data parallel)")
    func perDeviceInputs() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)

        let exe = try client.compile(Self.addModule, numPartitions: 2, useSPMDPartitioning: true)

        let d0 = try #require(client.device(at: 0))
        let d1 = try #require(client.device(at: 1))
        func buf(_ v: [Float], on dev: PJRTDevice) throws -> PJRTBuffer {
            try client.createBuffer(v, shape: [4], elementType: .float32, device: dev)
        }
        let zeros: [Float] = [0, 0, 0, 0]
        let inputs = [
            [try buf([1, 2, 3, 4], on: d0), try buf(zeros, on: d0)],
            [try buf([100, 200, 300, 400], on: d1), try buf(zeros, on: d1)],
        ]

        let outs = try exe.executeMultiDevice(inputsPerDevice: inputs)
        #expect(outs.count == 2)
        #expect(try outs[0][0].toFloatArray() == [1, 2, 3, 4])
        #expect(try outs[1][0].toFloatArray() == [100, 200, 300, 400])
    }
}
