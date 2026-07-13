// Magma - Buffer distribution tests
// scatter (shard host data along axis 0), replicate (copy to every device), and
// gather (concat per-device shards back to host). These feed the multi-device
// execute paths (DDP replicate params, SPMD shard data + gather outputs).
// CPU-emulated devices, no OOM.

import Testing
@testable import XLARuntime

@Suite("Buffer Distribution Tests", .serialized)
struct BufferDistributionTests {
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    @Test("scatter shards along axis 0, gather reassembles")
    func scatterGatherRoundTrip() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 4)

        // [8,2] scattered across 4 devices -> [2,2] each.
        let host = (1...16).map(Float.init)
        let shards = try client.scatterAlongAxis0(host, shape: [8, 2], elementType: .float32, count: 4)
        #expect(shards.count == 4)
        #expect(shards.allSatisfy { $0.shape == [2, 2] })
        // Device 0 holds rows 0-1, device 3 holds rows 6-7.
        #expect(try shards[0].toFloatArray() == [1, 2, 3, 4])
        #expect(try shards[3].toFloatArray() == [13, 14, 15, 16])

        let (gathered, shape) = try client.gatherAlongAxis0(shards)
        #expect(shape == [8, 2])
        #expect(gathered == host)
    }

    @Test("replicate copies the same data to every device")
    func replicateCopies() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 3)

        let data: [Float] = [10, 20, 30, 40]
        let copies = try client.replicate(data, shape: [4], elementType: .float32, count: 3)
        #expect(copies.count == 3)
        for c in copies { #expect(try c.toFloatArray() == data) }
    }

    @Test("scatter rejects an indivisible leading dimension")
    func scatterRejectsIndivisible() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        // 5 rows can't split evenly across 2 devices.
        #expect(throws: XLAError.self) {
            _ = try client.scatterAlongAxis0((1...10).map(Float.init),
                                             shape: [5, 2], elementType: .float32, count: 2)
        }
    }
}
