// Magma - CPU multi-device emulation tests
// The XLA CPU plugin can expose N virtual devices via its "cpu_device_count"
// create option. This lets all multi-device logic (meshes, sharding, collectives,
// multi-device execute) be developed and tested on CPU with no physical
// accelerators and no OOM risk. This suite establishes that emulation works.

import Testing
@testable import XLARuntime

@Suite("CPU Multi-Device Emulation Tests", .serialized)
struct CpuMultiDeviceTests {
    /// Whether the CPU plugin is present at all.
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    @Test("a CPU client can expose N virtual devices")
    func eightVirtualDevices() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")

        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 8)
        #expect(client.deviceCount == 8)
        #expect(client.devices.count == 8)
        // Distinct device ids, all addressable via device(at:).
        let ids = Set(client.devices.map(\.id))
        #expect(ids.count == 8)
        for i in 0..<8 { #expect(client.device(at: i) != nil) }
    }

    @Test("device count is configurable")
    func configurableCount() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")

        let two = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        #expect(two.deviceCount == 2)

        let four = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 4)
        #expect(four.deviceCount == 4)
    }

    @Test("nil cpuDeviceCount keeps the default single-device client")
    func defaultUnchanged() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")

        let client = try PJRTClient.create(backend: .cpu)   // no option
        #expect(client.deviceCount >= 1)
    }
}
