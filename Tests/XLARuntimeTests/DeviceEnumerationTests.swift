// Magma - Device enumeration tests
// Baseline coverage that a client exposes its addressable devices (the
// foundation every multi-device path builds on). Runs on the CPU plugin; a plain
// CPU client has exactly one device (multi-device emulation is covered
// separately).

import Testing
@testable import XLARuntime

@Suite("Device Enumeration Tests", .serialized)
struct DeviceEnumerationTests {
    static let cpuClient: PJRTClient? = try? PJRTClient.create(backend: .cpu)

    @Test("a client exposes its addressable devices")
    func enumerate() throws {
        let client = try #require(Self.cpuClient, "CPU PJRT plugin not available")

        #expect(client.deviceCount == client.devices.count)
        #expect(client.deviceCount >= 1)
        #expect(client.defaultDevice != nil)
        #expect(client.device(at: 0) != nil)
        #expect(client.device(at: client.deviceCount) == nil)   // out of range

        // Every enumerated device reports a non-empty kind and matches the list.
        for (i, device) in client.devices.enumerated() {
            #expect(!device.kind.isEmpty)
            #expect(client.device(at: i) === device)
        }
    }
}
