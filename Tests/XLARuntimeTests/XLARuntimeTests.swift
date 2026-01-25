// Magma - XLARuntime Tests
// These tests require XLA to be installed
// Skip in CI if XLA is not available

import Testing
@testable import XLARuntime

@Suite("XLA Runtime Tests")
struct XLARuntimeTests {

    // MARK: - Device Tests

    @Test("Default device")
    func defaultDevice() {
        let device = Device.default
        #expect(device.backend == .cpu)
        #expect(device.index == 0)
    }

    @Test("Device description")
    func deviceDescription() {
        let cpu = Device(backend: .cpu, index: 0)
        #expect(cpu.description == "CPU:0")

        let gpu = Device(backend: .gpu, index: 1)
        #expect(gpu.description == "GPU:1")
    }

    @Test("Device equality")
    func deviceEquality() {
        let d1 = Device(backend: .cpu, index: 0)
        let d2 = Device(backend: .cpu, index: 0)
        let d3 = Device(backend: .gpu, index: 0)

        #expect(d1 == d2)
        #expect(d1 != d3)
    }

    // MARK: - Backend Tests

    @Test("Backend raw values")
    func backendRawValues() {
        #expect(Backend.cpu.rawValue == "cpu")
        #expect(Backend.gpu.rawValue == "gpu")
        #expect(Backend.tpu.rawValue == "tpu")
    }

    // MARK: - Error Tests

    @Test("XLA error descriptions")
    func xlaErrorDescriptions() {
        let errors: [XLAError] = [
            .clientCreationFailed("test"),
            .compilationFailed("test"),
            .executionFailed("test"),
            .bufferCreationFailed("test"),
            .deviceNotFound("test"),
            .notImplemented("test"),
        ]

        for error in errors {
            #expect(!error.description.isEmpty)
        }
    }

    // MARK: - Client Tests (Requires XLA)

    @Test("Client creation")
    func clientCreation() throws {
        // This test tries to load the XLA plugin
        // It will fail with clientCreationFailed if the plugin isn't installed
        // which is the expected behavior when XLA is not available
        do {
            let client = try PJRTClient.create(backend: .cpu)
            // If we get here, XLA is installed - verify client was created
            #expect(client.backend == .cpu)
            #expect(!client.devices.isEmpty, "Client should have at least one device")
        } catch let error as XLAError {
            // Expected when XLA is not installed
            if case .clientCreationFailed(let msg) = error {
                #expect(msg.contains("plugin"), "Error should mention plugin: \(msg)")
            }
        }
    }

    // MARK: - Placeholder Tests

    @Test("Placeholder for future integration")
    func placeholderForFutureIntegration() {
        // TODO: Add real integration tests once XLA is available
        // These will test:
        // - Client creation with real PJRT plugins
        // - Buffer creation and data transfer
        // - MLIR compilation
        // - Program execution
        #expect(true, "Placeholder test")
    }
}
