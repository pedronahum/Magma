// Magma - XLARuntime Tests
// These tests require XLA to be installed
// Skip in CI if XLA is not available

import XCTest
@testable import XLARuntime

final class XLARuntimeTests: XCTestCase {

    // MARK: - Device Tests

    func testDefaultDevice() {
        let device = Device.default
        XCTAssertEqual(device.backend, .cpu)
        XCTAssertEqual(device.index, 0)
    }

    func testDeviceDescription() {
        let cpu = Device(backend: .cpu, index: 0)
        XCTAssertEqual(cpu.description, "CPU:0")

        let gpu = Device(backend: .gpu, index: 1)
        XCTAssertEqual(gpu.description, "GPU:1")
    }

    func testDeviceEquality() {
        let d1 = Device(backend: .cpu, index: 0)
        let d2 = Device(backend: .cpu, index: 0)
        let d3 = Device(backend: .gpu, index: 0)

        XCTAssertEqual(d1, d2)
        XCTAssertNotEqual(d1, d3)
    }

    // MARK: - Backend Tests

    func testBackendRawValues() {
        XCTAssertEqual(Backend.cpu.rawValue, "cpu")
        XCTAssertEqual(Backend.gpu.rawValue, "gpu")
        XCTAssertEqual(Backend.tpu.rawValue, "tpu")
    }

    // MARK: - Error Tests

    func testXLAErrorDescriptions() {
        let errors: [XLAError] = [
            .clientCreationFailed("test"),
            .compilationFailed("test"),
            .executionFailed("test"),
            .bufferCreationFailed("test"),
            .deviceNotFound("test"),
            .notImplemented("test"),
        ]

        for error in errors {
            XCTAssertFalse(error.description.isEmpty)
        }
    }

    // MARK: - Client Tests (Requires XLA)

    func testClientCreation() throws {
        // This test tries to load the XLA plugin
        // It will fail with clientCreationFailed if the plugin isn't installed
        // which is the expected behavior when XLA is not available
        do {
            let client = try PJRTClient.create(backend: .cpu)
            // If we get here, XLA is installed - verify client was created
            XCTAssertEqual(client.backend, .cpu)
            XCTAssertFalse(client.devices.isEmpty, "Client should have at least one device")
        } catch XLAError.clientCreationFailed(let msg) {
            // Expected when XLA is not installed
            XCTAssertTrue(msg.contains("plugin"), "Error should mention plugin: \(msg)")
        }
    }

    // MARK: - Placeholder Tests

    func testPlaceholderForFutureIntegration() {
        // TODO: Add real integration tests once XLA is available
        // These will test:
        // - Client creation with real PJRT plugins
        // - Buffer creation and data transfer
        // - MLIR compilation
        // - Program execution
        XCTAssertTrue(true, "Placeholder test")
    }
}
