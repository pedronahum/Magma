// Plugin availability probes shared by the suites that need a real PJRT
// backend. Suites gate on these with `.enabled(if:)`, so on a machine without
// the plugin they are reported as skipped rather than failed.
//
// Only one PJRT plugin can be loaded per process, and a CUDA client reserves
// most of device memory, so the backend under test is chosen explicitly with
// MAGMA_TEST_BACKEND (`cpu`, the default, or `gpu`). The probe for the backend
// that was not selected returns false without creating a client:
//
//   swift test --no-parallel                                   # CPU suites
//   MAGMA_TEST_BACKEND=gpu swift test --no-parallel --filter GPU  # GPU suites

import Foundation
import Testing
@testable import XLARuntime

enum PluginAvailability {
    /// The backend selected for this test run (`MAGMA_TEST_BACKEND`, default `cpu`).
    static let selectedBackend: String =
        ProcessInfo.processInfo.environment["MAGMA_TEST_BACKEND"]?.lowercased() ?? "cpu"

    /// Whether CPU suites should run: CPU selected and a CPU client can be created.
    static let cpu: Bool =
        selectedBackend == "cpu" && (try? PJRTClient.create(backend: .cpu)) != nil

    /// Whether GPU suites should run: GPU selected and a GPU client can be created.
    static let gpu: Bool =
        selectedBackend == "gpu" && (try? PJRTClient.create(backend: .gpu)) != nil
}
