// Magma - Plugin mismatch tests
// PJRT_LoadPlugin previously returned OK for ANY request once a plugin was
// loaded, so asking for a second (different) backend silently handed back the
// already-loaded one. It must now reject a different path.
//
// .serialized: creating a GPU client reserves most of unified memory.

import Foundation
import Testing
import CXLARuntime
@testable import XLARuntime

@Suite("Plugin Mismatch Tests", .serialized)
struct PluginMismatchTests {
    static let gpuAvailable: Bool = { (try? PJRTClient.create(backend: .gpu)) != nil }()

    @Test("a different plugin path is rejected once one is loaded")
    func differentPathRejected() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")
        _ = try PJRTClient.create(backend: .gpu)   // ensure a plugin is loaded

        // Requesting a different plugin must fail loudly, not return OK.
        let code = PJRT_LoadPlugin("/nonexistent/some_other_plugin.so")
        #expect(code != SW_PJRT_Error_OK)
    }

    @Test("the same plugin path still reloads OK")
    func samePathOK() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")
        _ = try PJRTClient.create(backend: .gpu)

        guard let base = ProcessInfo.processInfo.environment["MAGMA_XLA_PATH"] else {
            return   // exact-path check needs the env; skip otherwise
        }
        let path = "\(base)/pjrt_c_api_gpu_plugin.so"
        #expect(PJRT_LoadPlugin(path) == SW_PJRT_Error_OK)
    }
}
