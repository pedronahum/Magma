// Plugin availability probes shared by the suites that need a real PJRT
// backend. Suites gate on these with `.enabled(if:)`, so on a machine without
// the plugin they are reported as skipped rather than failed.

import Testing
@testable import XLARuntime

enum PluginAvailability {
    /// Whether a CPU PJRT client can be created in this process.
    static let cpu: Bool = (try? PJRTClient.create(backend: .cpu)) != nil

    /// Whether a GPU PJRT client can be created in this process. Only one
    /// plugin can be loaded per process, so this is false when a CPU-backed
    /// suite loaded its plugin first; run the GPU suites in their own invocation.
    static let gpu: Bool = (try? PJRTClient.create(backend: .gpu)) != nil
}
