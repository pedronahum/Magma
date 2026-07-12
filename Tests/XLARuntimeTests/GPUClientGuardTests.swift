// Magma - GPU client guard tests
// Verify the concurrent-accelerator-client guard: at most one live GPU client at
// a time (unless MAGMA_ALLOW_CONCURRENT_ACCEL_CLIENTS is set), so a stray second
// client cannot over-commit unified memory and freeze the machine.
//
// These tests hold at most ONE real GPU client at a time — the refused second
// client throws before reserving any memory — so they are safe to run.
// Run this suite in its own invocation: a retained GPU client from another suite
// in the same process would (correctly) make the first client here be refused.

import Testing
@testable import XLARuntime

@Suite("GPU Client Guard Tests", .serialized)
struct GPUClientGuardTests {
    static let gpuAvailable: Bool = { (try? PJRTClient.create(backend: .gpu)) != nil }()

    @Test("a second concurrent GPU client is refused")
    func secondConcurrentClientRefused() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")

        let first = try PJRTClient.create(backend: .gpu)   // holds the one slot
        #expect(first.backend == .gpu)

        // A second concurrent client must be refused — and refused *before* it
        // reserves any device memory, so this does not double the reservation.
        #expect(throws: XLAError.self) {
            _ = try PJRTClient.create(backend: .gpu)
        }

        withExtendedLifetime(first) {}   // keep `first` alive across the attempt
    }

    @Test("a new GPU client succeeds once the previous one is released")
    func slotReleasedAfterScope() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")

        do {
            let a = try PJRTClient.create(backend: .gpu)
            #expect(a.backend == .gpu)
        }   // `a` released here -> slot freed

        // With the slot freed, a fresh client must be creatable again.
        let b = try PJRTClient.create(backend: .gpu)
        #expect(b.backend == .gpu)
    }
}
