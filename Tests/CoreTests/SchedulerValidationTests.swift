// Magma - LR scheduler validation tests
//
// Degenerate scheduler configurations used to divide by zero (StepLR) or
// produce a NaN learning rate (CosineAnnealingLR, WarmupCosineScheduler).
// They now trap at construction with a clear message.

import Foundation
import Testing
@testable import Magma

@Suite("LR scheduler validation")
struct SchedulerValidationTests {

    @Test("StepLR with stepSize 0 traps at construction")
    func stepLRZeroStepSize() async {
        await #expect(processExitsWith: .failure) {
            _ = optim.StepLR(baseLR: 0.1, stepSize: 0)
        }
    }

    @Test("CosineAnnealingLR with totalEpochs 0 traps at construction")
    func cosineZeroEpochs() async {
        await #expect(processExitsWith: .failure) {
            _ = optim.CosineAnnealingLR(baseLR: 0.1, totalEpochs: 0)
        }
    }

    @Test("WarmupCosineScheduler needs totalSteps > warmupSteps")
    func warmupCosineDegenerate() async {
        await #expect(processExitsWith: .failure) {
            _ = optim.WarmupCosineScheduler(baseLR: 0.1, warmupSteps: 10, totalSteps: 10)
        }
        await #expect(processExitsWith: .failure) {
            _ = optim.WarmupLR(baseLR: 0.1, warmupSteps: -1)
        }
    }

    @Test("Valid schedules produce finite learning rates through the end")
    func validSchedulesStayFinite() {
        var cosine = optim.WarmupCosineScheduler(baseLR: 1, warmupSteps: 2, totalSteps: 3, minLR: 0.1)
        var lrs: [Float] = []
        for _ in 0..<5 {
            lrs.append(cosine.currentLR)
            cosine.step()
        }
        #expect(lrs.allSatisfy { $0.isFinite })
        #expect(abs(lrs[0] - 0.5) < 1e-6)   // warmup 1/2
        #expect(abs(lrs[2] - 1.0) < 1e-6)   // decay starts at baseLR
        #expect(abs(lrs[4] - 0.1) < 1e-6)   // clamped at totalSteps -> minLR

        var step = optim.StepLR(baseLR: 1, stepSize: 2, gamma: 0.5)
        var stepLRs: [Float] = []
        for _ in 0..<5 {
            stepLRs.append(step.currentLR)
            step.step()
        }
        #expect(stepLRs == [1, 1, 0.5, 0.5, 0.25])
    }
}
