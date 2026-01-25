// Magma - Advanced Activation Tests
// Tests for Phase 4 activation functions

import XCTest
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - Leaky ReLU Tests

final class LeakyReLUTests: XCTestCase {

    func testLeakyReluTensor() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.leakyRelu()
        XCTAssertEqual(result.shape, [2, 3])
    }

    func testLeakyReluLayer() {
        let layer = nn.LeakyReLU()
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        XCTAssertEqual(result.shape, [32, 128])
    }

    func testLeakyReluCustomSlope() {
        let layer = nn.LeakyReLU(negativeSlope: 0.2)
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        XCTAssertEqual(result.shape, [32, 128])
    }
}

// MARK: - SiLU Tests

final class SiLUTests: XCTestCase {

    func testSiluTensor() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.silu()
        XCTAssertEqual(result.shape, [2, 3])
    }

    func testSiluLayer() {
        let layer = nn.SiLU()
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        XCTAssertEqual(result.shape, [32, 128])
    }
}

// MARK: - ELU Tests

final class ELUTests: XCTestCase {

    func testEluTensor() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.elu()
        XCTAssertEqual(result.shape, [2, 3])
    }

    func testEluLayer() {
        let layer = nn.ELU()
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        XCTAssertEqual(result.shape, [32, 128])
    }

    func testEluCustomAlpha() {
        let layer = nn.ELU(alpha: 0.5)
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        XCTAssertEqual(result.shape, [32, 128])
    }
}

// MARK: - Hardtanh Tests

final class HardtanhTests: XCTestCase {

    func testHardtanhTensor() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.hardtanh()
        XCTAssertEqual(result.shape, [2, 3])
    }

    func testHardtanhLayer() {
        let layer = nn.Hardtanh()
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        XCTAssertEqual(result.shape, [32, 128])
    }

    func testHardtanhCustomRange() {
        let layer = nn.Hardtanh(minVal: -2.0, maxVal: 2.0)
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        XCTAssertEqual(result.shape, [32, 128])
    }

    func testClampFunction() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.clamp(min: 0.0, max: 1.0)
        XCTAssertEqual(result.shape, [2, 3])
    }
}

// MARK: - Abs and Pow Tests

final class MathOpsTests: XCTestCase {

    func testAbs() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.abs()
        XCTAssertEqual(result.shape, [2, 3])
    }

    func testPow() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.pow(2.0)
        XCTAssertEqual(result.shape, [2, 3])
    }

    func testPowFractional() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.pow(0.5)
        XCTAssertEqual(result.shape, [2, 3])
    }
}

// MARK: - Learning Rate Scheduler Tests

final class LRSchedulerTests: XCTestCase {

    func testStepLR() {
        var scheduler = optim.StepLR(baseLR: 0.1, stepSize: 3, gamma: 0.1)

        XCTAssertEqual(scheduler.currentLR, 0.1, accuracy: 1e-6)

        scheduler.step()
        scheduler.step()
        XCTAssertEqual(scheduler.currentLR, 0.1, accuracy: 1e-6)  // Before step size

        scheduler.step()
        XCTAssertEqual(scheduler.currentLR, 0.01, accuracy: 1e-6)  // After step size

        scheduler.reset()
        XCTAssertEqual(scheduler.currentLR, 0.1, accuracy: 1e-6)
    }

    func testExponentialLR() {
        var scheduler = optim.ExponentialLR(baseLR: 0.1, gamma: 0.9)

        XCTAssertEqual(scheduler.currentLR, 0.1, accuracy: 1e-6)

        scheduler.step()
        XCTAssertEqual(scheduler.currentLR, 0.09, accuracy: 1e-6)

        scheduler.step()
        XCTAssertEqual(scheduler.currentLR, 0.081, accuracy: 1e-6)
    }

    func testCosineAnnealingLR() {
        var scheduler = optim.CosineAnnealingLR(baseLR: 0.1, totalEpochs: 10, minLR: 0.0)

        XCTAssertEqual(scheduler.currentLR, 0.1, accuracy: 1e-6)  // Start

        for _ in 0..<5 {
            scheduler.step()
        }
        XCTAssertEqual(scheduler.currentLR, 0.05, accuracy: 1e-6)  // Midpoint

        for _ in 0..<5 {
            scheduler.step()
        }
        XCTAssertEqual(scheduler.currentLR, 0.0, accuracy: 1e-6)  // End
    }

    func testWarmupLR() {
        var scheduler = optim.WarmupLR(baseLR: 0.1, warmupSteps: 5)

        XCTAssertEqual(scheduler.currentLR, 0.02, accuracy: 1e-6)  // Step 0

        scheduler.step()
        XCTAssertEqual(scheduler.currentLR, 0.04, accuracy: 1e-6)  // Step 1

        for _ in 0..<10 {
            scheduler.step()
        }
        XCTAssertEqual(scheduler.currentLR, 0.1, accuracy: 1e-6)  // After warmup
    }

    func testWarmupCosineScheduler() {
        var scheduler = optim.WarmupCosineScheduler(
            baseLR: 0.1,
            warmupSteps: 10,
            totalSteps: 100,
            minLR: 0.0
        )

        // Warmup phase
        XCTAssertEqual(scheduler.currentLR, 0.01, accuracy: 1e-6)  // Step 0

        for _ in 0..<10 {
            scheduler.step()
        }
        XCTAssertEqual(scheduler.currentLR, 0.1, accuracy: 1e-6)  // After warmup

        // Decay phase
        for _ in 0..<45 {
            scheduler.step()
        }
        XCTAssertEqual(scheduler.currentLR, 0.05, accuracy: 0.01)  // ~Midpoint
    }
}

// MARK: - Integration Tests

final class AdvancedActivationIntegrationTests: XCTestCase {

    func testModelWithAdvancedActivations() {
        let model = nn.sequential {
            nn.Linear(inputSize: 784, outputSize: 256)
            nn.LeakyReLU(negativeSlope: 0.1)
            nn.Linear(inputSize: 256, outputSize: 128)
            nn.SiLU()
            nn.Linear(inputSize: 128, outputSize: 10)
        }

        let x = Tensor<Float>.randn([32, 784])
        let output = model(x)
        XCTAssertEqual(output.shape, [32, 10])
    }

    func testTrainingWithScheduler() {
        let model = nn.sequential {
            nn.Linear(inputSize: 10, outputSize: 5)
            nn.LeakyReLU()
        }

        let params = model.parameters()
        var optimizer = optim.Adam(parameters: params, lr: 0.01)
        var scheduler = optim.StepLR(baseLR: 0.01, stepSize: 2, gamma: 0.5)

        let x = Tensor<Float>.ones([8, 10])

        // Simulate 5 epochs
        for epoch in 0..<5 {
            let _ = model(x)

            // Update optimizer LR from scheduler
            optimizer.learningRate = scheduler.currentLR

            // Optimizer step
            let grads = params.map { Tensor<Float>.ones($0.shape) }
            optimizer.step(grads)

            scheduler.step()
        }

        // After 4 steps, LR should be 0.01 * 0.5^2 = 0.0025
        XCTAssertEqual(scheduler.currentLR, 0.0025, accuracy: 1e-6)
    }
}
