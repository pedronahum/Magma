// Magma - Advanced Activation Tests
// Tests for Phase 4 activation functions

import Testing
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - Leaky ReLU Tests

@Suite("Leaky ReLU Tests")
struct LeakyReLUTests {

    @Test("Leaky ReLU tensor operation")
    func leakyReluTensor() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.leakyRelu()
        #expect(result.shape == [2, 3])
    }

    @Test("Leaky ReLU layer")
    func leakyReluLayer() {
        let layer = nn.LeakyReLU()
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        #expect(result.shape == [32, 128])
    }

    @Test("Leaky ReLU custom slope")
    func leakyReluCustomSlope() {
        let layer = nn.LeakyReLU(negativeSlope: 0.2)
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        #expect(result.shape == [32, 128])
    }
}

// MARK: - SiLU Tests

@Suite("SiLU Tests")
struct SiLUTests {

    @Test("SiLU tensor operation")
    func siluTensor() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.silu()
        #expect(result.shape == [2, 3])
    }

    @Test("SiLU layer")
    func siluLayer() {
        let layer = nn.SiLU()
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        #expect(result.shape == [32, 128])
    }
}

// MARK: - ELU Tests

@Suite("ELU Tests")
struct ELUTests {

    @Test("ELU tensor operation")
    func eluTensor() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.elu()
        #expect(result.shape == [2, 3])
    }

    @Test("ELU layer")
    func eluLayer() {
        let layer = nn.ELU()
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        #expect(result.shape == [32, 128])
    }

    @Test("ELU custom alpha")
    func eluCustomAlpha() {
        let layer = nn.ELU(alpha: 0.5)
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        #expect(result.shape == [32, 128])
    }
}

// MARK: - Hardtanh Tests

@Suite("Hardtanh Tests")
struct HardtanhTests {

    @Test("Hardtanh tensor operation")
    func hardtanhTensor() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.hardtanh()
        #expect(result.shape == [2, 3])
    }

    @Test("Hardtanh layer")
    func hardtanhLayer() {
        let layer = nn.Hardtanh()
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        #expect(result.shape == [32, 128])
    }

    @Test("Hardtanh custom range")
    func hardtanhCustomRange() {
        let layer = nn.Hardtanh(minVal: -2.0, maxVal: 2.0)
        let x = Tensor<Float>.ones([32, 128])
        let result = layer(x)
        #expect(result.shape == [32, 128])
    }

    @Test("Clamp function")
    func clampFunction() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.clamp(min: 0.0, max: 1.0)
        #expect(result.shape == [2, 3])
    }
}

// MARK: - Abs and Pow Tests

@Suite("Math Ops Tests")
struct MathOpsTests {

    @Test("Abs operation")
    func abs() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.abs()
        #expect(result.shape == [2, 3])
    }

    @Test("Pow operation")
    func pow() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.pow(2.0)
        #expect(result.shape == [2, 3])
    }

    @Test("Pow fractional exponent")
    func powFractional() {
        let x = Tensor<Float>.ones([2, 3])
        let result = x.pow(0.5)
        #expect(result.shape == [2, 3])
    }
}

// MARK: - Learning Rate Scheduler Tests

@Suite("LR Scheduler Tests")
struct LRSchedulerTests {

    @Test("Step LR scheduler")
    func stepLR() {
        var scheduler = optim.StepLR(baseLR: 0.1, stepSize: 3, gamma: 0.1)

        #expect(abs(scheduler.currentLR - 0.1) < 1e-6)

        scheduler.step()
        scheduler.step()
        #expect(abs(scheduler.currentLR - 0.1) < 1e-6)  // Before step size

        scheduler.step()
        #expect(abs(scheduler.currentLR - 0.01) < 1e-6)  // After step size

        scheduler.reset()
        #expect(abs(scheduler.currentLR - 0.1) < 1e-6)
    }

    @Test("Exponential LR scheduler")
    func exponentialLR() {
        var scheduler = optim.ExponentialLR(baseLR: 0.1, gamma: 0.9)

        #expect(abs(scheduler.currentLR - 0.1) < 1e-6)

        scheduler.step()
        #expect(abs(scheduler.currentLR - 0.09) < 1e-6)

        scheduler.step()
        #expect(abs(scheduler.currentLR - 0.081) < 1e-6)
    }

    @Test("Cosine annealing LR scheduler")
    func cosineAnnealingLR() {
        var scheduler = optim.CosineAnnealingLR(baseLR: 0.1, totalEpochs: 10, minLR: 0.0)

        #expect(abs(scheduler.currentLR - 0.1) < 1e-6)  // Start

        for _ in 0..<5 {
            scheduler.step()
        }
        #expect(abs(scheduler.currentLR - 0.05) < 1e-6)  // Midpoint

        for _ in 0..<5 {
            scheduler.step()
        }
        #expect(abs(scheduler.currentLR - 0.0) < 1e-6)  // End
    }

    @Test("Warmup LR scheduler")
    func warmupLR() {
        var scheduler = optim.WarmupLR(baseLR: 0.1, warmupSteps: 5)

        #expect(abs(scheduler.currentLR - 0.02) < 1e-6)  // Step 0

        scheduler.step()
        #expect(abs(scheduler.currentLR - 0.04) < 1e-6)  // Step 1

        for _ in 0..<10 {
            scheduler.step()
        }
        #expect(abs(scheduler.currentLR - 0.1) < 1e-6)  // After warmup
    }

    @Test("Warmup cosine scheduler")
    func warmupCosineScheduler() {
        var scheduler = optim.WarmupCosineScheduler(
            baseLR: 0.1,
            warmupSteps: 10,
            totalSteps: 100,
            minLR: 0.0
        )

        // Warmup phase
        #expect(abs(scheduler.currentLR - 0.01) < 1e-6)  // Step 0

        for _ in 0..<10 {
            scheduler.step()
        }
        #expect(abs(scheduler.currentLR - 0.1) < 1e-6)  // After warmup

        // Decay phase
        for _ in 0..<45 {
            scheduler.step()
        }
        #expect(abs(scheduler.currentLR - 0.05) < 0.01)  // ~Midpoint
    }
}

// MARK: - Integration Tests

@Suite("Advanced Activation Integration Tests")
struct AdvancedActivationIntegrationTests {

    @Test("Model with advanced activations")
    func modelWithAdvancedActivations() {
        let model = nn.sequential {
            nn.Linear(inputSize: 784, outputSize: 256)
            nn.LeakyReLU(negativeSlope: 0.1)
            nn.Linear(inputSize: 256, outputSize: 128)
            nn.SiLU()
            nn.Linear(inputSize: 128, outputSize: 10)
        }

        let x = Tensor<Float>.randn([32, 784])
        let output = model(x)
        #expect(output.shape == [32, 10])
    }

    @Test("Training with scheduler")
    func trainingWithScheduler() {
        let model = nn.sequential {
            nn.Linear(inputSize: 10, outputSize: 5)
            nn.LeakyReLU()
        }

        let params = model.parameters()
        var optimizer = optim.Adam(parameters: params, lr: 0.01)
        var scheduler = optim.StepLR(baseLR: 0.01, stepSize: 2, gamma: 0.5)

        let x = Tensor<Float>.ones([8, 10])

        // Simulate 5 epochs
        for _ in 0..<5 {
            let _ = model(x)

            // Update optimizer LR from scheduler
            optimizer.learningRate = scheduler.currentLR

            // Optimizer step
            let grads = params.map { Tensor<Float>.ones($0.shape) }
            optimizer.step(grads)

            scheduler.step()
        }

        // After 4 steps, LR should be 0.01 * 0.5^2 = 0.0025
        #expect(abs(scheduler.currentLR - 0.0025) < 1e-6)
    }
}
