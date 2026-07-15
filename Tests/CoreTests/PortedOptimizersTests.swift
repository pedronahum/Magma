// Magma - Ported Optimizers Tests
// Tests for optimizers ported from S4TF (RMSProp, AdaGrad, AdaDelta)

import Testing
@testable import Magma
@testable import LazyTensor

// MARK: - RMSProp Optimizer Tests

@Suite("RMSProp Optimizer Tests")
struct RMSPropOptimizerTests {

    @Test("RMSProp creation")
    func rmspropCreation() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        let optimizer = optim.RMSProp(parameters: [param])

        #expect(optimizer.learningRate == 0.01)  // Default lr
        #expect(optimizer.rho == 0.99)  // Default rho
        #expect(optimizer.momentum == 0.0)
        #expect(!optimizer.centered)
    }

    @Test("RMSProp basic")
    func rmspropBasic() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("RMSProp multiple steps")
    func rmspropMultipleSteps() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        for _ in 0..<5 {
            optimizer.step([grad])
        }

        #expect(param.value.shape == [2, 3])
    }

    @Test("RMSProp with momentum")
    func rmspropWithMomentum() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01, momentum: 0.9)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])
        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("RMSProp centered")
    func rmspropCentered() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01, centered: true)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("RMSProp weight decay")
    func rmspropWeightDecay() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01, weightDecay: 0.01)

        let grad = Tensor<Float>.zeros([2, 3])

        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("RMSProp zero grad")
    func rmspropZeroGrad() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])
        optimizer.resetState()
        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("RMSProp multiple parameters")
    func rmspropMultipleParameters() {
        let param1 = Parameter(Tensor<Float>.ones([2, 3]), name: "w1")
        let param2 = Parameter(Tensor<Float>.ones([3, 4]), name: "w2")
        var optimizer = optim.RMSProp(parameters: [param1, param2], lr: 0.01)

        let grad1 = Tensor<Float>.ones([2, 3])
        let grad2 = Tensor<Float>.ones([3, 4])

        optimizer.step([grad1, grad2])

        #expect(param1.value.shape == [2, 3])
        #expect(param2.value.shape == [3, 4])
    }

    @Test("RMSProp custom rho")
    func rmspropCustomRho() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01, rho: 0.95)

        #expect(optimizer.rho == 0.95)

        let grad = Tensor<Float>.ones([2, 3])
        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }
}

// MARK: - AdaGrad Optimizer Tests

@Suite("AdaGrad Optimizer Tests")
struct AdaGradOptimizerTests {

    @Test("AdaGrad creation")
    func adagradCreation() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        let optimizer = optim.AdaGrad(parameters: [param])

        #expect(optimizer.learningRate == 0.01)  // Default lr
        #expect(optimizer.initialAccumulatorValue == 0.0)
    }

    @Test("AdaGrad basic")
    func adagradBasic() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaGrad(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("AdaGrad multiple steps")
    func adagradMultipleSteps() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaGrad(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        // AdaGrad accumulates squared gradients, so effective learning rate decreases
        for _ in 0..<5 {
            optimizer.step([grad])
        }

        #expect(param.value.shape == [2, 3])
    }

    @Test("AdaGrad weight decay")
    func adagradWeightDecay() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaGrad(parameters: [param], lr: 0.01, weightDecay: 0.01)

        let grad = Tensor<Float>.zeros([2, 3])

        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("AdaGrad initial accumulator")
    func adagradInitialAccumulator() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaGrad(parameters: [param], lr: 0.01, initialAccumulatorValue: 0.1)

        #expect(optimizer.initialAccumulatorValue == 0.1)

        let grad = Tensor<Float>.ones([2, 3])
        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("AdaGrad zero grad")
    func adagradZeroGrad() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaGrad(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])
        optimizer.resetState()  // Reset accumulator
        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("AdaGrad multiple parameters")
    func adagradMultipleParameters() {
        let param1 = Parameter(Tensor<Float>.ones([2, 3]), name: "w1")
        let param2 = Parameter(Tensor<Float>.ones([3, 4]), name: "w2")
        var optimizer = optim.AdaGrad(parameters: [param1, param2], lr: 0.01)

        let grad1 = Tensor<Float>.ones([2, 3])
        let grad2 = Tensor<Float>.ones([3, 4])

        optimizer.step([grad1, grad2])

        #expect(param1.value.shape == [2, 3])
        #expect(param2.value.shape == [3, 4])
    }
}

// MARK: - AdaDelta Optimizer Tests

@Suite("AdaDelta Optimizer Tests")
struct AdaDeltaOptimizerTests {

    @Test("AdaDelta creation")
    func adadeltaCreation() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        let optimizer = optim.AdaDelta(parameters: [param])

        #expect(optimizer.learningRate == 1.0)  // Default lr for AdaDelta
        #expect(optimizer.rho == 0.9)
    }

    @Test("AdaDelta basic")
    func adadeltaBasic() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaDelta(parameters: [param])

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("AdaDelta multiple steps")
    func adadeltaMultipleSteps() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaDelta(parameters: [param])

        let grad = Tensor<Float>.ones([2, 3])

        for _ in 0..<5 {
            optimizer.step([grad])
        }

        #expect(param.value.shape == [2, 3])
    }

    @Test("AdaDelta weight decay")
    func adadeltaWeightDecay() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaDelta(parameters: [param], weightDecay: 0.01)

        let grad = Tensor<Float>.zeros([2, 3])

        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("AdaDelta custom rho")
    func adadeltaCustomRho() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaDelta(parameters: [param], rho: 0.95)

        #expect(optimizer.rho == 0.95)

        let grad = Tensor<Float>.ones([2, 3])
        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("AdaDelta zero grad")
    func adadeltaZeroGrad() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaDelta(parameters: [param])

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])
        optimizer.resetState()
        optimizer.step([grad])

        #expect(param.value.shape == [2, 3])
    }

    @Test("AdaDelta multiple parameters")
    func adadeltaMultipleParameters() {
        let param1 = Parameter(Tensor<Float>.ones([2, 3]), name: "w1")
        let param2 = Parameter(Tensor<Float>.ones([3, 4]), name: "w2")
        var optimizer = optim.AdaDelta(parameters: [param1, param2])

        let grad1 = Tensor<Float>.ones([2, 3])
        let grad2 = Tensor<Float>.ones([3, 4])

        optimizer.step([grad1, grad2])

        #expect(param1.value.shape == [2, 3])
        #expect(param2.value.shape == [3, 4])
    }

    @Test("AdaDelta with model")
    func adadeltaWithModel() {
        // Test with a real model
        let fc = nn.Linear(inputSize: 10, outputSize: 5)
        var optimizer = optim.AdaDelta(parameters: fc.parameters())

        let grads = fc.parameters().map { param in
            Tensor<Float>.ones(param.shape)
        }

        optimizer.step(grads)

        #expect(fc.weight.shape == [5, 10])
        #expect(fc.bias.shape == [5])
    }
}

// MARK: - Optimizer Integration Tests

@Suite("Ported Optimizer Integration Tests")
struct PortedOptimizerIntegrationTests {

    @Test("RMSProp with sequential model")
    func rmspropWithSequentialModel() {
        let model = nn.sequential {
            nn.Linear(inputSize: 784, outputSize: 256)
            nn.ReLU()
            nn.Linear(inputSize: 256, outputSize: 10)
        }

        var optimizer = optim.RMSProp(parameters: model.parameters(), lr: 0.001)

        let input = Tensor<Float>.ones([32, 784])
        let output = model(input)
        #expect(output.shape == [32, 10])

        // Create gradients and step
        let grads = model.parameters().map { Tensor<Float>.ones($0.shape) }
        optimizer.step(grads)

        // Model should still work after optimization step
        let output2 = model(input)
        #expect(output2.shape == [32, 10])
    }

    @Test("AdaGrad with sequential model")
    func adagradWithSequentialModel() {
        let model = nn.sequential {
            nn.Linear(inputSize: 100, outputSize: 50)
            nn.ReLU()
            nn.Linear(inputSize: 50, outputSize: 10)
        }

        var optimizer = optim.AdaGrad(parameters: model.parameters(), lr: 0.01)

        let input = Tensor<Float>.ones([16, 100])
        let output = model(input)
        #expect(output.shape == [16, 10])

        let grads = model.parameters().map { Tensor<Float>.ones($0.shape) }
        optimizer.step(grads)

        let output2 = model(input)
        #expect(output2.shape == [16, 10])
    }

    @Test("Optimizer skips non-grad params")
    func optimizerSkipsNonGradParams() {
        let trainable = Parameter(Tensor<Float>.ones([2, 3]), requiresGrad: true, name: "trainable")
        let frozen = Parameter(Tensor<Float>.ones([3, 4]), requiresGrad: false, name: "frozen")

        var rmsprop = optim.RMSProp(parameters: [trainable, frozen], lr: 0.01)
        var adagrad = optim.AdaGrad(parameters: [trainable, frozen], lr: 0.01)
        var adadelta = optim.AdaDelta(parameters: [trainable, frozen])

        let grad1 = Tensor<Float>.ones([2, 3])
        let grad2 = Tensor<Float>.ones([3, 4])

        rmsprop.step([grad1, grad2])
        adagrad.step([grad1, grad2])
        adadelta.step([grad1, grad2])

        // All optimizers should work without error
        #expect(trainable.value.shape == [2, 3])
        #expect(frozen.value.shape == [3, 4])
    }
}
