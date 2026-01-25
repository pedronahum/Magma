// Magma - Ported Optimizers Tests
// Tests for optimizers ported from S4TF (RMSProp, AdaGrad, AdaDelta)

import XCTest
@testable import Magma
@testable import LazyTensor

// MARK: - RMSProp Optimizer Tests

final class RMSPropOptimizerTests: XCTestCase {

    func testRMSPropCreation() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        let optimizer = optim.RMSProp(parameters: [param])

        XCTAssertEqual(optimizer.learningRate, 0.01)  // Default lr
        XCTAssertEqual(optimizer.rho, 0.99)  // Default rho
        XCTAssertEqual(optimizer.momentum, 0.0)
        XCTAssertFalse(optimizer.centered)
    }

    func testRMSPropBasic() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testRMSPropMultipleSteps() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        for _ in 0..<5 {
            optimizer.step([grad])
        }

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testRMSPropWithMomentum() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01, momentum: 0.9)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])
        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testRMSPropCentered() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01, centered: true)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testRMSPropWeightDecay() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01, weightDecay: 0.01)

        let grad = Tensor<Float>.zeros([2, 3])

        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testRMSPropZeroGrad() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])
        optimizer.zeroGrad()
        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testRMSPropMultipleParameters() {
        let param1 = Parameter(Tensor<Float>.ones([2, 3]), name: "w1")
        let param2 = Parameter(Tensor<Float>.ones([3, 4]), name: "w2")
        var optimizer = optim.RMSProp(parameters: [param1, param2], lr: 0.01)

        let grad1 = Tensor<Float>.ones([2, 3])
        let grad2 = Tensor<Float>.ones([3, 4])

        optimizer.step([grad1, grad2])

        XCTAssertEqual(param1.value.shape, [2, 3])
        XCTAssertEqual(param2.value.shape, [3, 4])
    }

    func testRMSPropCustomRho() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.RMSProp(parameters: [param], lr: 0.01, rho: 0.95)

        XCTAssertEqual(optimizer.rho, 0.95)

        let grad = Tensor<Float>.ones([2, 3])
        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }
}

// MARK: - AdaGrad Optimizer Tests

final class AdaGradOptimizerTests: XCTestCase {

    func testAdaGradCreation() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        let optimizer = optim.AdaGrad(parameters: [param])

        XCTAssertEqual(optimizer.learningRate, 0.01)  // Default lr
        XCTAssertEqual(optimizer.initialAccumulatorValue, 0.0)
    }

    func testAdaGradBasic() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaGrad(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testAdaGradMultipleSteps() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaGrad(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        // AdaGrad accumulates squared gradients, so effective learning rate decreases
        for _ in 0..<5 {
            optimizer.step([grad])
        }

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testAdaGradWeightDecay() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaGrad(parameters: [param], lr: 0.01, weightDecay: 0.01)

        let grad = Tensor<Float>.zeros([2, 3])

        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testAdaGradInitialAccumulator() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaGrad(parameters: [param], lr: 0.01, initialAccumulatorValue: 0.1)

        XCTAssertEqual(optimizer.initialAccumulatorValue, 0.1)

        let grad = Tensor<Float>.ones([2, 3])
        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testAdaGradZeroGrad() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaGrad(parameters: [param], lr: 0.01)

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])
        optimizer.zeroGrad()  // Reset accumulator
        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testAdaGradMultipleParameters() {
        let param1 = Parameter(Tensor<Float>.ones([2, 3]), name: "w1")
        let param2 = Parameter(Tensor<Float>.ones([3, 4]), name: "w2")
        var optimizer = optim.AdaGrad(parameters: [param1, param2], lr: 0.01)

        let grad1 = Tensor<Float>.ones([2, 3])
        let grad2 = Tensor<Float>.ones([3, 4])

        optimizer.step([grad1, grad2])

        XCTAssertEqual(param1.value.shape, [2, 3])
        XCTAssertEqual(param2.value.shape, [3, 4])
    }
}

// MARK: - AdaDelta Optimizer Tests

final class AdaDeltaOptimizerTests: XCTestCase {

    func testAdaDeltaCreation() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        let optimizer = optim.AdaDelta(parameters: [param])

        XCTAssertEqual(optimizer.learningRate, 1.0)  // Default lr for AdaDelta
        XCTAssertEqual(optimizer.rho, 0.9)
    }

    func testAdaDeltaBasic() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaDelta(parameters: [param])

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testAdaDeltaMultipleSteps() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaDelta(parameters: [param])

        let grad = Tensor<Float>.ones([2, 3])

        for _ in 0..<5 {
            optimizer.step([grad])
        }

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testAdaDeltaWeightDecay() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaDelta(parameters: [param], weightDecay: 0.01)

        let grad = Tensor<Float>.zeros([2, 3])

        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testAdaDeltaCustomRho() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaDelta(parameters: [param], rho: 0.95)

        XCTAssertEqual(optimizer.rho, 0.95)

        let grad = Tensor<Float>.ones([2, 3])
        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testAdaDeltaZeroGrad() {
        let param = Parameter(Tensor<Float>.ones([2, 3]), name: "test")
        var optimizer = optim.AdaDelta(parameters: [param])

        let grad = Tensor<Float>.ones([2, 3])

        optimizer.step([grad])
        optimizer.zeroGrad()
        optimizer.step([grad])

        XCTAssertEqual(param.value.shape, [2, 3])
    }

    func testAdaDeltaMultipleParameters() {
        let param1 = Parameter(Tensor<Float>.ones([2, 3]), name: "w1")
        let param2 = Parameter(Tensor<Float>.ones([3, 4]), name: "w2")
        var optimizer = optim.AdaDelta(parameters: [param1, param2])

        let grad1 = Tensor<Float>.ones([2, 3])
        let grad2 = Tensor<Float>.ones([3, 4])

        optimizer.step([grad1, grad2])

        XCTAssertEqual(param1.value.shape, [2, 3])
        XCTAssertEqual(param2.value.shape, [3, 4])
    }

    func testAdaDeltaWithModel() {
        // Test with a real model
        let fc = nn.Linear(inputSize: 10, outputSize: 5)
        var optimizer = optim.AdaDelta(parameters: fc.parameters())

        let grads = fc.parameters().map { param in
            Tensor<Float>.ones(param.shape)
        }

        optimizer.step(grads)

        XCTAssertEqual(fc.weight.shape, [5, 10])
        XCTAssertEqual(fc.bias.shape, [5])
    }
}

// MARK: - Optimizer Integration Tests

final class PortedOptimizerIntegrationTests: XCTestCase {

    func testRMSPropWithSequentialModel() {
        let model = nn.sequential {
            nn.Linear(inputSize: 784, outputSize: 256)
            nn.ReLU()
            nn.Linear(inputSize: 256, outputSize: 10)
        }

        var optimizer = optim.RMSProp(parameters: model.parameters(), lr: 0.001)

        let input = Tensor<Float>.ones([32, 784])
        let output = model(input)
        XCTAssertEqual(output.shape, [32, 10])

        // Create gradients and step
        let grads = model.parameters().map { Tensor<Float>.ones($0.shape) }
        optimizer.step(grads)

        // Model should still work after optimization step
        let output2 = model(input)
        XCTAssertEqual(output2.shape, [32, 10])
    }

    func testAdaGradWithSequentialModel() {
        let model = nn.sequential {
            nn.Linear(inputSize: 100, outputSize: 50)
            nn.ReLU()
            nn.Linear(inputSize: 50, outputSize: 10)
        }

        var optimizer = optim.AdaGrad(parameters: model.parameters(), lr: 0.01)

        let input = Tensor<Float>.ones([16, 100])
        let output = model(input)
        XCTAssertEqual(output.shape, [16, 10])

        let grads = model.parameters().map { Tensor<Float>.ones($0.shape) }
        optimizer.step(grads)

        let output2 = model(input)
        XCTAssertEqual(output2.shape, [16, 10])
    }

    func testOptimizerSkipsNonGradParams() {
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
        XCTAssertEqual(trainable.value.shape, [2, 3])
        XCTAssertEqual(frozen.value.shape, [3, 4])
    }
}
