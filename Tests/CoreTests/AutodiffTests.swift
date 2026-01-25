// Magma - Autodiff Tests
// Tests for automatic differentiation functionality

import XCTest
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - Basic Differentiable Conformance Tests

final class DifferentiableConformanceTests: XCTestCase {

    func testTensorIsDifferentiable() {
        let x = Tensor<Float>.ones([2, 3])

        // Tensor should conform to Differentiable
        // If it compiles, conformance is verified
        XCTAssertEqual(x.shape, [2, 3])
    }

    func testTensorIsAdditiveArithmetic() {
        let x = Tensor<Float>.ones([2, 3])
        let y = Tensor<Float>.ones([2, 3])

        // AdditiveArithmetic operations
        let sum = x + y
        XCTAssertEqual(sum.shape, [2, 3])

        let zero = Tensor<Float>.zero
        XCTAssertEqual(zero.shape, [])
    }

    func testTensorEquatable() {
        let x = Tensor<Float>.ones([2, 3])
        let y = x  // Same handle

        // Identity equality
        XCTAssertEqual(x, y)
    }
}

// MARK: - Gradient Computation Tests

final class GradientComputationTests: XCTestCase {

    func testSumGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.sum()
        }

        // Gradient of sum is all ones (broadcast to original shape)
        XCTAssertEqual(grad.shape, [2, 3])
    }

    func testMeanGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.mean()
        }

        // Gradient of mean is 1/n broadcast to original shape
        XCTAssertEqual(grad.shape, [2, 3])
    }

    func testAdditionGradient() {
        let x = Tensor<Float>.ones([2, 3])
        let y = Tensor<Float>.ones([2, 3])

        let (gradX, gradY) = gradient(at: x, y) { a, b in
            (a + b).sum()
        }

        // Both gradients should have same shape
        XCTAssertEqual(gradX.shape, [2, 3])
        XCTAssertEqual(gradY.shape, [2, 3])
    }

    func testMultiplicationGradient() {
        let x = Tensor<Float>.ones([2, 3])
        let y = Tensor<Float>.ones([2, 3])

        let (gradX, gradY) = gradient(at: x, y) { a, b in
            (a * b).sum()
        }

        // d(sum(a*b))/da = b, d(sum(a*b))/db = a
        XCTAssertEqual(gradX.shape, [2, 3])
        XCTAssertEqual(gradY.shape, [2, 3])
    }

    func testValueWithGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let (value, grad) = valueWithGradient(at: x) { t in
            t.sum()
        }

        XCTAssertEqual(value.shape, [])  // Scalar output
        XCTAssertEqual(grad.shape, [2, 3])
    }
}

// MARK: - Activation Gradient Tests

final class ActivationGradientTests: XCTestCase {

    func testReluGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.relu().sum()
        }

        XCTAssertEqual(grad.shape, [2, 3])
    }

    func testSigmoidGradient() {
        let x = Tensor<Float>.zeros([2, 3])

        let grad = gradient(at: x) { t in
            t.sigmoid().sum()
        }

        // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        XCTAssertEqual(grad.shape, [2, 3])
    }

    func testTanhGradient() {
        let x = Tensor<Float>.zeros([2, 3])

        let grad = gradient(at: x) { t in
            t.tanh().sum()
        }

        // tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1
        XCTAssertEqual(grad.shape, [2, 3])
    }

    func testExpGradient() {
        let x = Tensor<Float>.zeros([2, 3])

        let grad = gradient(at: x) { t in
            t.exp().sum()
        }

        // exp'(x) = exp(x)
        // exp'(0) = 1
        XCTAssertEqual(grad.shape, [2, 3])
    }

    func testLogGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.log().sum()
        }

        // log'(x) = 1/x
        // log'(1) = 1
        XCTAssertEqual(grad.shape, [2, 3])
    }
}

// MARK: - Matrix Operation Gradient Tests

final class MatrixGradientTests: XCTestCase {

    func testMatmulGradient() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.ones([3, 4])

        let (gradA, gradB) = gradient(at: a, b) { x, y in
            x.matmul(y).sum()
        }

        // d(sum(A@B))/dA has shape [2, 3]
        // d(sum(A@B))/dB has shape [3, 4]
        XCTAssertEqual(gradA.shape, [2, 3])
        XCTAssertEqual(gradB.shape, [3, 4])
    }

    func testTransposeGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.transpose().sum()
        }

        XCTAssertEqual(grad.shape, [2, 3])
    }

    func testReshapeGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.reshape([6]).sum()
        }

        XCTAssertEqual(grad.shape, [2, 3])
    }
}

// MARK: - Chained Operations Gradient Tests

final class ChainedGradientTests: XCTestCase {

    func testLinearLayerGradient() {
        // y = relu(x @ w + b)
        let x = Tensor<Float>.ones([32, 784])
        let w = Tensor<Float>.ones([784, 256])

        let grad = gradient(at: w) { weight in
            let output = x.matmul(weight)
            return output.relu().sum()
        }

        XCTAssertEqual(grad.shape, [784, 256])
    }

    func testMultiLayerGradient() {
        // y = relu(relu(x @ w1) @ w2)
        let x = Tensor<Float>.ones([32, 10])
        let w1 = Tensor<Float>.ones([10, 20])
        let w2 = Tensor<Float>.ones([20, 5])

        let (gradW1, gradW2) = gradient(at: w1, w2) { weight1, weight2 in
            let h = x.matmul(weight1).relu()
            let out = h.matmul(weight2).relu()
            return out.sum()
        }

        XCTAssertEqual(gradW1.shape, [10, 20])
        XCTAssertEqual(gradW2.shape, [20, 5])
    }

    func testMSELossGradient() {
        let prediction = Tensor<Float>.ones([32, 10])
        let target = Tensor<Float>.zeros([32, 10])

        let grad = gradient(at: prediction) { pred in
            let diff = pred - target
            return (diff * diff).mean()
        }

        XCTAssertEqual(grad.shape, [32, 10])
    }
}

// MARK: - Differentiable Function Tests

final class DifferentiableFunctionTests: XCTestCase {

    @differentiable(reverse)
    func simpleForward(_ x: Tensor<Float>) -> Tensor<Float> {
        x.relu().sum()
    }

    func testDifferentiableFunction() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x, of: simpleForward)

        XCTAssertEqual(grad.shape, [2, 3])
    }

    @differentiable(reverse)
    func mlpForward(
        _ input: Tensor<Float>,
        _ w1: Tensor<Float>,
        _ w2: Tensor<Float>
    ) -> Tensor<Float> {
        let h = input.matmul(w1).relu()
        return h.matmul(w2).sum()
    }

    func testDifferentiableMLP() {
        let x = Tensor<Float>.ones([32, 10])
        let w1 = Tensor<Float>.ones([10, 20])
        let w2 = Tensor<Float>.ones([20, 5])

        // Get gradients through differentiable function
        let (_, pb) = valueWithPullback(at: x, w1, w2, of: mlpForward)
        let (gradX, gradW1, gradW2) = pb(Tensor<Float>.ones([], on: .default))

        XCTAssertEqual(gradX.shape, [32, 10])
        XCTAssertEqual(gradW1.shape, [10, 20])
        XCTAssertEqual(gradW2.shape, [20, 5])
    }
}
