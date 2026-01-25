// Magma - Autodiff Tests
// Tests for automatic differentiation functionality

import Testing
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - Basic Differentiable Conformance Tests

@Suite("Differentiable Conformance Tests")
struct DifferentiableConformanceTests {

    @Test("Tensor is Differentiable")
    func tensorIsDifferentiable() {
        let x = Tensor<Float>.ones([2, 3])

        // Tensor should conform to Differentiable
        // If it compiles, conformance is verified
        #expect(x.shape == [2, 3])
    }

    @Test("Tensor is AdditiveArithmetic")
    func tensorIsAdditiveArithmetic() {
        let x = Tensor<Float>.ones([2, 3])
        let y = Tensor<Float>.ones([2, 3])

        // AdditiveArithmetic operations
        let sum = x + y
        #expect(sum.shape == [2, 3])

        let zero = Tensor<Float>.zero
        #expect(zero.shape == [])
    }

    @Test("Tensor Equatable")
    func tensorEquatable() {
        let x = Tensor<Float>.ones([2, 3])
        let y = x  // Same handle

        // Identity equality
        #expect(x == y)
    }
}

// MARK: - Gradient Computation Tests

@Suite("Gradient Computation Tests")
struct GradientComputationTests {

    @Test("Sum gradient")
    func sumGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.sum()
        }

        // Gradient of sum is all ones (broadcast to original shape)
        #expect(grad.shape == [2, 3])
    }

    @Test("Mean gradient")
    func meanGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.mean()
        }

        // Gradient of mean is 1/n broadcast to original shape
        #expect(grad.shape == [2, 3])
    }

    @Test("Addition gradient")
    func additionGradient() {
        let x = Tensor<Float>.ones([2, 3])
        let y = Tensor<Float>.ones([2, 3])

        let (gradX, gradY) = gradient(at: x, y) { a, b in
            (a + b).sum()
        }

        // Both gradients should have same shape
        #expect(gradX.shape == [2, 3])
        #expect(gradY.shape == [2, 3])
    }

    @Test("Multiplication gradient")
    func multiplicationGradient() {
        let x = Tensor<Float>.ones([2, 3])
        let y = Tensor<Float>.ones([2, 3])

        let (gradX, gradY) = gradient(at: x, y) { a, b in
            (a * b).sum()
        }

        // d(sum(a*b))/da = b, d(sum(a*b))/db = a
        #expect(gradX.shape == [2, 3])
        #expect(gradY.shape == [2, 3])
    }

    @Test("Value with gradient")
    func valueWithGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let (value, grad) = Magma.valueWithGradient(at: x) { t in
            t.sum()
        }

        #expect(value.shape == [])  // Scalar output
        #expect(grad.shape == [2, 3])
    }
}

// MARK: - Activation Gradient Tests

@Suite("Activation Gradient Tests")
struct ActivationGradientTests {

    @Test("ReLU gradient")
    func reluGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.relu().sum()
        }

        #expect(grad.shape == [2, 3])
    }

    @Test("Sigmoid gradient")
    func sigmoidGradient() {
        let x = Tensor<Float>.zeros([2, 3])

        let grad = gradient(at: x) { t in
            t.sigmoid().sum()
        }

        // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        #expect(grad.shape == [2, 3])
    }

    @Test("Tanh gradient")
    func tanhGradient() {
        let x = Tensor<Float>.zeros([2, 3])

        let grad = gradient(at: x) { t in
            t.tanh().sum()
        }

        // tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1
        #expect(grad.shape == [2, 3])
    }

    @Test("Exp gradient")
    func expGradient() {
        let x = Tensor<Float>.zeros([2, 3])

        let grad = gradient(at: x) { t in
            t.exp().sum()
        }

        // exp'(x) = exp(x)
        // exp'(0) = 1
        #expect(grad.shape == [2, 3])
    }

    @Test("Log gradient")
    func logGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.log().sum()
        }

        // log'(x) = 1/x
        // log'(1) = 1
        #expect(grad.shape == [2, 3])
    }
}

// MARK: - Matrix Operation Gradient Tests

@Suite("Matrix Gradient Tests")
struct MatrixGradientTests {

    @Test("Matmul gradient")
    func matmulGradient() {
        let a = Tensor<Float>.ones([2, 3])
        let b = Tensor<Float>.ones([3, 4])

        let (gradA, gradB) = gradient(at: a, b) { x, y in
            x.matmul(y).sum()
        }

        // d(sum(A@B))/dA has shape [2, 3]
        // d(sum(A@B))/dB has shape [3, 4]
        #expect(gradA.shape == [2, 3])
        #expect(gradB.shape == [3, 4])
    }

    @Test("Transpose gradient")
    func transposeGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.transpose().sum()
        }

        #expect(grad.shape == [2, 3])
    }

    @Test("Reshape gradient")
    func reshapeGradient() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x) { t in
            t.reshape([6]).sum()
        }

        #expect(grad.shape == [2, 3])
    }
}

// MARK: - Chained Operations Gradient Tests

@Suite("Chained Gradient Tests")
struct ChainedGradientTests {

    @Test("Linear layer gradient")
    func linearLayerGradient() {
        // y = relu(x @ w + b)
        let x = Tensor<Float>.ones([32, 784])
        let w = Tensor<Float>.ones([784, 256])

        let grad = gradient(at: w) { weight in
            let output = x.matmul(weight)
            return output.relu().sum()
        }

        #expect(grad.shape == [784, 256])
    }

    @Test("Multi-layer gradient")
    func multiLayerGradient() {
        // y = relu(relu(x @ w1) @ w2)
        let x = Tensor<Float>.ones([32, 10])
        let w1 = Tensor<Float>.ones([10, 20])
        let w2 = Tensor<Float>.ones([20, 5])

        let (gradW1, gradW2) = gradient(at: w1, w2) { weight1, weight2 in
            let h = x.matmul(weight1).relu()
            let out = h.matmul(weight2).relu()
            return out.sum()
        }

        #expect(gradW1.shape == [10, 20])
        #expect(gradW2.shape == [20, 5])
    }

    @Test("MSE loss gradient")
    func mseLossGradient() {
        let prediction = Tensor<Float>.ones([32, 10])
        let target = Tensor<Float>.zeros([32, 10])

        let grad = gradient(at: prediction) { pred in
            let diff = pred - target
            return (diff * diff).mean()
        }

        #expect(grad.shape == [32, 10])
    }
}

// MARK: - Differentiable Function Tests

@differentiable(reverse)
func simpleForward(_ x: Tensor<Float>) -> Tensor<Float> {
    x.relu().sum()
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

@Suite("Differentiable Function Tests")
struct DifferentiableFunctionTests {

    @Test("Differentiable function")
    func differentiableFunction() {
        let x = Tensor<Float>.ones([2, 3])

        let grad = gradient(at: x, of: simpleForward)

        #expect(grad.shape == [2, 3])
    }

    @Test("Differentiable MLP")
    func differentiableMLP() {
        let x = Tensor<Float>.ones([32, 10])
        let w1 = Tensor<Float>.ones([10, 20])
        let w2 = Tensor<Float>.ones([20, 5])

        // Get gradients through differentiable function
        let (_, pb) = valueWithPullback(at: x, w1, w2, of: mlpForward)
        let (gradX, gradW1, gradW2) = pb(Tensor<Float>.ones([], on: .default))

        #expect(gradX.shape == [32, 10])
        #expect(gradW1.shape == [10, 20])
        #expect(gradW2.shape == [20, 5])
    }
}
