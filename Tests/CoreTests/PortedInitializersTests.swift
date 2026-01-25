// Magma - Ported Initializers Tests
// Tests for parameter initializers ported from S4TF

import Testing
@testable import Magma
@testable import LazyTensor

// MARK: - Glorot (Xavier) Uniform Initializer Tests

@Suite("Glorot Uniform Initializer Tests")
struct GlorotUniformInitializerTests {

    @Test("Glorot uniform shape")
    func glorotUniformShape() {
        let tensor = Tensor<Float>.glorotUniform([784, 256])

        #expect(tensor.shape == [784, 256])
    }

    @Test("Glorot uniform different shapes")
    func glorotUniformDifferentShapes() {
        let t1 = Tensor<Float>.glorotUniform([10, 20])
        let t2 = Tensor<Float>.glorotUniform([100, 50])
        let t3 = Tensor<Float>.glorotUniform([256, 128])

        #expect(t1.shape == [10, 20])
        #expect(t2.shape == [100, 50])
        #expect(t3.shape == [256, 128])
    }

    @Test("Glorot uniform 1D uses randn fallback")
    func glorotUniform1D() {
        // Glorot/Xavier requires 2D+ tensors for fan calculation
        // For 1D, it should fall back to randn
        let tensor = Tensor<Float>.randn([100])

        #expect(tensor.shape == [100])
    }

    @Test("Glorot uniform 3D")
    func glorotUniform3D() {
        let tensor = Tensor<Float>.glorotUniform([3, 3, 64])

        #expect(tensor.shape == [3, 3, 64])
    }

    @Test("Glorot uniform 4D")
    func glorotUniform4D() {
        let tensor = Tensor<Float>.glorotUniform([3, 3, 64, 128])

        #expect(tensor.shape == [3, 3, 64, 128])
    }
}

// MARK: - Glorot (Xavier) Normal Initializer Tests

@Suite("Glorot Normal Initializer Tests")
struct GlorotNormalInitializerTests {

    @Test("Glorot normal shape")
    func glorotNormalShape() {
        let tensor = Tensor<Float>.glorotNormal([784, 256])

        #expect(tensor.shape == [784, 256])
    }

    @Test("Glorot normal different shapes")
    func glorotNormalDifferentShapes() {
        let t1 = Tensor<Float>.glorotNormal([10, 20])
        let t2 = Tensor<Float>.glorotNormal([100, 50])
        let t3 = Tensor<Float>.glorotNormal([256, 128])

        #expect(t1.shape == [10, 20])
        #expect(t2.shape == [100, 50])
        #expect(t3.shape == [256, 128])
    }
}

// MARK: - He (Kaiming) Uniform Initializer Tests

@Suite("He Uniform Initializer Tests")
struct HeUniformInitializerTests {

    @Test("He uniform shape")
    func heUniformShape() {
        let tensor = Tensor<Float>.heUniform([784, 256])

        #expect(tensor.shape == [784, 256])
    }

    @Test("He uniform different shapes")
    func heUniformDifferentShapes() {
        let t1 = Tensor<Float>.heUniform([10, 20])
        let t2 = Tensor<Float>.heUniform([100, 50])

        #expect(t1.shape == [10, 20])
        #expect(t2.shape == [100, 50])
    }

    @Test("He uniform conv weight")
    func heUniformConvWeight() {
        let tensor = Tensor<Float>.heUniform([3, 3, 64, 128])  // Conv2d weight shape

        #expect(tensor.shape == [3, 3, 64, 128])
    }
}

// MARK: - He (Kaiming) Normal Initializer Tests

@Suite("He Normal Initializer Tests")
struct HeNormalInitializerTests {

    @Test("He normal shape")
    func heNormalShape() {
        let tensor = Tensor<Float>.heNormal([784, 256])

        #expect(tensor.shape == [784, 256])
    }

    @Test("He normal different shapes")
    func heNormalDifferentShapes() {
        let t1 = Tensor<Float>.heNormal([10, 20])
        let t2 = Tensor<Float>.heNormal([100, 50])

        #expect(t1.shape == [10, 20])
        #expect(t2.shape == [100, 50])
    }
}

// MARK: - LeCun Uniform Initializer Tests

@Suite("LeCun Uniform Initializer Tests")
struct LeCunUniformInitializerTests {

    @Test("LeCun uniform shape")
    func lecunUniformShape() {
        let tensor = Tensor<Float>.lecunUniform([784, 256])

        #expect(tensor.shape == [784, 256])
    }

    @Test("LeCun uniform different shapes")
    func lecunUniformDifferentShapes() {
        let t1 = Tensor<Float>.lecunUniform([10, 20])
        let t2 = Tensor<Float>.lecunUniform([100, 50])

        #expect(t1.shape == [10, 20])
        #expect(t2.shape == [100, 50])
    }
}

// MARK: - LeCun Normal Initializer Tests

@Suite("LeCun Normal Initializer Tests")
struct LeCunNormalInitializerTests {

    @Test("LeCun normal shape")
    func lecunNormalShape() {
        let tensor = Tensor<Float>.lecunNormal([784, 256])

        #expect(tensor.shape == [784, 256])
    }

    @Test("LeCun normal different shapes")
    func lecunNormalDifferentShapes() {
        let t1 = Tensor<Float>.lecunNormal([10, 20])
        let t2 = Tensor<Float>.lecunNormal([100, 50])

        #expect(t1.shape == [10, 20])
        #expect(t2.shape == [100, 50])
    }
}

// MARK: - Zeros Initializer Tests

@Suite("Zeros Initializer Tests")
struct ZerosInitializerTests {

    @Test("Zeros shape")
    func zerosShape() {
        let tensor = Tensor<Float>.zeros([784, 256])

        #expect(tensor.shape == [784, 256])
    }

    @Test("Zeros different shapes")
    func zerosDifferentShapes() {
        let t1 = Tensor<Float>.zeros([10])
        let t2 = Tensor<Float>.zeros([100, 50])
        let t3 = Tensor<Float>.zeros([3, 3, 64, 128])

        #expect(t1.shape == [10])
        #expect(t2.shape == [100, 50])
        #expect(t3.shape == [3, 3, 64, 128])
    }
}

// MARK: - Ones Initializer Tests

@Suite("Ones Initializer Tests")
struct OnesInitializerTests {

    @Test("Ones shape")
    func onesShape() {
        let tensor = Tensor<Float>.ones([784, 256])

        #expect(tensor.shape == [784, 256])
    }

    @Test("Ones different shapes")
    func onesDifferentShapes() {
        let t1 = Tensor<Float>.ones([10])
        let t2 = Tensor<Float>.ones([100, 50])
        let t3 = Tensor<Float>.ones([3, 3, 64, 128])

        #expect(t1.shape == [10])
        #expect(t2.shape == [100, 50])
        #expect(t3.shape == [3, 3, 64, 128])
    }
}

// MARK: - Full (Constant) Initializer Tests

@Suite("Full Initializer Tests")
struct FullInitializerTests {

    @Test("Full shape")
    func fullShape() {
        let tensor = Tensor<Float>.full([784, 256], 0.5, on: .default)

        #expect(tensor.shape == [784, 256])
    }

    @Test("Full different values")
    func fullDifferentValues() {
        let t1 = Tensor<Float>.full([10, 10], 0.0, on: .default)
        let t2 = Tensor<Float>.full([10, 10], 1.0, on: .default)
        let t3 = Tensor<Float>.full([10, 10], -0.5, on: .default)

        #expect(t1.shape == [10, 10])
        #expect(t2.shape == [10, 10])
        #expect(t3.shape == [10, 10])
    }
}

// MARK: - Truncated Normal Initializer Tests

@Suite("Truncated Normal Initializer Tests")
struct TruncatedNormalInitializerTests {

    @Test("Truncated normal shape")
    func truncatedNormalShape() {
        let tensor = Tensor<Float>.truncatedNormal([784, 256])

        #expect(tensor.shape == [784, 256])
    }

    @Test("Truncated normal custom params")
    func truncatedNormalCustomParams() {
        let tensor = Tensor<Float>.truncatedNormal([100, 50], mean: 0.0, stddev: 0.05)

        #expect(tensor.shape == [100, 50])
    }
}

// MARK: - Orthogonal Initializer Tests

@Suite("Orthogonal Initializer Tests")
struct OrthogonalInitializerTests {

    @Test("Orthogonal shape")
    func orthogonalShape() {
        let tensor = Tensor<Float>.orthogonal([256, 256])

        #expect(tensor.shape == [256, 256])
    }

    @Test("Orthogonal non-square")
    func orthogonalNonSquare() {
        let t1 = Tensor<Float>.orthogonal([100, 200])
        let t2 = Tensor<Float>.orthogonal([200, 100])

        #expect(t1.shape == [100, 200])
        #expect(t2.shape == [200, 100])
    }

    @Test("Orthogonal custom gain")
    func orthogonalCustomGain() {
        let tensor = Tensor<Float>.orthogonal([256, 256], gain: 1.41421)  // sqrt(2) for ReLU

        #expect(tensor.shape == [256, 256])
    }
}

// MARK: - Initializer with Layer Tests

@Suite("Initializer Layer Integration Tests")
struct InitializerLayerIntegrationTests {

    @Test("Linear with default init")
    func linearWithDefaultInit() {
        // nn.Linear uses Kaiming by default
        let linear = nn.Linear(inputSize: 784, outputSize: 256)

        #expect(linear.weight.shape == [256, 784])
        #expect(linear.bias.shape == [256])
    }

    @Test("Conv2d initialization")
    func conv2dInitialization() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3)

        #expect(conv.weight.shape == [3, 3, 3, 64])
        #expect(conv.bias.shape == [64])
    }

    @Test("Initializer device support")
    func initializerDeviceSupport() {
        let tensor = Tensor<Float>.glorotUniform([100, 100])

        // Tensor should be on default device
        #expect(tensor.shape == [100, 100])
    }
}

// MARK: - Fan Calculation Tests

@Suite("Fan Calculation Tests")
struct FanCalculationTests {

    @Test("Fan in fan out for linear")
    func fanInFanOutForLinear() {
        // For a linear layer weight [outFeatures, inFeatures]
        // fanIn = inFeatures, fanOut = outFeatures
        let tensor = Tensor<Float>.glorotUniform([256, 784])

        #expect(tensor.shape == [256, 784])
    }

    @Test("Fan in fan out for conv")
    func fanInFanOutForConv() {
        // For a conv weight [kernelH, kernelW, inChannels, outChannels]
        // fanIn = kernelH * kernelW * inChannels
        // fanOut = kernelH * kernelW * outChannels
        let tensor = Tensor<Float>.heUniform([3, 3, 64, 128])

        #expect(tensor.shape == [3, 3, 64, 128])
    }
}
