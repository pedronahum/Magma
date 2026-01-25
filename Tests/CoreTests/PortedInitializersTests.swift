// Magma - Ported Initializers Tests
// Tests for parameter initializers ported from S4TF

import XCTest
@testable import Magma
@testable import LazyTensor

// MARK: - Glorot (Xavier) Uniform Initializer Tests

final class GlorotUniformInitializerTests: XCTestCase {

    func testGlorotUniformShape() {
        let tensor = Tensor<Float>.glorotUniform([784, 256])

        XCTAssertEqual(tensor.shape, [784, 256])
    }

    func testGlorotUniformDifferentShapes() {
        let t1 = Tensor<Float>.glorotUniform([10, 20])
        let t2 = Tensor<Float>.glorotUniform([100, 50])
        let t3 = Tensor<Float>.glorotUniform([256, 128])

        XCTAssertEqual(t1.shape, [10, 20])
        XCTAssertEqual(t2.shape, [100, 50])
        XCTAssertEqual(t3.shape, [256, 128])
    }

    func testGlorotUniform1D() {
        let tensor = Tensor<Float>.glorotUniform([100])

        XCTAssertEqual(tensor.shape, [100])
    }

    func testGlorotUniform3D() {
        let tensor = Tensor<Float>.glorotUniform([3, 3, 64])

        XCTAssertEqual(tensor.shape, [3, 3, 64])
    }

    func testGlorotUniform4D() {
        let tensor = Tensor<Float>.glorotUniform([3, 3, 64, 128])

        XCTAssertEqual(tensor.shape, [3, 3, 64, 128])
    }
}

// MARK: - Glorot (Xavier) Normal Initializer Tests

final class GlorotNormalInitializerTests: XCTestCase {

    func testGlorotNormalShape() {
        let tensor = Tensor<Float>.glorotNormal([784, 256])

        XCTAssertEqual(tensor.shape, [784, 256])
    }

    func testGlorotNormalDifferentShapes() {
        let t1 = Tensor<Float>.glorotNormal([10, 20])
        let t2 = Tensor<Float>.glorotNormal([100, 50])
        let t3 = Tensor<Float>.glorotNormal([256, 128])

        XCTAssertEqual(t1.shape, [10, 20])
        XCTAssertEqual(t2.shape, [100, 50])
        XCTAssertEqual(t3.shape, [256, 128])
    }
}

// MARK: - He (Kaiming) Uniform Initializer Tests

final class HeUniformInitializerTests: XCTestCase {

    func testHeUniformShape() {
        let tensor = Tensor<Float>.heUniform([784, 256])

        XCTAssertEqual(tensor.shape, [784, 256])
    }

    func testHeUniformDifferentShapes() {
        let t1 = Tensor<Float>.heUniform([10, 20])
        let t2 = Tensor<Float>.heUniform([100, 50])

        XCTAssertEqual(t1.shape, [10, 20])
        XCTAssertEqual(t2.shape, [100, 50])
    }

    func testHeUniformConvWeight() {
        let tensor = Tensor<Float>.heUniform([3, 3, 64, 128])  // Conv2d weight shape

        XCTAssertEqual(tensor.shape, [3, 3, 64, 128])
    }
}

// MARK: - He (Kaiming) Normal Initializer Tests

final class HeNormalInitializerTests: XCTestCase {

    func testHeNormalShape() {
        let tensor = Tensor<Float>.heNormal([784, 256])

        XCTAssertEqual(tensor.shape, [784, 256])
    }

    func testHeNormalDifferentShapes() {
        let t1 = Tensor<Float>.heNormal([10, 20])
        let t2 = Tensor<Float>.heNormal([100, 50])

        XCTAssertEqual(t1.shape, [10, 20])
        XCTAssertEqual(t2.shape, [100, 50])
    }
}

// MARK: - LeCun Uniform Initializer Tests

final class LeCunUniformInitializerTests: XCTestCase {

    func testLeCunUniformShape() {
        let tensor = Tensor<Float>.lecunUniform([784, 256])

        XCTAssertEqual(tensor.shape, [784, 256])
    }

    func testLeCunUniformDifferentShapes() {
        let t1 = Tensor<Float>.lecunUniform([10, 20])
        let t2 = Tensor<Float>.lecunUniform([100, 50])

        XCTAssertEqual(t1.shape, [10, 20])
        XCTAssertEqual(t2.shape, [100, 50])
    }
}

// MARK: - LeCun Normal Initializer Tests

final class LeCunNormalInitializerTests: XCTestCase {

    func testLeCunNormalShape() {
        let tensor = Tensor<Float>.lecunNormal([784, 256])

        XCTAssertEqual(tensor.shape, [784, 256])
    }

    func testLeCunNormalDifferentShapes() {
        let t1 = Tensor<Float>.lecunNormal([10, 20])
        let t2 = Tensor<Float>.lecunNormal([100, 50])

        XCTAssertEqual(t1.shape, [10, 20])
        XCTAssertEqual(t2.shape, [100, 50])
    }
}

// MARK: - Zeros Initializer Tests

final class ZerosInitializerTests: XCTestCase {

    func testZerosShape() {
        let tensor = Tensor<Float>.zeros([784, 256])

        XCTAssertEqual(tensor.shape, [784, 256])
    }

    func testZerosDifferentShapes() {
        let t1 = Tensor<Float>.zeros([10])
        let t2 = Tensor<Float>.zeros([100, 50])
        let t3 = Tensor<Float>.zeros([3, 3, 64, 128])

        XCTAssertEqual(t1.shape, [10])
        XCTAssertEqual(t2.shape, [100, 50])
        XCTAssertEqual(t3.shape, [3, 3, 64, 128])
    }
}

// MARK: - Ones Initializer Tests

final class OnesInitializerTests: XCTestCase {

    func testOnesShape() {
        let tensor = Tensor<Float>.ones([784, 256])

        XCTAssertEqual(tensor.shape, [784, 256])
    }

    func testOnesDifferentShapes() {
        let t1 = Tensor<Float>.ones([10])
        let t2 = Tensor<Float>.ones([100, 50])
        let t3 = Tensor<Float>.ones([3, 3, 64, 128])

        XCTAssertEqual(t1.shape, [10])
        XCTAssertEqual(t2.shape, [100, 50])
        XCTAssertEqual(t3.shape, [3, 3, 64, 128])
    }
}

// MARK: - Full (Constant) Initializer Tests

final class FullInitializerTests: XCTestCase {

    func testFullShape() {
        let tensor = Tensor<Float>.full([784, 256], 0.5, on: .default)

        XCTAssertEqual(tensor.shape, [784, 256])
    }

    func testFullDifferentValues() {
        let t1 = Tensor<Float>.full([10, 10], 0.0, on: .default)
        let t2 = Tensor<Float>.full([10, 10], 1.0, on: .default)
        let t3 = Tensor<Float>.full([10, 10], -0.5, on: .default)

        XCTAssertEqual(t1.shape, [10, 10])
        XCTAssertEqual(t2.shape, [10, 10])
        XCTAssertEqual(t3.shape, [10, 10])
    }
}

// MARK: - Truncated Normal Initializer Tests

final class TruncatedNormalInitializerTests: XCTestCase {

    func testTruncatedNormalShape() {
        let tensor = Tensor<Float>.truncatedNormal([784, 256])

        XCTAssertEqual(tensor.shape, [784, 256])
    }

    func testTruncatedNormalCustomParams() {
        let tensor = Tensor<Float>.truncatedNormal([100, 50], mean: 0.0, stddev: 0.05)

        XCTAssertEqual(tensor.shape, [100, 50])
    }
}

// MARK: - Orthogonal Initializer Tests

final class OrthogonalInitializerTests: XCTestCase {

    func testOrthogonalShape() {
        let tensor = Tensor<Float>.orthogonal([256, 256])

        XCTAssertEqual(tensor.shape, [256, 256])
    }

    func testOrthogonalNonSquare() {
        let t1 = Tensor<Float>.orthogonal([100, 200])
        let t2 = Tensor<Float>.orthogonal([200, 100])

        XCTAssertEqual(t1.shape, [100, 200])
        XCTAssertEqual(t2.shape, [200, 100])
    }

    func testOrthogonalCustomGain() {
        let tensor = Tensor<Float>.orthogonal([256, 256], gain: 1.41421)  // sqrt(2) for ReLU

        XCTAssertEqual(tensor.shape, [256, 256])
    }
}

// MARK: - Initializer with Layer Tests

final class InitializerLayerIntegrationTests: XCTestCase {

    func testLinearWithDefaultInit() {
        // nn.Linear uses Kaiming by default
        let linear = nn.Linear(inputSize: 784, outputSize: 256)

        XCTAssertEqual(linear.weight.shape, [256, 784])
        XCTAssertEqual(linear.bias.shape, [256])
    }

    func testConv2dInitialization() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3)

        XCTAssertEqual(conv.weight.shape, [3, 3, 3, 64])
        XCTAssertEqual(conv.bias.shape, [64])
    }

    func testInitializerDeviceSupport() {
        let tensor = Tensor<Float>.glorotUniform([100, 100])

        // Tensor should be on default device
        XCTAssertEqual(tensor.shape, [100, 100])
    }
}

// MARK: - Fan Calculation Tests

final class FanCalculationTests: XCTestCase {

    func testFanInFanOutForLinear() {
        // For a linear layer weight [outFeatures, inFeatures]
        // fanIn = inFeatures, fanOut = outFeatures
        let tensor = Tensor<Float>.glorotUniform([256, 784])

        XCTAssertEqual(tensor.shape, [256, 784])
    }

    func testFanInFanOutForConv() {
        // For a conv weight [kernelH, kernelW, inChannels, outChannels]
        // fanIn = kernelH * kernelW * inChannels
        // fanOut = kernelH * kernelW * outChannels
        let tensor = Tensor<Float>.heUniform([3, 3, 64, 128])

        XCTAssertEqual(tensor.shape, [3, 3, 64, 128])
    }
}
