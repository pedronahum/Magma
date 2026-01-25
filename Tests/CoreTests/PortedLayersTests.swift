// Magma - Ported Layers Tests
// Tests for layers and components ported from S4TF

import XCTest
@testable import Magma
@testable import LazyTensor

// MARK: - Conv1d Tests

final class Conv1dTests: XCTestCase {

    func testConv1dCreation() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3)

        XCTAssertEqual(conv.inChannels, 3)
        XCTAssertEqual(conv.outChannels, 64)
        XCTAssertEqual(conv.kernelSize, 3)
        XCTAssertEqual(conv.stride, 1)
        XCTAssertEqual(conv.padding, 0)
        XCTAssertTrue(conv.useBias)
    }

    func testConv1dForward() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3)
        let input = Tensor<Float>.zeros([32, 100, 3])  // [batch, length, channels]

        let output = conv(input)

        // Output length = (100 + 2*0 - 3) / 1 + 1 = 98
        XCTAssertEqual(output.shape, [32, 98, 64])
    }

    func testConv1dWithPadding() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3, padding: 1)
        let input = Tensor<Float>.zeros([32, 100, 3])

        let output = conv(input)

        // Output length = (100 + 2*1 - 3) / 1 + 1 = 100
        XCTAssertEqual(output.shape, [32, 100, 64])
    }

    func testConv1dWithStride() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3, stride: 2)
        let input = Tensor<Float>.zeros([32, 100, 3])

        let output = conv(input)

        // Output length = (100 + 2*0 - 3) / 2 + 1 = 49
        XCTAssertEqual(output.shape, [32, 49, 64])
    }

    func testConv1dNoBias() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3, bias: false)

        XCTAssertFalse(conv.useBias)
        XCTAssertEqual(conv.parameters().count, 1)  // Only weight
    }

    func testConv1dParameters() {
        let convWithBias = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3, bias: true)
        let convNoBias = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3, bias: false)

        XCTAssertEqual(convWithBias.parameters().count, 2)
        XCTAssertEqual(convNoBias.parameters().count, 1)
    }

    func testConv1dWeightShape() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 5)

        // Weight shape: [kernelSize, inChannels, outChannels]
        XCTAssertEqual(conv.weight.shape, [5, 3, 64])
        XCTAssertEqual(conv.bias.shape, [64])
    }
}

// MARK: - ConvTranspose2d Tests

final class ConvTranspose2dTests: XCTestCase {

    func testConvTranspose2dCreation() {
        let convT = nn.ConvTranspose2d(inChannels: 64, outChannels: 32, kernelSize: 4)

        XCTAssertEqual(convT.inChannels, 64)
        XCTAssertEqual(convT.outChannels, 32)
        XCTAssertEqual(convT.kernelSize.0, 4)
        XCTAssertEqual(convT.kernelSize.1, 4)
        XCTAssertEqual(convT.stride.0, 1)
        XCTAssertEqual(convT.stride.1, 1)
        XCTAssertEqual(convT.padding.0, 0)
        XCTAssertEqual(convT.padding.1, 0)
        XCTAssertTrue(convT.useBias)
    }

    func testConvTranspose2dForward() {
        let convT = nn.ConvTranspose2d(inChannels: 64, outChannels: 32, kernelSize: 4, stride: 2, padding: 1)
        let input = Tensor<Float>.zeros([32, 14, 14, 64])  // [batch, height, width, channels]

        let output = convT(input)

        // Output = (input - 1) * stride - 2 * padding + kernel + outputPadding
        // = (14 - 1) * 2 - 2 * 1 + 4 + 0 = 28
        XCTAssertEqual(output.shape, [32, 28, 28, 32])
    }

    func testConvTranspose2dUpsampling() {
        // Common GAN-style upsampling: 4x4 -> 8x8
        let convT = nn.ConvTranspose2d(inChannels: 512, outChannels: 256, kernelSize: 4, stride: 2, padding: 1)
        let input = Tensor<Float>.zeros([16, 4, 4, 512])

        let output = convT(input)

        XCTAssertEqual(output.shape, [16, 8, 8, 256])
    }

    func testConvTranspose2dNoBias() {
        let convT = nn.ConvTranspose2d(inChannels: 64, outChannels: 32, kernelSize: 4, bias: false)

        XCTAssertFalse(convT.useBias)
        XCTAssertEqual(convT.parameters().count, 1)
    }

    func testConvTranspose2dWeightShape() {
        let convT = nn.ConvTranspose2d(inChannels: 64, outChannels: 32, kernelSize: 4)

        // Weight shape: [kernelH, kernelW, outChannels, inChannels] (reversed from regular conv)
        XCTAssertEqual(convT.weight.shape, [4, 4, 32, 64])
        XCTAssertEqual(convT.bias.shape, [32])
    }

    func testConvTranspose2dTupleKernel() {
        let convT = nn.ConvTranspose2d(
            inChannels: 64,
            outChannels: 32,
            kernelSize: (3, 5),
            stride: (2, 2),
            padding: (1, 2)
        )
        let input = Tensor<Float>.zeros([16, 7, 7, 64])

        let output = convT(input)

        // Height: (7 - 1) * 2 - 2 * 1 + 3 = 13
        // Width: (7 - 1) * 2 - 2 * 2 + 5 = 13
        XCTAssertEqual(output.shape, [16, 13, 13, 32])
    }
}

// MARK: - Global Pooling Tests

final class GlobalPoolingTests: XCTestCase {

    // GlobalAvgPool1d Tests

    func testGlobalAvgPool1dForward() {
        let pool = nn.GlobalAvgPool1d()
        let input = Tensor<Float>.zeros([32, 100, 64])  // [batch, length, channels]

        let output = pool(input)

        XCTAssertEqual(output.shape, [32, 64])
    }

    func testGlobalAvgPool1dNoParameters() {
        let pool = nn.GlobalAvgPool1d()
        XCTAssertEqual(pool.parameters().count, 0)
    }

    // GlobalAvgPool2d Tests

    func testGlobalAvgPool2dForward() {
        let pool = nn.GlobalAvgPool2d()
        let input = Tensor<Float>.zeros([32, 7, 7, 512])  // [batch, height, width, channels]

        let output = pool(input)

        XCTAssertEqual(output.shape, [32, 512])
    }

    func testGlobalAvgPool2dNoParameters() {
        let pool = nn.GlobalAvgPool2d()
        XCTAssertEqual(pool.parameters().count, 0)
    }

    // GlobalMaxPool1d Tests

    func testGlobalMaxPool1dForward() {
        let pool = nn.GlobalMaxPool1d()
        let input = Tensor<Float>.zeros([32, 100, 64])

        let output = pool(input)

        XCTAssertEqual(output.shape, [32, 64])
    }

    func testGlobalMaxPool1dNoParameters() {
        let pool = nn.GlobalMaxPool1d()
        XCTAssertEqual(pool.parameters().count, 0)
    }

    // GlobalMaxPool2d Tests

    func testGlobalMaxPool2dForward() {
        let pool = nn.GlobalMaxPool2d()
        let input = Tensor<Float>.zeros([32, 7, 7, 512])

        let output = pool(input)

        XCTAssertEqual(output.shape, [32, 512])
    }

    func testGlobalMaxPool2dNoParameters() {
        let pool = nn.GlobalMaxPool2d()
        XCTAssertEqual(pool.parameters().count, 0)
    }

    // Different spatial sizes

    func testGlobalAvgPool2dVariousSizes() {
        let pool = nn.GlobalAvgPool2d()

        let input1 = Tensor<Float>.zeros([16, 14, 14, 256])
        let input2 = Tensor<Float>.zeros([8, 28, 28, 128])
        let input3 = Tensor<Float>.zeros([4, 56, 56, 64])

        XCTAssertEqual(pool(input1).shape, [16, 256])
        XCTAssertEqual(pool(input2).shape, [8, 128])
        XCTAssertEqual(pool(input3).shape, [4, 64])
    }
}

// MARK: - Upsampling Tests

final class UpsamplingTests: XCTestCase {

    // Upsample1d Tests

    func testUpsample1dCreation() {
        let up = nn.Upsample1d(size: 2)
        XCTAssertEqual(up.size, 2)
    }

    func testUpsample1dForward() {
        let up = nn.Upsample1d(size: 2)
        let input = Tensor<Float>.zeros([32, 50, 64])  // [batch, length, channels]

        let output = up(input)

        XCTAssertEqual(output.shape, [32, 100, 64])
    }

    func testUpsample1dScale3() {
        let up = nn.Upsample1d(size: 3)
        let input = Tensor<Float>.zeros([16, 20, 128])

        let output = up(input)

        XCTAssertEqual(output.shape, [16, 60, 128])
    }

    func testUpsample1dNoParameters() {
        let up = nn.Upsample1d(size: 2)
        XCTAssertEqual(up.parameters().count, 0)
    }

    // Upsample2d Tests

    func testUpsample2dCreation() {
        let up = nn.Upsample2d(size: 2)
        XCTAssertEqual(up.size.0, 2)
        XCTAssertEqual(up.size.1, 2)
    }

    func testUpsample2dForward() {
        let up = nn.Upsample2d(size: 2)
        let input = Tensor<Float>.zeros([32, 14, 14, 64])  // [batch, height, width, channels]

        let output = up(input)

        XCTAssertEqual(output.shape, [32, 28, 28, 64])
    }

    func testUpsample2dScale4() {
        let up = nn.Upsample2d(size: 4)
        let input = Tensor<Float>.zeros([16, 7, 7, 512])

        let output = up(input)

        XCTAssertEqual(output.shape, [16, 28, 28, 512])
    }

    func testUpsample2dAsymmetric() {
        let up = nn.Upsample2d(size: (2, 3))
        let input = Tensor<Float>.zeros([8, 10, 10, 128])

        let output = up(input)

        XCTAssertEqual(output.shape, [8, 20, 30, 128])
    }

    func testUpsample2dNoParameters() {
        let up = nn.Upsample2d(size: 2)
        XCTAssertEqual(up.parameters().count, 0)
    }

    // UpsamplingNearest2d alias test
    func testUpsamplingNearest2dAlias() {
        let up: nn.UpsamplingNearest2d = nn.Upsample2d(size: 2)
        let input = Tensor<Float>.zeros([32, 14, 14, 64])

        let output = up(input)

        XCTAssertEqual(output.shape, [32, 28, 28, 64])
    }
}

// MARK: - Activation Layer Tests (Ported)

final class PortedActivationTests: XCTestCase {

    // SELU Tests

    func testSELULayer() {
        let selu = nn.SELU()
        let input = Tensor<Float>.zeros([32, 256])

        let output = selu(input)

        XCTAssertEqual(output.shape, [32, 256])
    }

    func testSELUNoParameters() {
        let selu = nn.SELU()
        XCTAssertEqual(selu.parameters().count, 0)
    }

    // Mish Tests

    func testMishLayer() {
        let mish = nn.Mish()
        let input = Tensor<Float>.zeros([32, 256])

        let output = mish(input)

        XCTAssertEqual(output.shape, [32, 256])
    }

    func testMishNoParameters() {
        let mish = nn.Mish()
        XCTAssertEqual(mish.parameters().count, 0)
    }

    // Softplus Tests

    func testSoftplusLayer() {
        let softplus = nn.Softplus()
        let input = Tensor<Float>.zeros([32, 256])

        let output = softplus(input)

        XCTAssertEqual(output.shape, [32, 256])
    }

    func testSoftplusWithBeta() {
        let softplus = nn.Softplus(beta: 2.0)
        XCTAssertEqual(softplus.beta, 2.0)
    }

    func testSoftplusNoParameters() {
        let softplus = nn.Softplus()
        XCTAssertEqual(softplus.parameters().count, 0)
    }

    // Softsign Tests

    func testSoftsignLayer() {
        let softsign = nn.Softsign()
        let input = Tensor<Float>.zeros([32, 256])

        let output = softsign(input)

        XCTAssertEqual(output.shape, [32, 256])
    }

    func testSoftsignNoParameters() {
        let softsign = nn.Softsign()
        XCTAssertEqual(softsign.parameters().count, 0)
    }

    // PReLU Tests

    func testPReLULayer() {
        let prelu = nn.PReLU()
        let input = Tensor<Float>.zeros([32, 256])

        let output = prelu(input)

        XCTAssertEqual(output.shape, [32, 256])
    }

    func testPReLUWithNumParameters() {
        let prelu = nn.PReLU(numParameters: 64)
        XCTAssertEqual(prelu.numParameters, 64)
    }

    func testPReLUHasParameters() {
        let prelu = nn.PReLU()
        // PReLU has learnable parameters
        XCTAssertEqual(prelu.parameters().count, 1)
    }

    func testPReLUWeightShape() {
        let prelu1 = nn.PReLU(numParameters: 1)
        let prelu64 = nn.PReLU(numParameters: 64)

        XCTAssertEqual(prelu1.weight.shape, [1])
        XCTAssertEqual(prelu64.weight.shape, [64])
    }
}

// MARK: - Normalization Layer Tests (Ported)

final class PortedNormalizationTests: XCTestCase {

    // GroupNorm Tests

    func testGroupNormCreation() {
        let gn = nn.GroupNorm(numGroups: 8, numChannels: 64)

        XCTAssertEqual(gn.numGroups, 8)
        XCTAssertEqual(gn.numChannels, 64)
        XCTAssertTrue(gn.affine)
    }

    func testGroupNormForward() {
        let gn = nn.GroupNorm(numGroups: 8, numChannels: 64)
        let input = Tensor<Float>.zeros([32, 14, 14, 64])  // [batch, height, width, channels]

        let output = gn(input)

        XCTAssertEqual(output.shape, [32, 14, 14, 64])
    }

    func testGroupNormParameters() {
        let gnAffine = nn.GroupNorm(numGroups: 8, numChannels: 64, affine: true)
        let gnNoAffine = nn.GroupNorm(numGroups: 8, numChannels: 64, affine: false)

        XCTAssertEqual(gnAffine.parameters().count, 2)  // weight and bias
        XCTAssertEqual(gnNoAffine.parameters().count, 0)
    }

    func testGroupNormWeightShape() {
        let gn = nn.GroupNorm(numGroups: 8, numChannels: 64)

        XCTAssertEqual(gn.weight.shape, [64])
        XCTAssertEqual(gn.bias.shape, [64])
    }

    // InstanceNorm2d Tests

    func testInstanceNorm2dCreation() {
        let inorm = nn.InstanceNorm2d(numFeatures: 64)

        XCTAssertEqual(inorm.numFeatures, 64)
        XCTAssertTrue(inorm.affine)
    }

    func testInstanceNorm2dForward() {
        let inorm = nn.InstanceNorm2d(numFeatures: 64)
        let input = Tensor<Float>.zeros([32, 14, 14, 64])

        let output = inorm(input)

        XCTAssertEqual(output.shape, [32, 14, 14, 64])
    }

    func testInstanceNorm2dParameters() {
        let inormAffine = nn.InstanceNorm2d(numFeatures: 64, affine: true)
        let inormNoAffine = nn.InstanceNorm2d(numFeatures: 64, affine: false)

        XCTAssertEqual(inormAffine.parameters().count, 2)
        XCTAssertEqual(inormNoAffine.parameters().count, 0)
    }
}
