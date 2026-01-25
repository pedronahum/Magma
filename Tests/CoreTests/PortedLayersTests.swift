// Magma - Ported Layers Tests
// Tests for layers and components ported from S4TF

import Testing
@testable import Magma
@testable import LazyTensor

// MARK: - Conv1d Tests

@Suite("Conv1d Tests")
struct Conv1dTests {

    @Test("Conv1d creation")
    func conv1dCreation() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3)

        #expect(conv.inChannels == 3)
        #expect(conv.outChannels == 64)
        #expect(conv.kernelSize == 3)
        #expect(conv.stride == 1)
        #expect(conv.padding == 0)
        #expect(conv.useBias)
    }

    @Test("Conv1d forward")
    func conv1dForward() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3)
        let input = Tensor<Float>.zeros([32, 100, 3])  // [batch, length, channels]

        let output = conv(input)

        // Output length = (100 + 2*0 - 3) / 1 + 1 = 98
        #expect(output.shape == [32, 98, 64])
    }

    @Test("Conv1d with padding")
    func conv1dWithPadding() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3, padding: 1)
        let input = Tensor<Float>.zeros([32, 100, 3])

        let output = conv(input)

        // Output length = (100 + 2*1 - 3) / 1 + 1 = 100
        #expect(output.shape == [32, 100, 64])
    }

    @Test("Conv1d with stride")
    func conv1dWithStride() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3, stride: 2)
        let input = Tensor<Float>.zeros([32, 100, 3])

        let output = conv(input)

        // Output length = (100 + 2*0 - 3) / 2 + 1 = 49
        #expect(output.shape == [32, 49, 64])
    }

    @Test("Conv1d no bias")
    func conv1dNoBias() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3, bias: false)

        #expect(!conv.useBias)
        #expect(conv.parameters().count == 1)  // Only weight
    }

    @Test("Conv1d parameters")
    func conv1dParameters() {
        let convWithBias = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3, bias: true)
        let convNoBias = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 3, bias: false)

        #expect(convWithBias.parameters().count == 2)
        #expect(convNoBias.parameters().count == 1)
    }

    @Test("Conv1d weight shape")
    func conv1dWeightShape() {
        let conv = nn.Conv1d(inChannels: 3, outChannels: 64, kernelSize: 5)

        // Weight shape: [kernelSize, inChannels, outChannels]
        #expect(conv.weight.shape == [5, 3, 64])
        #expect(conv.bias.shape == [64])
    }
}

// MARK: - ConvTranspose2d Tests

@Suite("ConvTranspose2d Tests")
struct ConvTranspose2dTests {

    @Test("ConvTranspose2d creation")
    func convTranspose2dCreation() {
        let convT = nn.ConvTranspose2d(inChannels: 64, outChannels: 32, kernelSize: 4)

        #expect(convT.inChannels == 64)
        #expect(convT.outChannels == 32)
        #expect(convT.kernelSize.0 == 4)
        #expect(convT.kernelSize.1 == 4)
        #expect(convT.stride.0 == 1)
        #expect(convT.stride.1 == 1)
        #expect(convT.padding.0 == 0)
        #expect(convT.padding.1 == 0)
        #expect(convT.useBias)
    }

    @Test("ConvTranspose2d forward")
    func convTranspose2dForward() {
        let convT = nn.ConvTranspose2d(inChannels: 64, outChannels: 32, kernelSize: 4, stride: 2, padding: 1)
        let input = Tensor<Float>.zeros([32, 14, 14, 64])  // [batch, height, width, channels]

        let output = convT(input)

        // Output = (input - 1) * stride - 2 * padding + kernel + outputPadding
        // = (14 - 1) * 2 - 2 * 1 + 4 + 0 = 28
        #expect(output.shape == [32, 28, 28, 32])
    }

    @Test("ConvTranspose2d upsampling")
    func convTranspose2dUpsampling() {
        // Common GAN-style upsampling: 4x4 -> 8x8
        let convT = nn.ConvTranspose2d(inChannels: 512, outChannels: 256, kernelSize: 4, stride: 2, padding: 1)
        let input = Tensor<Float>.zeros([16, 4, 4, 512])

        let output = convT(input)

        #expect(output.shape == [16, 8, 8, 256])
    }

    @Test("ConvTranspose2d no bias")
    func convTranspose2dNoBias() {
        let convT = nn.ConvTranspose2d(inChannels: 64, outChannels: 32, kernelSize: 4, bias: false)

        #expect(!convT.useBias)
        #expect(convT.parameters().count == 1)
    }

    @Test("ConvTranspose2d weight shape")
    func convTranspose2dWeightShape() {
        let convT = nn.ConvTranspose2d(inChannels: 64, outChannels: 32, kernelSize: 4)

        // Weight shape: [kernelH, kernelW, outChannels, inChannels] (reversed from regular conv)
        #expect(convT.weight.shape == [4, 4, 32, 64])
        #expect(convT.bias.shape == [32])
    }

    @Test("ConvTranspose2d tuple kernel")
    func convTranspose2dTupleKernel() {
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
        #expect(output.shape == [16, 13, 13, 32])
    }
}

// MARK: - Global Pooling Tests

@Suite("Global Pooling Tests")
struct GlobalPoolingTests {

    // GlobalAvgPool1d Tests

    @Test("GlobalAvgPool1d forward")
    func globalAvgPool1dForward() {
        let pool = nn.GlobalAvgPool1d()
        let input = Tensor<Float>.zeros([32, 100, 64])  // [batch, length, channels]

        let output = pool(input)

        #expect(output.shape == [32, 64])
    }

    @Test("GlobalAvgPool1d no parameters")
    func globalAvgPool1dNoParameters() {
        let pool = nn.GlobalAvgPool1d()
        #expect(pool.parameters().count == 0)
    }

    // GlobalAvgPool2d Tests

    @Test("GlobalAvgPool2d forward")
    func globalAvgPool2dForward() {
        let pool = nn.GlobalAvgPool2d()
        let input = Tensor<Float>.zeros([32, 7, 7, 512])  // [batch, height, width, channels]

        let output = pool(input)

        #expect(output.shape == [32, 512])
    }

    @Test("GlobalAvgPool2d no parameters")
    func globalAvgPool2dNoParameters() {
        let pool = nn.GlobalAvgPool2d()
        #expect(pool.parameters().count == 0)
    }

    // GlobalMaxPool1d Tests

    @Test("GlobalMaxPool1d forward")
    func globalMaxPool1dForward() {
        let pool = nn.GlobalMaxPool1d()
        let input = Tensor<Float>.zeros([32, 100, 64])

        let output = pool(input)

        #expect(output.shape == [32, 64])
    }

    @Test("GlobalMaxPool1d no parameters")
    func globalMaxPool1dNoParameters() {
        let pool = nn.GlobalMaxPool1d()
        #expect(pool.parameters().count == 0)
    }

    // GlobalMaxPool2d Tests

    @Test("GlobalMaxPool2d forward")
    func globalMaxPool2dForward() {
        let pool = nn.GlobalMaxPool2d()
        let input = Tensor<Float>.zeros([32, 7, 7, 512])

        let output = pool(input)

        #expect(output.shape == [32, 512])
    }

    @Test("GlobalMaxPool2d no parameters")
    func globalMaxPool2dNoParameters() {
        let pool = nn.GlobalMaxPool2d()
        #expect(pool.parameters().count == 0)
    }

    // Different spatial sizes

    @Test("GlobalAvgPool2d various sizes")
    func globalAvgPool2dVariousSizes() {
        let pool = nn.GlobalAvgPool2d()

        let input1 = Tensor<Float>.zeros([16, 14, 14, 256])
        let input2 = Tensor<Float>.zeros([8, 28, 28, 128])
        let input3 = Tensor<Float>.zeros([4, 56, 56, 64])

        #expect(pool(input1).shape == [16, 256])
        #expect(pool(input2).shape == [8, 128])
        #expect(pool(input3).shape == [4, 64])
    }
}

// MARK: - Upsampling Tests

@Suite("Upsampling Tests")
struct UpsamplingTests {

    // Upsample1d Tests

    @Test("Upsample1d creation")
    func upsample1dCreation() {
        let up = nn.Upsample1d(size: 2)
        #expect(up.size == 2)
    }

    @Test("Upsample1d forward")
    func upsample1dForward() {
        let up = nn.Upsample1d(size: 2)
        let input = Tensor<Float>.zeros([32, 50, 64])  // [batch, length, channels]

        let output = up(input)

        #expect(output.shape == [32, 100, 64])
    }

    @Test("Upsample1d scale 3")
    func upsample1dScale3() {
        let up = nn.Upsample1d(size: 3)
        let input = Tensor<Float>.zeros([16, 20, 128])

        let output = up(input)

        #expect(output.shape == [16, 60, 128])
    }

    @Test("Upsample1d no parameters")
    func upsample1dNoParameters() {
        let up = nn.Upsample1d(size: 2)
        #expect(up.parameters().count == 0)
    }

    // Upsample2d Tests

    @Test("Upsample2d creation")
    func upsample2dCreation() {
        let up = nn.Upsample2d(size: 2)
        #expect(up.size.0 == 2)
        #expect(up.size.1 == 2)
    }

    @Test("Upsample2d forward")
    func upsample2dForward() {
        let up = nn.Upsample2d(size: 2)
        let input = Tensor<Float>.zeros([32, 14, 14, 64])  // [batch, height, width, channels]

        let output = up(input)

        #expect(output.shape == [32, 28, 28, 64])
    }

    @Test("Upsample2d scale 4")
    func upsample2dScale4() {
        let up = nn.Upsample2d(size: 4)
        let input = Tensor<Float>.zeros([16, 7, 7, 512])

        let output = up(input)

        #expect(output.shape == [16, 28, 28, 512])
    }

    @Test("Upsample2d asymmetric")
    func upsample2dAsymmetric() {
        let up = nn.Upsample2d(size: (2, 3))
        let input = Tensor<Float>.zeros([8, 10, 10, 128])

        let output = up(input)

        #expect(output.shape == [8, 20, 30, 128])
    }

    @Test("Upsample2d no parameters")
    func upsample2dNoParameters() {
        let up = nn.Upsample2d(size: 2)
        #expect(up.parameters().count == 0)
    }

    // UpsamplingNearest2d alias test
    @Test("UpsamplingNearest2d alias")
    func upsamplingNearest2dAlias() {
        let up: nn.UpsamplingNearest2d = nn.Upsample2d(size: 2)
        let input = Tensor<Float>.zeros([32, 14, 14, 64])

        let output = up(input)

        #expect(output.shape == [32, 28, 28, 64])
    }
}

// MARK: - Activation Layer Tests (Ported)

@Suite("Ported Activation Tests")
struct PortedActivationTests {

    // SELU Tests

    @Test("SELU layer")
    func seluLayer() {
        let selu = nn.SELU()
        let input = Tensor<Float>.zeros([32, 256])

        let output = selu(input)

        #expect(output.shape == [32, 256])
    }

    @Test("SELU no parameters")
    func seluNoParameters() {
        let selu = nn.SELU()
        #expect(selu.parameters().count == 0)
    }

    // Mish Tests

    @Test("Mish layer")
    func mishLayer() {
        let mish = nn.Mish()
        let input = Tensor<Float>.zeros([32, 256])

        let output = mish(input)

        #expect(output.shape == [32, 256])
    }

    @Test("Mish no parameters")
    func mishNoParameters() {
        let mish = nn.Mish()
        #expect(mish.parameters().count == 0)
    }

    // Softplus Tests

    @Test("Softplus layer")
    func softplusLayer() {
        let softplus = nn.Softplus()
        let input = Tensor<Float>.zeros([32, 256])

        let output = softplus(input)

        #expect(output.shape == [32, 256])
    }

    @Test("Softplus with beta")
    func softplusWithBeta() {
        let softplus = nn.Softplus(beta: 2.0)
        #expect(softplus.beta == 2.0)
    }

    @Test("Softplus no parameters")
    func softplusNoParameters() {
        let softplus = nn.Softplus()
        #expect(softplus.parameters().count == 0)
    }

    // Softsign Tests

    @Test("Softsign layer")
    func softsignLayer() {
        let softsign = nn.Softsign()
        let input = Tensor<Float>.zeros([32, 256])

        let output = softsign(input)

        #expect(output.shape == [32, 256])
    }

    @Test("Softsign no parameters")
    func softsignNoParameters() {
        let softsign = nn.Softsign()
        #expect(softsign.parameters().count == 0)
    }

    // PReLU Tests

    @Test("PReLU layer")
    func preluLayer() {
        let prelu = nn.PReLU()
        let input = Tensor<Float>.zeros([32, 256])

        let output = prelu(input)

        #expect(output.shape == [32, 256])
    }

    @Test("PReLU with num parameters")
    func preluWithNumParameters() {
        let prelu = nn.PReLU(numParameters: 64)
        #expect(prelu.numParameters == 64)
    }

    @Test("PReLU has parameters")
    func preluHasParameters() {
        let prelu = nn.PReLU()
        // PReLU has learnable parameters
        #expect(prelu.parameters().count == 1)
    }

    @Test("PReLU weight shape")
    func preluWeightShape() {
        let prelu1 = nn.PReLU(numParameters: 1)
        let prelu64 = nn.PReLU(numParameters: 64)

        #expect(prelu1.weight.shape == [1])
        #expect(prelu64.weight.shape == [64])
    }
}

// MARK: - Normalization Layer Tests (Ported)

@Suite("Ported Normalization Tests")
struct PortedNormalizationTests {

    // GroupNorm Tests

    @Test("GroupNorm creation")
    func groupNormCreation() {
        let gn = nn.GroupNorm(numGroups: 8, numChannels: 64)

        #expect(gn.numGroups == 8)
        #expect(gn.numChannels == 64)
        #expect(gn.affine)
    }

    @Test("GroupNorm forward")
    func groupNormForward() {
        let gn = nn.GroupNorm(numGroups: 8, numChannels: 64)
        let input = Tensor<Float>.zeros([32, 14, 14, 64])  // [batch, height, width, channels]

        let output = gn(input)

        #expect(output.shape == [32, 14, 14, 64])
    }

    @Test("GroupNorm parameters")
    func groupNormParameters() {
        let gnAffine = nn.GroupNorm(numGroups: 8, numChannels: 64, affine: true)
        let gnNoAffine = nn.GroupNorm(numGroups: 8, numChannels: 64, affine: false)

        #expect(gnAffine.parameters().count == 2)  // weight and bias
        #expect(gnNoAffine.parameters().count == 0)
    }

    @Test("GroupNorm weight shape")
    func groupNormWeightShape() {
        let gn = nn.GroupNorm(numGroups: 8, numChannels: 64)

        #expect(gn.weight.shape == [64])
        #expect(gn.bias.shape == [64])
    }

    // InstanceNorm2d Tests

    @Test("InstanceNorm2d creation")
    func instanceNorm2dCreation() {
        let inorm = nn.InstanceNorm2d(numFeatures: 64)

        #expect(inorm.numFeatures == 64)
        #expect(inorm.affine)
    }

    @Test("InstanceNorm2d forward")
    func instanceNorm2dForward() {
        let inorm = nn.InstanceNorm2d(numFeatures: 64)
        let input = Tensor<Float>.zeros([32, 14, 14, 64])

        let output = inorm(input)

        #expect(output.shape == [32, 14, 14, 64])
    }

    @Test("InstanceNorm2d parameters")
    func instanceNorm2dParameters() {
        let inormAffine = nn.InstanceNorm2d(numFeatures: 64, affine: true)
        let inormNoAffine = nn.InstanceNorm2d(numFeatures: 64, affine: false)

        #expect(inormAffine.parameters().count == 2)
        #expect(inormNoAffine.parameters().count == 0)
    }
}
