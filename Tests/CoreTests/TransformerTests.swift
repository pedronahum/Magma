// Magma - Transformer Component Tests
// Tests for LayerNorm, TransformerEncoderLayer, and Positional Encoding

import XCTest
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - LayerNorm Tests

final class LayerNormTests: XCTestCase {

    func testLayerNormCreation() {
        let ln = nn.LayerNorm(512)
        XCTAssertEqual(ln.normalizedShape, [512])
        XCTAssertEqual(ln.eps, 1e-5)
        XCTAssertTrue(ln.elementwiseAffine)
    }

    func testLayerNormCreationMultiDim() {
        let ln = nn.LayerNorm(normalizedShape: [10, 512])
        XCTAssertEqual(ln.normalizedShape, [10, 512])
    }

    func testLayerNormParameters() {
        let ln = nn.LayerNorm(256)
        let params = ln.parameters()
        XCTAssertEqual(params.count, 2)  // weight and bias
        XCTAssertEqual(params[0].shape, [256])  // weight (gamma)
        XCTAssertEqual(params[1].shape, [256])  // bias (beta)
    }

    func testLayerNormNoAffine() {
        let ln = nn.LayerNorm(256, elementwiseAffine: false)
        let params = ln.parameters()
        XCTAssertEqual(params.count, 0)  // No learnable parameters
    }

    func testLayerNormForward2D() {
        let ln = nn.LayerNorm(64)
        let x = Tensor<Float>.randn([32, 64])
        let y = ln(x)
        XCTAssertEqual(y.shape, [32, 64])
    }

    func testLayerNormForward3D() {
        let ln = nn.LayerNorm(512)
        let x = Tensor<Float>.randn([4, 16, 512])
        let y = ln(x)
        XCTAssertEqual(y.shape, [4, 16, 512])
    }

    func testLayerNormForward4D() {
        // Normalize over last dimension only
        let ln = nn.LayerNorm(64)
        let x = Tensor<Float>.randn([2, 8, 16, 64])
        let y = ln(x)
        XCTAssertEqual(y.shape, [2, 8, 16, 64])
    }

    func testLayerNormMultipleDims() {
        // Normalize over last 2 dimensions
        let ln = nn.LayerNorm(normalizedShape: [16, 64])
        let x = Tensor<Float>.randn([2, 8, 16, 64])
        let y = ln(x)
        XCTAssertEqual(y.shape, [2, 8, 16, 64])
    }
}

// MARK: - Mean and Variance Tests

final class MeanVarianceTests: XCTestCase {

    func testMeanWithDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let mean = x.mean(dims: [2], keepDims: true)
        XCTAssertEqual(mean.shape, [4, 8, 1])
    }

    func testMeanWithDimsNoKeep() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let mean = x.mean(dims: [2], keepDims: false)
        XCTAssertEqual(mean.shape, [4, 8])
    }

    func testMeanMultipleDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let mean = x.mean(dims: [1, 2], keepDims: true)
        XCTAssertEqual(mean.shape, [4, 1, 1])
    }

    func testVarianceWithDims() {
        let x = Tensor<Float>.randn([4, 8, 16])
        let variance = x.variance(dims: [2], keepDims: true)
        XCTAssertEqual(variance.shape, [4, 8, 1])
    }

    func testVarianceWithDimsNoKeep() {
        let x = Tensor<Float>.randn([4, 8, 16])
        let variance = x.variance(dims: [2], keepDims: false)
        XCTAssertEqual(variance.shape, [4, 8])
    }

    func testVarianceMultipleDims() {
        let x = Tensor<Float>.randn([4, 8, 16])
        let variance = x.variance(dims: [1, 2], keepDims: true)
        XCTAssertEqual(variance.shape, [4, 1, 1])
    }

    func testMeanNegativeIndex() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let mean = x.mean(dims: [-1], keepDims: true)
        XCTAssertEqual(mean.shape, [4, 8, 1])
    }

    func testVarianceNegativeIndex() {
        let x = Tensor<Float>.randn([4, 8, 16])
        let variance = x.variance(dims: [-1], keepDims: true)
        XCTAssertEqual(variance.shape, [4, 8, 1])
    }
}

// MARK: - Transformer Encoder Layer Tests

final class TransformerEncoderLayerTests: XCTestCase {

    func testTransformerEncoderLayerCreation() {
        let layer = nn.TransformerEncoderLayer(dModel: 512, nHead: 8)
        XCTAssertEqual(layer.dModel, 512)
        XCTAssertEqual(layer.nHead, 8)
        XCTAssertEqual(layer.dimFeedforward, 2048)  // 4 * dModel
    }

    func testTransformerEncoderLayerCustomFFN() {
        let layer = nn.TransformerEncoderLayer(dModel: 256, nHead: 4, dimFeedforward: 1024)
        XCTAssertEqual(layer.dModel, 256)
        XCTAssertEqual(layer.dimFeedforward, 1024)
    }

    func testTransformerEncoderLayerParameters() {
        let layer = nn.TransformerEncoderLayer(dModel: 128, nHead: 4)
        let params = layer.parameters()

        // Self-attention: 4 weights + 4 biases = 8
        // Norm1: 2 (weight + bias)
        // Norm2: 2 (weight + bias)
        // Linear1: 2 (weight + bias)
        // Linear2: 2 (weight + bias)
        // Total: 8 + 2 + 2 + 2 + 2 = 16
        XCTAssertEqual(params.count, 16)
    }

    func testTransformerEncoderLayerForward() {
        let layer = nn.TransformerEncoderLayer(dModel: 128, nHead: 4)
        let x = Tensor<Float>.randn([2, 16, 128])  // [batch, seq, embed]
        let y = layer(x)
        XCTAssertEqual(y.shape, [2, 16, 128])
    }

    func testTransformerEncoderLayerPreLN() {
        let layer = nn.TransformerEncoderLayer(dModel: 128, nHead: 4, normFirst: true)
        XCTAssertTrue(layer.normFirst)
        let x = Tensor<Float>.randn([2, 16, 128])
        let y = layer(x)
        XCTAssertEqual(y.shape, [2, 16, 128])
    }

    func testTransformerEncoderLayerGELU() {
        let layer = nn.TransformerEncoderLayer(dModel: 128, nHead: 4, activation: .gelu)
        let x = Tensor<Float>.randn([2, 16, 128])
        let y = layer(x)
        XCTAssertEqual(y.shape, [2, 16, 128])
    }

    func testTransformerEncoderLayerWithMask() {
        let layer = nn.TransformerEncoderLayer(dModel: 128, nHead: 4)
        let x = Tensor<Float>.randn([2, 16, 128])
        let mask = Tensor<Float>.ones([2, 4, 16, 16])  // [batch, heads, seq, seq]
        let y = layer.forward(src: x, srcMask: mask)
        XCTAssertEqual(y.shape, [2, 16, 128])
    }
}

// MARK: - Positional Encoding Tests

final class PositionalEncodingTests: XCTestCase {

    func testSinusoidalPositionalEncodingCreation() {
        let pe = nn.SinusoidalPositionalEncoding(dModel: 512, maxLen: 1000)
        XCTAssertEqual(pe.dModel, 512)
        XCTAssertEqual(pe.maxLen, 1000)
        XCTAssertEqual(pe.pe.shape, [1, 1000, 512])
    }

    func testSinusoidalPositionalEncodingForward() {
        let pe = nn.SinusoidalPositionalEncoding(dModel: 256, maxLen: 500)
        let x = Tensor<Float>.randn([4, 100, 256])  // seq < maxLen
        let y = pe(x)
        XCTAssertEqual(y.shape, [4, 100, 256])
    }

    func testSinusoidalPositionalEncodingFullLength() {
        let pe = nn.SinusoidalPositionalEncoding(dModel: 64, maxLen: 50)
        let x = Tensor<Float>.randn([2, 50, 64])  // seq == maxLen
        let y = pe(x)
        XCTAssertEqual(y.shape, [2, 50, 64])
    }

    func testSinusoidalPositionalEncodingNoParameters() {
        let pe = nn.SinusoidalPositionalEncoding(dModel: 128)
        let params = pe.parameters()
        XCTAssertEqual(params.count, 0)  // Sinusoidal PE has no learned parameters
    }

    func testLearnedPositionalEmbeddingCreation() {
        let pe = nn.LearnedPositionalEmbedding(dModel: 512, maxLen: 512)
        XCTAssertEqual(pe.dModel, 512)
        XCTAssertEqual(pe.maxLen, 512)
    }

    func testLearnedPositionalEmbeddingForward() {
        let pe = nn.LearnedPositionalEmbedding(dModel: 256, maxLen: 200)
        let x = Tensor<Float>.randn([4, 100, 256])
        let y = pe(x)
        XCTAssertEqual(y.shape, [4, 100, 256])
    }

    func testLearnedPositionalEmbeddingParameters() {
        let pe = nn.LearnedPositionalEmbedding(dModel: 128, maxLen: 100)
        let params = pe.parameters()
        XCTAssertEqual(params.count, 1)  // Just the embedding
        XCTAssertEqual(params[0].shape, [1, 100, 128])
    }
}

// MARK: - Transformer Decoder Layer Tests

final class TransformerDecoderLayerTests: XCTestCase {

    func testTransformerDecoderLayerCreation() {
        let layer = nn.TransformerDecoderLayer(dModel: 512, nHead: 8)
        XCTAssertEqual(layer.dModel, 512)
        XCTAssertEqual(layer.nHead, 8)
        XCTAssertEqual(layer.dimFeedforward, 2048)  // 4 * dModel
    }

    func testTransformerDecoderLayerCustomFFN() {
        let layer = nn.TransformerDecoderLayer(dModel: 256, nHead: 4, dimFeedforward: 1024)
        XCTAssertEqual(layer.dModel, 256)
        XCTAssertEqual(layer.dimFeedforward, 1024)
    }

    func testTransformerDecoderLayerParameters() {
        let layer = nn.TransformerDecoderLayer(dModel: 128, nHead: 4)
        let params = layer.parameters()

        // Self-attention: 4 weights + 4 biases = 8
        // Cross-attention: 4 weights + 4 biases = 8
        // Norm1: 2 (weight + bias)
        // Norm2: 2 (weight + bias)
        // Norm3: 2 (weight + bias)
        // Linear1: 2 (weight + bias)
        // Linear2: 2 (weight + bias)
        // Total: 8 + 8 + 2 + 2 + 2 + 2 + 2 = 26
        XCTAssertEqual(params.count, 26)
    }

    func testTransformerDecoderLayerForward() {
        let layer = nn.TransformerDecoderLayer(dModel: 128, nHead: 4)
        let tgt = Tensor<Float>.randn([2, 10, 128])  // [batch, tgt_seq, embed]
        let memory = Tensor<Float>.randn([2, 16, 128])  // [batch, src_seq, embed]
        let y = layer(tgt: tgt, memory: memory)
        XCTAssertEqual(y.shape, [2, 10, 128])
    }

    func testTransformerDecoderLayerPreLN() {
        let layer = nn.TransformerDecoderLayer(dModel: 128, nHead: 4, normFirst: true)
        XCTAssertTrue(layer.normFirst)
        let tgt = Tensor<Float>.randn([2, 10, 128])
        let memory = Tensor<Float>.randn([2, 16, 128])
        let y = layer(tgt: tgt, memory: memory)
        XCTAssertEqual(y.shape, [2, 10, 128])
    }

    func testTransformerDecoderLayerGELU() {
        let layer = nn.TransformerDecoderLayer(dModel: 128, nHead: 4, activation: .gelu)
        let tgt = Tensor<Float>.randn([2, 10, 128])
        let memory = Tensor<Float>.randn([2, 16, 128])
        let y = layer(tgt: tgt, memory: memory)
        XCTAssertEqual(y.shape, [2, 10, 128])
    }

    func testTransformerDecoderLayerWithCausalMask() {
        let layer = nn.TransformerDecoderLayer(dModel: 128, nHead: 4)
        let tgt = Tensor<Float>.randn([2, 10, 128])
        let memory = Tensor<Float>.randn([2, 16, 128])
        let causalMask = nn.generateCausalMask(size: 10)  // Causal mask for target
        let y = layer.forward(tgt: tgt, memory: memory, tgtMask: causalMask)
        XCTAssertEqual(y.shape, [2, 10, 128])
    }

    func testTransformerDecoderLayerWithMemoryMask() {
        let layer = nn.TransformerDecoderLayer(dModel: 128, nHead: 4)
        let tgt = Tensor<Float>.randn([2, 10, 128])
        let memory = Tensor<Float>.randn([2, 16, 128])
        let memoryMask = Tensor<Float>.zeros([10, 16])  // Attend to all encoder positions
        let y = layer.forward(tgt: tgt, memory: memory, memoryMask: memoryMask)
        XCTAssertEqual(y.shape, [2, 10, 128])
    }

    func testTransformerDecoderLayerDifferentSeqLens() {
        // Test with different source and target sequence lengths
        let layer = nn.TransformerDecoderLayer(dModel: 64, nHead: 2)
        let tgt = Tensor<Float>.randn([4, 8, 64])   // shorter target
        let memory = Tensor<Float>.randn([4, 32, 64])  // longer source
        let y = layer(tgt: tgt, memory: memory)
        XCTAssertEqual(y.shape, [4, 8, 64])
    }
}

// MARK: - Causal Mask Tests

final class CausalMaskTests: XCTestCase {

    func testGenerateCausalMaskShape() {
        let mask = nn.generateCausalMask(size: 10)
        XCTAssertEqual(mask.shape, [10, 10])
    }

    func testGenerateCausalMaskSmall() {
        let mask = nn.generateCausalMask(size: 4)
        XCTAssertEqual(mask.shape, [4, 4])
        // The mask should be lower triangular with 0s (attend) and -inf (don't attend)
    }

    func testGenerateCausalMaskLarge() {
        let mask = nn.generateCausalMask(size: 512)
        XCTAssertEqual(mask.shape, [512, 512])
    }
}

// MARK: - Integration Tests

final class TransformerIntegrationTests: XCTestCase {

    func testTransformerEncoderStack() {
        // Stack multiple encoder layers
        let dModel = 128
        let nHead = 4
        let numLayers = 3

        var layers: [nn.TransformerEncoderLayer] = []
        for _ in 0..<numLayers {
            layers.append(nn.TransformerEncoderLayer(dModel: dModel, nHead: nHead))
        }

        var x = Tensor<Float>.randn([2, 16, dModel])
        for layer in layers {
            x = layer(x)
        }
        XCTAssertEqual(x.shape, [2, 16, dModel])
    }

    func testTransformerWithPositionalEncoding() {
        let dModel = 128
        let seqLen = 32
        let batch = 4

        let posEnc = nn.SinusoidalPositionalEncoding(dModel: dModel, maxLen: 100)
        let encoder = nn.TransformerEncoderLayer(dModel: dModel, nHead: 4)

        var x = Tensor<Float>.randn([batch, seqLen, dModel])
        x = posEnc(x)  // Add positional encoding
        let y = encoder(x)

        XCTAssertEqual(y.shape, [batch, seqLen, dModel])
    }

    func testLayerNormInTransformer() {
        // Test that LayerNorm works correctly within transformer
        let dModel = 64
        let ln = nn.LayerNorm(dModel)
        let encoder = nn.TransformerEncoderLayer(dModel: dModel, nHead: 2)

        let x = Tensor<Float>.randn([2, 8, dModel])
        let normalized = ln(x)
        let encoded = encoder(normalized)

        XCTAssertEqual(encoded.shape, [2, 8, dModel])
    }

    func testPreLNTransformerEncoder() {
        // Pre-LN is often more stable for deep transformers
        let dModel = 128
        let numLayers = 6

        var layers: [nn.TransformerEncoderLayer] = []
        for _ in 0..<numLayers {
            layers.append(nn.TransformerEncoderLayer(
                dModel: dModel,
                nHead: 4,
                normFirst: true  // Pre-LN
            ))
        }

        var x = Tensor<Float>.randn([2, 16, dModel])
        for layer in layers {
            x = layer(x)
        }

        // Final layer norm for Pre-LN architecture
        let finalLN = nn.LayerNorm(dModel)
        x = finalLN(x)

        XCTAssertEqual(x.shape, [2, 16, dModel])
    }

    func testTransformerParameterCount() {
        let dModel = 256
        let nHead = 8
        let dimFF = 1024

        let encoder = nn.TransformerEncoderLayer(
            dModel: dModel,
            nHead: nHead,
            dimFeedforward: dimFF
        )

        let params = encoder.parameters()
        var totalParams = 0
        for p in params {
            totalParams += p.elementCount
        }

        // Expected:
        // Self-attention Q,K,V,O weights: 4 * dModel * dModel = 4 * 256 * 256 = 262144
        // Self-attention biases: 4 * dModel = 4 * 256 = 1024
        // LayerNorm1: 2 * dModel = 512
        // LayerNorm2: 2 * dModel = 512
        // Linear1: dModel * dimFF + dimFF = 256 * 1024 + 1024 = 263168
        // Linear2: dimFF * dModel + dModel = 1024 * 256 + 256 = 262400
        // Total ≈ 789760

        XCTAssertTrue(totalParams > 700000)  // Sanity check
        XCTAssertTrue(totalParams < 900000)
    }

    func testFullEncoderDecoderTransformer() {
        // Test a complete encoder-decoder transformer architecture
        let dModel = 128
        let nHead = 4
        let numEncoderLayers = 2
        let numDecoderLayers = 2

        // Build encoder stack
        var encoderLayers: [nn.TransformerEncoderLayer] = []
        for _ in 0..<numEncoderLayers {
            encoderLayers.append(nn.TransformerEncoderLayer(dModel: dModel, nHead: nHead))
        }

        // Build decoder stack
        var decoderLayers: [nn.TransformerDecoderLayer] = []
        for _ in 0..<numDecoderLayers {
            decoderLayers.append(nn.TransformerDecoderLayer(dModel: dModel, nHead: nHead))
        }

        // Positional encoding
        let posEnc = nn.SinusoidalPositionalEncoding(dModel: dModel, maxLen: 100)

        // Source and target sequences
        var src = Tensor<Float>.randn([2, 20, dModel])  // [batch, src_seq, embed]
        var tgt = Tensor<Float>.randn([2, 10, dModel])  // [batch, tgt_seq, embed]

        // Add positional encoding
        src = posEnc(src)
        tgt = posEnc(tgt)

        // Encode source
        for layer in encoderLayers {
            src = layer(src)
        }
        let memory = src

        // Decode target with causal mask
        let causalMask = nn.generateCausalMask(size: 10)
        for layer in decoderLayers {
            tgt = layer.forward(tgt: tgt, memory: memory, tgtMask: causalMask)
        }

        XCTAssertEqual(tgt.shape, [2, 10, dModel])
    }

    func testDecoderStack() {
        // Stack multiple decoder layers
        let dModel = 128
        let nHead = 4
        let numLayers = 3

        var layers: [nn.TransformerDecoderLayer] = []
        for _ in 0..<numLayers {
            layers.append(nn.TransformerDecoderLayer(dModel: dModel, nHead: nHead))
        }

        var tgt = Tensor<Float>.randn([2, 16, dModel])
        let memory = Tensor<Float>.randn([2, 32, dModel])

        for layer in layers {
            tgt = layer(tgt: tgt, memory: memory)
        }
        XCTAssertEqual(tgt.shape, [2, 16, dModel])
    }

    func testDecoderParameterCount() {
        let dModel = 256
        let nHead = 8
        let dimFF = 1024

        let decoder = nn.TransformerDecoderLayer(
            dModel: dModel,
            nHead: nHead,
            dimFeedforward: dimFF
        )

        let params = decoder.parameters()
        var totalParams = 0
        for p in params {
            totalParams += p.elementCount
        }

        // Expected:
        // Self-attention Q,K,V,O weights: 4 * dModel * dModel = 262144
        // Self-attention biases: 4 * dModel = 1024
        // Cross-attention Q,K,V,O weights: 4 * dModel * dModel = 262144
        // Cross-attention biases: 4 * dModel = 1024
        // LayerNorm1: 2 * dModel = 512
        // LayerNorm2: 2 * dModel = 512
        // LayerNorm3: 2 * dModel = 512
        // Linear1: dModel * dimFF + dimFF = 263168
        // Linear2: dimFF * dModel + dModel = 262400
        // Total ≈ 1053440

        XCTAssertTrue(totalParams > 1000000)  // Sanity check
        XCTAssertTrue(totalParams < 1200000)
    }
}
