// Magma - Transformer Component Tests
// Tests for LayerNorm, TransformerEncoderLayer, and Positional Encoding

import Testing
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - LayerNorm Tests

@Suite("LayerNorm Tests")
struct LayerNormTests {

    @Test("LayerNorm creation")
    func layerNormCreation() {
        let ln = nn.LayerNorm(512)
        #expect(ln.normalizedShape == [512])
        #expect(ln.eps == 1e-5)
        #expect(ln.elementwiseAffine)
    }

    @Test("LayerNorm creation multi dim")
    func layerNormCreationMultiDim() {
        let ln = nn.LayerNorm(normalizedShape: [10, 512])
        #expect(ln.normalizedShape == [10, 512])
    }

    @Test("LayerNorm parameters")
    func layerNormParameters() {
        let ln = nn.LayerNorm(256)
        let params = ln.parameters()
        #expect(params.count == 2)  // weight and bias
        #expect(params[0].shape == [256])  // weight (gamma)
        #expect(params[1].shape == [256])  // bias (beta)
    }

    @Test("LayerNorm no affine")
    func layerNormNoAffine() {
        let ln = nn.LayerNorm(256, elementwiseAffine: false)
        let params = ln.parameters()
        #expect(params.count == 0)  // No learnable parameters
    }

    @Test("LayerNorm forward 2D")
    func layerNormForward2D() {
        let ln = nn.LayerNorm(64)
        let x = Tensor<Float>.randn([32, 64])
        let y = ln(x)
        #expect(y.shape == [32, 64])
    }

    @Test("LayerNorm forward 3D")
    func layerNormForward3D() {
        let ln = nn.LayerNorm(512)
        let x = Tensor<Float>.randn([4, 16, 512])
        let y = ln(x)
        #expect(y.shape == [4, 16, 512])
    }

    @Test("LayerNorm forward 4D")
    func layerNormForward4D() {
        // Normalize over last dimension only
        let ln = nn.LayerNorm(64)
        let x = Tensor<Float>.randn([2, 8, 16, 64])
        let y = ln(x)
        #expect(y.shape == [2, 8, 16, 64])
    }

    @Test("LayerNorm multiple dims")
    func layerNormMultipleDims() {
        // Normalize over last 2 dimensions
        let ln = nn.LayerNorm(normalizedShape: [16, 64])
        let x = Tensor<Float>.randn([2, 8, 16, 64])
        let y = ln(x)
        #expect(y.shape == [2, 8, 16, 64])
    }
}

// MARK: - Mean and Variance Tests

@Suite("Mean Variance Tests")
struct MeanVarianceTests {

    @Test("Mean with dims")
    func meanWithDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let mean = x.mean(dims: [2], keepDims: true)
        #expect(mean.shape == [4, 8, 1])
    }

    @Test("Mean with dims no keep")
    func meanWithDimsNoKeep() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let mean = x.mean(dims: [2], keepDims: false)
        #expect(mean.shape == [4, 8])
    }

    @Test("Mean multiple dims")
    func meanMultipleDims() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let mean = x.mean(dims: [1, 2], keepDims: true)
        #expect(mean.shape == [4, 1, 1])
    }

    @Test("Variance with dims")
    func varianceWithDims() {
        let x = Tensor<Float>.randn([4, 8, 16])
        let variance = x.variance(dims: [2], keepDims: true)
        #expect(variance.shape == [4, 8, 1])
    }

    @Test("Variance with dims no keep")
    func varianceWithDimsNoKeep() {
        let x = Tensor<Float>.randn([4, 8, 16])
        let variance = x.variance(dims: [2], keepDims: false)
        #expect(variance.shape == [4, 8])
    }

    @Test("Variance multiple dims")
    func varianceMultipleDims() {
        let x = Tensor<Float>.randn([4, 8, 16])
        let variance = x.variance(dims: [1, 2], keepDims: true)
        #expect(variance.shape == [4, 1, 1])
    }

    @Test("Mean negative index")
    func meanNegativeIndex() {
        let x = Tensor<Float>.ones([4, 8, 16])
        let mean = x.mean(dims: [-1], keepDims: true)
        #expect(mean.shape == [4, 8, 1])
    }

    @Test("Variance negative index")
    func varianceNegativeIndex() {
        let x = Tensor<Float>.randn([4, 8, 16])
        let variance = x.variance(dims: [-1], keepDims: true)
        #expect(variance.shape == [4, 8, 1])
    }
}

// MARK: - Transformer Encoder Layer Tests

@Suite("Transformer Encoder Layer Tests")
struct TransformerEncoderLayerTests {

    @Test("TransformerEncoderLayer creation")
    func transformerEncoderLayerCreation() {
        let layer = nn.TransformerEncoderLayer(dModel: 512, nHead: 8)
        #expect(layer.dModel == 512)
        #expect(layer.nHead == 8)
        #expect(layer.dimFeedforward == 2048)  // 4 * dModel
    }

    @Test("TransformerEncoderLayer custom FFN")
    func transformerEncoderLayerCustomFFN() {
        let layer = nn.TransformerEncoderLayer(dModel: 256, nHead: 4, dimFeedforward: 1024)
        #expect(layer.dModel == 256)
        #expect(layer.dimFeedforward == 1024)
    }

    @Test("TransformerEncoderLayer parameters")
    func transformerEncoderLayerParameters() {
        let layer = nn.TransformerEncoderLayer(dModel: 128, nHead: 4)
        let params = layer.parameters()

        // Self-attention: 4 weights + 4 biases = 8
        // Norm1: 2 (weight + bias)
        // Norm2: 2 (weight + bias)
        // Linear1: 2 (weight + bias)
        // Linear2: 2 (weight + bias)
        // Total: 8 + 2 + 2 + 2 + 2 = 16
        #expect(params.count == 16)
    }

    @Test("TransformerEncoderLayer forward")
    func transformerEncoderLayerForward() {
        let layer = nn.TransformerEncoderLayer(dModel: 128, nHead: 4)
        let x = Tensor<Float>.randn([2, 16, 128])  // [batch, seq, embed]
        let y = layer(x)
        #expect(y.shape == [2, 16, 128])
    }

    @Test("TransformerEncoderLayer pre-LN")
    func transformerEncoderLayerPreLN() {
        let layer = nn.TransformerEncoderLayer(dModel: 128, nHead: 4, normFirst: true)
        #expect(layer.normFirst)
        let x = Tensor<Float>.randn([2, 16, 128])
        let y = layer(x)
        #expect(y.shape == [2, 16, 128])
    }

    @Test("TransformerEncoderLayer GELU")
    func transformerEncoderLayerGELU() {
        let layer = nn.TransformerEncoderLayer(dModel: 128, nHead: 4, activation: .gelu)
        let x = Tensor<Float>.randn([2, 16, 128])
        let y = layer(x)
        #expect(y.shape == [2, 16, 128])
    }

    @Test("TransformerEncoderLayer with mask")
    func transformerEncoderLayerWithMask() {
        let layer = nn.TransformerEncoderLayer(dModel: 128, nHead: 4)
        let x = Tensor<Float>.randn([2, 16, 128])
        let mask = Tensor<Float>.ones([2, 4, 16, 16])  // [batch, heads, seq, seq]
        let y = layer.forward(src: x, srcMask: mask)
        #expect(y.shape == [2, 16, 128])
    }
}

// MARK: - Positional Encoding Tests

@Suite("Positional Encoding Tests")
struct PositionalEncodingTests {

    @Test("SinusoidalPositionalEncoding creation")
    func sinusoidalPositionalEncodingCreation() {
        let pe = nn.SinusoidalPositionalEncoding(dModel: 512, maxLen: 1000)
        #expect(pe.dModel == 512)
        #expect(pe.maxLen == 1000)
        #expect(pe.pe.shape == [1, 1000, 512])
    }

    @Test("SinusoidalPositionalEncoding forward")
    func sinusoidalPositionalEncodingForward() {
        let pe = nn.SinusoidalPositionalEncoding(dModel: 256, maxLen: 500)
        let x = Tensor<Float>.randn([4, 100, 256])  // seq < maxLen
        let y = pe(x)
        #expect(y.shape == [4, 100, 256])
    }

    @Test("SinusoidalPositionalEncoding full length")
    func sinusoidalPositionalEncodingFullLength() {
        let pe = nn.SinusoidalPositionalEncoding(dModel: 64, maxLen: 50)
        let x = Tensor<Float>.randn([2, 50, 64])  // seq == maxLen
        let y = pe(x)
        #expect(y.shape == [2, 50, 64])
    }

    @Test("SinusoidalPositionalEncoding no parameters")
    func sinusoidalPositionalEncodingNoParameters() {
        let pe = nn.SinusoidalPositionalEncoding(dModel: 128)
        let params = pe.parameters()
        #expect(params.count == 0)  // Sinusoidal PE has no learned parameters
    }

    @Test("LearnedPositionalEmbedding creation")
    func learnedPositionalEmbeddingCreation() {
        let pe = nn.LearnedPositionalEmbedding(dModel: 512, maxLen: 512)
        #expect(pe.dModel == 512)
        #expect(pe.maxLen == 512)
    }

    @Test("LearnedPositionalEmbedding forward")
    func learnedPositionalEmbeddingForward() {
        let pe = nn.LearnedPositionalEmbedding(dModel: 256, maxLen: 200)
        let x = Tensor<Float>.randn([4, 100, 256])
        let y = pe(x)
        #expect(y.shape == [4, 100, 256])
    }

    @Test("LearnedPositionalEmbedding parameters")
    func learnedPositionalEmbeddingParameters() {
        let pe = nn.LearnedPositionalEmbedding(dModel: 128, maxLen: 100)
        let params = pe.parameters()
        #expect(params.count == 1)  // Just the embedding
        #expect(params[0].shape == [1, 100, 128])
    }
}

// MARK: - Transformer Decoder Layer Tests

@Suite("Transformer Decoder Layer Tests")
struct TransformerDecoderLayerTests {

    @Test("TransformerDecoderLayer creation")
    func transformerDecoderLayerCreation() {
        let layer = nn.TransformerDecoderLayer(dModel: 512, nHead: 8)
        #expect(layer.dModel == 512)
        #expect(layer.nHead == 8)
        #expect(layer.dimFeedforward == 2048)  // 4 * dModel
    }

    @Test("TransformerDecoderLayer custom FFN")
    func transformerDecoderLayerCustomFFN() {
        let layer = nn.TransformerDecoderLayer(dModel: 256, nHead: 4, dimFeedforward: 1024)
        #expect(layer.dModel == 256)
        #expect(layer.dimFeedforward == 1024)
    }

    @Test("TransformerDecoderLayer parameters")
    func transformerDecoderLayerParameters() {
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
        #expect(params.count == 26)
    }

    @Test("TransformerDecoderLayer forward")
    func transformerDecoderLayerForward() {
        let layer = nn.TransformerDecoderLayer(dModel: 128, nHead: 4)
        let tgt = Tensor<Float>.randn([2, 10, 128])  // [batch, tgt_seq, embed]
        let memory = Tensor<Float>.randn([2, 16, 128])  // [batch, src_seq, embed]
        let y = layer(tgt: tgt, memory: memory)
        #expect(y.shape == [2, 10, 128])
    }

    @Test("TransformerDecoderLayer pre-LN")
    func transformerDecoderLayerPreLN() {
        let layer = nn.TransformerDecoderLayer(dModel: 128, nHead: 4, normFirst: true)
        #expect(layer.normFirst)
        let tgt = Tensor<Float>.randn([2, 10, 128])
        let memory = Tensor<Float>.randn([2, 16, 128])
        let y = layer(tgt: tgt, memory: memory)
        #expect(y.shape == [2, 10, 128])
    }

    @Test("TransformerDecoderLayer GELU")
    func transformerDecoderLayerGELU() {
        let layer = nn.TransformerDecoderLayer(dModel: 128, nHead: 4, activation: .gelu)
        let tgt = Tensor<Float>.randn([2, 10, 128])
        let memory = Tensor<Float>.randn([2, 16, 128])
        let y = layer(tgt: tgt, memory: memory)
        #expect(y.shape == [2, 10, 128])
    }

    @Test("TransformerDecoderLayer with causal mask")
    func transformerDecoderLayerWithCausalMask() {
        let layer = nn.TransformerDecoderLayer(dModel: 128, nHead: 4)
        let tgt = Tensor<Float>.randn([2, 10, 128])
        let memory = Tensor<Float>.randn([2, 16, 128])
        let causalMask = nn.generateCausalMask(size: 10)  // Causal mask for target
        let y = layer.forward(tgt: tgt, memory: memory, tgtMask: causalMask)
        #expect(y.shape == [2, 10, 128])
    }

    @Test("TransformerDecoderLayer with memory mask")
    func transformerDecoderLayerWithMemoryMask() {
        let layer = nn.TransformerDecoderLayer(dModel: 128, nHead: 4)
        let tgt = Tensor<Float>.randn([2, 10, 128])
        let memory = Tensor<Float>.randn([2, 16, 128])
        let memoryMask = Tensor<Float>.zeros([10, 16])  // Attend to all encoder positions
        let y = layer.forward(tgt: tgt, memory: memory, memoryMask: memoryMask)
        #expect(y.shape == [2, 10, 128])
    }

    @Test("TransformerDecoderLayer different seq lens")
    func transformerDecoderLayerDifferentSeqLens() {
        // Test with different source and target sequence lengths
        let layer = nn.TransformerDecoderLayer(dModel: 64, nHead: 2)
        let tgt = Tensor<Float>.randn([4, 8, 64])   // shorter target
        let memory = Tensor<Float>.randn([4, 32, 64])  // longer source
        let y = layer(tgt: tgt, memory: memory)
        #expect(y.shape == [4, 8, 64])
    }
}

// MARK: - Causal Mask Tests

@Suite("Causal Mask Tests")
struct CausalMaskTests {

    @Test("Generate causal mask shape")
    func generateCausalMaskShape() {
        let mask = nn.generateCausalMask(size: 10)
        #expect(mask.shape == [10, 10])
    }

    @Test("Generate causal mask small")
    func generateCausalMaskSmall() {
        let mask = nn.generateCausalMask(size: 4)
        #expect(mask.shape == [4, 4])
        // The mask should be lower triangular with 0s (attend) and -inf (don't attend)
    }

    @Test("Generate causal mask large")
    func generateCausalMaskLarge() {
        let mask = nn.generateCausalMask(size: 512)
        #expect(mask.shape == [512, 512])
    }
}

// MARK: - Integration Tests

@Suite("Transformer Integration Tests")
struct TransformerIntegrationTests {

    @Test("Transformer encoder stack")
    func transformerEncoderStack() {
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
        #expect(x.shape == [2, 16, dModel])
    }

    @Test("Transformer with positional encoding")
    func transformerWithPositionalEncoding() {
        let dModel = 128
        let seqLen = 32
        let batch = 4

        let posEnc = nn.SinusoidalPositionalEncoding(dModel: dModel, maxLen: 100)
        let encoder = nn.TransformerEncoderLayer(dModel: dModel, nHead: 4)

        var x = Tensor<Float>.randn([batch, seqLen, dModel])
        x = posEnc(x)  // Add positional encoding
        let y = encoder(x)

        #expect(y.shape == [batch, seqLen, dModel])
    }

    @Test("LayerNorm in transformer")
    func layerNormInTransformer() {
        // Test that LayerNorm works correctly within transformer
        let dModel = 64
        let ln = nn.LayerNorm(dModel)
        let encoder = nn.TransformerEncoderLayer(dModel: dModel, nHead: 2)

        let x = Tensor<Float>.randn([2, 8, dModel])
        let normalized = ln(x)
        let encoded = encoder(normalized)

        #expect(encoded.shape == [2, 8, dModel])
    }

    @Test("Pre-LN transformer encoder")
    func preLNTransformerEncoder() {
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

        #expect(x.shape == [2, 16, dModel])
    }

    @Test("Transformer parameter count")
    func transformerParameterCount() {
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

        #expect(totalParams > 700000)  // Sanity check
        #expect(totalParams < 900000)
    }

    @Test("Full encoder-decoder transformer")
    func fullEncoderDecoderTransformer() {
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

        #expect(tgt.shape == [2, 10, dModel])
    }

    @Test("Decoder stack")
    func decoderStack() {
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
        #expect(tgt.shape == [2, 16, dModel])
    }

    @Test("Decoder parameter count")
    func decoderParameterCount() {
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

        #expect(totalParams > 1000000)  // Sanity check
        #expect(totalParams < 1200000)
    }
}
