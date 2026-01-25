// Magma - Attention Tests
// Tests for MultiheadAttention and related components

import Testing
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - Tensor Operation Tests

@Suite("Tensor Transpose Tests")
struct TensorTransposeTests {

    @Test("Transpose with specific dimensions")
    func transposeWithDims() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let result = x.transpose(0, 2)
        #expect(result.shape == [4, 3, 2])
    }

    @Test("Transpose last two dimensions")
    func transposeLastTwo() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let result = x.transposeLastTwo()
        #expect(result.shape == [2, 4, 3])
    }

    @Test("Transpose last two dimensions 4D")
    func transposeLastTwo4D() {
        let x = Tensor<Float>.ones([2, 8, 10, 64])
        let result = x.transposeLastTwo()
        #expect(result.shape == [2, 8, 64, 10])
    }

    @Test("Transpose with negative indices")
    func transposeNegativeIndices() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let result = x.transpose(-2, -1)
        #expect(result.shape == [2, 4, 3])
    }
}

// MARK: - Batched MatMul Tests

@Suite("Batched Matmul Tests")
struct BatchedMatmulTests {

    @Test("Batched matmul 3D")
    func batchedMatmul3D() {
        let a = Tensor<Float>.ones([4, 10, 32])
        let b = Tensor<Float>.ones([4, 32, 64])
        let result = a.batchedMatmul(b)
        #expect(result.shape == [4, 10, 64])
    }

    @Test("Batched matmul 4D")
    func batchedMatmul4D() {
        let a = Tensor<Float>.ones([2, 8, 10, 64])
        let b = Tensor<Float>.ones([2, 8, 64, 10])
        let result = a.batchedMatmul(b)
        #expect(result.shape == [2, 8, 10, 10])
    }

    @Test("Batched matmul attention pattern")
    func batchedMatmulAttentionPattern() {
        // Typical attention pattern: [batch, heads, seq, dim] @ [batch, heads, dim, seq]
        let batch = 4
        let heads = 8
        let seqLen = 32
        let headDim = 64

        let q = Tensor<Float>.randn([batch, heads, seqLen, headDim])
        let k = Tensor<Float>.randn([batch, heads, seqLen, headDim])

        // K^T
        let kT = k.transposeLastTwo()
        #expect(kT.shape == [batch, heads, headDim, seqLen])

        // Q @ K^T -> attention scores
        let scores = q.batchedMatmul(kT)
        #expect(scores.shape == [batch, heads, seqLen, seqLen])
    }
}

// MARK: - Masked Fill Tests

@Suite("Masked Fill Tests")
struct MaskedFillTests {

    @Test("Masked fill operation")
    func maskedFill() {
        let x = Tensor<Float>.ones([2, 3])
        let mask = Tensor<Float>.zeros([2, 3])
        let fillValue = Tensor<Float>.full([2, 3], -1e9, on: .default)
        let result = x.maskedFill(mask: mask, value: fillValue)
        #expect(result.shape == [2, 3])
    }
}

// MARK: - Scaled Dot-Product Attention Tests

@Suite("Scaled Dot-Product Attention Tests")
struct ScaledDotProductAttentionTests {

    @Test("Scaled dot-product attention 3D")
    func scaledDotProductAttention3D() {
        let batch = 4
        let seqLen = 16
        let embedDim = 64

        let query = Tensor<Float>.randn([batch, seqLen, embedDim])
        let key = Tensor<Float>.randn([batch, seqLen, embedDim])
        let value = Tensor<Float>.randn([batch, seqLen, embedDim])

        let (output, weights) = nn.scaledDotProductAttention(
            query: query,
            key: key,
            value: value
        )

        #expect(output.shape == [batch, seqLen, embedDim])
        #expect(weights.shape == [batch, seqLen, seqLen])
    }

    @Test("Scaled dot-product attention 4D")
    func scaledDotProductAttention4D() {
        // With multiple heads
        let batch = 4
        let numHeads = 8
        let seqLen = 16
        let headDim = 64

        let query = Tensor<Float>.randn([batch, numHeads, seqLen, headDim])
        let key = Tensor<Float>.randn([batch, numHeads, seqLen, headDim])
        let value = Tensor<Float>.randn([batch, numHeads, seqLen, headDim])

        let (output, weights) = nn.scaledDotProductAttention(
            query: query,
            key: key,
            value: value
        )

        #expect(output.shape == [batch, numHeads, seqLen, headDim])
        #expect(weights.shape == [batch, numHeads, seqLen, seqLen])
    }

    @Test("Scaled dot-product attention with mask")
    func scaledDotProductAttentionWithMask() {
        let batch = 2
        let numHeads = 4
        let seqLen = 8
        let headDim = 32

        let query = Tensor<Float>.randn([batch, numHeads, seqLen, headDim])
        let key = Tensor<Float>.randn([batch, numHeads, seqLen, headDim])
        let value = Tensor<Float>.randn([batch, numHeads, seqLen, headDim])

        // Create a causal mask (upper triangular = masked)
        let mask = Tensor<Float>.ones([batch, numHeads, seqLen, seqLen])

        let (output, weights) = nn.scaledDotProductAttention(
            query: query,
            key: key,
            value: value,
            mask: mask
        )

        #expect(output.shape == [batch, numHeads, seqLen, headDim])
        #expect(weights.shape == [batch, numHeads, seqLen, seqLen])
    }
}

// MARK: - MultiheadAttention Tests

@Suite("Multihead Attention Tests")
struct MultiheadAttentionTests {

    @Test("Creation")
    func multiheadAttentionCreation() {
        let attention = nn.MultiheadAttention(embedDim: 512, numHeads: 8)

        #expect(attention.embedDim == 512)
        #expect(attention.numHeads == 8)
        #expect(attention.headDim == 64)
    }

    @Test("Parameters")
    func multiheadAttentionParameters() {
        let attention = nn.MultiheadAttention(embedDim: 256, numHeads: 4)
        let params = attention.parameters()

        // 4 weight matrices + 4 bias vectors = 8 parameters
        #expect(params.count == 8)

        // Check weight shapes
        #expect(attention.wQ.shape == [256, 256])
        #expect(attention.wK.shape == [256, 256])
        #expect(attention.wV.shape == [256, 256])
        #expect(attention.wO.shape == [256, 256])
    }

    @Test("Parameters without bias")
    func multiheadAttentionParametersNoBias() {
        let attention = nn.MultiheadAttention(embedDim: 256, numHeads: 4, bias: false)
        let params = attention.parameters()

        // 4 weight matrices, no biases
        #expect(params.count == 4)
    }

    @Test("Self attention")
    func multiheadAttentionSelfAttention() {
        let attention = nn.MultiheadAttention(embedDim: 128, numHeads: 4)
        let batch = 2
        let seqLen = 10

        let x = Tensor<Float>.randn([batch, seqLen, 128])

        // Self-attention: query = key = value = x
        let output = attention(x)

        #expect(output.shape == [batch, seqLen, 128])
    }

    @Test("Forward pass")
    func multiheadAttentionForward() {
        let attention = nn.MultiheadAttention(embedDim: 256, numHeads: 8)
        let batch = 4
        let seqLen = 16

        let query = Tensor<Float>.randn([batch, seqLen, 256])
        let key = Tensor<Float>.randn([batch, seqLen, 256])
        let value = Tensor<Float>.randn([batch, seqLen, 256])

        let (output, weights) = attention.forward(
            query: query,
            key: key,
            value: value,
            mask: nil
        )

        #expect(output.shape == [batch, seqLen, 256])
        #expect(weights.shape == [batch, 8, seqLen, seqLen])
    }

    @Test("Cross attention")
    func multiheadAttentionCrossAttention() {
        let attention = nn.MultiheadAttention(embedDim: 512, numHeads: 8)
        let batch = 2

        // Query from decoder: [batch, tgtLen, embedDim]
        let query = Tensor<Float>.randn([batch, 10, 512])
        // Key/Value from encoder: [batch, srcLen, embedDim]
        let key = Tensor<Float>.randn([batch, 20, 512])
        let value = Tensor<Float>.randn([batch, 20, 512])

        let (output, weights) = attention.forward(
            query: query,
            key: key,
            value: value,
            mask: nil
        )

        #expect(output.shape == [batch, 10, 512])
        #expect(weights.shape == [batch, 8, 10, 20])
    }
}

// MARK: - Integration Tests

@Suite("Attention Integration Tests")
struct AttentionIntegrationTests {

    @Test("Transformer encoder block pattern")
    func transformerEncoderBlockPattern() {
        // Test a simplified transformer encoder block pattern:
        // x -> MultiheadAttention -> Add & Norm -> FFN -> Add & Norm

        let embedDim = 256
        let numHeads = 4
        let batch = 2
        let seqLen = 16

        // Self-attention layer
        let selfAttn = nn.MultiheadAttention(embedDim: embedDim, numHeads: numHeads)

        // FFN layers
        let ffn1 = nn.Linear(inputSize: embedDim, outputSize: embedDim * 4)
        let ffn2 = nn.Linear(inputSize: embedDim * 4, outputSize: embedDim)

        // Input
        let x = Tensor<Float>.randn([batch, seqLen, embedDim])

        // Self-attention with residual
        let attnOut = selfAttn(x)
        let afterAttn = x + attnOut  // Residual connection

        // FFN with residual
        // Reshape for Linear: [batch * seqLen, embedDim]
        let flatX = afterAttn.reshape([batch * seqLen, embedDim])
        let ffnMid = ffn1(flatX).relu()
        let ffnOut = ffn2(ffnMid)
        let ffnReshaped = ffnOut.reshape([batch, seqLen, embedDim])
        let output = afterAttn + ffnReshaped  // Residual connection

        #expect(output.shape == [batch, seqLen, embedDim])
    }

    @Test("Attention with different sequence lengths")
    func attentionWithDifferentSequenceLengths() {
        let attention = nn.MultiheadAttention(embedDim: 128, numHeads: 4)

        // Different sequence lengths for query vs key/value
        let query = Tensor<Float>.randn([2, 8, 128])   // tgtLen = 8
        let key = Tensor<Float>.randn([2, 32, 128])    // srcLen = 32
        let value = Tensor<Float>.randn([2, 32, 128])

        let (output, weights) = attention.forward(
            query: query,
            key: key,
            value: value,
            mask: nil
        )

        #expect(output.shape == [2, 8, 128])
        #expect(weights.shape == [2, 4, 8, 32])
    }
}
