// Magma - Attention Tests
// Tests for MultiheadAttention and related components

import XCTest
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - Tensor Operation Tests

final class TensorTransposeTests: XCTestCase {

    func testTransposeWithDims() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let result = x.transpose(0, 2)
        XCTAssertEqual(result.shape, [4, 3, 2])
    }

    func testTransposeLastTwo() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let result = x.transposeLastTwo()
        XCTAssertEqual(result.shape, [2, 4, 3])
    }

    func testTransposeLastTwo4D() {
        let x = Tensor<Float>.ones([2, 8, 10, 64])
        let result = x.transposeLastTwo()
        XCTAssertEqual(result.shape, [2, 8, 64, 10])
    }

    func testTransposeNegativeIndices() {
        let x = Tensor<Float>.ones([2, 3, 4])
        let result = x.transpose(-2, -1)
        XCTAssertEqual(result.shape, [2, 4, 3])
    }
}

// MARK: - Batched MatMul Tests

final class BatchedMatmulTests: XCTestCase {

    func testBatchedMatmul3D() {
        let a = Tensor<Float>.ones([4, 10, 32])
        let b = Tensor<Float>.ones([4, 32, 64])
        let result = a.batchedMatmul(b)
        XCTAssertEqual(result.shape, [4, 10, 64])
    }

    func testBatchedMatmul4D() {
        let a = Tensor<Float>.ones([2, 8, 10, 64])
        let b = Tensor<Float>.ones([2, 8, 64, 10])
        let result = a.batchedMatmul(b)
        XCTAssertEqual(result.shape, [2, 8, 10, 10])
    }

    func testBatchedMatmulAttentionPattern() {
        // Typical attention pattern: [batch, heads, seq, dim] @ [batch, heads, dim, seq]
        let batch = 4
        let heads = 8
        let seqLen = 32
        let headDim = 64

        let q = Tensor<Float>.randn([batch, heads, seqLen, headDim])
        let k = Tensor<Float>.randn([batch, heads, seqLen, headDim])

        // K^T
        let kT = k.transposeLastTwo()
        XCTAssertEqual(kT.shape, [batch, heads, headDim, seqLen])

        // Q @ K^T -> attention scores
        let scores = q.batchedMatmul(kT)
        XCTAssertEqual(scores.shape, [batch, heads, seqLen, seqLen])
    }
}

// MARK: - Masked Fill Tests

final class MaskedFillTests: XCTestCase {

    func testMaskedFill() {
        let x = Tensor<Float>.ones([2, 3])
        let mask = Tensor<Float>.zeros([2, 3])
        let fillValue = Tensor<Float>.full([2, 3], -1e9, on: .default)
        let result = x.maskedFill(mask: mask, value: fillValue)
        XCTAssertEqual(result.shape, [2, 3])
    }
}

// MARK: - Scaled Dot-Product Attention Tests

final class ScaledDotProductAttentionTests: XCTestCase {

    func testScaledDotProductAttention3D() {
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

        XCTAssertEqual(output.shape, [batch, seqLen, embedDim])
        XCTAssertEqual(weights.shape, [batch, seqLen, seqLen])
    }

    func testScaledDotProductAttention4D() {
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

        XCTAssertEqual(output.shape, [batch, numHeads, seqLen, headDim])
        XCTAssertEqual(weights.shape, [batch, numHeads, seqLen, seqLen])
    }

    func testScaledDotProductAttentionWithMask() {
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

        XCTAssertEqual(output.shape, [batch, numHeads, seqLen, headDim])
        XCTAssertEqual(weights.shape, [batch, numHeads, seqLen, seqLen])
    }
}

// MARK: - MultiheadAttention Tests

final class MultiheadAttentionTests: XCTestCase {

    func testMultiheadAttentionCreation() {
        let attention = nn.MultiheadAttention(embedDim: 512, numHeads: 8)

        XCTAssertEqual(attention.embedDim, 512)
        XCTAssertEqual(attention.numHeads, 8)
        XCTAssertEqual(attention.headDim, 64)
    }

    func testMultiheadAttentionParameters() {
        let attention = nn.MultiheadAttention(embedDim: 256, numHeads: 4)
        let params = attention.parameters()

        // 4 weight matrices + 4 bias vectors = 8 parameters
        XCTAssertEqual(params.count, 8)

        // Check weight shapes
        XCTAssertEqual(attention.wQ.shape, [256, 256])
        XCTAssertEqual(attention.wK.shape, [256, 256])
        XCTAssertEqual(attention.wV.shape, [256, 256])
        XCTAssertEqual(attention.wO.shape, [256, 256])
    }

    func testMultiheadAttentionParametersNoBias() {
        let attention = nn.MultiheadAttention(embedDim: 256, numHeads: 4, bias: false)
        let params = attention.parameters()

        // 4 weight matrices, no biases
        XCTAssertEqual(params.count, 4)
    }

    func testMultiheadAttentionSelfAttention() {
        let attention = nn.MultiheadAttention(embedDim: 128, numHeads: 4)
        let batch = 2
        let seqLen = 10

        let x = Tensor<Float>.randn([batch, seqLen, 128])

        // Self-attention: query = key = value = x
        let output = attention(x)

        XCTAssertEqual(output.shape, [batch, seqLen, 128])
    }

    func testMultiheadAttentionForward() {
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

        XCTAssertEqual(output.shape, [batch, seqLen, 256])
        XCTAssertEqual(weights.shape, [batch, 8, seqLen, seqLen])
    }

    func testMultiheadAttentionCrossAttention() {
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

        XCTAssertEqual(output.shape, [batch, 10, 512])
        XCTAssertEqual(weights.shape, [batch, 8, 10, 20])
    }
}

// MARK: - Integration Tests

final class AttentionIntegrationTests: XCTestCase {

    func testTransformerEncoderBlockPattern() {
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

        XCTAssertEqual(output.shape, [batch, seqLen, embedDim])
    }

    func testAttentionWithDifferentSequenceLengths() {
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

        XCTAssertEqual(output.shape, [2, 8, 128])
        XCTAssertEqual(weights.shape, [2, 4, 8, 32])
    }
}
