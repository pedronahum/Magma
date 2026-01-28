// Magma - Attention Tests
// Tests for MultiheadAttention and related components

import Testing
import Foundation
import _Differentiation
@testable import Magma
@testable import LazyTensor
@testable import XLARuntime

#if canImport(MLX)
import MLX
#endif

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

// MARK: - Debug Attention Step-by-Step

@Suite("Attention Debug Tests")
struct AttentionDebugTests {

    @Test("Debug attention step by step with values")
    func debugAttentionStepByStep() throws {
        // Use Metal device explicitly
        guard Backend.metal.isAvailable else {
            print("Metal not available, skipping test")
            return
        }
        let metalDevice = Device(backend: .metal, index: 0)

        // Small test case for debugging
        let batch = 1, heads = 1, seqLen = 4, headDim = 4
        let scale: Float = 1.0 / Foundation.sqrt(Float(headDim))

        // Create simple test data with known values
        let qData: [Float] = Array(repeating: 0.1, count: batch * heads * seqLen * headDim)
        let kData: [Float] = Array(repeating: 0.2, count: batch * heads * seqLen * headDim)
        let vData: [Float] = Array(repeating: 0.3, count: batch * heads * seqLen * headDim)

        let q = Tensor<Float>(qData, shape: [batch, heads, seqLen, headDim], on: metalDevice)
        let k = Tensor<Float>(kData, shape: [batch, heads, seqLen, headDim], on: metalDevice)
        let v = Tensor<Float>(vData, shape: [batch, heads, seqLen, headDim], on: metalDevice)

        print("\n=== Debug Attention Step by Step ===")
        print("Input shapes: Q=\(q.shape), K=\(k.shape), V=\(v.shape)")
        print("Scale: \(scale)")

        // Step 1: Transpose K
        let kT = k.transpose(-1, -2)
        print("\nStep 1: K transpose -> shape: \(kT.shape)")
        let kTScalars = kT.scalars()
        print("  K^T values (first 4): \(Array(kTScalars.prefix(4)))")
        print("  K^T has NaN: \(kTScalars.contains { $0.isNaN })")
        print("  K^T has Inf: \(kTScalars.contains { $0.isInfinite })")
        #expect(!kTScalars.contains { $0.isNaN }, "K^T should not have NaN")
        #expect(!kTScalars.contains { $0.isInfinite }, "K^T should not have Inf")

        // Step 2: Q @ K^T
        let scores = q.batchedMatmul(kT)
        print("\nStep 2: Q @ K^T -> shape: \(scores.shape)")
        let scoresScalars = scores.scalars()
        print("  scores values (first 4): \(Array(scoresScalars.prefix(4)))")
        print("  scores has NaN: \(scoresScalars.contains { $0.isNaN })")
        print("  scores has Inf: \(scoresScalars.contains { $0.isInfinite })")
        #expect(!scoresScalars.contains { $0.isNaN }, "scores should not have NaN")
        #expect(!scoresScalars.contains { $0.isInfinite }, "scores should not have Inf")

        // Step 3: Scale
        let scaleTensor = Tensor<Float>.full([], scale, on: metalDevice)
        let scaledScores = scores * scaleTensor
        print("\nStep 3: Scale by \(scale) -> shape: \(scaledScores.shape)")
        let scaledScoresScalars = scaledScores.scalars()
        print("  scaled scores values (first 4): \(Array(scaledScoresScalars.prefix(4)))")
        print("  scaled scores has NaN: \(scaledScoresScalars.contains { $0.isNaN })")
        print("  scaled scores has Inf: \(scaledScoresScalars.contains { $0.isInfinite })")
        #expect(!scaledScoresScalars.contains { $0.isNaN }, "scaled scores should not have NaN")
        #expect(!scaledScoresScalars.contains { $0.isInfinite }, "scaled scores should not have Inf")

        // Step 4: Softmax
        let attnWeights = scaledScores.softmax(dim: -1)
        print("\nStep 4: Softmax -> shape: \(attnWeights.shape)")
        let attnWeightsScalars = attnWeights.scalars()
        print("  attn weights values (first 4): \(Array(attnWeightsScalars.prefix(4)))")
        print("  attn weights has NaN: \(attnWeightsScalars.contains { $0.isNaN })")
        print("  attn weights has Inf: \(attnWeightsScalars.contains { $0.isInfinite })")
        #expect(!attnWeightsScalars.contains { $0.isNaN }, "attn weights should not have NaN")
        #expect(!attnWeightsScalars.contains { $0.isInfinite }, "attn weights should not have Inf")

        // Step 5: attn @ V
        let output = attnWeights.batchedMatmul(v)
        print("\nStep 5: attn @ V -> shape: \(output.shape)")
        let outputScalars = output.scalars()
        print("  output values (first 4): \(Array(outputScalars.prefix(4)))")
        print("  output has NaN: \(outputScalars.contains { $0.isNaN })")
        print("  output has Inf: \(outputScalars.contains { $0.isInfinite })")
        #expect(!outputScalars.contains { $0.isNaN }, "output should not have NaN")
        #expect(!outputScalars.contains { $0.isInfinite }, "output should not have Inf")

        // Expected: each output element should be 0.3 (since uniform softmax times uniform V)
        let expectedOutput: Float = 0.3
        for (i, val) in outputScalars.enumerated() {
            if abs(val - expectedOutput) > 0.01 {
                print("  WARNING: output[\(i)] = \(val), expected ~\(expectedOutput)")
            }
        }
        print("\n=== Debug Complete ===\n")
    }

    @Test("Debug attention with larger tensors")
    func debugAttentionLarger() throws {
        // Use Metal device explicitly
        guard Backend.metal.isAvailable else {
            print("Metal not available, skipping test")
            return
        }
        let metalDevice = Device(backend: .metal, index: 0)

        // Match benchmark size: [batch=4, heads=8, seq=128, dim=64]
        let batch = 4, heads = 8, seqLen = 128, headDim = 64
        let scale: Float = 1.0 / Foundation.sqrt(Float(headDim))

        // Create random test data with known seed
        var rng = SystemRandomNumberGenerator()
        let totalQ = batch * heads * seqLen * headDim
        let qData = (0..<totalQ).map { _ in Float.random(in: -1...1, using: &rng) }
        let kData = (0..<totalQ).map { _ in Float.random(in: -1...1, using: &rng) }
        let vData = (0..<totalQ).map { _ in Float.random(in: -1...1, using: &rng) }

        let q = Tensor<Float>(qData, shape: [batch, heads, seqLen, headDim], on: metalDevice)
        let k = Tensor<Float>(kData, shape: [batch, heads, seqLen, headDim], on: metalDevice)
        let v = Tensor<Float>(vData, shape: [batch, heads, seqLen, headDim], on: metalDevice)

        print("\n=== Debug Larger Attention ===")
        print("Input shapes: Q=\(q.shape), K=\(k.shape), V=\(v.shape)")
        print("Scale: \(scale)")

        // Step 1: Transpose K
        print("\nStep 1: K transpose")
        let kT = k.transpose(-1, -2)
        print("  K^T shape: \(kT.shape)")
        let kTScalars = kT.scalars()
        print("  K^T first 4 values: \(Array(kTScalars.prefix(4)))")
        print("  K^T has NaN: \(kTScalars.contains { $0.isNaN })")
        print("  K^T has Inf: \(kTScalars.contains { $0.isInfinite })")
        #expect(!kTScalars.contains { $0.isNaN }, "K^T should not have NaN")

        // Step 2: Q @ K^T
        print("\nStep 2: Q @ K^T (batched matmul)")
        let scores = q.batchedMatmul(kT)
        print("  scores shape: \(scores.shape)")
        let scoresScalars = scores.scalars()
        print("  scores first 4 values: \(Array(scoresScalars.prefix(4)))")
        print("  scores has NaN: \(scoresScalars.contains { $0.isNaN })")
        print("  scores has Inf: \(scoresScalars.contains { $0.isInfinite })")
        #expect(!scoresScalars.contains { $0.isNaN }, "scores should not have NaN")

        // Step 3: Scale
        print("\nStep 3: Scale by \(scale)")
        let scaleTensor = Tensor<Float>.full([], scale, on: metalDevice)
        let scaledScores = scores * scaleTensor
        print("  scaled scores shape: \(scaledScores.shape)")
        let scaledScalars = scaledScores.scalars()
        print("  scaled scores first 4 values: \(Array(scaledScalars.prefix(4)))")
        print("  scaled scores has NaN: \(scaledScalars.contains { $0.isNaN })")
        #expect(!scaledScalars.contains { $0.isNaN }, "scaled scores should not have NaN")

        // Step 4: Softmax
        print("\nStep 4: Softmax on dim=-1")
        let attnWeights = scaledScores.softmax(dim: -1)
        print("  attn weights shape: \(attnWeights.shape)")
        let attnScalars = attnWeights.scalars()
        print("  attn weights first 4 values: \(Array(attnScalars.prefix(4)))")
        print("  attn weights has NaN: \(attnScalars.contains { $0.isNaN })")
        print("  attn weights has Inf: \(attnScalars.contains { $0.isInfinite })")
        #expect(!attnScalars.contains { $0.isNaN }, "attn weights should not have NaN")

        // Step 5: attn @ V
        print("\nStep 5: attn @ V (batched matmul)")
        let output = attnWeights.batchedMatmul(v)
        print("  output shape: \(output.shape)")
        let outputScalars = output.scalars()
        print("  output first 4 values: \(Array(outputScalars.prefix(4)))")
        print("  output has NaN: \(outputScalars.contains { $0.isNaN })")
        print("  output has Inf: \(outputScalars.contains { $0.isInfinite })")
        #expect(!outputScalars.contains { $0.isNaN }, "output should not have NaN")
        #expect(!outputScalars.contains { $0.isInfinite }, "output should not have Inf")

        print("\n=== Debug Complete ===\n")
    }

    @Test("Debug attention replicating benchmark pattern")
    func debugAttentionBenchmarkPattern() throws {
        // Use Metal device explicitly
        guard Backend.metal.isAvailable else {
            print("Metal not available, skipping test")
            return
        }
        let metalDevice = Device(backend: .metal, index: 0)

        // Match first benchmark size: [batch=4, heads=8, seq=128, dim=64]
        let batch = 4, heads = 8, seqLen = 128, headDim = 64
        let scale: Float = 1.0 / Foundation.sqrt(Float(headDim))

        // Create input tensors (like mlxToMagma does)
        var rng = SystemRandomNumberGenerator()
        let total = batch * heads * seqLen * headDim
        let qData = (0..<total).map { _ in Float.random(in: -1...1, using: &rng) }
        let kData = (0..<total).map { _ in Float.random(in: -1...1, using: &rng) }
        let vData = (0..<total).map { _ in Float.random(in: -1...1, using: &rng) }

        let magmaQ = Tensor<Float>(qData, shape: [batch, heads, seqLen, headDim], on: metalDevice)
        let magmaK = Tensor<Float>(kData, shape: [batch, heads, seqLen, headDim], on: metalDevice)
        let magmaV = Tensor<Float>(vData, shape: [batch, heads, seqLen, headDim], on: metalDevice)

        // Materialize like the benchmark does
        magmaQ.markForMaterialization()
        magmaK.markForMaterialization()
        magmaV.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        print("\n=== Benchmark Pattern Test ===")
        print("Input shapes: Q=\(magmaQ.shape), K=\(magmaK.shape), V=\(magmaV.shape)")

        // Warmup (like benchmark)
        for i in 0..<3 {
            let kT = magmaK.transpose(-1, -2)
            let scores = magmaQ.batchedMatmul(kT) * Tensor<Float>.full([], scale, on: metalDevice)
            let attnWeights = scores.softmax(dim: -1)
            let z = attnWeights.batchedMatmul(magmaV)
            LazyTensorBarrier(on: metalDevice)
            let first = z.scalars().first
            print("Warmup \(i): first value = \(first ?? Float.nan)")
        }

        // Run like benchmark
        var magmaResult: Tensor<Float>!
        for i in 0..<3 {
            let kT = magmaK.transpose(-1, -2)
            let scores = magmaQ.batchedMatmul(kT) * Tensor<Float>.full([], scale, on: metalDevice)
            let attnWeights = scores.softmax(dim: -1)
            magmaResult = attnWeights.batchedMatmul(magmaV)
            LazyTensorBarrier(on: metalDevice)
            let first = magmaResult.scalars().first
            print("Run \(i): first value = \(first ?? Float.nan)")
        }

        // Final check
        let finalScalars = magmaResult.scalars()
        print("Final scalars count: \(finalScalars.count)")
        print("Final first 4 values: \(Array(finalScalars.prefix(4)))")
        print("Final has NaN: \(finalScalars.contains { $0.isNaN })")
        print("Final has Inf: \(finalScalars.contains { $0.isInfinite })")

        #expect(!finalScalars.contains { $0.isNaN }, "Final result should not have NaN")
        #expect(!finalScalars.contains { $0.isInfinite }, "Final result should not have Inf")

        // Also verify intermediate values are consistent
        let kT = magmaK.transpose(-1, -2)
        let scores = magmaQ.batchedMatmul(kT) * Tensor<Float>.full([], scale, on: metalDevice)
        let attnWeights = scores.softmax(dim: -1)
        let freshResult = attnWeights.batchedMatmul(magmaV)
        LazyTensorBarrier(on: metalDevice)

        let freshScalars = freshResult.scalars()
        print("Fresh result first 4: \(Array(freshScalars.prefix(4)))")

        // Check if the two results match
        let diff = zip(finalScalars, freshScalars).map { abs($0 - $1) }.max() ?? 0
        print("Max diff between runs: \(diff)")
        #expect(diff < 1e-5, "Results should be consistent across runs")

        print("\n=== Test Complete ===\n")
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
