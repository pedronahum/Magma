// Magma - batchedMatmul materialization.
// batchedMatmul lowers to stablehlo.dot_general, which used to be emitted in a
// form the StableHLO parser rejects: every 3D/4D batched matmul (and so every
// attention / MultiheadAttention / Transformer output) failed to compile and
// scalars() returned []. Shape-only tests never noticed.
//
// .serialized: materialization drives the PJRT backend.

import Testing
@testable import Magma

/// Host reference: [batch..., m, k] @ [batch..., k, n] with a flattened batch.
private func refBatchedMatmul(_ a: [Float], _ b: [Float], batch: Int, m: Int, k: Int, n: Int) -> [Float] {
    var out = [Float](repeating: 0, count: batch * m * n)
    for bi in 0..<batch { for i in 0..<m { for j in 0..<n {
        var acc: Float = 0
        for p in 0..<k { acc += a[(bi * m + i) * k + p] * b[(bi * k + p) * n + j] }
        out[(bi * m + i) * n + j] = acc
    } } }
    return out
}

@Suite("batchedMatmul values", .serialized)
struct BatchedMatmulValueTests {

    @Test("3D and 4D batched matmul match a host reference", arguments: [[2], [2, 3]])
    func matchesReference(batchDims: [Int]) {
        let (m, k, n) = (4, 3, 5)
        let batch = batchDims.reduce(1, *)
        let a = (0..<(batch * m * k)).map { Float($0 % 7) * 0.1 - 0.3 }
        let b = (0..<(batch * k * n)).map { Float($0 % 5) * 0.2 - 0.4 }
        let y = Tensor<Float>(a, shape: batchDims + [m, k])
            .batchedMatmul(Tensor<Float>(b, shape: batchDims + [k, n]))
        #expect(y.shape == batchDims + [m, n])
        let got = y.scalars()
        let expected = refBatchedMatmul(a, b, batch: batch, m: m, k: k, n: n)
        #expect(got.count == expected.count)
        let maxDiff = zip(got, expected).map { abs($0 - $1) }.max() ?? .infinity
        #expect(maxDiff < 1e-5)
    }

    @Test("MultiheadAttention output materializes")
    func attentionMaterializes() {
        let mha = nn.MultiheadAttention(embedDim: 8, numHeads: 2)
        let x = Tensor<Float>((0..<48).map { Float($0 % 7) * 0.1 - 0.3 }, shape: [2, 3, 8])
        let values = mha(x).scalars()
        #expect(values.count == 48)
        #expect(values.allSatisfy { $0.isFinite })
    }
}
