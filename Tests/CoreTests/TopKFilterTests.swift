// Magma - top-k filter tests
// topKFilter must keep only the k largest logits (rest get ~0 probability),
// not ignore k and return the full softmax.
//
// .serialized: materialization creates a GPU PJRT client (see XLAGPUSmokeTests).

import Testing
@testable import Magma
@testable import LazyTensor

@Suite("TopKFilter Tests", .serialized)
struct TopKFilterTests {

    @Test("keeps exactly the top-k, zeroes the rest")
    func topKMasks() {
        // Distinct logits; top-2 are 5 (idx 3) and 4 (idx 4).
        let logits = Tensor<Float>([1, 3, 2, 5, 4], shape: [1, 5])
        let probs = logits.topKFilter(k: 2).scalars()

        // Only the two largest keep probability mass.
        #expect(probs[0] < 1e-6)
        #expect(probs[1] < 1e-6)
        #expect(probs[2] < 1e-6)
        #expect(probs[3] > 0.5)          // e^5 / (e^5 + e^4) ~ 0.731
        #expect(probs[4] > 0.2)          // e^4 / (e^5 + e^4) ~ 0.269
        // Valid distribution.
        #expect(abs(probs.reduce(0, +) - 1.0) < 1e-4)
    }

    @Test("k == vocab keeps everything (plain softmax)")
    func kEqualsVocab() {
        let logits = Tensor<Float>([1, 2, 3], shape: [1, 3])
        let probs = logits.topKFilter(k: 3).scalars()
        #expect(probs.allSatisfy { $0 > 0 })
        #expect(abs(probs.reduce(0, +) - 1.0) < 1e-4)
    }

    @Test("k == 1 keeps only the argmax")
    func kEqualsOne() {
        let logits = Tensor<Float>([1, 9, 2, 3], shape: [1, 4])
        let probs = logits.topKFilter(k: 1).scalars()
        #expect(probs[1] > 0.99)         // all mass on the max
        #expect(probs[0] < 1e-3)
        #expect(probs[2] < 1e-3)
        #expect(probs[3] < 1e-3)
    }
}
