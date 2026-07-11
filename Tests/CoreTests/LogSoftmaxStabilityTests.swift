// Magma - LogSoftmax numerical stability tests
// logSoftmax must be computed as x - logsumexp(x), which stays finite for
// extreme logits, rather than log(softmax(x)) which underflows to -inf.
//
// .serialized: materialization creates a GPU PJRT client (see XLAGPUSmokeTests).

import Foundation
import Testing
@testable import Magma
@testable import LazyTensor

@Suite("LogSoftmax Stability Tests", .serialized)
struct LogSoftmaxStabilityTests {

    @Test("stays finite for extreme logits")
    func extremeLogits() {
        // log(softmax([0, 1000]))[0] would be log(0) = -inf under the old impl.
        let x = Tensor<Float>([0, 1000], shape: [1, 2])
        let out = x.logSoftmax(dim: -1).scalars()

        #expect(out.allSatisfy { $0.isFinite })
        #expect(abs(out[1] - 0) < 1e-2)          // dominant logit -> log-prob ~ 0
        #expect(abs(out[0] - (-1000)) < 1.0)     // suppressed logit -> ~ -1000
    }

    @Test("matches reference values for normal logits")
    func normalLogits() {
        let x = Tensor<Float>([1, 2, 3], shape: [1, 3])
        let out = x.logSoftmax(dim: -1).scalars()

        // logsumexp([1,2,3]) = 3 + log(e^-2 + e^-1 + 1) ~ 3.40761
        let expected: [Float] = [-2.40761, -1.40761, -0.40761]
        for i in 0..<3 { #expect(abs(out[i] - expected[i]) < 1e-3) }

        // exp(logSoftmax) must be a valid distribution summing to 1.
        let probSum = out.reduce(Float(0)) { $0 + expf($1) }
        #expect(abs(probSum - 1.0) < 1e-4)
    }

    @Test("row-independent for batched input")
    func batched() {
        let x = Tensor<Float>([1, 2, 3,  10, 20, 30], shape: [2, 3])
        let out = x.logSoftmax(dim: -1).scalars()
        #expect(out.allSatisfy { $0.isFinite })
        // Each row is a valid log-distribution.
        let row0 = Array(out[0..<3]).reduce(Float(0)) { $0 + expf($1) }
        let row1 = Array(out[3..<6]).reduce(Float(0)) { $0 + expf($1) }
        #expect(abs(row0 - 1.0) < 1e-4)
        #expect(abs(row1 - 1.0) < 1e-4)
    }
}
