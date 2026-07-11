// Magma - BatchNorm train/eval + running-stat tests
// Verifies the reworked BatchNorm: training-mode batch normalization with a
// running-statistics EMA update, inference mode using those running stats,
// train()/eval() propagation through containers, and buffer serialization.
//
// .serialized: materializing tensors creates a GPU PJRT client (see
// XLAGPUSmokeTests) and concurrent clients OOM a unified-memory machine.

import Testing
@testable import Magma
@testable import LazyTensor

@Suite("BatchNorm Train/Eval Tests", .serialized)
struct BatchNormTrainEvalTests {

    private func approxEqual(_ a: [Float], _ b: [Float], tol: Float = 1e-4) -> Bool {
        guard a.count == b.count else { return false }
        for i in 0..<a.count where abs(a[i] - b[i]) > tol { return false }
        return true
    }

    @Test("training updates running mean/var via EMA")
    func trainingUpdatesRunningStats() {
        var bn = nn.BatchNorm1d(numFeatures: 3)   // training == true by default

        // Every row identical => per-feature batch mean = [1,2,3], batch var = 0.
        let x = Tensor<Float>([1, 2, 3,  1, 2, 3,  1, 2, 3,  1, 2, 3], shape: [4, 3])
        _ = bn.forward(x)

        // momentum 0.1: runningMean = 0.9*0 + 0.1*[1,2,3] = [0.1,0.2,0.3]
        //               runningVar  = 0.9*1 + 0.1*0       = [0.9,0.9,0.9]
        #expect(approxEqual(bn.runningMean.value.scalars(), [0.1, 0.2, 0.3]))
        #expect(approxEqual(bn.runningVar.value.scalars(), [0.9, 0.9, 0.9]))
    }

    @Test("eval mode uses running stats and does not update them")
    func evalDoesNotUpdate() {
        var bn = nn.BatchNorm1d(numFeatures: 2)
        let x = Tensor<Float>([2, 4,  2, 4], shape: [2, 2])
        _ = bn.forward(x)                       // one training step
        let afterTrain = bn.runningMean.value.scalars()

        bn.eval()
        // Inference forward with different data must NOT change running stats.
        let x2 = Tensor<Float>([100, 100,  -100, -100], shape: [2, 2])
        let out = bn.forward(x2)
        _ = out.scalars()                       // force materialization
        #expect(approxEqual(bn.runningMean.value.scalars(), afterTrain))
    }

    @Test("eval normalizes with running stats")
    func evalUsesRunningStats() {
        var bn = nn.BatchNorm1d(numFeatures: 1)
        // Drive running mean/var to known values with a few identical batches.
        let x = Tensor<Float>([5, 5, 5, 5], shape: [4, 1])
        for _ in 0..<3 { _ = bn.forward(x) }
        let rm = bn.runningMean.value.scalars()[0]
        let rv = bn.runningVar.value.scalars()[0]

        bn.eval()
        let probe = Tensor<Float>([9], shape: [1, 1])
        let out = bn.forward(probe).scalars()[0]
        // (9 - rm) / sqrt(rv + eps), affine identity (gamma=1, beta=0).
        let expected = (9 - rm) / (rv + 1e-5).squareRoot()
        #expect(abs(out - expected) < 1e-3)
    }

    @Test("train/eval propagates through Sequential to Dropout")
    func trainEvalPropagation() {
        var model = nn.Sequential(nn.AnyLayer(nn.Dropout(p: 0.5)))
        let x = Tensor<Float>([1, 2, 3, 4], shape: [2, 2])

        // In eval, Dropout is a pass-through: output equals input exactly.
        model.eval()
        let evalOut = model.forward(x).scalars()
        #expect(approxEqual(evalOut, x.scalars()))

        // Back in train mode, Dropout scales/zeros => output differs from input.
        model.train()
        let trainOut = model.forward(x).scalars()
        #expect(!approxEqual(trainOut, x.scalars()))
    }

    @Test("running stats survive stateDict round-trip")
    func runningStatsSerialization() throws {
        var bn = nn.BatchNorm1d(numFeatures: 3)
        let x = Tensor<Float>([1, 2, 3,  4, 5, 6], shape: [2, 3])
        _ = bn.forward(x)
        let savedMean = bn.runningMean.value.scalars()
        let savedVar = bn.runningVar.value.scalars()

        let sd = bn.stateDict()
        // Buffers are present under the "buffer." namespace.
        #expect(sd.keys.contains { $0.hasPrefix("buffer.") })

        var restored = nn.BatchNorm1d(numFeatures: 3)   // fresh: running mean = 0
        try restored.loadStateDict(sd, strict: true)
        #expect(approxEqual(restored.runningMean.value.scalars(), savedMean))
        #expect(approxEqual(restored.runningVar.value.scalars(), savedVar))
    }
}
