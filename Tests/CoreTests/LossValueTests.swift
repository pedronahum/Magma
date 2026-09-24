// Magma - Loss value tests
//
// Losses checked against values computed by hand, not just output shapes,
// including crossEntropy's one-hot/probability targets and the stable form of
// binaryCrossEntropyWithLogits at large |logits|.

import Foundation
import Testing
import _Differentiation
@testable import Magma

@Suite("Loss values", .enabled(if: PluginAvailability.cpu, "CPU PJRT plugin not available"))
struct LossValueTests {

    private func close(_ a: Float, _ b: Float, tol: Float = 1e-5) -> Bool {
        abs(a - b) <= tol * max(1, abs(b))
    }

    private func close(_ a: [Float], _ b: [Float], tol: Float = 1e-5) -> Bool {
        a.count == b.count && zip(a, b).allSatisfy { close($0, $1, tol: tol) }
    }

    // logits row 0 = [1, 2, 3]: logsumexp = 3 + log(1 + e^-1 + e^-2)
    // logits row 1 = [1, 1, 1]: logsumexp = 1 + log 3
    private let logits = Tensor<Float>([1, 2, 3,  1, 1, 1], shape: [2, 3])
    private let lse0: Float = 3 + Foundation.log(1 + Foundation.exp(-1) + Foundation.exp(-2))
    private let lse1: Float = 1 + Foundation.log(3)

    @Test("mse is the mean squared difference")
    func mseValue() {
        let loss = nn.functional.mse(Tensor<Float>([1, 2, 3, 4], shape: [4]),
                                     Tensor<Float>([2, 2, 1, 4], shape: [4]))
        #expect(close(loss.scalars()[0], 1.25))   // (1 + 0 + 4 + 0) / 4
    }

    @Test("crossEntropy with class-index targets")
    func crossEntropyClassIndices() {
        let targets = Tensor<Float>([2, 0], shape: [2])
        let loss = nn.functional.crossEntropy(logits, targets).scalars()[0]
        let expected = ((lse0 - 3) + (lse1 - 1)) / 2
        #expect(close(loss, expected))
    }

    @Test("crossEntropy with one-hot targets matches class indices")
    func crossEntropyOneHot() {
        let oneHot = Tensor<Float>([0, 0, 1,  1, 0, 0], shape: [2, 3])
        let indices = Tensor<Float>([2, 0], shape: [2])
        let fromOneHot = nn.functional.crossEntropy(logits, oneHot).scalars()[0]
        let fromIndices = nn.functional.crossEntropy(logits, indices).scalars()[0]
        #expect(close(fromOneHot, fromIndices))
    }

    @Test("crossEntropy with soft probability targets")
    func crossEntropySoftTargets() {
        let probs = Tensor<Float>([0.5, 0.5, 0,  0, 0, 1], shape: [2, 3])
        let loss = nn.functional.crossEntropy(logits, probs).scalars()[0]
        let row0 = 0.5 * (lse0 - 1) + 0.5 * (lse0 - 2)
        let row1 = lse1 - 1
        #expect(close(loss, (row0 + row1) / 2))
    }

    // Exit tests need Swift 6.2+; older toolchains skip these checks.
    #if compiler(>=6.2)
    @Test("crossEntropy rejects probability targets with the wrong class count")
    func crossEntropyClassMismatchTraps() async {
        await #expect(processExitsWith: .failure) {
            let logits = Tensor<Float>.zeros([2, 3])
            _ = nn.functional.crossEntropy(logits, Tensor<Float>.zeros([2, 4]))
        }
    }
    #endif

    @Test("binaryCrossEntropyWithLogits matches the closed form")
    func bceWithLogitsValue() {
        let x = Tensor<Float>([0, 2, -3], shape: [3])
        let y = Tensor<Float>([1, 0, 1], shape: [3])
        let loss = nn.functional.binaryCrossEntropyWithLogits(x, y).scalars()[0]
        // max(x,0) - x*y + log(1 + exp(-|x|)) per element
        let e0 = Foundation.log(Float(2))
        let e1 = 2 + Foundation.log(1 + Foundation.exp(Float(-2)))
        let e2 = 3 + Foundation.log(1 + Foundation.exp(Float(-3)))
        #expect(close(loss, (e0 + e1 + e2) / 3))
    }

    @Test("binaryCrossEntropyWithLogits stays exact for large logits")
    func bceWithLogitsLargeLogits() {
        // Confidently wrong: loss is |x| (the sigmoid+clamp form saturated at ~16).
        let wrong = nn.functional.binaryCrossEntropyWithLogits(
            Tensor<Float>([100, -100], shape: [2]), Tensor<Float>([0, 1], shape: [2]))
        #expect(close(wrong.scalars()[0], 100))
        // Confidently right: loss is ~0, not a clamped epsilon floor.
        let right = nn.functional.binaryCrossEntropyWithLogits(
            Tensor<Float>([100, -100], shape: [2]), Tensor<Float>([1, 0], shape: [2]))
        #expect(abs(right.scalars()[0]) < 1e-6)
    }

    @Test("binaryCrossEntropyWithLogits gradient is (sigmoid(x) - y) / n")
    func bceWithLogitsGradient() {
        // x == 0 is where the relu/abs kinks sit (zero-initialized output layers).
        let xs: [Float] = [2, -3, 100, -100, 0, 0]
        let ys: [Float] = [0, 1, 0, 1, 0, 1]
        let x = Tensor<Float>(xs, shape: [6])
        let y = Tensor<Float>(ys, shape: [6])
        let grad = gradient(at: x) { nn.functional.binaryCrossEntropyWithLogits($0, y) }
        let expected: [Float] = zip(xs, ys).map { xi, yi in
            let sigmoid: Float = 1 / (1 + Foundation.exp(-xi))
            return (sigmoid - yi) / 6
        }
        #expect(close(grad.scalars(), expected, tol: 1e-4))
    }

    @Test("huberLoss is quadratic inside delta and linear outside")
    func huberValue() {
        let loss = huberLoss(predicted: Tensor<Float>([1, 2, 3, 4], shape: [4]),
                             expected: Tensor<Float>([2, 2, 1, 4], shape: [4]),
                             delta: 1)
        // errors [1, 0, -2, 0] -> 0.5, 0, 0.5 * 1 + 1 * (2 - 1), 0
        #expect(close(loss.scalars()[0], 2.0))
    }

    @Test("kullbackLeiblerDivergence matches sum(p * log(p / q))")
    func klDivergenceValue() {
        let loss = kullbackLeiblerDivergence(predicted: Tensor<Float>([0.25, 0.75], shape: [2]),
                                             expected: Tensor<Float>([0.5, 0.5], shape: [2]))
        let expected = 0.5 * Foundation.log(Float(2)) + 0.5 * Foundation.log(Float(0.5 / 0.75))
        #expect(close(loss.scalars()[0], expected))
    }

    // Exit tests need Swift 6.2+; older toolchains skip these checks.
    #if compiler(>=6.2)
    @Test("softmaxCrossEntropyWithLabels rejects out-of-range labels")
    func sparseLabelsOutOfRangeTraps() async {
        await #expect(processExitsWith: .failure) {
            _ = softmaxCrossEntropyWithLabels(logits: Tensor<Float>.zeros([2, 3]), labels: [0, 3], numClasses: 3)
        }
        await #expect(processExitsWith: .failure) {
            _ = softmaxCrossEntropyWithLabels(logits: Tensor<Float>.zeros([2, 3]), labels: [0], numClasses: 3)
        }
    }
    #endif
}
