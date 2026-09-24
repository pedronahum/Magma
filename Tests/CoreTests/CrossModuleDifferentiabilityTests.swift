// Magma - Cross-module differentiability tests
//
// Swift can only differentiate a function from another module if that function
// is declared `@differentiable(reverse)` or has a registered `@derivative`.
// This file is compiled into the test module and imports Magma *without*
// `@testable`, so every `gradient(at:)` call below is exactly what user code in
// a separate package sees: if an op lost its differentiability attribute, this
// file would stop compiling ("expression is not differentiable").
//
// Each gradient is also compared against a central-difference numerical
// gradient at points away from kinks (relu/abs/max/clamp/hinge boundaries),
// weighting outputs with distinct coefficients so that a pullback that routes
// cotangents to the wrong element is caught.
//
// (`_Differentiation` is imported as user code would to spell
// `@differentiable` closure types.)
//
// .serialized: materialization drives the PJRT backend.

import Testing
import _Differentiation
import Magma

// MARK: - Helpers

/// Distinct, mixed-sign values away from 0, ±1 and ±1.5 (the kinks of relu,
/// leakyRelu, elu, hardtanh and clamp(-1.5, 1.5)).
private let mixed = Tensor<Float>([-1.7, -0.6, 0.3, 1.2, 2.1, -2.4], shape: [2, 3])
/// Distinct positive values (for log, pow, poisson, KL, MSLE, ...).
private let positive = Tensor<Float>([0.4, 1.3, 0.7, 2.2, 0.9, 1.6], shape: [2, 3])
/// Distinct values in (0, 1) (for binary cross-entropy).
private let probabilities = Tensor<Float>([0.2, 0.7, 0.45, 0.9, 0.35, 0.6], shape: [2, 3])
/// Per-element weights, so `(f(x) * weights).sum()` gives each output element
/// a different cotangent.
private let weights = Tensor<Float>([0.5, -1.25, 2.0, 0.75, -0.3, 1.5], shape: [2, 3])

private let atol: Float = 2e-3
private let rtol: Float = 2e-2

/// Autodiff gradient of `f` at `x` must match the numerical gradient.
private func expectGradientMatches(
    _ f: @differentiable(reverse) (Tensor<Float>) -> Tensor<Float>,
    at x: Tensor<Float>,
    eps: Float = 1e-2,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let result = gradcheck(f, input: x, eps: eps, atol: atol, rtol: rtol)
    #expect(result.passed,
            "gradcheck failed: maxAbsDiff \(result.maxAbsDiff), maxRelDiff \(result.maxRelDiff)",
            sourceLocation: sourceLocation)
}

/// Autodiff gradients of `f` w.r.t. both inputs must match the numerical ones.
private func expectGradientsMatch(
    _ f: @differentiable(reverse) (Tensor<Float>, Tensor<Float>) -> Tensor<Float>,
    at x: Tensor<Float>, _ y: Tensor<Float>,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let (rx, ry) = gradcheck(f, input1: x, input2: y, atol: atol, rtol: rtol)
    #expect(rx.passed, "first input: maxAbsDiff \(rx.maxAbsDiff), maxRelDiff \(rx.maxRelDiff)",
            sourceLocation: sourceLocation)
    #expect(ry.passed, "second input: maxAbsDiff \(ry.maxAbsDiff), maxRelDiff \(ry.maxRelDiff)",
            sourceLocation: sourceLocation)
}

/// Element-wise ops: weight each output element differently, then sum.
private func expectElementwiseGradientMatches(
    _ f: @escaping @differentiable(reverse) (Tensor<Float>) -> Tensor<Float>,
    at x: Tensor<Float> = mixed,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    expectGradientMatches({ (f($0) * weights).sum() }, at: x, sourceLocation: sourceLocation)
}

@Suite("Cross-module differentiability", .serialized,
       .enabled(if: PluginAvailability.cpu, "CPU PJRT plugin not available"))
struct CrossModuleDifferentiabilityTests {

    // MARK: - Tensor activations

    @Test func logSoftmax() {
        expectElementwiseGradientMatches { $0.logSoftmax(dim: -1) }
        expectElementwiseGradientMatches { $0.logSoftmax(dim: 0) }
        expectElementwiseGradientMatches { $0.logSoftmax() }
    }

    @Test func leakyRelu() {
        expectElementwiseGradientMatches { $0.leakyRelu() }
        expectElementwiseGradientMatches { $0.leakyRelu(negativeSlope: 0.2) }
    }

    @Test func silu() { expectElementwiseGradientMatches { $0.silu() } }

    @Test func elu() {
        expectElementwiseGradientMatches { $0.elu() }
        expectElementwiseGradientMatches { $0.elu(alpha: 1.3) }
    }

    @Test func hardtanh() {
        expectElementwiseGradientMatches { $0.hardtanh() }
        expectElementwiseGradientMatches { $0.hardtanh(minVal: -2, maxVal: 0.5) }
    }

    @Test func selu() { expectElementwiseGradientMatches { $0.selu() } }

    @Test("selu applies its scale: scale * elu(x, alpha)")
    func seluValues() {
        let out = Tensor<Float>([1, -1], shape: [2]).selu().scalars()
        let scale: Float = 1.0507009873554804934193349852946
        let alpha: Float = 1.6732632423543772848170429916717
        #expect(Swift.abs(out[0] - scale) < 1e-5)
        #expect(Swift.abs(out[1] - scale * alpha * (0.36787944117 - 1)) < 1e-5)
    }

    @Test func mish() { expectElementwiseGradientMatches { $0.mish() } }

    @Test func softplus() {
        expectElementwiseGradientMatches { $0.softplus() }
        expectElementwiseGradientMatches { Magma.softplus($0) }
    }

    @Test("softplus stays finite with a finite gradient for large inputs")
    func softplusLargeInputs() {
        let x = Tensor<Float>([-200, 200], shape: [2])
        let values = x.softplus().scalars()
        #expect(Swift.abs(values[0]) < 1e-6)
        #expect(Swift.abs(values[1] - 200) < 1e-3)
        let grad = gradient(at: x) { $0.softplus().sum() }.scalars()
        #expect(grad.allSatisfy { $0.isFinite })
        #expect(Swift.abs(grad[0]) < 1e-6 && Swift.abs(grad[1] - 1) < 1e-6)
    }

    @Test func softsign() { expectElementwiseGradientMatches { $0.softsign() } }

    @Test func gelu() { expectElementwiseGradientMatches { $0.gelu() } }

    @Test func pow() {
        expectElementwiseGradientMatches { $0.pow(3) }
        expectElementwiseGradientMatches({ $0.pow(2.5) }, at: positive)
        expectElementwiseGradientMatches({ $0.pow(-1) }, at: positive)
        expectElementwiseGradientMatches { $0.pow(0) }
    }

    @Test("pow computes x^exponent")
    func powValues() {
        let x = Tensor<Float>([2, 3, 0.5], shape: [3])
        #expect(x.pow(3).scalars() == [8, 27, 0.125])
        #expect(x.pow(-1).scalars() == [0.5, Float(1) / 3, 2])
    }

    @Test func clamp() {
        expectElementwiseGradientMatches { $0.clamp(min: -1.5, max: 1.5) }
    }

    @Test func squaredAndLog1p() {
        expectElementwiseGradientMatches { $0.squared() }
        expectElementwiseGradientMatches({ $0.log1p() }, at: positive)
    }

    // MARK: - Reductions

    @Test func maxReductions() {
        expectGradientMatches({ $0.max() }, at: mixed)
        expectGradientMatches({ ($0.max(dims: [1]) * Tensor<Float>([0.5, -2], shape: [2])).sum() }, at: mixed)
        expectGradientMatches({ ($0.max(dims: [0], keepDims: true) * Tensor<Float>([1, 2, 3], shape: [1, 3])).sum() }, at: mixed)
        expectGradientMatches({ ($0.max(alongAxes: -1) * Tensor<Float>([0.5, -2], shape: [2])).sum() }, at: mixed)
        expectGradientMatches({ ($0.max(alongAxes: [0, 1])) }, at: mixed)
    }

    @Test func minReductions() {
        expectGradientMatches({ ($0.min(alongAxes: 1) * Tensor<Float>([0.5, -2], shape: [2])).sum() }, at: mixed)
        expectGradientMatches({ ($0.min(alongAxes: [0]) * Tensor<Float>([1, 2, 3], shape: [3])).sum() }, at: mixed)
    }

    @Test("max reduction splits the gradient evenly among ties")
    func maxTies() {
        let x = Tensor<Float>([1, 3, 3, 2], shape: [4])
        let grad = gradient(at: x) { $0.max() }.scalars()
        #expect(grad == [0, 0.5, 0.5, 0])
    }

    @Test func sumAlongAxes() {
        expectGradientMatches({ ($0.sum(alongAxes: 1) * Tensor<Float>([0.5, -2], shape: [2])).sum() }, at: mixed)
        expectGradientMatches({ ($0.sum(alongAxes: [0]) * Tensor<Float>([1, 2, 3], shape: [3])).sum() }, at: mixed)
    }

    @Test func variance() {
        expectGradientMatches({ ($0.variance(dims: [1]) * Tensor<Float>([0.5, -2], shape: [2])).sum() }, at: mixed)
        expectGradientMatches({ ($0.variance(dims: [0], keepDims: true, unbiased: false) * Tensor<Float>([1, 2, 3], shape: [1, 3])).sum() }, at: mixed)
        expectGradientMatches({ $0.variance(dims: [0, 1]).sum() }, at: mixed)
    }

    // MARK: - Masking and selection

    @Test func maskedFill() {
        let mask = Tensor<Float>([1, 0, 0, 1, 0, 1], shape: [2, 3])
        expectElementwiseGradientMatches { $0.maskedFill(mask: mask, value: positive) }
        expectElementwiseGradientMatches { mixed.maskedFill(mask: mask, value: $0) }
    }

    @Test func whereSelect() {
        let mask = Tensor<Float>([0, 1, 1, 0, 1, 0], shape: [2, 3])
        expectElementwiseGradientMatches { Tensor.where_(mask, $0, positive) }
        expectElementwiseGradientMatches { Tensor.where_(mask, positive, $0) }
        expectElementwiseGradientMatches { $0.where_(mask, positive) }
        expectElementwiseGradientMatches { positive.where_(mask, $0) }
    }

    @Test func maskedSelect() {
        let mask = Tensor<Float>([0, 1, 1, 0, 1, 0], shape: [2, 3])
        expectGradientMatches({ ($0.maskedSelect(mask) * weights.reshape([6])).sum() }, at: mixed)
    }

    @Test func triangular() {
        let square = Tensor<Float>([0.3, -1.2, 0.8, 1.1, -0.4, 2.0, 0.6, -0.9, 1.4], shape: [3, 3])
        let w = Tensor<Float>([1, 2, 3, 4, 5, 6, 7, 8, 9], shape: [3, 3])
        expectGradientMatches({ ($0.tril() * w).sum() }, at: square)
        expectGradientMatches({ ($0.triu(k: 1) * w).sum() }, at: square)
    }

    // MARK: - Shape manipulation

    @Test func transposeLastTwo() {
        let w = Tensor<Float>([1, 2, 3, 4, 5, 6], shape: [3, 2])
        expectGradientMatches({ ($0.transposeLastTwo() * w).sum() }, at: mixed)
    }

    @Test func expandAndSqueeze() {
        expectElementwiseGradientMatches { $0.expandingShape(at: 1).squeeze(dim: 1) }
        expectGradientMatches({ ($0.expandingShape(at: 0) * weights.reshape([1, 2, 3])).sum() }, at: mixed)
        let column = Tensor<Float>([0.3, -1.2, 0.8], shape: [3, 1])
        expectGradientMatches({ ($0.squeeze() * Tensor<Float>([1, 2, 3], shape: [3])).sum() }, at: column)
        expectGradientMatches({ ($0.squeeze(dim: -1) * Tensor<Float>([1, 2, 3], shape: [3])).sum() }, at: column)
    }

    @Test func concatAndStack() {
        let w4 = Tensor<Float>((0..<12).map { Float($0) - 5.5 }, shape: [4, 3])
        let w6 = Tensor<Float>((0..<12).map { Float($0) * 0.5 - 2 }, shape: [2, 6])
        let w223 = Tensor<Float>((0..<12).map { Float($0) - 5.5 }, shape: [2, 2, 3])
        let w232 = Tensor<Float>((0..<12).map { Float($0) * 0.5 - 2 }, shape: [2, 3, 2])
        expectGradientsMatch({ (Tensor.concat([$0, $1], axis: 0) * w4).sum() }, at: mixed, positive)
        expectGradientsMatch({ (Tensor.concat([$0, $1], axis: -1) * w6).sum() }, at: mixed, positive)
        expectGradientsMatch({ ($0.concat(with: [$1]) * w4).sum() }, at: mixed, positive)
        expectGradientsMatch({ (Tensor.stack([$0, $1]) * w223).sum() }, at: mixed, positive)
        expectGradientsMatch({ (Tensor.stack([$0, $1], axis: -1) * w232).sum() }, at: mixed, positive)
        expectGradientsMatch({ ($0.stack(with: [$1], axis: 1) * w223).sum() }, at: mixed, positive)
        expectGradientsMatch({ (data.stack([$0, $1]) * w223).sum() }, at: mixed, positive)
    }

    @Test("dtype and device conversions pass the gradient through")
    func conversions() {
        expectElementwiseGradientMatches { $0.to(device: .default) }
        // The cotangent itself makes the bfloat16 round trip, so allow its rounding.
        let grad = gradient(at: mixed) { ($0.toReducedPrecision().toFullPrecision() * weights).sum() }
        for (g, w) in zip(grad.scalars(), weights.scalars()) { #expect(Swift.abs(g - w) < 1e-2) }
    }

    // MARK: - Free functions

    @Test func freeElementwiseFunctions() {
        expectElementwiseGradientMatches { Magma.abs($0) }
        expectElementwiseGradientMatches({ Magma.log($0) }, at: positive)
        expectElementwiseGradientMatches { Magma.exp($0) }
        expectGradientsMatch({ (Magma.max($0, $1) * weights).sum() }, at: mixed, positive)
        expectGradientsMatch({ (Magma.min($0, $1) * weights).sum() }, at: mixed, positive)
    }

    // MARK: - nn.functional

    @Test func functionalActivations() {
        expectElementwiseGradientMatches { nn.functional.relu($0) }
        expectElementwiseGradientMatches { nn.functional.sigmoid($0) }
        expectElementwiseGradientMatches { nn.functional.tanh($0) }
        expectElementwiseGradientMatches { nn.functional.gelu($0) }
        expectElementwiseGradientMatches { nn.functional.softmax($0) }
        expectElementwiseGradientMatches { nn.functional.softmax($0, dim: 0) }
        expectElementwiseGradientMatches { nn.functional.logSoftmax($0) }
    }

    @Test func functionalMSE() {
        expectGradientsMatch({ nn.functional.mse($0, $1) }, at: mixed, positive)
    }

    @Test("crossEntropy with class-index targets")
    func functionalCrossEntropyIndices() {
        let targets = Tensor<Float>([2, 0], shape: [2])
        expectGradientMatches({ nn.functional.crossEntropy($0, targets) }, at: mixed)
    }

    @Test("crossEntropy with one-hot and soft targets")
    func functionalCrossEntropyProbabilities() {
        let oneHot = Tensor<Float>([0, 0, 1, 1, 0, 0], shape: [2, 3])
        expectGradientMatches({ nn.functional.crossEntropy($0, oneHot) }, at: mixed)
        let soft = Tensor<Float>([0.2, 0.3, 0.5, 0.6, 0.1, 0.3], shape: [2, 3])
        expectGradientMatches({ nn.functional.crossEntropy($0, soft) }, at: mixed)
    }

    @Test func functionalBinaryCrossEntropy() {
        let target = Tensor<Float>([1, 0, 1, 0, 0, 1], shape: [2, 3])
        expectGradientMatches({ nn.functional.binaryCrossEntropy($0, target) }, at: probabilities, eps: 1e-3)
        expectGradientMatches({ nn.functional.binaryCrossEntropyWithLogits($0, target) }, at: mixed)
    }

    @Test func functionalNLLLoss() {
        let targets = Tensor<Float>([1, 2], shape: [2])
        expectGradientMatches({ nn.functional.nllLoss($0.logSoftmax(), targets) }, at: mixed)
        expectGradientMatches({ nn.functional.nllLoss($0, targets) }, at: mixed)
    }

    // MARK: - Loss.swift losses

    @Test func regressionLosses() {
        expectGradientsMatch({ l1Loss(predicted: $0, expected: $1) }, at: mixed, positive)
        expectGradientsMatch({ l1Loss(predicted: $0, expected: $1, reduction: .mean) }, at: mixed, positive)
        expectGradientsMatch({ (l1Loss(predicted: $0, expected: $1, reduction: .none) * weights).sum() }, at: mixed, positive)
        expectGradientsMatch({ l2Loss(predicted: $0, expected: $1) }, at: mixed, positive)
        expectGradientsMatch({ meanAbsoluteError(predicted: $0, expected: $1) }, at: mixed, positive)
        expectGradientsMatch({ meanSquaredError(predicted: $0, expected: $1) }, at: mixed, positive)
        expectGradientsMatch({ meanSquaredLogarithmicError(predicted: $0, expected: $1) },
                             at: positive, probabilities)
        expectGradientsMatch({ meanAbsolutePercentageError(predicted: $0, expected: $1) },
                             at: mixed, positive)
        expectGradientsMatch({ logCoshLoss(predicted: $0, expected: $1) }, at: mixed, positive)
        // |error| = [2.1, 1.9, 0.4, 1.0, 1.2, 4.0]: on both sides of delta = 1.5, away from it.
        expectGradientsMatch({ huberLoss(predicted: $0, expected: $1, delta: 1.5) }, at: mixed, positive)
    }

    @Test func hingeLosses() {
        let labels = Tensor<Float>([1, -1, 1, -1, 1, -1], shape: [2, 3])
        // 1 - y * p = [2.7, 0.4, 0.7, 2.2, -1.1, -1.4]: no kinks.
        expectGradientMatches({ hingeLoss(predicted: $0, expected: labels) }, at: mixed)
        expectGradientMatches({ squaredHingeLoss(predicted: $0, expected: labels) }, at: mixed)
        let oneHot = Tensor<Float>([0, 0, 1, 1, 0, 0], shape: [2, 3])
        expectGradientMatches({ categoricalHingeLoss(predicted: $0, expected: oneHot) }, at: positive)
    }

    @Test func probabilisticLosses() {
        expectGradientMatches({ poissonLoss(predicted: $0, expected: probabilities) }, at: positive)
        expectGradientMatches({ kullbackLeiblerDivergence(predicted: $0, expected: probabilities) }, at: positive)
    }

    @Test func softmaxCrossEntropyLosses() {
        let oneHot = Tensor<Float>([0, 0, 1, 1, 0, 0], shape: [2, 3])
        expectGradientMatches({ softmaxCrossEntropy(logits: $0, probabilities: oneHot) }, at: mixed)
        expectGradientMatches({ softmaxCrossEntropy(logits: $0, probabilities: oneHot, reduction: .sum) }, at: mixed)
        expectGradientMatches({ softmaxCrossEntropyWithLabels(logits: $0, labels: [2, 0], numClasses: 3) }, at: mixed)
    }

    @Test func sigmoidCrossEntropyLoss() {
        let labels = Tensor<Float>([1, 0, 1, 0, 0, 1], shape: [2, 3])
        expectGradientMatches({ sigmoidCrossEntropy(logits: $0, labels: labels) }, at: mixed)
    }

    @Test func similarityLosses() {
        expectGradientsMatch({ (cosineSimilarity($0, $1) * Tensor<Float>([0.5, -2], shape: [2])).sum() },
                             at: mixed, positive)
        expectGradientsMatch({ cosineDistance(predicted: $0, expected: $1) }, at: mixed, positive)
    }

    @Test func metricLearningLosses() {
        let anchor = Tensor<Float>([0.1, 0.4, -0.2, 0.3, 0.0, 0.5], shape: [2, 3])
        let near = Tensor<Float>([0.2, 0.1, -0.1, 0.9, 0.3, 0.2], shape: [2, 3])
        let far = Tensor<Float>([1.5, -0.8, 0.6, -1.1, 1.2, 0.9], shape: [2, 3])
        // One similar and one dissimilar pair, with the dissimilar one inside the margin.
        let labels = Tensor<Float>([1, 0], shape: [2])
        expectGradientsMatch({ contrastiveLoss(anchor: $0, sample: $1, labels: labels, margin: 2) },
                             at: anchor, near)
        // posDistance - negDistance + margin is positive for both rows.
        expectGradientsMatch({ tripletMarginLoss(anchor: $0, positive: $1, negative: far, margin: 2) },
                             at: anchor, near)
        expectGradientMatches({ tripletMarginLoss(anchor: anchor, positive: near, negative: $0, margin: 2) },
                              at: far)
    }
}
