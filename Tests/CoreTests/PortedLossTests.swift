// Magma - Ported Loss Functions Tests
// Tests for loss functions ported from S4TF

import XCTest
@testable import Magma
@testable import LazyTensor

// MARK: - L1 Loss Tests

final class L1LossTests: XCTestCase {

    func testL1LossMeanReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l1Loss(predicted: predicted, expected: expected, reduction: .mean)

        // L1 loss of all 1s vs all 0s = mean(|1 - 0|) = 1.0
        XCTAssertEqual(loss.shape, [])  // Scalar output
    }

    func testL1LossSumReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l1Loss(predicted: predicted, expected: expected, reduction: .sum)

        XCTAssertEqual(loss.shape, [])  // Scalar output
    }

    func testL1LossNoReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l1Loss(predicted: predicted, expected: expected, reduction: .none)

        XCTAssertEqual(loss.shape, [32, 10])  // Same shape as input
    }
}

// MARK: - L2 Loss Tests

final class L2LossTests: XCTestCase {

    func testL2LossMeanReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l2Loss(predicted: predicted, expected: expected, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }

    func testL2LossSumReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l2Loss(predicted: predicted, expected: expected, reduction: .sum)

        XCTAssertEqual(loss.shape, [])
    }

    func testL2LossNoReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l2Loss(predicted: predicted, expected: expected, reduction: .none)

        XCTAssertEqual(loss.shape, [32, 10])
    }
}

// MARK: - Mean Absolute Error Tests

final class MeanAbsoluteErrorTests: XCTestCase {

    func testMeanAbsoluteError() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = meanAbsoluteError(predicted: predicted, expected: expected)

        XCTAssertEqual(loss.shape, [])
    }
}

// MARK: - Mean Squared Error Tests

final class MeanSquaredErrorTests: XCTestCase {

    func testMeanSquaredError() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = meanSquaredError(predicted: predicted, expected: expected)

        XCTAssertEqual(loss.shape, [])
    }
}

// MARK: - Hinge Loss Tests

final class HingeLossTests: XCTestCase {

    func testHingeLoss() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.ones([32, 10])  // Labels should be -1 or 1

        let loss = hingeLoss(predicted: predicted, expected: expected, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }

    func testHingeLossNoReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.ones([32, 10])

        let loss = hingeLoss(predicted: predicted, expected: expected, reduction: .none)

        XCTAssertEqual(loss.shape, [32, 10])
    }
}

// MARK: - Squared Hinge Loss Tests

final class SquaredHingeLossTests: XCTestCase {

    func testSquaredHingeLoss() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.ones([32, 10])

        let loss = squaredHingeLoss(predicted: predicted, expected: expected, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }
}

// MARK: - Huber Loss Tests

final class HuberLossTests: XCTestCase {

    func testHuberLoss() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = huberLoss(predicted: predicted, expected: expected, delta: 1.0, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }

    func testHuberLossCustomDelta() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = huberLoss(predicted: predicted, expected: expected, delta: 0.5, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }

    func testHuberLossNoReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = huberLoss(predicted: predicted, expected: expected, delta: 1.0, reduction: .none)

        XCTAssertEqual(loss.shape, [32, 10])
    }
}

// MARK: - Log-Cosh Loss Tests

final class LogCoshLossTests: XCTestCase {

    func testLogCoshLoss() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = logCoshLoss(predicted: predicted, expected: expected, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }

    func testLogCoshLossNoReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = logCoshLoss(predicted: predicted, expected: expected, reduction: .none)

        XCTAssertEqual(loss.shape, [32, 10])
    }
}

// MARK: - Poisson Loss Tests

final class PoissonLossTests: XCTestCase {

    func testPoissonLoss() {
        let predicted = Tensor<Float>.ones([32, 10])  // Must be positive
        let expected = Tensor<Float>.ones([32, 10])

        let loss = poissonLoss(predicted: predicted, expected: expected, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }
}

// MARK: - Softmax Cross Entropy Tests

final class SoftmaxCrossEntropyTests: XCTestCase {

    func testSoftmaxCrossEntropy() {
        let logits = Tensor<Float>.zeros([32, 10])
        let labels = Tensor<Float>.zeros([32, 10])

        let loss = softmaxCrossEntropy(logits: logits, probabilities: labels, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }

    func testSoftmaxCrossEntropyNoReduction() {
        let logits = Tensor<Float>.zeros([32, 10])
        let labels = Tensor<Float>.zeros([32, 10])

        let loss = softmaxCrossEntropy(logits: logits, probabilities: labels, reduction: .none)

        XCTAssertEqual(loss.shape, [32])  // One loss per sample
    }
}

// MARK: - Sigmoid Cross Entropy Tests

final class SigmoidCrossEntropyTests: XCTestCase {

    func testSigmoidCrossEntropy() {
        let logits = Tensor<Float>.zeros([32, 10])
        let labels = Tensor<Float>.zeros([32, 10])

        let loss = sigmoidCrossEntropy(logits: logits, labels: labels, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }

    func testSigmoidCrossEntropyNoReduction() {
        let logits = Tensor<Float>.zeros([32, 10])
        let labels = Tensor<Float>.zeros([32, 10])

        let loss = sigmoidCrossEntropy(logits: logits, labels: labels, reduction: .none)

        XCTAssertEqual(loss.shape, [32, 10])
    }
}

// MARK: - KL Divergence Tests

final class KLDivergenceTests: XCTestCase {

    func testKullbackLeiblerDivergence() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.ones([32, 10])

        let loss = kullbackLeiblerDivergence(predicted: predicted, expected: expected, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }
}

// MARK: - Cosine Distance Tests

final class CosineDistanceTests: XCTestCase {

    func testCosineDistance() {
        let predicted = Tensor<Float>.ones([32, 128])
        let expected = Tensor<Float>.ones([32, 128])

        let loss = cosineDistance(predicted: predicted, expected: expected, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }

    func testCosineDistanceNoReduction() {
        let predicted = Tensor<Float>.ones([32, 128])
        let expected = Tensor<Float>.ones([32, 128])

        let loss = cosineDistance(predicted: predicted, expected: expected, reduction: .none)

        XCTAssertEqual(loss.shape, [32])
    }
}

// MARK: - Contrastive Loss Tests

final class ContrastiveLossTests: XCTestCase {

    func testContrastiveLoss() {
        let anchor = Tensor<Float>.ones([32, 128])
        let sample = Tensor<Float>.ones([32, 128])
        let labels = Tensor<Float>.ones([32])  // 1 for similar pairs

        let loss = contrastiveLoss(anchor: anchor, sample: sample, labels: labels, margin: 1.0, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }

    func testContrastiveLossCustomMargin() {
        let anchor = Tensor<Float>.ones([32, 128])
        let sample = Tensor<Float>.ones([32, 128])
        let labels = Tensor<Float>.zeros([32])  // 0 for dissimilar pairs

        let loss = contrastiveLoss(anchor: anchor, sample: sample, labels: labels, margin: 2.0, reduction: .mean)

        XCTAssertEqual(loss.shape, [])
    }
}

// MARK: - Triplet Margin Loss Tests

final class TripletMarginLossTests: XCTestCase {

    func testTripletMarginLoss() {
        let anchor = Tensor<Float>.ones([32, 128])
        let positive = Tensor<Float>.ones([32, 128])
        let negative = Tensor<Float>.zeros([32, 128])

        let loss = tripletMarginLoss(
            anchor: anchor,
            positive: positive,
            negative: negative,
            margin: 1.0,
            reduction: .mean
        )

        XCTAssertEqual(loss.shape, [])
    }

    func testTripletMarginLossCustomMargin() {
        let anchor = Tensor<Float>.ones([32, 128])
        let positive = Tensor<Float>.ones([32, 128])
        let negative = Tensor<Float>.zeros([32, 128])

        let loss = tripletMarginLoss(
            anchor: anchor,
            positive: positive,
            negative: negative,
            margin: 0.5,
            reduction: .mean
        )

        XCTAssertEqual(loss.shape, [])
    }
}

// MARK: - Loss Reduction Tests

final class LossReductionTests: XCTestCase {

    func testAllReductionTypesForL1() {
        let predicted = Tensor<Float>.ones([16, 8])
        let expected = Tensor<Float>.zeros([16, 8])

        let meanLoss = l1Loss(predicted: predicted, expected: expected, reduction: .mean)
        let sumLoss = l1Loss(predicted: predicted, expected: expected, reduction: .sum)
        let noneLoss = l1Loss(predicted: predicted, expected: expected, reduction: .none)

        XCTAssertEqual(meanLoss.shape, [])
        XCTAssertEqual(sumLoss.shape, [])
        XCTAssertEqual(noneLoss.shape, [16, 8])
    }

    func testAllReductionTypesForL2() {
        let predicted = Tensor<Float>.ones([16, 8])
        let expected = Tensor<Float>.zeros([16, 8])

        let meanLoss = l2Loss(predicted: predicted, expected: expected, reduction: .mean)
        let sumLoss = l2Loss(predicted: predicted, expected: expected, reduction: .sum)
        let noneLoss = l2Loss(predicted: predicted, expected: expected, reduction: .none)

        XCTAssertEqual(meanLoss.shape, [])
        XCTAssertEqual(sumLoss.shape, [])
        XCTAssertEqual(noneLoss.shape, [16, 8])
    }
}
