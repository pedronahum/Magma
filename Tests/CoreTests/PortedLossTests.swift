// Magma - Ported Loss Functions Tests
// Tests for loss functions ported from S4TF

import Testing
@testable import Magma
@testable import LazyTensor

// MARK: - L1 Loss Tests

@Suite("L1 Loss Tests")
struct L1LossTests {

    @Test("L1 loss mean reduction")
    func l1LossMeanReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l1Loss(predicted: predicted, expected: expected, reduction: .mean)

        // L1 loss of all 1s vs all 0s = mean(|1 - 0|) = 1.0
        #expect(loss.shape == [])  // Scalar output
    }

    @Test("L1 loss sum reduction")
    func l1LossSumReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l1Loss(predicted: predicted, expected: expected, reduction: .sum)

        #expect(loss.shape == [])  // Scalar output
    }

    @Test("L1 loss no reduction")
    func l1LossNoReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l1Loss(predicted: predicted, expected: expected, reduction: .none)

        #expect(loss.shape == [32, 10])  // Same shape as input
    }
}

// MARK: - L2 Loss Tests

@Suite("L2 Loss Tests")
struct L2LossTests {

    @Test("L2 loss mean reduction")
    func l2LossMeanReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l2Loss(predicted: predicted, expected: expected, reduction: .mean)

        #expect(loss.shape == [])
    }

    @Test("L2 loss sum reduction")
    func l2LossSumReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l2Loss(predicted: predicted, expected: expected, reduction: .sum)

        #expect(loss.shape == [])
    }

    @Test("L2 loss no reduction")
    func l2LossNoReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = l2Loss(predicted: predicted, expected: expected, reduction: .none)

        #expect(loss.shape == [32, 10])
    }
}

// MARK: - Mean Absolute Error Tests

@Suite("Mean Absolute Error Tests")
struct MeanAbsoluteErrorTests {

    @Test("Mean absolute error")
    func meanAbsoluteError_() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = meanAbsoluteError(predicted: predicted, expected: expected)

        #expect(loss.shape == [])
    }
}

// MARK: - Mean Squared Error Tests

@Suite("Mean Squared Error Tests")
struct MeanSquaredErrorTests {

    @Test("Mean squared error")
    func meanSquaredError_() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = meanSquaredError(predicted: predicted, expected: expected)

        #expect(loss.shape == [])
    }
}

// MARK: - Hinge Loss Tests

@Suite("Hinge Loss Tests")
struct HingeLossTests {

    @Test("Hinge loss")
    func hingeLoss_() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.ones([32, 10])  // Labels should be -1 or 1

        let loss = hingeLoss(predicted: predicted, expected: expected, reduction: .mean)

        #expect(loss.shape == [])
    }

    @Test("Hinge loss no reduction")
    func hingeLossNoReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.ones([32, 10])

        let loss = hingeLoss(predicted: predicted, expected: expected, reduction: .none)

        #expect(loss.shape == [32, 10])
    }
}

// MARK: - Squared Hinge Loss Tests

@Suite("Squared Hinge Loss Tests")
struct SquaredHingeLossTests {

    @Test("Squared hinge loss")
    func squaredHingeLoss_() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.ones([32, 10])

        let loss = squaredHingeLoss(predicted: predicted, expected: expected, reduction: .mean)

        #expect(loss.shape == [])
    }
}

// MARK: - Huber Loss Tests

@Suite("Huber Loss Tests")
struct HuberLossTests {

    @Test("Huber loss")
    func huberLoss_() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = huberLoss(predicted: predicted, expected: expected, delta: 1.0, reduction: .mean)

        #expect(loss.shape == [])
    }

    @Test("Huber loss custom delta")
    func huberLossCustomDelta() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = huberLoss(predicted: predicted, expected: expected, delta: 0.5, reduction: .mean)

        #expect(loss.shape == [])
    }

    @Test("Huber loss no reduction")
    func huberLossNoReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = huberLoss(predicted: predicted, expected: expected, delta: 1.0, reduction: .none)

        #expect(loss.shape == [32, 10])
    }
}

// MARK: - Log-Cosh Loss Tests

@Suite("Log-Cosh Loss Tests")
struct LogCoshLossTests {

    @Test("Log-cosh loss")
    func logCoshLoss_() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = logCoshLoss(predicted: predicted, expected: expected, reduction: .mean)

        #expect(loss.shape == [])
    }

    @Test("Log-cosh loss no reduction")
    func logCoshLossNoReduction() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.zeros([32, 10])

        let loss = logCoshLoss(predicted: predicted, expected: expected, reduction: .none)

        #expect(loss.shape == [32, 10])
    }
}

// MARK: - Poisson Loss Tests

@Suite("Poisson Loss Tests")
struct PoissonLossTests {

    @Test("Poisson loss")
    func poissonLoss_() {
        let predicted = Tensor<Float>.ones([32, 10])  // Must be positive
        let expected = Tensor<Float>.ones([32, 10])

        let loss = poissonLoss(predicted: predicted, expected: expected, reduction: .mean)

        #expect(loss.shape == [])
    }
}

// MARK: - Softmax Cross Entropy Tests

@Suite("Softmax Cross Entropy Tests")
struct SoftmaxCrossEntropyTests {

    @Test("Softmax cross entropy")
    func softmaxCrossEntropy_() {
        let logits = Tensor<Float>.zeros([32, 10])
        let labels = Tensor<Float>.zeros([32, 10])

        let loss = softmaxCrossEntropy(logits: logits, probabilities: labels, reduction: .mean)

        #expect(loss.shape == [])
    }

    @Test("Softmax cross entropy no reduction")
    func softmaxCrossEntropyNoReduction() {
        let logits = Tensor<Float>.zeros([32, 10])
        let labels = Tensor<Float>.zeros([32, 10])

        let loss = softmaxCrossEntropy(logits: logits, probabilities: labels, reduction: .none)

        #expect(loss.shape == [32])  // One loss per sample
    }
}

// MARK: - Sigmoid Cross Entropy Tests

@Suite("Sigmoid Cross Entropy Tests")
struct SigmoidCrossEntropyTests {

    @Test("Sigmoid cross entropy")
    func sigmoidCrossEntropy_() {
        let logits = Tensor<Float>.zeros([32, 10])
        let labels = Tensor<Float>.zeros([32, 10])

        let loss = sigmoidCrossEntropy(logits: logits, labels: labels, reduction: .mean)

        #expect(loss.shape == [])
    }

    @Test("Sigmoid cross entropy no reduction")
    func sigmoidCrossEntropyNoReduction() {
        let logits = Tensor<Float>.zeros([32, 10])
        let labels = Tensor<Float>.zeros([32, 10])

        let loss = sigmoidCrossEntropy(logits: logits, labels: labels, reduction: .none)

        #expect(loss.shape == [32, 10])
    }
}

// MARK: - KL Divergence Tests

@Suite("KL Divergence Tests")
struct KLDivergenceTests {

    @Test("Kullback-Leibler divergence")
    func kullbackLeiblerDivergence_() {
        let predicted = Tensor<Float>.ones([32, 10])
        let expected = Tensor<Float>.ones([32, 10])

        let loss = kullbackLeiblerDivergence(predicted: predicted, expected: expected, reduction: .mean)

        #expect(loss.shape == [])
    }
}

// MARK: - Cosine Distance Tests

@Suite("Cosine Distance Tests")
struct CosineDistanceTests {

    @Test("Cosine distance")
    func cosineDistance_() {
        let predicted = Tensor<Float>.ones([32, 128])
        let expected = Tensor<Float>.ones([32, 128])

        let loss = cosineDistance(predicted: predicted, expected: expected, reduction: .mean)

        #expect(loss.shape == [])
    }

    @Test("Cosine distance no reduction")
    func cosineDistanceNoReduction() {
        let predicted = Tensor<Float>.ones([32, 128])
        let expected = Tensor<Float>.ones([32, 128])

        let loss = cosineDistance(predicted: predicted, expected: expected, reduction: .none)

        #expect(loss.shape == [32])
    }
}

// MARK: - Contrastive Loss Tests

@Suite("Contrastive Loss Tests")
struct ContrastiveLossTests {

    @Test("Contrastive loss")
    func contrastiveLoss_() {
        let anchor = Tensor<Float>.ones([32, 128])
        let sample = Tensor<Float>.ones([32, 128])
        let labels = Tensor<Float>.ones([32])  // 1 for similar pairs

        let loss = contrastiveLoss(anchor: anchor, sample: sample, labels: labels, margin: 1.0, reduction: .mean)

        #expect(loss.shape == [])
    }

    @Test("Contrastive loss custom margin")
    func contrastiveLossCustomMargin() {
        let anchor = Tensor<Float>.ones([32, 128])
        let sample = Tensor<Float>.ones([32, 128])
        let labels = Tensor<Float>.zeros([32])  // 0 for dissimilar pairs

        let loss = contrastiveLoss(anchor: anchor, sample: sample, labels: labels, margin: 2.0, reduction: .mean)

        #expect(loss.shape == [])
    }
}

// MARK: - Triplet Margin Loss Tests

@Suite("Triplet Margin Loss Tests")
struct TripletMarginLossTests {

    @Test("Triplet margin loss")
    func tripletMarginLoss_() {
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

        #expect(loss.shape == [])
    }

    @Test("Triplet margin loss custom margin")
    func tripletMarginLossCustomMargin() {
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

        #expect(loss.shape == [])
    }
}

// MARK: - Loss Reduction Tests

@Suite("Loss Reduction Tests")
struct LossReductionTests {

    @Test("All reduction types for L1")
    func allReductionTypesForL1() {
        let predicted = Tensor<Float>.ones([16, 8])
        let expected = Tensor<Float>.zeros([16, 8])

        let meanLoss = l1Loss(predicted: predicted, expected: expected, reduction: .mean)
        let sumLoss = l1Loss(predicted: predicted, expected: expected, reduction: .sum)
        let noneLoss = l1Loss(predicted: predicted, expected: expected, reduction: .none)

        #expect(meanLoss.shape == [])
        #expect(sumLoss.shape == [])
        #expect(noneLoss.shape == [16, 8])
    }

    @Test("All reduction types for L2")
    func allReductionTypesForL2() {
        let predicted = Tensor<Float>.ones([16, 8])
        let expected = Tensor<Float>.zeros([16, 8])

        let meanLoss = l2Loss(predicted: predicted, expected: expected, reduction: .mean)
        let sumLoss = l2Loss(predicted: predicted, expected: expected, reduction: .sum)
        let noneLoss = l2Loss(predicted: predicted, expected: expected, reduction: .none)

        #expect(meanLoss.shape == [])
        #expect(sumLoss.shape == [])
        #expect(noneLoss.shape == [16, 8])
    }
}
