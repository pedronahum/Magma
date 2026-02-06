// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
// Copyright 2024-2026 Pedro N. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Loss functions for neural networks.
// Ported from Swift for TensorFlow (S4TF).

import Foundation
import _Differentiation
import LazyTensor
import StableHLO
import XLARuntime

// MARK: - Helper Operations

extension Tensor where Scalar == Float {

    /// Element-wise squared
    public func squared() -> Tensor {
        self * self
    }

    /// Sum along specified axes
    ///
    /// - Parameter axes: The axes to reduce over. Negative indices are supported.
    /// - Returns: Tensor with sum computed along specified axes.
    public func sum(alongAxes axes: [Int]) -> Tensor {
        // Normalize negative indices
        let normalizedAxes = axes.map { $0 < 0 ? rank + $0 : $0 }

        // Compute output shape (removing reduced dimensions)
        let newShape = shape.enumerated()
            .filter { !normalizedAxes.contains($0.offset) }
            .map { $0.element }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: newShape.isEmpty ? [] : newShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .reduceSum, inputs: [self.handle], attributes: ["axes": normalizedAxes])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Sum along a single axis
    public func sum(alongAxes axis: Int) -> Tensor {
        sum(alongAxes: [axis])
    }

    /// Maximum along specified axes
    ///
    /// - Parameter axes: The axes to reduce over. Negative indices are supported.
    /// - Returns: Tensor with maximum computed along specified axes.
    public func max(alongAxes axes: [Int]) -> Tensor {
        // Normalize negative indices
        let normalizedAxes = axes.map { $0 < 0 ? rank + $0 : $0 }

        // Compute output shape (removing reduced dimensions)
        let newShape = shape.enumerated()
            .filter { !normalizedAxes.contains($0.offset) }
            .map { $0.element }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: newShape.isEmpty ? [] : newShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .reduceMax, inputs: [self.handle], attributes: ["axes": normalizedAxes])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Maximum along a single axis
    public func max(alongAxes axis: Int) -> Tensor {
        max(alongAxes: [axis])
    }

    /// Minimum along specified axes
    ///
    /// - Parameter axes: The axes to reduce over. Negative indices are supported.
    /// - Returns: Tensor with minimum computed along specified axes.
    public func min(alongAxes axes: [Int]) -> Tensor {
        // Normalize negative indices
        let normalizedAxes = axes.map { $0 < 0 ? rank + $0 : $0 }

        // Compute output shape (removing reduced dimensions)
        let newShape = shape.enumerated()
            .filter { !normalizedAxes.contains($0.offset) }
            .map { $0.element }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: newShape.isEmpty ? [] : newShape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .reduceMin, inputs: [self.handle], attributes: ["axes": normalizedAxes])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// Minimum along a single axis
    public func min(alongAxes axis: Int) -> Tensor {
        min(alongAxes: [axis])
    }

    /// Softplus activation: log(1 + exp(x))
    ///
    /// This is a smooth approximation to ReLU.
    public func softplus() -> Tensor {
        // softplus(x) = log(1 + exp(x))
        // For numerical stability with large x: softplus(x) ≈ x when x >> 0
        // Using log1p for better stability: softplus(x) = log1p(exp(x))
        self.exp().log1p()
    }

    /// Log of 1 plus x: log(1 + x)
    ///
    /// For small x, this is more numerically stable than log(1 + x).
    /// Note: Uses the identity log1p(x) = log(1 + x).
    public func log1p() -> Tensor {
        let one = Tensor<Float>.ones(shape, on: device)
        return (self + one).log()
    }

    /// Add a dimension at the specified axis
    public func expandingShape(at axis: Int) -> Tensor {
        let normalizedAxis = axis < 0 ? rank + axis + 1 : axis
        var newShape = shape
        newShape.insert(1, at: normalizedAxis)
        return reshape(newShape)
    }

    /// Remove a dimension of size 1 at the specified axis.
    ///
    /// Similar to PyTorch's squeeze(dim).
    ///
    /// - Parameter dim: The dimension to remove. Must have size 1.
    /// - Returns: Tensor with the specified dimension removed.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>.ones([2, 1, 3])
    /// let squeezed = x.squeeze(dim: 1)  // shape: [2, 3]
    /// ```
    public func squeeze(dim: Int) -> Tensor {
        let normalizedDim = dim < 0 ? rank + dim : dim
        precondition(normalizedDim >= 0 && normalizedDim < rank,
            "squeeze dim \(dim) out of bounds for rank \(rank)")
        precondition(shape[normalizedDim] == 1,
            "squeeze requires dim \(dim) to have size 1, but got size \(shape[normalizedDim])")
        var newShape = shape
        newShape.remove(at: normalizedDim)
        return reshape(newShape.isEmpty ? [1] : newShape)
    }

    /// Remove all dimensions of size 1.
    ///
    /// Similar to PyTorch's squeeze() with no arguments.
    ///
    /// - Returns: Tensor with all size-1 dimensions removed.
    ///
    /// Example:
    /// ```swift
    /// let x = Tensor<Float>.ones([1, 2, 1, 3, 1])
    /// let squeezed = x.squeeze()  // shape: [2, 3]
    /// ```
    public func squeeze() -> Tensor {
        let newShape = shape.filter { $0 != 1 }
        return reshape(newShape.isEmpty ? [1] : newShape)
    }
}

// MARK: - Free Functions for Element-wise Operations

/// Element-wise absolute value
public func abs(_ x: Tensor<Float>) -> Tensor<Float> {
    x.abs()
}

/// Element-wise natural logarithm
public func log(_ x: Tensor<Float>) -> Tensor<Float> {
    x.log()
}

/// Element-wise exponential
public func exp(_ x: Tensor<Float>) -> Tensor<Float> {
    x.exp()
}

/// Element-wise maximum of two tensors
public func max(_ x: Tensor<Float>, _ y: Tensor<Float>) -> Tensor<Float> {
    let id = TensorRegistry.shared.nextTensorId()
    let handle = LazyTensorHandle(
        id: id,
        shape: x.shape,
        dtype: x.dtype,
        device: x.device
    )
    handle.irNode = .operation(op: .maximum, inputs: [x.handle, y.handle], attributes: [:])
    TensorRegistry.shared.registerPending(handle)
    return Tensor(handle: handle)
}

/// Element-wise minimum of two tensors
public func min(_ x: Tensor<Float>, _ y: Tensor<Float>) -> Tensor<Float> {
    let id = TensorRegistry.shared.nextTensorId()
    let handle = LazyTensorHandle(
        id: id,
        shape: x.shape,
        dtype: x.dtype,
        device: x.device
    )
    handle.irNode = .operation(op: .minimum, inputs: [x.handle, y.handle], attributes: [:])
    TensorRegistry.shared.registerPending(handle)
    return Tensor(handle: handle)
}

/// Softplus activation: log(1 + exp(x))
public func softplus(_ x: Tensor<Float>) -> Tensor<Float> {
    x.softplus()
}

// MARK: - Reduction Type

/// Specifies how to reduce element-wise losses to a scalar.
public enum LossReduction {
    /// Sum all loss values
    case sum
    /// Mean of all loss values
    case mean
    /// No reduction, return element-wise losses
    case none
}

// MARK: - Loss Functions

/// Computes the L1 loss between `expected` and `predicted`.
///
/// `loss = reduction(abs(expected - predicted))`
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The L1 loss.
public func l1Loss(
    predicted: Tensor<Float>,
    expected: Tensor<Float>,
    reduction: LossReduction = .sum
) -> Tensor<Float> {
    let elementwise = abs(expected - predicted)
    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

/// Computes the L2 loss between `expected` and `predicted`.
///
/// `loss = reduction(square(expected - predicted))`
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The L2 loss.
public func l2Loss(
    predicted: Tensor<Float>,
    expected: Tensor<Float>,
    reduction: LossReduction = .sum
) -> Tensor<Float> {
    let elementwise = (expected - predicted).squared()
    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

/// Computes the mean of absolute difference between labels and predictions.
///
/// `loss = mean(abs(expected - predicted))`
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
/// - Returns: The mean absolute error.
public func meanAbsoluteError(
    predicted: Tensor<Float>,
    expected: Tensor<Float>
) -> Tensor<Float> {
    l1Loss(predicted: predicted, expected: expected, reduction: .mean)
}

/// Computes the mean of squares of errors between labels and predictions.
///
/// `loss = mean(square(expected - predicted))`
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
/// - Returns: The mean squared error.
public func meanSquaredError(
    predicted: Tensor<Float>,
    expected: Tensor<Float>
) -> Tensor<Float> {
    l2Loss(predicted: predicted, expected: expected, reduction: .mean)
}

/// Computes the mean squared logarithmic error between `predicted` and `expected`.
///
/// `loss = mean(square(log(expected + 1) - log(predicted + 1)))`
///
/// - Note: Negative tensor entries will be clamped at `0` to avoid undefined
///   logarithmic behavior, as `log(_:)` is undefined for negative reals.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
/// - Returns: The mean squared logarithmic error.
public func meanSquaredLogarithmicError(
    predicted: Tensor<Float>,
    expected: Tensor<Float>
) -> Tensor<Float> {
    let zero = Tensor<Float>.zeros(predicted.shape, on: predicted.device)
    let one = Tensor<Float>.ones(predicted.shape, on: predicted.device)
    let logPredicted = log(max(predicted, zero) + one)
    let logExpected = log(max(expected, zero) + one)
    return l2Loss(predicted: logPredicted, expected: logExpected, reduction: .mean)
}

/// Computes the mean absolute percentage error between `predicted` and `expected`.
///
/// `loss = 100 * mean(abs((expected - predicted) / abs(expected)))`
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
/// - Returns: The mean absolute percentage error.
public func meanAbsolutePercentageError(
    predicted: Tensor<Float>,
    expected: Tensor<Float>
) -> Tensor<Float> {
    let hundred = Tensor<Float>([100.0], shape: [], on: predicted.device)
    return hundred * (abs(expected - predicted) / abs(expected)).mean()
}

/// Computes the hinge loss between `predicted` and `expected`.
///
/// `loss = reduction(max(0, 1 - predicted * expected))`
///
/// - Note: `expected` values are expected to be -1 or 1.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets (-1 or 1).
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The hinge loss.
public func hingeLoss(
    predicted: Tensor<Float>,
    expected: Tensor<Float>,
    reduction: LossReduction = .mean
) -> Tensor<Float> {
    let zero = Tensor<Float>.zeros(predicted.shape, on: predicted.device)
    let one = Tensor<Float>.ones(predicted.shape, on: predicted.device)
    let elementwise = max(zero, one - expected * predicted)
    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

/// Computes the squared hinge loss between `predicted` and `expected`.
///
/// `loss = reduction(square(max(0, 1 - predicted * expected)))`
///
/// - Note: `expected` values are expected to be -1 or 1.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets (-1 or 1).
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The squared hinge loss.
public func squaredHingeLoss(
    predicted: Tensor<Float>,
    expected: Tensor<Float>,
    reduction: LossReduction = .mean
) -> Tensor<Float> {
    let elementwise = hingeLoss(predicted: predicted, expected: expected, reduction: .none).squared()
    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

/// Computes the categorical hinge loss between `predicted` and `expected`.
///
/// `loss = maximum(negative - positive + 1, 0)`
/// where `negative = max((1 - expected) * predicted)` and
/// `positive = sum(predicted * expected)`
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values (one-hot encoded targets).
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The categorical hinge loss.
public func categoricalHingeLoss(
    predicted: Tensor<Float>,
    expected: Tensor<Float>,
    reduction: LossReduction = .mean
) -> Tensor<Float> {
    let one = Tensor<Float>.ones(predicted.shape, on: predicted.device)
    let positive = (expected * predicted).sum(alongAxes: -1)
    let negative = ((one - expected) * predicted).max(alongAxes: -1)

    // Create scalars for comparison
    let zeroScalar = Tensor<Float>.zeros(positive.shape, on: predicted.device)
    let oneScalar = Tensor<Float>.ones(positive.shape, on: predicted.device)

    let elementwise = max(zeroScalar, negative - positive + oneScalar)
    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

/// Computes the logarithm of the hyperbolic cosine of the prediction error.
///
/// `logcosh = log((exp(x) + exp(-x))/2)`,
/// where x is the error `predicted - expected`
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The log-cosh loss.
public func logCoshLoss(
    predicted: Tensor<Float>,
    expected: Tensor<Float>,
    reduction: LossReduction = .mean
) -> Tensor<Float> {
    let x = predicted - expected
    let negTwo = Tensor<Float>([-2.0], shape: [], on: predicted.device)
    let logTwo = Tensor<Float>([Foundation.log(2.0)], shape: [], on: predicted.device)

    // logcosh(x) = x + softplus(-2x) - log(2)
    let elementwise = x + softplus(negTwo.broadcast(to: x.shape) * x) - logTwo.broadcast(to: x.shape)
    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

/// Computes the Poisson loss between predicted and expected.
///
/// The Poisson loss is the mean of the elements of the tensor
/// `predicted - expected * log(predicted)`.
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The Poisson loss.
public func poissonLoss(
    predicted: Tensor<Float>,
    expected: Tensor<Float>,
    reduction: LossReduction = .mean
) -> Tensor<Float> {
    let elementwise = predicted - expected * log(predicted)
    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

/// Computes Kullback-Leibler divergence loss between `expected` and `predicted`.
///
/// `loss = reduction(expected * log(expected / predicted))`
///
/// - Parameters:
///   - predicted: Predicted probability distribution.
///   - expected: Expected probability distribution (target).
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The KL divergence.
public func kullbackLeiblerDivergence(
    predicted: Tensor<Float>,
    expected: Tensor<Float>,
    reduction: LossReduction = .sum
) -> Tensor<Float> {
    let elementwise = expected * log(expected / predicted)
    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

/// Computes the softmax cross entropy (categorical cross entropy) between logits and probabilities.
///
/// Use this crossentropy loss function when there are two or more label classes.
/// Labels should be provided in a `one_hot` representation.
/// There should be `# classes` floating point values per feature.
///
/// - Parameters:
///   - logits: Unscaled log probabilities from a neural network.
///   - probabilities: Probability values that correspond to the correct output.
///                    Each row must be a valid probability distribution.
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The softmax cross entropy loss.
public func softmaxCrossEntropy(
    logits: Tensor<Float>,
    probabilities: Tensor<Float>,
    reduction: LossReduction = .mean
) -> Tensor<Float> {
    // Numerically stable computation:
    // loss = -sum(labels * log_softmax(logits))
    // log_softmax(x) = x - log(sum(exp(x)))

    // For numerical stability, subtract the max from logits
    let maxLogits = logits.max(alongAxes: -1).expandingShape(at: -1).broadcast(to: logits.shape)
    let shiftedLogits = logits - maxLogits

    // Compute log(sum(exp(shifted_logits)))
    let expLogits = shiftedLogits.exp()
    let sumExp = expLogits.sum(alongAxes: -1).expandingShape(at: -1).broadcast(to: logits.shape)
    let logSumExp = sumExp.log()

    // log_softmax = shifted_logits - log_sum_exp
    let logSoftmax = shiftedLogits - logSumExp

    // Cross entropy = -sum(probabilities * log_softmax) along class axis
    let crossEntropyPerSample = -(probabilities * logSoftmax).sum(alongAxes: -1)

    switch reduction {
    case .sum: return crossEntropyPerSample.sum()
    case .mean: return crossEntropyPerSample.mean()
    case .none: return crossEntropyPerSample
    }
}

/// Computes the sparse softmax cross entropy (categorical cross entropy) between logits and labels.
///
/// Use this crossentropy loss function when there are two or more label classes.
/// Labels are provided as integers (class indices).
/// There should be `# classes` floating point values per feature for `logits`
/// and a single integer value per feature for `labels`.
///
/// - Note: This version requires converting integer labels to one-hot encoding first.
///         For a more efficient version, use the gather operation when available.
///
/// - Parameters:
///   - logits: Unnormalized log probabilities from a neural network, shape [batch, numClasses].
///   - labels: Class indices (integers), shape [batch].
///   - numClasses: The number of classes.
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The sparse softmax cross entropy loss.
public func softmaxCrossEntropyWithLabels(
    logits: Tensor<Float>,
    labels: [Int],
    numClasses: Int,
    reduction: LossReduction = .mean
) -> Tensor<Float> {
    // Convert labels to one-hot encoding
    let batchSize = logits.shape[0]
    var oneHotData = Array(repeating: Float(0), count: batchSize * numClasses)
    for (i, label) in labels.enumerated() {
        oneHotData[i * numClasses + label] = 1.0
    }
    let oneHot = Tensor<Float>(oneHotData, shape: [batchSize, numClasses], on: logits.device)

    return softmaxCrossEntropy(logits: logits, probabilities: oneHot, reduction: reduction)
}

/// Computes the sigmoid cross entropy (binary cross entropy) between logits and labels.
///
/// Use this cross-entropy loss when there are only two label classes (assumed to
/// be 0 and 1). For each example, there should be a single floating-point value
/// per prediction.
///
/// - Parameters:
///   - logits: The unscaled output of a neural network.
///   - labels: Float values (0.0 or 1.0) that correspond to the correct output.
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The sigmoid cross entropy loss.
public func sigmoidCrossEntropy(
    logits: Tensor<Float>,
    labels: Tensor<Float>,
    reduction: LossReduction = .mean
) -> Tensor<Float> {
    // This numerically stable implementation is based on the TensorFlow Python API.
    // loss = max(logits, 0) - logits * labels + log(1 + exp(-|logits|))

    let zero = Tensor<Float>.zeros(logits.shape, on: logits.device)
    let maxLogitsWithZero = max(logits, zero)

    // Custom abs for gradient computation at 0
    let negAbsLogits = max(logits, -logits)

    // log(1 + exp(-|logits|)) = log1p(exp(-|logits|))
    let logTerm = (-negAbsLogits).exp().log1p()

    let elementwise = maxLogitsWithZero - logits * labels + logTerm

    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

/// Computes the Huber loss between `predicted` and `expected`.
///
/// For each value `x` in `error = expected - predicted`:
/// - `0.5 * x^2` if `|x| <= δ`.
/// - `0.5 * δ^2 + δ * (|x| - δ)` otherwise.
///
/// Huber loss is less sensitive to outliers than MSE.
///
/// - Source: [Wikipedia article](https://en.wikipedia.org/wiki/Huber_loss)
///
/// - Parameters:
///   - predicted: Predicted outputs from a neural network.
///   - expected: Expected values, i.e. targets, that correspond to the correct output.
///   - delta: A floating point scalar representing the point where the Huber loss function
///            changes from quadratic to linear.
///   - reduction: Reduction to apply on the computed element-wise loss values.
/// - Returns: The Huber loss.
public func huberLoss(
    predicted: Tensor<Float>,
    expected: Tensor<Float>,
    delta: Float,
    reduction: LossReduction = .sum
) -> Tensor<Float> {
    let error = expected - predicted
    let absError = abs(error)

    let deltaT = Tensor<Float>.full(absError.shape, delta, on: predicted.device)
    let quadratic = min(absError, deltaT)
    let linear = absError - quadratic

    let half = Tensor<Float>([0.5], shape: [], on: predicted.device)
    let elementwise = half.broadcast(to: quadratic.shape) * quadratic * quadratic + deltaT * linear

    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

// MARK: - Contrastive and Metric Learning Losses

/// Computes the cosine similarity between two tensors.
///
/// `similarity = sum(a * b) / (norm(a) * norm(b))`
///
/// - Parameters:
///   - a: First tensor.
///   - b: Second tensor.
///   - epsilon: Small value for numerical stability.
/// - Returns: Cosine similarity value(s).
public func cosineSimilarity(
    _ a: Tensor<Float>,
    _ b: Tensor<Float>,
    epsilon: Float = 1e-8
) -> Tensor<Float> {
    let dotProduct = (a * b).sum(alongAxes: -1)
    let normA = (a * a).sum(alongAxes: -1).sqrt()
    let normB = (b * b).sum(alongAxes: -1).sqrt()
    let eps = Tensor<Float>.full(dotProduct.shape, epsilon, on: a.device)
    return dotProduct / (normA * normB + eps)
}

/// Computes the cosine distance (1 - cosine similarity) between two tensors.
///
/// - Parameters:
///   - predicted: Predicted embedding.
///   - expected: Expected embedding.
///   - reduction: Reduction to apply.
/// - Returns: The cosine distance.
public func cosineDistance(
    predicted: Tensor<Float>,
    expected: Tensor<Float>,
    reduction: LossReduction = .mean
) -> Tensor<Float> {
    let one = Tensor<Float>.ones([], on: predicted.device)
    let elementwise = one - cosineSimilarity(predicted, expected)
    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

/// Computes the contrastive loss.
///
/// Used in Siamese networks for learning similarity metrics.
/// `loss = (1 - y) * d^2 + y * max(margin - d, 0)^2`
/// where `y` is 1 for similar pairs, 0 for dissimilar pairs, and `d` is the distance.
///
/// - Parameters:
///   - predicted: Predicted embeddings, shape [batch, 2, embedding_dim].
///   - expected: Labels indicating whether pairs are similar (1) or dissimilar (0).
///   - margin: The margin for dissimilar pairs.
///   - reduction: Reduction to apply.
/// - Returns: The contrastive loss.
public func contrastiveLoss(
    anchor: Tensor<Float>,
    sample: Tensor<Float>,
    labels: Tensor<Float>,  // 1 for similar, 0 for dissimilar
    margin: Float = 1.0,
    reduction: LossReduction = .mean
) -> Tensor<Float> {
    // Compute Euclidean distance
    let diff = anchor - sample
    let distance = (diff * diff).sum(alongAxes: -1).sqrt()

    let one = Tensor<Float>.ones(distance.shape, on: anchor.device)
    let zero = Tensor<Float>.zeros(distance.shape, on: anchor.device)
    let marginT = Tensor<Float>.full(distance.shape, margin, on: anchor.device)

    // Similar pairs: minimize distance
    let similarLoss = labels * distance.squared()

    // Dissimilar pairs: maximize distance up to margin
    let dissimilarLoss = (one - labels) * max(marginT - distance, zero).squared()

    let half = Tensor<Float>([0.5], shape: [], on: anchor.device)
    let elementwise = half.broadcast(to: distance.shape) * (similarLoss + dissimilarLoss)

    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

/// Computes the triplet margin loss.
///
/// Used for learning embeddings where similar samples are closer than dissimilar ones.
/// `loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)`
///
/// - Parameters:
///   - anchor: Anchor embeddings.
///   - positive: Positive (similar) embeddings.
///   - negative: Negative (dissimilar) embeddings.
///   - margin: Minimum desired margin between positive and negative distances.
///   - reduction: Reduction to apply.
/// - Returns: The triplet margin loss.
public func tripletMarginLoss(
    anchor: Tensor<Float>,
    positive: Tensor<Float>,
    negative: Tensor<Float>,
    margin: Float = 1.0,
    reduction: LossReduction = .mean
) -> Tensor<Float> {
    // Compute distances
    let posDiff = anchor - positive
    let negDiff = anchor - negative

    let posDistance = (posDiff * posDiff).sum(alongAxes: -1).sqrt()
    let negDistance = (negDiff * negDiff).sum(alongAxes: -1).sqrt()

    let zero = Tensor<Float>.zeros(posDistance.shape, on: anchor.device)
    let marginT = Tensor<Float>.full(posDistance.shape, margin, on: anchor.device)

    let elementwise = max(posDistance - negDistance + marginT, zero)

    switch reduction {
    case .sum: return elementwise.sum()
    case .mean: return elementwise.mean()
    case .none: return elementwise
    }
}

// MARK: - Naming Conventions
//
// The loss functions above follow standard naming conventions:
//
// | Magma                    | PyTorch                      | TensorFlow/Keras           |
// |--------------------------------|------------------------------|----------------------------|
// | l1Loss                         | L1Loss                       | MeanAbsoluteError (mae)    |
// | l2Loss                         | MSELoss                      | MeanSquaredError (mse)     |
// | meanAbsoluteError              | L1Loss(reduction='mean')     | MeanAbsoluteError          |
// | meanSquaredError               | MSELoss(reduction='mean')    | MeanSquaredError           |
// | hingeLoss                      | HingeEmbeddingLoss           | Hinge                      |
// | squaredHingeLoss               | -                            | SquaredHinge               |
// | categoricalHingeLoss           | -                            | CategoricalHinge           |
// | logCoshLoss                    | -                            | LogCosh                    |
// | poissonLoss                    | PoissonNLLLoss               | Poisson                    |
// | kullbackLeiblerDivergence      | KLDivLoss                    | KLDivergence               |
// | softmaxCrossEntropy            | CrossEntropyLoss             | CategoricalCrossentropy    |
// | sigmoidCrossEntropy            | BCEWithLogitsLoss            | BinaryCrossentropy         |
// | huberLoss                      | HuberLoss                    | Huber                      |
// | cosineSimilarity               | CosineSimilarity             | CosineSimilarity           |
// | cosineDistance                 | CosineEmbeddingLoss          | CosineSimilarity (neg)     |
// | contrastiveLoss                | -                            | -                          |
// | tripletMarginLoss              | TripletMarginLoss            | -                          |
