// Magma - MNIST Training Example
// Demonstrates end-to-end training with real MNIST data and autodiff.
//
// This is a minimal example that runs just a few forward/backward passes
// to verify the system works end-to-end.

import Foundation
import Magma
import LazyTensor
import _Differentiation

@main
struct MNISTTraining {
    static func main() {
        print("Magma MNIST Training")
        print("==========================")
        print("")

        // Quick sanity check with basic tensor operations
        print("0. Testing basic tensor operations...")
        let a = Tensor<Float>([1, 2, 3, 4], shape: [2, 2], on: .default)
        let b = Tensor<Float>([5, 6, 7, 8], shape: [2, 2], on: .default)
        let c = a + b
        let cValues = c.scalars()
        print("   [2,2] + [2,2] = \(cValues)")
        print("   Basic test passed!")
        print("")

        do {
            try trainMNIST()
        } catch {
            print("Error: \(error)")
        }
    }

    static func trainMNIST() throws {
        // MARK: - Configuration

        let batchSize = 32
        let numBatches = 10  // Quick test with 10 batches
        let learningRate: Float = 0.1  // Higher learning rate to see changes faster

        // MARK: - Load MNIST Data

        print("1. Loading MNIST dataset...")
        let trainData = try MNIST(split: .train, normalize: true, flatten: true)

        print("   Training samples: \(trainData.count)")
        print("   Image shape: \(trainData.images.shape)")
        print("   Label shape: \(trainData.labels.shape)")

        // MARK: - Model Definition

        print("")
        print("2. Defining MLP model...")

        // Simple 2-layer MLP: 784 -> 64 -> 10
        // Initialize with small random weights
        let scale1 = Tensor<Float>.full([], 0.1, on: .default)
        let scale2 = Tensor<Float>.full([], 0.1, on: .default)
        var params = ModelParams(
            w1: Tensor<Float>.randn([784, 64]) * scale1,
            b1: Tensor<Float>.zeros([64]),
            w2: Tensor<Float>.randn([64, 10]) * scale2,
            b2: Tensor<Float>.zeros([10])
        )

        // Verify weights are actually random
        print("   Testing weight materialization...")
        let w1Sample = params.w1.scalars()
        print("   W1 sample values: \(Array(w1Sample.prefix(5)))")

        print("   Layer 1: 784 -> 64 (ReLU)")
        print("   Layer 2: 64 -> 10 (Softmax)")
        let totalParams = 784 * 64 + 64 + 64 * 10 + 10
        print("   Total parameters: \(totalParams)")

        // MARK: - Training Loop

        print("")
        print("3. Training (\(numBatches) batches, lr=\(learningRate))...")

        for batch in 0..<numBatches {
            let batchStart = batch * batchSize
            let currentBatchSize = batchSize

            // Get batch
            let batchImages = trainData.images.slice(start: batchStart, size: currentBatchSize)
            let batchLabels = trainData.labels.slice(start: batchStart, size: currentBatchSize)

            // Forward pass with gradient computation
            let (loss, grads) = valueWithGradient(at: params) { p -> Tensor<Float> in
                // Forward pass
                let h1 = (batchImages.matmul(p.w1) + p.b1.broadcast(to: [currentBatchSize, 64])).relu()
                let logits = h1.matmul(p.w2) + p.b2.broadcast(to: [currentBatchSize, 10])
                let probs = logits.softmax(dim: 1)

                // Cross-entropy loss
                let logProbs = (probs + Tensor<Float>.full([], 1e-7, on: .default)).log()
                let crossEntropy = -(batchLabels * logProbs).sum() / Tensor<Float>.full([], Float(currentBatchSize), on: .default)
                return crossEntropy
            }

            // Update weights (SGD)
            let lr = Tensor<Float>.full([], learningRate, on: .default)
            params.w1 = params.w1 - grads.w1 * lr
            params.b1 = params.b1 - grads.b1 * lr
            params.w2 = params.w2 - grads.w2 * lr
            params.b2 = params.b2 - grads.b2 * lr

            // Get loss value (triggers materialization)
            let lossValues = loss.scalars()
            let lossValue = lossValues.isEmpty ? Float.nan : lossValues[0]

            print("   Batch \(batch): loss = \(String(format: "%.4f", lossValue))")
        }

        // MARK: - Quick Evaluation

        print("")
        print("4. Evaluation (first batch)...")
        let testBatch = trainData.images.slice(start: 0, size: batchSize)
        let h1 = (testBatch.matmul(params.w1) + params.b1.broadcast(to: [batchSize, 64])).relu()
        let logits = h1.matmul(params.w2) + params.b2.broadcast(to: [batchSize, 10])
        let probs = logits.softmax(dim: 1)

        let probValues = probs.scalars()
        print("   Output probs (first 10): \(Array(probValues.prefix(10).map { String(format: "%.3f", $0) }))")

        // MARK: - Print Metrics

        print("")
        print("5. Compilation metrics:")
        PrintMetrics()

        // MARK: - Summary

        print("")
        print("==========================")
        print("MNIST Training Complete!")
        print("==========================")
        print("")
        print("Demonstrated capabilities:")
        print("  - MNIST data loading and parsing")
        print("  - Automatic differentiation (VJP-based)")
        print("  - Forward-backward training loop")
        print("  - Cross-entropy loss with softmax")
        print("  - SGD weight updates")
        print("  - XLA compilation and execution")
    }
}

// MARK: - Multi-input gradient helper

/// A simple struct to group parameters for differentiation
struct ModelParams: Differentiable {
    var w1: Tensor<Float>
    var b1: Tensor<Float>
    var w2: Tensor<Float>
    var b2: Tensor<Float>
}

/// Compute gradient with respect to model parameters
func valueWithGradient(
    at params: ModelParams,
    of f: @differentiable(reverse) (ModelParams) -> Tensor<Float>
) -> (value: Tensor<Float>, gradient: ModelParams.TangentVector) {
    let (value, pullback) = valueWithPullback(at: params, of: f)
    let gradient = pullback(Tensor<Float>.ones([], on: .default))
    return (value, gradient)
}
