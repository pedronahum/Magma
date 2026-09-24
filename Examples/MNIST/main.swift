// Magma - MNIST Training Example
//
// A short smoke-test training loop on real MNIST data: a 784 -> 64 -> 10 MLP
// trained with Swift-native reverse-mode autodiff and plain SGD for a few
// hundred mini-batches (a fraction of one epoch), then evaluated on the MNIST
// test split. It checks that data loading, autodiff and XLA execution work end
// to end; it is not a tuned classifier (expect roughly 85-90% test accuracy
// with the defaults and ~20 s on a CPU plugin; not state-of-the-art numbers).
//
// Run:
//   MAGMA_XLA_PATH=/opt/xla/lib swift run MNISTExample [batches]
//
//   batches  number of 64-image training batches (default 200)
//
// Requirements:
//   - A PJRT plugin (pjrt_c_api_cpu_plugin.so, or the gpu/tpu variant) in
//     $MAGMA_XLA_PATH.
//   - Network access on the first run: the dataset (~11 MB) is downloaded from
//     https://storage.googleapis.com/cvdf-datasets/mnist/ and decompressed with
//     gzip into ~/.magma/data/mnist. Later runs read that cache offline. If a
//     download was interrupted, delete that directory and run again.

import Foundation
import Magma
import LazyTensor
import XLARuntime
import _Differentiation

@main
struct MNISTTraining {
    static func main() {
        requireBackend(target: "MNISTExample")

        var numBatches = 200
        if CommandLine.arguments.count > 1 {
            guard let n = Int(CommandLine.arguments[1]), n > 0 else {
                print("Usage: MNISTExample [batches]   (batches: positive integer, default 200)")
                exit(1)
            }
            numBatches = n
        }

        print("Magma MNIST Training (smoke test)")
        print("=================================")
        print("")

        do {
            try trainMNIST(numBatches: numBatches)
        } catch {
            print("Error: \(error)")
            print("The first run downloads MNIST into ~/.magma/data/mnist and needs network access.")
            exit(1)
        }
    }

    static func trainMNIST(numBatches: Int) throws {
        // MARK: - Configuration

        let batchSize = 64
        let learningRate: Float = 0.1

        // MARK: - Load MNIST Data

        print("1. Loading MNIST dataset...")
        let trainData = try MNIST(split: .train, normalize: true, flatten: true)
        let testData = try MNIST(split: .test, normalize: true, flatten: true, oneHot: false)

        print("   Training samples: \(trainData.count)")
        print("   Test samples:     \(testData.count)")
        print("   Image shape: \(trainData.images.shape)")

        let maxBatches = trainData.count / batchSize
        let batches = min(numBatches, maxBatches)

        // MARK: - Model Definition

        print("")
        print("2. Defining MLP model...")

        // Simple 2-layer MLP: 784 -> 64 -> 10, small random initial weights.
        let scale = Tensor<Float>.full([], 0.1, on: .default)
        var params = ModelParams(
            w1: Tensor<Float>.randn([784, 64]) * scale,
            b1: Tensor<Float>.zeros([64]),
            w2: Tensor<Float>.randn([64, 10]) * scale,
            b2: Tensor<Float>.zeros([10])
        )

        print("   Layer 1: 784 -> 64 (ReLU)")
        print("   Layer 2: 64 -> 10 (Softmax)")
        let totalParams = 784 * 64 + 64 + 64 * 10 + 10
        print("   Total parameters: \(totalParams)")

        // MARK: - Training Loop

        print("")
        print("3. Training (\(batches) batches of \(batchSize), lr=\(learningRate))...")

        let lr = Tensor<Float>.full([], learningRate, on: .default)
        let reportEvery = max(1, batches / 10)
        for batch in 0..<batches {
            let batchImages = trainData.images.slice(start: batch * batchSize, size: batchSize)
            let batchLabels = trainData.labels.slice(start: batch * batchSize, size: batchSize)

            let (loss, grads) = valueWithGradient(at: params) { p -> Tensor<Float> in
                let probs = forward(p, batchImages).softmax(dim: 1)

                // Cross-entropy loss
                let logProbs = (probs + Tensor<Float>.full([], 1e-7, on: .default)).log()
                return -(batchLabels * logProbs).sum() / Tensor<Float>.full([], Float(batchSize), on: .default)
            }

            // SGD update
            params.w1 = params.w1 - grads.w1 * lr
            params.b1 = params.b1 - grads.b1 * lr
            params.w2 = params.w2 - grads.w2 * lr
            params.b2 = params.b2 - grads.b2 * lr

            if batch % reportEvery == 0 || batch == batches - 1 {
                // Reading the loss triggers compilation and execution.
                print("   Batch \(String(format: "%4d", batch)): loss = \(String(format: "%.4f", loss.item()))")
            }
        }

        // MARK: - Evaluation

        print("")
        print("4. Evaluation on the test split (\(testData.count) images)...")
        let logits = forward(params, testData.images).scalars()
        let labels = testData.labels.scalars()
        guard logits.count == testData.count * 10, labels.count == testData.count else {
            throw MNISTExampleError.executionFailed
        }
        var correct = 0
        for i in 0..<testData.count {
            let row = logits[(i * 10)..<(i * 10 + 10)]
            let predicted = row.indices.max { row[$0] < row[$1] }! - i * 10
            if predicted == Int(labels[i]) { correct += 1 }
        }
        let accuracy = Double(correct) / Double(testData.count) * 100
        print("   Test accuracy: \(String(format: "%.2f", accuracy))% (\(correct)/\(testData.count))")

        // MARK: - Print Metrics

        print("")
        print("5. Compilation metrics:")
        PrintMetrics()

        print("")
        print("Done.")
    }
}

/// MLP forward pass returning logits.
@differentiable(reverse, wrt: p)
func forward(_ p: ModelParams, _ images: Tensor<Float>) -> Tensor<Float> {
    let n = images.shape[0]
    let h1 = (images.matmul(p.w1) + p.b1.broadcast(to: [n, 64])).relu()
    return h1.matmul(p.w2) + p.b2.broadcast(to: [n, 10])
}

enum MNISTExampleError: Error, CustomStringConvertible {
    case executionFailed

    var description: String {
        "evaluation produced no values (did the PJRT plugin fail to compile or execute?)"
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

// MARK: - Backend preflight

/// Exits with an actionable message when no execution backend is installed.
///
/// Without a PJRT plugin every compile fails and reads return no values, so the
/// example would otherwise crash or print NaNs.
func requireBackend(target: String) {
    guard Backend.availableBackends.isEmpty else { return }
    #if os(macOS)
    let ext = "dylib"
    #else
    let ext = "so"
    #endif
    let path = ProcessInfo.processInfo.environment["MAGMA_XLA_PATH"]
    let searched = path.map { "MAGMA_XLA_PATH=\($0)" }
        ?? "MAGMA_XLA_PATH is not set; searched the default locations such as /opt/xla/lib"
    let message = """
        error: no execution backend found (\(searched)).
        Set MAGMA_XLA_PATH to the directory that contains pjrt_c_api_cpu_plugin.\(ext)
        (or pjrt_c_api_gpu_plugin.\(ext) / pjrt_c_api_tpu_plugin.\(ext)), for example:
            MAGMA_XLA_PATH=/opt/xla/lib swift run \(target)

        """
    FileHandle.standardError.write(Data(message.utf8))
    exit(1)
}
