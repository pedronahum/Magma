// Magma - Module Tests
// Tests for neural network module system

import XCTest
@testable import Magma
@testable import LazyTensor

// MARK: - Parameter Tests

final class ParameterTests: XCTestCase {

    func testParameterCreation() {
        let tensor = Tensor<Float>.zeros([10, 20])
        let param = Parameter(tensor)

        XCTAssertEqual(param.shape, [10, 20])
        XCTAssertEqual(param.elementCount, 200)
        XCTAssertTrue(param.requiresGrad)
        XCTAssertNil(param.name)
    }

    func testParameterWithName() {
        let tensor = Tensor<Float>.ones([5, 5])
        let param = Parameter(tensor, name: "weight")

        XCTAssertEqual(param.name, "weight")
        XCTAssertTrue(param.requiresGrad)
    }

    func testParameterWithoutGrad() {
        let tensor = Tensor<Float>.zeros([3, 3])
        let param = Parameter(tensor, requiresGrad: false)

        XCTAssertFalse(param.requiresGrad)
    }
}

// MARK: - Linear Layer Tests

final class LinearLayerTests: XCTestCase {

    func testLinearCreation() {
        let linear = nn.Linear(inputSize: 784, outputSize: 256)

        XCTAssertEqual(linear.weight.shape, [256, 784])
        XCTAssertEqual(linear.bias.shape, [256])
        XCTAssertTrue(linear.useBias)
    }

    func testLinearNoBias() {
        let linear = nn.Linear(inputSize: 100, outputSize: 50, bias: false)

        XCTAssertEqual(linear.weight.shape, [50, 100])
        XCTAssertFalse(linear.useBias)
    }

    func testLinearForward() {
        let linear = nn.Linear(inputSize: 784, outputSize: 256)
        let input = Tensor<Float>.zeros([32, 784])

        let output = linear(input)

        XCTAssertEqual(output.shape, [32, 256])
    }

    func testLinearParameters() {
        let linearWithBias = nn.Linear(inputSize: 10, outputSize: 5, bias: true)
        let linearNoBias = nn.Linear(inputSize: 10, outputSize: 5, bias: false)

        XCTAssertEqual(linearWithBias.parameters().count, 2)
        XCTAssertEqual(linearNoBias.parameters().count, 1)
    }

    func testLinearBatchedInput() {
        let linear = nn.Linear(inputSize: 100, outputSize: 50)

        // Different batch sizes
        let input1 = Tensor<Float>.zeros([1, 100])
        let input2 = Tensor<Float>.zeros([64, 100])
        let input3 = Tensor<Float>.zeros([128, 100])

        XCTAssertEqual(linear(input1).shape, [1, 50])
        XCTAssertEqual(linear(input2).shape, [64, 50])
        XCTAssertEqual(linear(input3).shape, [128, 50])
    }
}

// MARK: - Activation Layer Tests

final class ActivationLayerTests: XCTestCase {

    func testReLULayer() {
        let relu = nn.ReLU()
        let input = Tensor<Float>.zeros([10, 20])

        let output = relu(input)

        XCTAssertEqual(output.shape, [10, 20])
    }

    func testSigmoidLayer() {
        let sigmoid = nn.Sigmoid()
        let input = Tensor<Float>.zeros([5, 5])

        let output = sigmoid(input)

        XCTAssertEqual(output.shape, [5, 5])
    }

    func testTanhLayer() {
        let tanh = nn.Tanh()
        let input = Tensor<Float>.zeros([3, 4, 5])

        let output = tanh(input)

        XCTAssertEqual(output.shape, [3, 4, 5])
    }

    func testGELULayer() {
        let gelu = nn.GELU()
        let input = Tensor<Float>.zeros([10, 10])

        let output = gelu(input)

        XCTAssertEqual(output.shape, [10, 10])
    }

    func testActivationsHaveNoParameters() {
        XCTAssertEqual(nn.ReLU().parameters().count, 0)
        XCTAssertEqual(nn.Sigmoid().parameters().count, 0)
        XCTAssertEqual(nn.Tanh().parameters().count, 0)
        XCTAssertEqual(nn.GELU().parameters().count, 0)
    }
}

// MARK: - Flatten Layer Tests

final class FlattenLayerTests: XCTestCase {

    func testFlattenDefault() {
        let flatten = nn.Flatten()
        let input = Tensor<Float>.zeros([32, 7, 7, 64])

        let output = flatten(input)

        // Default startDim=1: [batch, flattened]
        XCTAssertEqual(output.shape, [32, 7 * 7 * 64])
    }

    func testFlattenCustomStartDim() {
        let flatten = nn.Flatten(startDim: 2)
        let input = Tensor<Float>.zeros([2, 3, 4, 5])

        let output = flatten(input)

        // Flatten from dim 2: [2, 3, 4*5]
        XCTAssertEqual(output.shape, [2, 3, 20])
    }

    func testFlattenAlreadyFlat() {
        let flatten = nn.Flatten()
        let input = Tensor<Float>.zeros([10, 20])

        let output = flatten(input)

        XCTAssertEqual(output.shape, [10, 20])
    }
}

// MARK: - Dropout Layer Tests

final class DropoutLayerTests: XCTestCase {

    func testDropoutCreation() {
        let dropout = nn.Dropout(p: 0.5)

        XCTAssertEqual(dropout.p, 0.5)
        XCTAssertTrue(dropout.training)
    }

    func testDropoutShapePreserved() {
        let dropout = nn.Dropout(p: 0.5)
        let input = Tensor<Float>.zeros([32, 256])

        let output = dropout(input)

        XCTAssertEqual(output.shape, [32, 256])
    }

    func testDropoutNoParameters() {
        let dropout = nn.Dropout(p: 0.3)
        XCTAssertEqual(dropout.parameters().count, 0)
    }
}

// MARK: - Sequential Tests

final class SequentialTests: XCTestCase {

    func testSequentialEmpty() {
        let seq = nn.Sequential()
        let input = Tensor<Float>.zeros([10, 20])

        let output = seq(input)

        XCTAssertEqual(output.shape, [10, 20])
    }

    func testSequentialSingleLayer() {
        let seq = nn.Sequential(nn.AnyLayer(nn.ReLU()))
        let input = Tensor<Float>.zeros([10, 20])

        let output = seq(input)

        XCTAssertEqual(output.shape, [10, 20])
    }

    func testSequentialMLP() {
        let model = nn.Sequential(
            nn.AnyLayer(nn.Linear(inputSize: 784, outputSize: 256)),
            nn.AnyLayer(nn.ReLU()),
            nn.AnyLayer(nn.Linear(inputSize: 256, outputSize: 128)),
            nn.AnyLayer(nn.ReLU()),
            nn.AnyLayer(nn.Linear(inputSize: 128, outputSize: 10))
        )

        let input = Tensor<Float>.zeros([32, 784])
        let output = model(input)

        XCTAssertEqual(output.shape, [32, 10])
    }

    func testSequentialParameters() {
        let model = nn.Sequential(
            nn.AnyLayer(nn.Linear(inputSize: 10, outputSize: 5)),
            nn.AnyLayer(nn.ReLU()),
            nn.AnyLayer(nn.Linear(inputSize: 5, outputSize: 2))
        )

        let params = model.parameters()

        // 2 Linear layers with bias: 2 + 2 = 4 parameters
        XCTAssertEqual(params.count, 4)
    }

    func testSequentialResultBuilder() {
        let model = nn.sequential {
            nn.Linear(inputSize: 784, outputSize: 256)
            nn.ReLU()
            nn.Linear(inputSize: 256, outputSize: 10)
        }

        let input = Tensor<Float>.zeros([32, 784])
        let output = model(input)

        XCTAssertEqual(output.shape, [32, 10])
    }

    func testSequentialAddLayer() {
        var model = nn.Sequential()
        model.add(nn.AnyLayer(nn.Linear(inputSize: 10, outputSize: 5)))
        model.add(nn.AnyLayer(nn.ReLU()))

        let input = Tensor<Float>.zeros([4, 10])
        let output = model(input)

        XCTAssertEqual(output.shape, [4, 5])
    }
}

// MARK: - Loss Function Tests

final class LossFunctionTests: XCTestCase {

    func testMSELoss() {
        let prediction = Tensor<Float>.zeros([32, 10])
        let target = Tensor<Float>.ones([32, 10])

        let loss = nn.functional.mse(prediction, target)

        // MSE returns a scalar
        XCTAssertEqual(loss.shape, [])
    }

    func testCrossEntropyLoss() {
        let logits = Tensor<Float>.zeros([32, 10])
        let targets = Tensor<Float>.zeros([32, 10])  // One-hot targets

        let loss = nn.functional.crossEntropy(logits, targets)

        XCTAssertEqual(loss.shape, [])
    }

    func testBinaryCrossEntropyLoss() {
        let prediction = Tensor<Float>.zeros([32, 1])
        let target = Tensor<Float>.ones([32, 1])

        let loss = nn.functional.binaryCrossEntropy(prediction, target)

        XCTAssertEqual(loss.shape, [])
    }

    func testNLLLoss() {
        let logProbs = Tensor<Float>.zeros([32, 10])
        let targets = Tensor<Float>.zeros([32, 10])

        let loss = nn.functional.nllLoss(logProbs, targets)

        XCTAssertEqual(loss.shape, [])
    }
}

// MARK: - Tensor Operations Tests

final class NewTensorOperationsTests: XCTestCase {

    func testGelu() {
        let input = Tensor<Float>.zeros([10, 20])
        let output = input.gelu()

        XCTAssertEqual(output.shape, [10, 20])
    }

    func testSoftmax() {
        let input = Tensor<Float>.zeros([32, 10])
        let output = input.softmax(dim: -1)

        XCTAssertEqual(output.shape, [32, 10])
    }

    func testLogSoftmax() {
        let input = Tensor<Float>.zeros([32, 10])
        let output = input.logSoftmax(dim: -1)

        XCTAssertEqual(output.shape, [32, 10])
    }

    func testLog() {
        let input = Tensor<Float>.ones([5, 5])
        let output = input.log()

        XCTAssertEqual(output.shape, [5, 5])
    }

    func testExp() {
        let input = Tensor<Float>.zeros([5, 5])
        let output = input.exp()

        XCTAssertEqual(output.shape, [5, 5])
    }

    func testNegated() {
        let input = Tensor<Float>.ones([3, 3])
        let output = input.negated()

        XCTAssertEqual(output.shape, [3, 3])
    }

    func testNegationOperator() {
        let input = Tensor<Float>.ones([3, 3])
        let output = -input

        XCTAssertEqual(output.shape, [3, 3])
    }

    func testBroadcast() {
        let input = Tensor<Float>.ones([10])
        let output = input.broadcast(to: [32, 10])

        XCTAssertEqual(output.shape, [32, 10])
    }

    func testUniform() {
        let tensor = Tensor<Float>.uniform(low: -1.0, high: 1.0, shape: [10, 20])

        XCTAssertEqual(tensor.shape, [10, 20])
    }
}

// MARK: - Conv2d Tests

final class Conv2dLayerTests: XCTestCase {

    func testConv2dCreation() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3)

        XCTAssertEqual(conv.inChannels, 3)
        XCTAssertEqual(conv.outChannels, 64)
        XCTAssertEqual(conv.kernelSize.0, 3)
        XCTAssertEqual(conv.kernelSize.1, 3)
        XCTAssertEqual(conv.stride.0, 1)
        XCTAssertEqual(conv.stride.1, 1)
        XCTAssertEqual(conv.padding.0, 0)
        XCTAssertEqual(conv.padding.1, 0)
        XCTAssertTrue(conv.useBias)
        XCTAssertEqual(conv.weight.shape, [3, 3, 3, 64])
        XCTAssertEqual(conv.bias.shape, [64])
    }

    func testConv2dNoBias() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 32, kernelSize: 3, bias: false)

        XCTAssertFalse(conv.useBias)
        XCTAssertEqual(conv.parameters().count, 1)
    }

    func testConv2dForward() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3)
        let input = Tensor<Float>.zeros([32, 224, 224, 3])

        let output = conv(input)

        // Output: [batch, (224-3)/1+1, (224-3)/1+1, 64] = [32, 222, 222, 64]
        XCTAssertEqual(output.shape, [32, 222, 222, 64])
    }

    func testConv2dWithPadding() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3, padding: 1)
        let input = Tensor<Float>.zeros([32, 224, 224, 3])

        let output = conv(input)

        // With padding=1, output size stays the same (same padding)
        XCTAssertEqual(output.shape, [32, 224, 224, 64])
    }

    func testConv2dWithStride() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3, stride: 2, padding: 1)
        let input = Tensor<Float>.zeros([32, 224, 224, 3])

        let output = conv(input)

        // With stride=2, output is halved
        XCTAssertEqual(output.shape, [32, 112, 112, 64])
    }

    func testConv2dParameters() {
        let convWithBias = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3, bias: true)
        let convNoBias = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3, bias: false)

        XCTAssertEqual(convWithBias.parameters().count, 2)
        XCTAssertEqual(convNoBias.parameters().count, 1)
    }
}

// MARK: - Pooling Layer Tests

final class PoolingLayerTests: XCTestCase {

    func testMaxPool2d() {
        let pool = nn.MaxPool2d(kernelSize: 2)
        let input = Tensor<Float>.zeros([32, 224, 224, 64])

        let output = pool(input)

        XCTAssertEqual(output.shape, [32, 112, 112, 64])
    }

    func testMaxPool2dWithStride() {
        let pool = nn.MaxPool2d(kernelSize: 3, stride: 2, padding: 1)
        let input = Tensor<Float>.zeros([32, 224, 224, 64])

        let output = pool(input)

        XCTAssertEqual(output.shape, [32, 112, 112, 64])
    }

    func testAvgPool2d() {
        let pool = nn.AvgPool2d(kernelSize: 2)
        let input = Tensor<Float>.zeros([32, 224, 224, 64])

        let output = pool(input)

        XCTAssertEqual(output.shape, [32, 112, 112, 64])
    }

    func testAdaptiveAvgPool2d() {
        let pool = nn.AdaptiveAvgPool2d(outputSize: 1)
        let input = Tensor<Float>.zeros([32, 7, 7, 512])

        let output = pool(input)

        XCTAssertEqual(output.shape, [32, 1, 1, 512])
    }

    func testAdaptiveAvgPool2dNonSquare() {
        let pool = nn.AdaptiveAvgPool2d(outputSize: (2, 3))
        let input = Tensor<Float>.zeros([32, 14, 21, 256])

        let output = pool(input)

        XCTAssertEqual(output.shape, [32, 2, 3, 256])
    }

    func testPoolingNoParameters() {
        XCTAssertEqual(nn.MaxPool2d(kernelSize: 2).parameters().count, 0)
        XCTAssertEqual(nn.AvgPool2d(kernelSize: 2).parameters().count, 0)
        XCTAssertEqual(nn.AdaptiveAvgPool2d(outputSize: 1).parameters().count, 0)
    }
}

// MARK: - BatchNorm Tests

final class BatchNormLayerTests: XCTestCase {

    func testBatchNorm2dCreation() {
        let bn = nn.BatchNorm2d(numFeatures: 64)

        XCTAssertEqual(bn.numFeatures, 64)
        XCTAssertEqual(bn.eps, 1e-5)
        XCTAssertEqual(bn.momentum, 0.1)
        XCTAssertTrue(bn.affine)
        XCTAssertEqual(bn.weight.shape, [64])
        XCTAssertEqual(bn.bias.shape, [64])
    }

    func testBatchNorm2dForward() {
        let bn = nn.BatchNorm2d(numFeatures: 64)
        let input = Tensor<Float>.zeros([32, 224, 224, 64])

        let output = bn(input)

        // BatchNorm preserves shape
        XCTAssertEqual(output.shape, [32, 224, 224, 64])
    }

    func testBatchNorm2dParameters() {
        let bnAffine = nn.BatchNorm2d(numFeatures: 64, affine: true)
        let bnNoAffine = nn.BatchNorm2d(numFeatures: 64, affine: false)

        // Affine: weight + bias
        XCTAssertEqual(bnAffine.parameters().count, 2)
        // No affine: no learnable parameters
        XCTAssertEqual(bnNoAffine.parameters().count, 0)
    }

    func testBatchNorm1d() {
        let bn = nn.BatchNorm1d(numFeatures: 256)
        let input = Tensor<Float>.zeros([32, 256])

        let output = bn(input)

        XCTAssertEqual(output.shape, [32, 256])
        XCTAssertEqual(bn.parameters().count, 2)
    }
}

// MARK: - CNN End-to-End Test

final class CNNEndToEndTests: XCTestCase {

    func testSimpleCNN() {
        // Build a simple CNN
        let conv1 = nn.Conv2d(inChannels: 3, outChannels: 32, kernelSize: 3, padding: 1)
        let bn1 = nn.BatchNorm2d(numFeatures: 32)
        let pool1 = nn.MaxPool2d(kernelSize: 2)

        let conv2 = nn.Conv2d(inChannels: 32, outChannels: 64, kernelSize: 3, padding: 1)
        let bn2 = nn.BatchNorm2d(numFeatures: 64)
        let pool2 = nn.MaxPool2d(kernelSize: 2)

        // Forward pass
        var x = Tensor<Float>.zeros([32, 32, 32, 3])

        // Block 1
        x = conv1(x)       // [32, 32, 32, 32]
        x = bn1(x)         // [32, 32, 32, 32]
        x = x.relu()       // [32, 32, 32, 32]
        x = pool1(x)       // [32, 16, 16, 32]

        XCTAssertEqual(x.shape, [32, 16, 16, 32])

        // Block 2
        x = conv2(x)       // [32, 16, 16, 64]
        x = bn2(x)         // [32, 16, 16, 64]
        x = x.relu()       // [32, 16, 16, 64]
        x = pool2(x)       // [32, 8, 8, 64]

        XCTAssertEqual(x.shape, [32, 8, 8, 64])
    }

    func testCNNToMLP() {
        // CNN feature extractor -> Flatten -> MLP classifier
        let conv = nn.Conv2d(inChannels: 1, outChannels: 32, kernelSize: 5)
        let pool = nn.MaxPool2d(kernelSize: 2)
        let flatten = nn.Flatten()
        let fc = nn.Linear(inputSize: 32 * 12 * 12, outputSize: 10)

        // MNIST-like input: [batch, 28, 28, 1]
        var x = Tensor<Float>.zeros([64, 28, 28, 1])

        x = conv(x)        // [64, 24, 24, 32]
        XCTAssertEqual(x.shape, [64, 24, 24, 32])

        x = pool(x)        // [64, 12, 12, 32]
        XCTAssertEqual(x.shape, [64, 12, 12, 32])

        x = x.relu()
        x = flatten(x)     // [64, 32*12*12]
        XCTAssertEqual(x.shape, [64, 32 * 12 * 12])

        x = fc(x)          // [64, 10]
        XCTAssertEqual(x.shape, [64, 10])
    }

    func testVGGStyleBlock() {
        // VGG-style: conv-relu-conv-relu-pool
        let block = nn.sequential {
            nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3, padding: 1)
            nn.ReLU()
            nn.Conv2d(inChannels: 64, outChannels: 64, kernelSize: 3, padding: 1)
            nn.ReLU()
            nn.MaxPool2d(kernelSize: 2)
        }

        let input = Tensor<Float>.zeros([16, 224, 224, 3])
        let output = block(input)

        XCTAssertEqual(output.shape, [16, 112, 112, 64])

        // 2 conv layers with bias = 4 parameters
        XCTAssertEqual(block.parameters().count, 4)
    }
}

// MARK: - MLP End-to-End Test

final class MLPEndToEndTests: XCTestCase {

    func testSimpleMLP() {
        // Build a simple MLP
        let model = nn.sequential {
            nn.Linear(inputSize: 784, outputSize: 256)
            nn.ReLU()
            nn.Linear(inputSize: 256, outputSize: 128)
            nn.ReLU()
            nn.Linear(inputSize: 128, outputSize: 10)
        }

        // Forward pass
        let batch = Tensor<Float>.zeros([32, 784])
        let logits = model(batch)

        XCTAssertEqual(logits.shape, [32, 10])

        // Compute loss
        let targets = Tensor<Float>.zeros([32, 10])
        let loss = nn.functional.crossEntropy(logits, targets)

        XCTAssertEqual(loss.shape, [])

        // Count parameters
        let params = model.parameters()
        // Layer 1: 784*256 + 256 = 200,960
        // Layer 2: 256*128 + 128 = 32,896
        // Layer 3: 128*10 + 10 = 1,290
        // Total parameters: 6
        XCTAssertEqual(params.count, 6)
    }

    func testMLPWithFlatten() {
        // Simulating CNN output -> MLP
        let mlp = nn.sequential {
            nn.Flatten()
            nn.Linear(inputSize: 7 * 7 * 64, outputSize: 256)
            nn.ReLU()
            nn.Dropout(p: 0.5)
            nn.Linear(inputSize: 256, outputSize: 10)
        }

        // Input: feature map from CNN
        let features = Tensor<Float>.zeros([32, 7, 7, 64])
        let output = mlp(features)

        XCTAssertEqual(output.shape, [32, 10])
    }
}

// MARK: - Embedding Layer Tests

final class EmbeddingLayerTests: XCTestCase {

    func testEmbeddingCreation() {
        let embedding = nn.Embedding(numEmbeddings: 1000, embeddingDim: 256)

        XCTAssertEqual(embedding.numEmbeddings, 1000)
        XCTAssertEqual(embedding.embeddingDim, 256)
        XCTAssertEqual(embedding.weight.shape, [1000, 256])
        XCTAssertNil(embedding.paddingIdx)
    }

    func testEmbeddingWithPaddingIdx() {
        let embedding = nn.Embedding(numEmbeddings: 1000, embeddingDim: 128, paddingIdx: 0)

        XCTAssertEqual(embedding.paddingIdx, 0)
        XCTAssertEqual(embedding.weight.shape, [1000, 128])
    }

    func testEmbeddingForward1D() {
        let embedding = nn.Embedding(numEmbeddings: 100, embeddingDim: 32)

        // Look up 5 embeddings
        let indices = Tensor<Float>([0, 10, 50, 99, 5], shape: [5])
        let output = embedding(indices)

        // Output should be [5, 32]
        XCTAssertEqual(output.shape, [5, 32])
    }

    func testEmbeddingForward2D() {
        let embedding = nn.Embedding(numEmbeddings: 1000, embeddingDim: 256)

        // Batch of sequences: [batch=32, seqLen=128]
        let indices = Tensor<Float>.zeros([32, 128])
        let output = embedding(indices)

        // Output should be [32, 128, 256]
        XCTAssertEqual(output.shape, [32, 128, 256])
    }

    func testEmbeddingParameters() {
        let embedding = nn.Embedding(numEmbeddings: 1000, embeddingDim: 256)
        let params = embedding.parameters()

        XCTAssertEqual(params.count, 1)
        XCTAssertEqual(params[0].name, "weight")
    }
}

// MARK: - Checkpoint Tests

final class CheckpointTests: XCTestCase {

    func testCheckpointCreation() {
        let checkpoint = Checkpoint(description: "Test checkpoint")

        XCTAssertEqual(checkpoint.version, 1)
        XCTAssertEqual(checkpoint.description, "Test checkpoint")
        XCTAssertTrue(checkpoint.parameters.isEmpty)
    }

    func testSaveAndLoadLinear() throws {
        // Create and initialize a linear layer
        let original = nn.Linear(inputSize: 10, outputSize: 5)

        // Save to temp file
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_linear.checkpoint")

        try original.save(to: tempURL, description: "Linear layer test")

        // Create new layer with same shape and load
        var loaded = nn.Linear(inputSize: 10, outputSize: 5)
        try loaded.load(from: tempURL)

        // Verify shapes match
        XCTAssertEqual(loaded.weight.shape, original.weight.shape)
        XCTAssertEqual(loaded.bias.shape, original.bias.shape)

        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }

    func testSaveAndLoadMLP() throws {
        // Create MLP
        let original = nn.sequential {
            nn.Linear(inputSize: 784, outputSize: 128)
            nn.ReLU()
            nn.Linear(inputSize: 128, outputSize: 10)
        }

        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_mlp.checkpoint")

        try original.save(to: tempURL)

        var loaded = nn.sequential {
            nn.Linear(inputSize: 784, outputSize: 128)
            nn.ReLU()
            nn.Linear(inputSize: 128, outputSize: 10)
        }
        try loaded.load(from: tempURL)

        // Verify parameter count
        XCTAssertEqual(loaded.parameters().count, original.parameters().count)

        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }

    func testStateDict() {
        let linear = nn.Linear(inputSize: 10, outputSize: 5)
        let stateDict = linear.stateDict()

        XCTAssertEqual(stateDict.count, 2)
        XCTAssertNotNil(stateDict["weight"])
        XCTAssertNotNil(stateDict["bias"])
        XCTAssertEqual(stateDict["weight"]?.shape, [5, 10])
        XCTAssertEqual(stateDict["bias"]?.shape, [5])
    }

    func testLoadStateDict() throws {
        let original = nn.Linear(inputSize: 10, outputSize: 5)
        let stateDict = original.stateDict()

        var loaded = nn.Linear(inputSize: 10, outputSize: 5)
        try loaded.loadStateDict(stateDict)

        // Both should have same parameter shapes
        XCTAssertEqual(loaded.weight.shape, original.weight.shape)
        XCTAssertEqual(loaded.bias.shape, original.bias.shape)
    }

    func testLoadStateDictStrict() throws {
        let linear = nn.Linear(inputSize: 10, outputSize: 5)

        // Missing key
        let incompleteDict: StateDict = ["weight": Tensor<Float>.zeros([5, 10])]

        var loaded = nn.Linear(inputSize: 10, outputSize: 5)
        XCTAssertThrowsError(try loaded.loadStateDict(incompleteDict, strict: true))

        // Should work with strict=false
        XCTAssertNoThrow(try loaded.loadStateDict(incompleteDict, strict: false))
    }

    func testParameterMismatchError() throws {
        let linear2Params = nn.Linear(inputSize: 10, outputSize: 5, bias: true)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_mismatch.checkpoint")

        try linear2Params.save(to: tempURL)

        // Try to load into model with different parameter count
        var linear1Param = nn.Linear(inputSize: 10, outputSize: 5, bias: false)
        XCTAssertThrowsError(try linear1Param.load(from: tempURL))

        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }

    func testBinaryCheckpointSaveAndLoad() throws {
        let original = nn.Linear(inputSize: 20, outputSize: 10)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_linear.bin")

        try BinaryCheckpoint.save(original, to: tempURL)

        var loaded = nn.Linear(inputSize: 20, outputSize: 10)
        try BinaryCheckpoint.load(&loaded, from: tempURL)

        // Verify shapes match
        XCTAssertEqual(loaded.weight.shape, original.weight.shape)
        XCTAssertEqual(loaded.bias.shape, original.bias.shape)

        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }

    func testCheckpointWithEmbedding() throws {
        let original = nn.Embedding(numEmbeddings: 100, embeddingDim: 32)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_embedding.checkpoint")

        try original.save(to: tempURL)

        var loaded = nn.Embedding(numEmbeddings: 100, embeddingDim: 32)
        try loaded.load(from: tempURL)

        XCTAssertEqual(loaded.weight.shape, original.weight.shape)

        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }
}
