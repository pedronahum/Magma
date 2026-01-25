// Magma - Module Tests
// Tests for neural network module system

import Testing
import Foundation
@testable import Magma
@testable import LazyTensor

// MARK: - Parameter Tests

@Suite("Parameter Tests")
struct ParameterTests {

    @Test("Parameter creation")
    func parameterCreation() {
        let tensor = Tensor<Float>.zeros([10, 20])
        let param = Parameter(tensor)

        #expect(param.shape == [10, 20])
        #expect(param.elementCount == 200)
        #expect(param.requiresGrad)
        #expect(param.name == nil)
    }

    @Test("Parameter with name")
    func parameterWithName() {
        let tensor = Tensor<Float>.ones([5, 5])
        let param = Parameter(tensor, name: "weight")

        #expect(param.name == "weight")
        #expect(param.requiresGrad)
    }

    @Test("Parameter without grad")
    func parameterWithoutGrad() {
        let tensor = Tensor<Float>.zeros([3, 3])
        let param = Parameter(tensor, requiresGrad: false)

        #expect(!param.requiresGrad)
    }
}

// MARK: - Linear Layer Tests

@Suite("Linear Layer Tests")
struct LinearLayerTests {

    @Test("Linear creation")
    func linearCreation() {
        let linear = nn.Linear(inputSize: 784, outputSize: 256)

        #expect(linear.weight.shape == [256, 784])
        #expect(linear.bias.shape == [256])
        #expect(linear.useBias)
    }

    @Test("Linear no bias")
    func linearNoBias() {
        let linear = nn.Linear(inputSize: 100, outputSize: 50, bias: false)

        #expect(linear.weight.shape == [50, 100])
        #expect(!linear.useBias)
    }

    @Test("Linear forward")
    func linearForward() {
        let linear = nn.Linear(inputSize: 784, outputSize: 256)
        let input = Tensor<Float>.zeros([32, 784])

        let output = linear(input)

        #expect(output.shape == [32, 256])
    }

    @Test("Linear parameters")
    func linearParameters() {
        let linearWithBias = nn.Linear(inputSize: 10, outputSize: 5, bias: true)
        let linearNoBias = nn.Linear(inputSize: 10, outputSize: 5, bias: false)

        #expect(linearWithBias.parameters().count == 2)
        #expect(linearNoBias.parameters().count == 1)
    }

    @Test("Linear batched input")
    func linearBatchedInput() {
        let linear = nn.Linear(inputSize: 100, outputSize: 50)

        // Different batch sizes
        let input1 = Tensor<Float>.zeros([1, 100])
        let input2 = Tensor<Float>.zeros([64, 100])
        let input3 = Tensor<Float>.zeros([128, 100])

        #expect(linear(input1).shape == [1, 50])
        #expect(linear(input2).shape == [64, 50])
        #expect(linear(input3).shape == [128, 50])
    }
}

// MARK: - Activation Layer Tests

@Suite("Activation Layer Tests")
struct ActivationLayerTests {

    @Test("ReLU layer")
    func reluLayer() {
        let relu = nn.ReLU()
        let input = Tensor<Float>.zeros([10, 20])

        let output = relu(input)

        #expect(output.shape == [10, 20])
    }

    @Test("Sigmoid layer")
    func sigmoidLayer() {
        let sigmoid = nn.Sigmoid()
        let input = Tensor<Float>.zeros([5, 5])

        let output = sigmoid(input)

        #expect(output.shape == [5, 5])
    }

    @Test("Tanh layer")
    func tanhLayer() {
        let tanh = nn.Tanh()
        let input = Tensor<Float>.zeros([3, 4, 5])

        let output = tanh(input)

        #expect(output.shape == [3, 4, 5])
    }

    @Test("GELU layer")
    func geluLayer() {
        let gelu = nn.GELU()
        let input = Tensor<Float>.zeros([10, 10])

        let output = gelu(input)

        #expect(output.shape == [10, 10])
    }

    @Test("Activations have no parameters")
    func activationsHaveNoParameters() {
        #expect(nn.ReLU().parameters().count == 0)
        #expect(nn.Sigmoid().parameters().count == 0)
        #expect(nn.Tanh().parameters().count == 0)
        #expect(nn.GELU().parameters().count == 0)
    }
}

// MARK: - Flatten Layer Tests

@Suite("Flatten Layer Tests")
struct FlattenLayerTests {

    @Test("Flatten default")
    func flattenDefault() {
        let flatten = nn.Flatten()
        let input = Tensor<Float>.zeros([32, 7, 7, 64])

        let output = flatten(input)

        // Default startDim=1: [batch, flattened]
        #expect(output.shape == [32, 7 * 7 * 64])
    }

    @Test("Flatten custom startDim")
    func flattenCustomStartDim() {
        let flatten = nn.Flatten(startDim: 2)
        let input = Tensor<Float>.zeros([2, 3, 4, 5])

        let output = flatten(input)

        // Flatten from dim 2: [2, 3, 4*5]
        #expect(output.shape == [2, 3, 20])
    }

    @Test("Flatten already flat")
    func flattenAlreadyFlat() {
        let flatten = nn.Flatten()
        let input = Tensor<Float>.zeros([10, 20])

        let output = flatten(input)

        #expect(output.shape == [10, 20])
    }
}

// MARK: - Dropout Layer Tests

@Suite("Dropout Layer Tests")
struct DropoutLayerTests {

    @Test("Dropout creation")
    func dropoutCreation() {
        let dropout = nn.Dropout(p: 0.5)

        #expect(dropout.p == 0.5)
        #expect(dropout.training)
    }

    @Test("Dropout shape preserved")
    func dropoutShapePreserved() {
        let dropout = nn.Dropout(p: 0.5)
        let input = Tensor<Float>.zeros([32, 256])

        let output = dropout(input)

        #expect(output.shape == [32, 256])
    }

    @Test("Dropout no parameters")
    func dropoutNoParameters() {
        let dropout = nn.Dropout(p: 0.3)
        #expect(dropout.parameters().count == 0)
    }
}

// MARK: - Sequential Tests

@Suite("Sequential Tests")
struct SequentialTests {

    @Test("Sequential empty")
    func sequentialEmpty() {
        let seq = nn.Sequential()
        let input = Tensor<Float>.zeros([10, 20])

        let output = seq(input)

        #expect(output.shape == [10, 20])
    }

    @Test("Sequential single layer")
    func sequentialSingleLayer() {
        let seq = nn.Sequential(nn.AnyLayer(nn.ReLU()))
        let input = Tensor<Float>.zeros([10, 20])

        let output = seq(input)

        #expect(output.shape == [10, 20])
    }

    @Test("Sequential MLP")
    func sequentialMLP() {
        let model = nn.Sequential(
            nn.AnyLayer(nn.Linear(inputSize: 784, outputSize: 256)),
            nn.AnyLayer(nn.ReLU()),
            nn.AnyLayer(nn.Linear(inputSize: 256, outputSize: 128)),
            nn.AnyLayer(nn.ReLU()),
            nn.AnyLayer(nn.Linear(inputSize: 128, outputSize: 10))
        )

        let input = Tensor<Float>.zeros([32, 784])
        let output = model(input)

        #expect(output.shape == [32, 10])
    }

    @Test("Sequential parameters")
    func sequentialParameters() {
        let model = nn.Sequential(
            nn.AnyLayer(nn.Linear(inputSize: 10, outputSize: 5)),
            nn.AnyLayer(nn.ReLU()),
            nn.AnyLayer(nn.Linear(inputSize: 5, outputSize: 2))
        )

        let params = model.parameters()

        // 2 Linear layers with bias: 2 + 2 = 4 parameters
        #expect(params.count == 4)
    }

    @Test("Sequential result builder")
    func sequentialResultBuilder() {
        let model = nn.sequential {
            nn.Linear(inputSize: 784, outputSize: 256)
            nn.ReLU()
            nn.Linear(inputSize: 256, outputSize: 10)
        }

        let input = Tensor<Float>.zeros([32, 784])
        let output = model(input)

        #expect(output.shape == [32, 10])
    }

    @Test("Sequential add layer")
    func sequentialAddLayer() {
        var model = nn.Sequential()
        model.add(nn.AnyLayer(nn.Linear(inputSize: 10, outputSize: 5)))
        model.add(nn.AnyLayer(nn.ReLU()))

        let input = Tensor<Float>.zeros([4, 10])
        let output = model(input)

        #expect(output.shape == [4, 5])
    }
}

// MARK: - Loss Function Tests

@Suite("Loss Function Tests")
struct LossFunctionTests {

    @Test("MSE loss")
    func mseLoss() {
        let prediction = Tensor<Float>.zeros([32, 10])
        let target = Tensor<Float>.ones([32, 10])

        let loss = nn.functional.mse(prediction, target)

        // MSE returns a scalar
        #expect(loss.shape == [])
    }

    @Test("Cross entropy loss")
    func crossEntropyLoss() {
        let logits = Tensor<Float>.zeros([32, 10])
        let targets = Tensor<Float>.zeros([32])  // Class indices (not one-hot)

        let loss = nn.functional.crossEntropy(logits, targets)

        #expect(loss.shape == [])
    }

    @Test("Binary cross entropy loss")
    func binaryCrossEntropyLoss() {
        let prediction = Tensor<Float>.zeros([32, 1])
        let target = Tensor<Float>.ones([32, 1])

        let loss = nn.functional.binaryCrossEntropy(prediction, target)

        #expect(loss.shape == [])
    }

    @Test("NLL loss")
    func nllLoss() {
        let logProbs = Tensor<Float>.zeros([32, 10])
        let targets = Tensor<Float>.zeros([32])  // Class indices (not one-hot)

        let loss = nn.functional.nllLoss(logProbs, targets)

        #expect(loss.shape == [])
    }
}

// MARK: - Tensor Operations Tests

@Suite("New Tensor Operations Tests")
struct NewTensorOperationsTests {

    @Test("GELU")
    func gelu() {
        let input = Tensor<Float>.zeros([10, 20])
        let output = input.gelu()

        #expect(output.shape == [10, 20])
    }

    @Test("Softmax")
    func softmax() {
        let input = Tensor<Float>.zeros([32, 10])
        let output = input.softmax(dim: -1)

        #expect(output.shape == [32, 10])
    }

    @Test("Log softmax")
    func logSoftmax() {
        let input = Tensor<Float>.zeros([32, 10])
        let output = input.logSoftmax(dim: -1)

        #expect(output.shape == [32, 10])
    }

    @Test("Log")
    func log() {
        let input = Tensor<Float>.ones([5, 5])
        let output = input.log()

        #expect(output.shape == [5, 5])
    }

    @Test("Exp")
    func exp() {
        let input = Tensor<Float>.zeros([5, 5])
        let output = input.exp()

        #expect(output.shape == [5, 5])
    }

    @Test("Negated")
    func negated() {
        let input = Tensor<Float>.ones([3, 3])
        let output = input.negated()

        #expect(output.shape == [3, 3])
    }

    @Test("Negation operator")
    func negationOperator() {
        let input = Tensor<Float>.ones([3, 3])
        let output = -input

        #expect(output.shape == [3, 3])
    }

    @Test("Broadcast")
    func broadcast() {
        let input = Tensor<Float>.ones([10])
        let output = input.broadcast(to: [32, 10])

        #expect(output.shape == [32, 10])
    }

    @Test("Uniform")
    func uniform() {
        let tensor = Tensor<Float>.uniform(low: -1.0, high: 1.0, shape: [10, 20])

        #expect(tensor.shape == [10, 20])
    }
}

// MARK: - Conv2d Tests

@Suite("Conv2d Layer Tests")
struct Conv2dLayerTests {

    @Test("Conv2d creation")
    func conv2dCreation() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3)

        #expect(conv.inChannels == 3)
        #expect(conv.outChannels == 64)
        #expect(conv.kernelSize.0 == 3)
        #expect(conv.kernelSize.1 == 3)
        #expect(conv.stride.0 == 1)
        #expect(conv.stride.1 == 1)
        #expect(conv.padding.0 == 0)
        #expect(conv.padding.1 == 0)
        #expect(conv.useBias)
        #expect(conv.weight.shape == [3, 3, 3, 64])
        #expect(conv.bias.shape == [64])
    }

    @Test("Conv2d no bias")
    func conv2dNoBias() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 32, kernelSize: 3, bias: false)

        #expect(!conv.useBias)
        #expect(conv.parameters().count == 1)
    }

    @Test("Conv2d forward")
    func conv2dForward() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3)
        let input = Tensor<Float>.zeros([32, 224, 224, 3])

        let output = conv(input)

        // Output: [batch, (224-3)/1+1, (224-3)/1+1, 64] = [32, 222, 222, 64]
        #expect(output.shape == [32, 222, 222, 64])
    }

    @Test("Conv2d with padding")
    func conv2dWithPadding() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3, padding: 1)
        let input = Tensor<Float>.zeros([32, 224, 224, 3])

        let output = conv(input)

        // With padding=1, output size stays the same (same padding)
        #expect(output.shape == [32, 224, 224, 64])
    }

    @Test("Conv2d with stride")
    func conv2dWithStride() {
        let conv = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3, stride: 2, padding: 1)
        let input = Tensor<Float>.zeros([32, 224, 224, 3])

        let output = conv(input)

        // With stride=2, output is halved
        #expect(output.shape == [32, 112, 112, 64])
    }

    @Test("Conv2d parameters")
    func conv2dParameters() {
        let convWithBias = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3, bias: true)
        let convNoBias = nn.Conv2d(inChannels: 3, outChannels: 64, kernelSize: 3, bias: false)

        #expect(convWithBias.parameters().count == 2)
        #expect(convNoBias.parameters().count == 1)
    }
}

// MARK: - Pooling Layer Tests

@Suite("Pooling Layer Tests")
struct PoolingLayerTests {

    @Test("MaxPool2d")
    func maxPool2d() {
        let pool = nn.MaxPool2d(kernelSize: 2)
        let input = Tensor<Float>.zeros([32, 224, 224, 64])

        let output = pool(input)

        #expect(output.shape == [32, 112, 112, 64])
    }

    @Test("MaxPool2d with stride")
    func maxPool2dWithStride() {
        let pool = nn.MaxPool2d(kernelSize: 3, stride: 2, padding: 1)
        let input = Tensor<Float>.zeros([32, 224, 224, 64])

        let output = pool(input)

        #expect(output.shape == [32, 112, 112, 64])
    }

    @Test("AvgPool2d")
    func avgPool2d() {
        let pool = nn.AvgPool2d(kernelSize: 2)
        let input = Tensor<Float>.zeros([32, 224, 224, 64])

        let output = pool(input)

        #expect(output.shape == [32, 112, 112, 64])
    }

    @Test("AdaptiveAvgPool2d")
    func adaptiveAvgPool2d() {
        let pool = nn.AdaptiveAvgPool2d(outputSize: 1)
        let input = Tensor<Float>.zeros([32, 7, 7, 512])

        let output = pool(input)

        #expect(output.shape == [32, 1, 1, 512])
    }

    @Test("AdaptiveAvgPool2d non-square")
    func adaptiveAvgPool2dNonSquare() {
        let pool = nn.AdaptiveAvgPool2d(outputSize: (2, 3))
        let input = Tensor<Float>.zeros([32, 14, 21, 256])

        let output = pool(input)

        #expect(output.shape == [32, 2, 3, 256])
    }

    @Test("Pooling no parameters")
    func poolingNoParameters() {
        #expect(nn.MaxPool2d(kernelSize: 2).parameters().count == 0)
        #expect(nn.AvgPool2d(kernelSize: 2).parameters().count == 0)
        #expect(nn.AdaptiveAvgPool2d(outputSize: 1).parameters().count == 0)
    }
}

// MARK: - BatchNorm Tests

@Suite("BatchNorm Layer Tests")
struct BatchNormLayerTests {

    @Test("BatchNorm2d creation")
    func batchNorm2dCreation() {
        let bn = nn.BatchNorm2d(numFeatures: 64)

        #expect(bn.numFeatures == 64)
        #expect(bn.eps == 1e-5)
        #expect(bn.momentum == 0.1)
        #expect(bn.affine)
        #expect(bn.weight.shape == [64])
        #expect(bn.bias.shape == [64])
    }

    @Test("BatchNorm2d forward")
    func batchNorm2dForward() {
        let bn = nn.BatchNorm2d(numFeatures: 64)
        let input = Tensor<Float>.zeros([32, 224, 224, 64])

        let output = bn(input)

        // BatchNorm preserves shape
        #expect(output.shape == [32, 224, 224, 64])
    }

    @Test("BatchNorm2d parameters")
    func batchNorm2dParameters() {
        let bnAffine = nn.BatchNorm2d(numFeatures: 64, affine: true)
        let bnNoAffine = nn.BatchNorm2d(numFeatures: 64, affine: false)

        // Affine: weight + bias
        #expect(bnAffine.parameters().count == 2)
        // No affine: no learnable parameters
        #expect(bnNoAffine.parameters().count == 0)
    }

    @Test("BatchNorm1d")
    func batchNorm1d() {
        let bn = nn.BatchNorm1d(numFeatures: 256)
        let input = Tensor<Float>.zeros([32, 256])

        let output = bn(input)

        #expect(output.shape == [32, 256])
        #expect(bn.parameters().count == 2)
    }
}

// MARK: - CNN End-to-End Test

@Suite("CNN End-to-End Tests")
struct CNNEndToEndTests {

    @Test("Simple CNN")
    func simpleCNN() {
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

        #expect(x.shape == [32, 16, 16, 32])

        // Block 2
        x = conv2(x)       // [32, 16, 16, 64]
        x = bn2(x)         // [32, 16, 16, 64]
        x = x.relu()       // [32, 16, 16, 64]
        x = pool2(x)       // [32, 8, 8, 64]

        #expect(x.shape == [32, 8, 8, 64])
    }

    @Test("CNN to MLP")
    func cnnToMLP() {
        // CNN feature extractor -> Flatten -> MLP classifier
        let conv = nn.Conv2d(inChannels: 1, outChannels: 32, kernelSize: 5)
        let pool = nn.MaxPool2d(kernelSize: 2)
        let flatten = nn.Flatten()
        let fc = nn.Linear(inputSize: 32 * 12 * 12, outputSize: 10)

        // MNIST-like input: [batch, 28, 28, 1]
        var x = Tensor<Float>.zeros([64, 28, 28, 1])

        x = conv(x)        // [64, 24, 24, 32]
        #expect(x.shape == [64, 24, 24, 32])

        x = pool(x)        // [64, 12, 12, 32]
        #expect(x.shape == [64, 12, 12, 32])

        x = x.relu()
        x = flatten(x)     // [64, 32*12*12]
        #expect(x.shape == [64, 32 * 12 * 12])

        x = fc(x)          // [64, 10]
        #expect(x.shape == [64, 10])
    }

    @Test("VGG-style block")
    func vggStyleBlock() {
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

        #expect(output.shape == [16, 112, 112, 64])

        // 2 conv layers with bias = 4 parameters
        #expect(block.parameters().count == 4)
    }
}

// MARK: - MLP End-to-End Test

@Suite("MLP End-to-End Tests")
struct MLPEndToEndTests {

    @Test("Simple MLP")
    func simpleMLP() {
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

        #expect(logits.shape == [32, 10])

        // Compute loss
        let targets = Tensor<Float>.zeros([32])  // Class indices
        let loss = nn.functional.crossEntropy(logits, targets)

        #expect(loss.shape == [])

        // Count parameters
        let params = model.parameters()
        // Layer 1: 784*256 + 256 = 200,960
        // Layer 2: 256*128 + 128 = 32,896
        // Layer 3: 128*10 + 10 = 1,290
        // Total parameters: 6
        #expect(params.count == 6)
    }

    @Test("MLP with flatten")
    func mlpWithFlatten() {
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

        #expect(output.shape == [32, 10])
    }
}

// MARK: - Embedding Layer Tests

@Suite("Embedding Layer Tests")
struct EmbeddingLayerTests {

    @Test("Embedding creation")
    func embeddingCreation() {
        let embedding = nn.Embedding(numEmbeddings: 1000, embeddingDim: 256)

        #expect(embedding.numEmbeddings == 1000)
        #expect(embedding.embeddingDim == 256)
        #expect(embedding.weight.shape == [1000, 256])
        #expect(embedding.paddingIdx == nil)
    }

    @Test("Embedding with padding idx")
    func embeddingWithPaddingIdx() {
        let embedding = nn.Embedding(numEmbeddings: 1000, embeddingDim: 128, paddingIdx: 0)

        #expect(embedding.paddingIdx == 0)
        #expect(embedding.weight.shape == [1000, 128])
    }

    @Test("Embedding forward 1D")
    func embeddingForward1D() {
        let embedding = nn.Embedding(numEmbeddings: 100, embeddingDim: 32)

        // Look up 5 embeddings
        let indices = Tensor<Float>([0, 10, 50, 99, 5], shape: [5])
        let output = embedding(indices)

        // Output should be [5, 32]
        #expect(output.shape == [5, 32])
    }

    @Test("Embedding forward 2D")
    func embeddingForward2D() {
        let embedding = nn.Embedding(numEmbeddings: 1000, embeddingDim: 256)

        // Batch of sequences: [batch=32, seqLen=128]
        let indices = Tensor<Float>.zeros([32, 128])
        let output = embedding(indices)

        // Output should be [32, 128, 256]
        #expect(output.shape == [32, 128, 256])
    }

    @Test("Embedding parameters")
    func embeddingParameters() {
        let embedding = nn.Embedding(numEmbeddings: 1000, embeddingDim: 256)
        let params = embedding.parameters()

        #expect(params.count == 1)
        #expect(params[0].name == "weight")
    }
}

// MARK: - Checkpoint Tests

@Suite("Checkpoint Tests")
struct CheckpointTests {

    @Test("Checkpoint creation")
    func checkpointCreation() {
        let checkpoint = Checkpoint(description: "Test checkpoint")

        #expect(checkpoint.version == 1)
        #expect(checkpoint.description == "Test checkpoint")
        #expect(checkpoint.parameters.isEmpty)
    }

    @Test("Save and load linear")
    func saveAndLoadLinear() throws {
        // Create and initialize a linear layer
        let original = nn.Linear(inputSize: 10, outputSize: 5)

        // Save to temp file
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_linear.checkpoint")

        try original.save(to: tempURL, description: "Linear layer test")

        // Create new layer with same shape and load
        var loaded = nn.Linear(inputSize: 10, outputSize: 5)
        try loaded.load(from: tempURL)

        // Verify shapes match
        #expect(loaded.weight.shape == original.weight.shape)
        #expect(loaded.bias.shape == original.bias.shape)

        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }

    @Test("Save and load MLP")
    func saveAndLoadMLP() throws {
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
        #expect(loaded.parameters().count == original.parameters().count)

        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }

    @Test("State dict")
    func stateDict() {
        let linear = nn.Linear(inputSize: 10, outputSize: 5)
        let stateDict = linear.stateDict()

        #expect(stateDict.count == 2)
        #expect(stateDict["weight"] != nil)
        #expect(stateDict["bias"] != nil)
        #expect(stateDict["weight"]?.shape == [5, 10])
        #expect(stateDict["bias"]?.shape == [5])
    }

    @Test("Load state dict")
    func loadStateDict() throws {
        let original = nn.Linear(inputSize: 10, outputSize: 5)
        let stateDict = original.stateDict()

        var loaded = nn.Linear(inputSize: 10, outputSize: 5)
        try loaded.loadStateDict(stateDict)

        // Both should have same parameter shapes
        #expect(loaded.weight.shape == original.weight.shape)
        #expect(loaded.bias.shape == original.bias.shape)
    }

    @Test("Load state dict strict")
    func loadStateDictStrict() throws {
        let _ = nn.Linear(inputSize: 10, outputSize: 5)

        // Missing key
        let incompleteDict: StateDict = ["weight": Tensor<Float>.zeros([5, 10])]

        var loaded = nn.Linear(inputSize: 10, outputSize: 5)
        #expect(throws: Error.self) {
            try loaded.loadStateDict(incompleteDict, strict: true)
        }

        // Should work with strict=false
        try loaded.loadStateDict(incompleteDict, strict: false)
    }

    @Test("Parameter mismatch error")
    func parameterMismatchError() throws {
        let linear2Params = nn.Linear(inputSize: 10, outputSize: 5, bias: true)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_mismatch.checkpoint")

        try linear2Params.save(to: tempURL)

        // Try to load into model with different parameter count
        var linear1Param = nn.Linear(inputSize: 10, outputSize: 5, bias: false)
        #expect(throws: Error.self) {
            try linear1Param.load(from: tempURL)
        }

        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }

    @Test("Binary checkpoint save and load")
    func binaryCheckpointSaveAndLoad() throws {
        let original = nn.Linear(inputSize: 20, outputSize: 10)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_linear.bin")

        try BinaryCheckpoint.save(original, to: tempURL)

        var loaded = nn.Linear(inputSize: 20, outputSize: 10)
        try BinaryCheckpoint.load(&loaded, from: tempURL)

        // Verify shapes match
        #expect(loaded.weight.shape == original.weight.shape)
        #expect(loaded.bias.shape == original.bias.shape)

        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }

    @Test("Checkpoint with embedding")
    func checkpointWithEmbedding() throws {
        let original = nn.Embedding(numEmbeddings: 100, embeddingDim: 32)
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("test_embedding.checkpoint")

        try original.save(to: tempURL)

        var loaded = nn.Embedding(numEmbeddings: 100, embeddingDim: 32)
        try loaded.load(from: tempURL)

        #expect(loaded.weight.shape == original.weight.shape)

        // Clean up
        try? FileManager.default.removeItem(at: tempURL)
    }
}
