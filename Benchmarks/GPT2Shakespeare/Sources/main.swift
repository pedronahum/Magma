// GPT-2 Shakespeare Training Example
// Trains a character-level GPT-2 on Shakespeare text using Magma
//
// Reproduces Karpathy's nanoGPT character-level Shakespeare results.
//
// Usage:
//   1. python3 prepare_data.py    (downloads & tokenizes Shakespeare)
//   2. swift run -c release GPT2Shakespeare
//
// Expected: loss starts ~4.17 (ln(65)), decreases to ~1.5 over 5000 steps

import Foundation
import Magma
import LazyTensor
import XLARuntime

#if canImport(MetalHLO)
import MetalHLO

// MARK: - GPT-2 Configuration

struct GPT2Config {
    let vocabSize: Int
    let blockSize: Int
    let nLayer: Int
    let nHead: Int
    let nEmbd: Int
    let dropout: Float
    let bias: Bool

    /// Tiny model for Shakespeare character-level (matches nanoGPT)
    static let shakespeareTiny = GPT2Config(
        vocabSize: 65,
        blockSize: 256,
        nLayer: 6,
        nHead: 6,
        nEmbd: 384,
        dropout: 0.2,
        bias: true
    )

    /// Very small for quick testing
    static let test = GPT2Config(
        vocabSize: 65,
        blockSize: 64,
        nLayer: 2,
        nHead: 2,
        nEmbd: 64,
        dropout: 0.2,
        bias: true
    )
}

// MARK: - Training Hyperparameters

struct TrainConfig {
    let numSteps: Int
    let batchSize: Int
    let learningRate: Float
    let minLR: Float
    let warmupSteps: Int
    let weightDecay: Float
    let beta1: Float
    let beta2: Float
    let gradClipNorm: Float
    let evalInterval: Int
    let evalBatches: Int
    let logInterval: Int
    let generateInterval: Int
    let generateTokens: Int

    static let shakespeare = TrainConfig(
        numSteps: 5000,
        batchSize: 64,
        learningRate: 6e-4,
        minLR: 6e-5,
        warmupSteps: 100,
        weightDecay: 0.1,
        beta1: 0.9,
        beta2: 0.99,
        gradClipNorm: 1.0,
        evalInterval: 250,
        evalBatches: 10,
        logInterval: 10,
        generateInterval: 500,
        generateTokens: 200
    )

    static let quick = TrainConfig(
        numSteps: 50,
        batchSize: 4,
        learningRate: 1e-3,
        minLR: 1e-4,
        warmupSteps: 5,
        weightDecay: 0.0,
        beta1: 0.9,
        beta2: 0.999,
        gradClipNorm: 1.0,
        evalInterval: 25,
        evalBatches: 2,
        logInterval: 5,
        generateInterval: 25,
        generateTokens: 100
    )
}

// MARK: - Causal Self Attention

struct CausalSelfAttention {
    let config: GPT2Config
    let headDim: Int
    let scale: Float

    var wQ: nn.Linear
    var wK: nn.Linear
    var wV: nn.Linear
    var wO: nn.Linear

    init(config: GPT2Config, device: Device = .default) {
        self.config = config
        self.headDim = config.nEmbd / config.nHead
        self.scale = 1.0 / Float(headDim).squareRoot()

        self.wQ = nn.Linear(inputSize: config.nEmbd, outputSize: config.nEmbd, bias: config.bias, device: device)
        self.wK = nn.Linear(inputSize: config.nEmbd, outputSize: config.nEmbd, bias: config.bias, device: device)
        self.wV = nn.Linear(inputSize: config.nEmbd, outputSize: config.nEmbd, bias: config.bias, device: device)
        self.wO = nn.Linear(inputSize: config.nEmbd, outputSize: config.nEmbd, bias: config.bias, device: device)
    }

    func forward(_ x: Tensor<Float>) -> Tensor<Float> {
        let B = x.shape[0]
        let T = x.shape[1]
        let nHead = config.nHead

        let qProj = wQ(x)
        let kProj = wK(x)
        let vProj = wV(x)

        qProj.markForMaterialization()
        kProj.markForMaterialization()
        vProj.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        let q = qProj.reshape([B, T, nHead, headDim]).transpose(1, 2)
        let k = kProj.reshape([B, T, nHead, headDim]).transpose(1, 2)
        let v = vProj.reshape([B, T, nHead, headDim]).transpose(1, 2)

        q.markForMaterialization()
        k.markForMaterialization()
        v.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        let kT = k.transpose(-1, -2)
        let scores = q.batchedMatmul(kT) * Tensor<Float>.full([], scale, on: x.device)

        scores.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        var maskData = [Float](repeating: -1e9, count: T * T)
        for i in 0..<T {
            for j in 0...i {
                maskData[i * T + j] = 0.0
            }
        }
        let mask = Tensor<Float>(maskData, shape: [1, 1, T, T], on: x.device)
        let maskedScores = scores + mask.broadcast(to: scores.shape)

        maskedScores.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        let attnWeights = maskedScores.softmax(dim: -1)
        let out = attnWeights.batchedMatmul(v)

        out.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        let outConcat = out.transpose(1, 2).reshape([B, T, config.nEmbd])
        return wO(outConcat)
    }

    func parameters() -> [Parameter] {
        wQ.parameters() + wK.parameters() + wV.parameters() + wO.parameters()
    }
}

// MARK: - GPT-2 MLP

struct GPT2MLP {
    let nEmbd: Int
    var cFc: nn.Linear
    var cProj: nn.Linear

    init(config: GPT2Config, device: Device = .default) {
        self.nEmbd = config.nEmbd
        self.cFc = nn.Linear(inputSize: config.nEmbd, outputSize: 4 * config.nEmbd, bias: config.bias, device: device)
        self.cProj = nn.Linear(inputSize: 4 * config.nEmbd, outputSize: config.nEmbd, bias: config.bias, device: device)
    }

    func forward(_ x: Tensor<Float>) -> Tensor<Float> {
        let hidden = cFc(x).gelu()
        return cProj(hidden)
    }

    func parameters() -> [Parameter] {
        cFc.parameters() + cProj.parameters()
    }
}

// MARK: - GPT-2 Block

struct GPT2Block {
    var ln1: nn.LayerNorm
    var attn: CausalSelfAttention
    var ln2: nn.LayerNorm
    var mlp: GPT2MLP

    init(config: GPT2Config, device: Device = .default) {
        self.ln1 = nn.LayerNorm(normalizedShape: [config.nEmbd], device: device)
        self.attn = CausalSelfAttention(config: config, device: device)
        self.ln2 = nn.LayerNorm(normalizedShape: [config.nEmbd], device: device)
        self.mlp = GPT2MLP(config: config, device: device)
    }

    func forward(_ x: Tensor<Float>) -> Tensor<Float> {
        let normed1 = ln1(x)
        normed1.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        let attnOut = attn.forward(normed1)
        attnOut.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        let h = x + attnOut
        h.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        let normed2 = ln2(h)
        normed2.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        let mlpOut = mlp.forward(normed2)
        mlpOut.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        let result = h + mlpOut
        result.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        return result
    }

    func parameters() -> [Parameter] {
        ln1.parameters() + attn.parameters() + ln2.parameters() + mlp.parameters()
    }
}

// MARK: - GPT-2 Model

struct GPT2 {
    let config: GPT2Config

    var wte: nn.Embedding
    var wpe: nn.Embedding
    var blocks: [GPT2Block]
    var lnF: nn.LayerNorm
    var lmHead: nn.Linear

    init(config: GPT2Config, device: Device = .default) {
        self.config = config

        self.wte = nn.Embedding(numEmbeddings: config.vocabSize, embeddingDim: config.nEmbd, device: device)
        self.wpe = nn.Embedding(numEmbeddings: config.blockSize, embeddingDim: config.nEmbd, device: device)
        self.blocks = (0..<config.nLayer).map { _ in GPT2Block(config: config, device: device) }
        self.lnF = nn.LayerNorm(normalizedShape: [config.nEmbd], device: device)
        self.lmHead = nn.Linear(inputSize: config.nEmbd, outputSize: config.vocabSize, bias: false, device: device)
    }

    func forward(_ idx: Tensor<Float>) -> Tensor<Float> {
        let T = idx.shape[1]

        let tokEmb = wte(idx)
        tokEmb.markForMaterialization()
        LazyTensorBarrier(on: idx.device)

        let posIndices = Tensor<Float>.arange(T, on: idx.device).reshape([1, T])
        let posEmb = wpe(posIndices)
        posEmb.markForMaterialization()
        LazyTensorBarrier(on: idx.device)

        var x = tokEmb + posEmb.broadcast(to: tokEmb.shape)
        x.markForMaterialization()
        LazyTensorBarrier(on: idx.device)

        for block in blocks {
            x = block.forward(x)
            x.markForMaterialization()
            LazyTensorBarrier(on: idx.device)
        }

        let normed = lnF(x)
        normed.markForMaterialization()
        LazyTensorBarrier(on: idx.device)

        let logits = lmHead(normed)
        logits.markForMaterialization()
        LazyTensorBarrier(on: idx.device)

        return logits  // [B, T, vocabSize]
    }

    func parameters() -> [Parameter] {
        var params = wte.parameters() + wpe.parameters()
        for block in blocks { params += block.parameters() }
        return params + lnF.parameters() + lmHead.parameters()
    }

    func parameterCount() -> Int {
        parameters().reduce(0) { $0 + $1.value.elementCount }
    }
}

// MARK: - Differentiable Weight Structures

import _Differentiation

struct BlockWeights: Differentiable {
    var ln1W: Tensor<Float>
    var ln1B: Tensor<Float>
    var wQW: Tensor<Float>; var wQB: Tensor<Float>
    var wKW: Tensor<Float>; var wKB: Tensor<Float>
    var wVW: Tensor<Float>; var wVB: Tensor<Float>
    var wOW: Tensor<Float>; var wOB: Tensor<Float>
    var ln2W: Tensor<Float>
    var ln2B: Tensor<Float>
    var fcW: Tensor<Float>; var fcB: Tensor<Float>
    var projW: Tensor<Float>; var projB: Tensor<Float>
}

struct GPT2Weights: Differentiable {
    var wte: Tensor<Float>
    var wpe: Tensor<Float>
    var blocks: [BlockWeights]
    var lnFW: Tensor<Float>
    var lnFB: Tensor<Float>
    var lmHeadW: Tensor<Float>
}

// MARK: - Differentiable Primitives

@differentiable(reverse, wrt: (input, weight, bias))
func linear(_ input: Tensor<Float>, weight: Tensor<Float>, bias: Tensor<Float>) -> Tensor<Float> {
    let origShape = withoutDerivative(at: input.shape)
    let lastDim = withoutDerivative(at: origShape.last!)
    let batchElements = withoutDerivative(at: origShape.dropLast().reduce(1, *))
    let outDim = withoutDerivative(at: weight.shape[0])
    let flat = input.reshape([batchElements, lastDim])
    let out = flat.matmul(weight.transpose()) + bias.broadcast(to: [batchElements, outDim])
    return out.reshape(Array(origShape.dropLast()) + [outDim])
}

@differentiable(reverse, wrt: (input, weight, bias))
func layerNorm(_ input: Tensor<Float>, weight: Tensor<Float>, bias: Tensor<Float>, eps: Float = 1e-5) -> Tensor<Float> {
    let inputShape = withoutDerivative(at: input.shape)
    let dev = withoutDerivative(at: input.device)
    let mean = input.mean(dims: [-1], keepDims: true)
    let centered = input - mean.broadcast(to: inputShape)
    let variance = (centered * centered).mean(dims: [-1], keepDims: true)
    let epsT = withoutDerivative(at: Tensor<Float>.full(variance.shape, eps, on: dev))
    let std = (variance + epsT).sqrt()
    let normalized = centered / std.broadcast(to: inputShape)
    return normalized * weight.broadcast(to: inputShape) + bias.broadcast(to: inputShape)
}

@differentiable(reverse, wrt: table)
func embedding(_ table: Tensor<Float>, indices: Tensor<Float>) -> Tensor<Float> {
    let inputShape = withoutDerivative(at: indices.shape)
    let batchSize = withoutDerivative(at: inputShape.reduce(1, *))
    let flatIndices = withoutDerivative(at: indices.reshape([batchSize]))
    let gathered = table.gather(indices: flatIndices, axis: 0)
    let embDim = withoutDerivative(at: table.shape[1])
    return gathered.reshape(inputShape + [embDim])
}

/// Differentiable dropout with explicit VJP.
/// The custom derivative ensures the same mask is used in forward and backward,
/// and avoids intermediate broadcast operations from the auto-generated multiply VJP.
@differentiable(reverse, wrt: input)
func dropout(_ input: Tensor<Float>, rate: Float, device: Device) -> Tensor<Float> {
    if rate == 0 { return input }
    let mask = Tensor<Float>.randDevice(input.shape, on: device).greaterThan(rate)
    let scale = Tensor<Float>.full([], 1.0 / (1.0 - rate), on: device)
    return input * mask * scale
}

@derivative(of: dropout, wrt: input)
func vjpDropout(_ input: Tensor<Float>, rate: Float, device: Device)
    -> (value: Tensor<Float>, pullback: (Tensor<Float>) -> Tensor<Float>) {
    if rate == 0 {
        return (input, { $0 })
    }
    let mask = Tensor<Float>.randDevice(input.shape, on: device).greaterThan(rate)
    let scale = Tensor<Float>.full([], 1.0 / (1.0 - rate), on: device)
    let scaledMask = mask * scale
    return (input * scaledMask, { upstream in upstream * scaledMask })
}

/// Numerically stable cross-entropy loss.
/// Uses log-softmax trick: log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
@differentiable(reverse, wrt: logits)
func crossEntropyLoss(_ logits: Tensor<Float>, targets: Tensor<Float>, numClasses: Int, device: Device) -> Tensor<Float> {
    let logitsShape = withoutDerivative(at: logits.shape)

    // Subtract max for numerical stability (detached from gradient)
    let maxLogits = withoutDerivative(at:
        logits.max(alongAxes: -1).expandingShape(at: -1).broadcast(to: logitsShape))
    let shifted = logits - maxLogits

    // log(sum(exp(shifted)))
    let expShifted = shifted.exp()
    let sumExp = expShifted.sum(dims: [-1], keepDims: true)
    let logSumExp = sumExp.log().broadcast(to: logitsShape)

    // logSoftmax = shifted - logSumExp
    let logProbs = shifted - logSumExp

    // Select target log-probs via one-hot
    let oneHot = withoutDerivative(at: Tensor<Float>.oneHot(targets, numClasses: numClasses, on: device))
    let targetLogProbs = (logProbs * oneHot).sum(dims: [1], keepDims: false)
    return (-targetLogProbs).mean()
}

// MARK: - Differentiable Forward Pass

@differentiable(reverse, wrt: weights)
func gpt2Loss(
    weights: GPT2Weights,
    input: Tensor<Float>,
    targets: Tensor<Float>,
    causalMask: Tensor<Float>,
    config: GPT2Config,
    dropoutRate: Float
) -> Tensor<Float> {
    let B = withoutDerivative(at: input.shape[0])
    let T = withoutDerivative(at: input.shape[1])
    let nHead = withoutDerivative(at: config.nHead)
    let headDim = withoutDerivative(at: config.nEmbd / config.nHead)
    let nEmbd = withoutDerivative(at: config.nEmbd)
    let scale = withoutDerivative(at: 1.0 / Float(headDim).squareRoot())
    let vocabSize = withoutDerivative(at: config.vocabSize)
    let device = withoutDerivative(at: input.device)
    let drop = withoutDerivative(at: dropoutRate)

    let tokEmb = embedding(weights.wte, indices: input)
    let posIndices = withoutDerivative(at: Tensor<Float>.arange(T, on: device).reshape([1, T]))
    let posEmb = embedding(weights.wpe, indices: posIndices)
    let tokEmbShape = withoutDerivative(at: tokEmb.shape)
    var x = dropout(tokEmb + posEmb.broadcast(to: tokEmbShape), rate: drop, device: device)

    let blockCount = withoutDerivative(at: weights.blocks.count)
    for i in withoutDerivative(at: 0..<blockCount) {
        let block = weights.blocks[i]
        let normed1 = layerNorm(x, weight: block.ln1W, bias: block.ln1B)

        let q4d = linear(normed1, weight: block.wQW, bias: block.wQB)
            .reshape([B, T, nHead, headDim]).transpose(1, 2)
        let k4d = linear(normed1, weight: block.wKW, bias: block.wKB)
            .reshape([B, T, nHead, headDim]).transpose(1, 2)
        let v4d = linear(normed1, weight: block.wVW, bias: block.wVB)
            .reshape([B, T, nHead, headDim]).transpose(1, 2)

        let scaleTensor = withoutDerivative(at: Tensor<Float>.full([], scale, on: device))
        let scores = q4d.batchedMatmul(k4d.transpose(-1, -2)) * scaleTensor
        let scoresShape = withoutDerivative(at: scores.shape)
        let maskedScores = scores + causalMask.broadcast(to: scoresShape)
        let attnW = dropout(maskedScores.softmax(dim: -1), rate: drop, device: device)
        let attnOut = attnW.batchedMatmul(v4d)

        let concat = attnOut.transpose(1, 2).reshape([B, T, nEmbd])
        let projected = dropout(linear(concat, weight: block.wOW, bias: block.wOB), rate: drop, device: device)
        let h = x + projected

        let normed2 = layerNorm(h, weight: block.ln2W, bias: block.ln2B)
        let hidden = linear(normed2, weight: block.fcW, bias: block.fcB).gelu()
        let mlpOut = dropout(linear(hidden, weight: block.projW, bias: block.projB), rate: drop, device: device)
        x = h + mlpOut
    }

    let normed = layerNorm(x, weight: weights.lnFW, bias: weights.lnFB)
    let zeroBias = withoutDerivative(at: Tensor<Float>.zeros([vocabSize], on: device))
    let logits = linear(normed, weight: weights.lmHeadW, bias: zeroBias)
    let logitsFlat = logits.reshape([B * T, vocabSize])
    let targetsFlat = withoutDerivative(at: targets.reshape([B * T]))

    return crossEntropyLoss(logitsFlat, targets: targetsFlat, numClasses: vocabSize, device: device)
}

// MARK: - Weight Helpers

func extractWeights(from model: GPT2, config: GPT2Config) -> GPT2Weights {
    var blockWeights: [BlockWeights] = []
    for block in model.blocks {
        blockWeights.append(BlockWeights(
            ln1W: block.ln1.parameters()[0].value,
            ln1B: block.ln1.parameters()[1].value,
            wQW: block.attn.wQ.parameters()[0].value,
            wQB: block.attn.wQ.parameters()[1].value,
            wKW: block.attn.wK.parameters()[0].value,
            wKB: block.attn.wK.parameters()[1].value,
            wVW: block.attn.wV.parameters()[0].value,
            wVB: block.attn.wV.parameters()[1].value,
            wOW: block.attn.wO.parameters()[0].value,
            wOB: block.attn.wO.parameters()[1].value,
            ln2W: block.ln2.parameters()[0].value,
            ln2B: block.ln2.parameters()[1].value,
            fcW: block.mlp.cFc.parameters()[0].value,
            fcB: block.mlp.cFc.parameters()[1].value,
            projW: block.mlp.cProj.parameters()[0].value,
            projB: block.mlp.cProj.parameters()[1].value
        ))
    }
    return GPT2Weights(
        wte: model.wte.parameters()[0].value,
        wpe: model.wpe.parameters()[0].value,
        blocks: blockWeights,
        lnFW: model.lnF.parameters()[0].value,
        lnFB: model.lnF.parameters()[1].value,
        lmHeadW: model.lmHead.parameters()[0].value
    )
}

func flattenGradients(_ grad: GPT2Weights.TangentVector) -> [Tensor<Float>] {
    var grads: [Tensor<Float>] = []
    grads.append(grad.wte)
    grads.append(grad.wpe)
    for bg in grad.blocks.base {
        grads.append(contentsOf: [
            bg.ln1W, bg.ln1B,
            bg.wQW, bg.wQB, bg.wKW, bg.wKB, bg.wVW, bg.wVB, bg.wOW, bg.wOB,
            bg.ln2W, bg.ln2B,
            bg.fcW, bg.fcB, bg.projW, bg.projB
        ])
    }
    grads.append(contentsOf: [grad.lnFW, grad.lnFB, grad.lmHeadW])
    return grads
}

// MARK: - Data Loading

struct VocabMeta {
    let vocabSize: Int
    let stoi: [String: Int]
    let itos: [Int: String]

    func encode(_ text: String) -> [Int] {
        text.map { stoi[String($0)] ?? 0 }
    }

    func decode(_ ids: [Int]) -> String {
        ids.map { itos[$0] ?? "?" }.joined()
    }
}

func loadMeta(path: String) throws -> VocabMeta {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
          let vocabSize = json["vocab_size"] as? Int,
          let stoiRaw = json["stoi"] as? [String: Int],
          let itosRaw = json["itos"] as? [String: String] else {
        fatalError("Invalid meta.json format")
    }
    var itos: [Int: String] = [:]
    for (k, v) in itosRaw {
        if let key = Int(k) { itos[key] = v }
    }
    return VocabMeta(vocabSize: vocabSize, stoi: stoiRaw, itos: itos)
}

func getRandomBatch(
    from dataset: MemoryMappedTokenDataset,
    batchSize: Int,
    device: Device
) -> (input: Tensor<Float>, target: Tensor<Float>) {
    var inputBatch: [Float] = []
    var targetBatch: [Float] = []
    inputBatch.reserveCapacity(batchSize * dataset.sequenceLength)
    targetBatch.reserveCapacity(batchSize * dataset.sequenceLength)

    for _ in 0..<batchSize {
        let idx = Int.random(in: 0..<dataset.count)
        let (input, target) = dataset.getSample(at: idx)
        inputBatch.append(contentsOf: input.map { Float($0) })
        targetBatch.append(contentsOf: target.map { Float($0) })
    }

    let inputs = Tensor<Float>(inputBatch, shape: [batchSize, dataset.sequenceLength], on: device)
    let targets = Tensor<Float>(targetBatch, shape: [batchSize, dataset.sequenceLength], on: device)
    return (inputs, targets)
}

// MARK: - Text Generation

func generateSample(
    model: GPT2,
    meta: VocabMeta,
    config: GPT2Config,
    device: Device,
    numTokens: Int = 200,
    temperature: Float = 0.8
) -> String {
    // Start with a newline character
    let startToken = meta.stoi["\n"] ?? 0
    var tokens: [Float] = [Float(startToken)]

    for _ in 0..<numTokens {
        // Truncate to blockSize if needed
        let contextTokens = Array(tokens.suffix(config.blockSize))
        let seqLen = contextTokens.count

        let input = Tensor<Float>(contextTokens, shape: [1, seqLen], on: device)
        input.markForMaterialization()
        LazyTensorBarrier(on: device)

        // Forward pass (non-differentiable)
        let logits = model.forward(input)  // [1, seqLen, vocabSize]
        logits.markForMaterialization()
        LazyTensorBarrier(on: device)

        // Get last position logits: slice the last time step
        let lastLogits = logits.slice(axis: 1, start: seqLen - 1, stop: seqLen)  // [1, 1, V]
        let lastLogitsFlat = lastLogits.reshape([1, config.vocabSize])  // [1, V]

        // Apply temperature
        let scaled: Tensor<Float>
        if temperature != 1.0 {
            scaled = lastLogitsFlat / Tensor<Float>.full(lastLogitsFlat.shape, temperature, on: device)
        } else {
            scaled = lastLogitsFlat
        }

        // Softmax + multinomial sampling
        let probs = scaled.softmax(dim: -1)
        let nextToken = probs.multinomial(numSamples: 1)  // [1]
        nextToken.markForMaterialization()
        LazyTensorBarrier(on: device)

        let tokenId = Int(nextToken.scalars()[0])
        tokens.append(Float(tokenId))
    }

    return meta.decode(tokens.map { Int($0) })
}

// MARK: - Validation

func evaluateValidation(
    model: GPT2,
    config: GPT2Config,
    dataset: MemoryMappedTokenDataset,
    causalMask: Tensor<Float>,
    numBatches: Int,
    batchSize: Int,
    device: Device
) -> Float {
    var totalLoss: Float = 0

    for _ in 0..<numBatches {
        let (inputs, targets) = getRandomBatch(from: dataset, batchSize: batchSize, device: device)
        inputs.markForMaterialization()
        targets.markForMaterialization()
        LazyTensorBarrier(on: device)

        let weights = extractWeights(from: model, config: config)
        let loss = gpt2Loss(weights: weights, input: inputs, targets: targets,
                            causalMask: causalMask, config: config, dropoutRate: 0)
        loss.markForMaterialization()
        LazyTensorBarrier(on: device)
        totalLoss += loss.scalars()[0]
    }

    return totalLoss / Float(numBatches)
}

// MARK: - Training

func trainShakespeare() {
    print("GPT-2 Shakespeare Training")
    print("==========================\n")

    guard Backend.metal.isAvailable else {
        print("ERROR: Metal backend not available")
        return
    }

    let device = Device(backend: .metal, index: 0)
    print("Device: \(MetalBackend.deviceName ?? "unknown")\n")

    // Load data
    let dataDir = "data"
    let trainPath = "\(dataDir)/train.bin"
    let valPath = "\(dataDir)/val.bin"
    let metaPath = "\(dataDir)/meta.json"

    guard FileManager.default.fileExists(atPath: trainPath) else {
        print("ERROR: Training data not found at \(trainPath)")
        print("Run: python3 prepare_data.py")
        return
    }

    let meta: VocabMeta
    do {
        meta = try loadMeta(path: metaPath)
        print("Vocabulary: \(meta.vocabSize) characters")
    } catch {
        print("ERROR: Failed to load meta.json: \(error)")
        return
    }

    let modelConfig = GPT2Config.shakespeareTiny
    let trainCfg = TrainConfig.shakespeare

    print("Model config:")
    print("  Vocab size: \(modelConfig.vocabSize)")
    print("  Block size: \(modelConfig.blockSize)")
    print("  Layers: \(modelConfig.nLayer)")
    print("  Heads: \(modelConfig.nHead)")
    print("  Embedding dim: \(modelConfig.nEmbd)")
    print("Training config:")
    print("  Steps: \(trainCfg.numSteps)")
    print("  Batch size: \(trainCfg.batchSize)")
    print("  LR: \(trainCfg.learningRate) -> \(trainCfg.minLR)")
    print("  Warmup: \(trainCfg.warmupSteps) steps")
    print("  Weight decay: \(trainCfg.weightDecay)")
    print()

    // Load datasets
    let trainDataset: MemoryMappedTokenDataset
    let valDataset: MemoryMappedTokenDataset
    do {
        trainDataset = try MemoryMappedTokenDataset(path: trainPath, sequenceLength: modelConfig.blockSize)
        valDataset = try MemoryMappedTokenDataset(path: valPath, sequenceLength: modelConfig.blockSize)
        print("Train tokens: \(trainDataset.tokenCount)")
        print("Val tokens: \(valDataset.tokenCount)")
        print()
    } catch {
        print("ERROR: Failed to load datasets: \(error)")
        return
    }

    // Create model
    print("Creating model...")
    let model = GPT2(config: modelConfig, device: device)
    let paramCount = model.parameterCount()
    print("Total parameters: \(paramCount) (\(String(format: "%.2f", Float(paramCount) / 1_000_000))M)")

    print("Materializing weights...")
    for param in model.parameters() {
        param.value.markForMaterialization()
    }
    LazyTensorBarrier(on: device)
    print("Weights materialized.\n")

    // Build causal mask
    let seqLen = modelConfig.blockSize
    var maskData = [Float](repeating: -1e9, count: seqLen * seqLen)
    for i in 0..<seqLen {
        for j in 0...i {
            maskData[i * seqLen + j] = 0.0
        }
    }
    let causalMask = Tensor<Float>(maskData, shape: [1, 1, seqLen, seqLen], on: device)
    causalMask.markForMaterialization()
    LazyTensorBarrier(on: device)

    // Optimizer + scheduler
    var optimizer = optim.Adam(
        parameters: model.parameters(),
        lr: trainCfg.learningRate,
        beta1: trainCfg.beta1,
        beta2: trainCfg.beta2,
        weightDecay: trainCfg.weightDecay
    )
    var scheduler = optim.WarmupCosineScheduler(
        baseLR: trainCfg.learningRate,
        warmupSteps: trainCfg.warmupSteps,
        totalSteps: trainCfg.numSteps,
        minLR: trainCfg.minLR
    )

    setbuf(stdout, nil)

    print("Expected initial loss: ~\(String(format: "%.2f", log(Float(modelConfig.vocabSize))))")
    print("Training for \(trainCfg.numSteps) steps...\n")

    var bestValLoss: Float = Float.infinity
    let startTime = Date()

    for step in 0..<trainCfg.numSteps {
        let stepStart = Date()

        // Update learning rate
        optimizer.learningRate = scheduler.currentLR
        let currentLR = scheduler.currentLR

        // Get batch
        let (inputs, targets) = getRandomBatch(from: trainDataset, batchSize: trainCfg.batchSize, device: device)
        inputs.markForMaterialization()
        targets.markForMaterialization()
        LazyTensorBarrier(on: device)

        // Forward + backward
        let weights = extractWeights(from: model, config: modelConfig)
        let (loss, grad) = valueWithGradient(at: weights) { w -> Tensor<Float> in
            gpt2Loss(weights: w, input: inputs, targets: targets,
                     causalMask: causalMask, config: modelConfig, dropoutRate: modelConfig.dropout)
        }

        loss.markForMaterialization()
        LazyTensorBarrier(on: device)
        let lossVal = loss.scalars()[0]

        // Flatten and materialize gradients
        let grads = flattenGradients(grad)
        for g in grads { g.markForMaterialization() }
        LazyTensorBarrier(on: device)

        // Gradient clipping
        let (clippedGrads, gradNorm) = optim.clipGradNorm(grads, maxNorm: trainCfg.gradClipNorm)
        for g in clippedGrads { g.markForMaterialization() }
        LazyTensorBarrier(on: device)

        // Optimizer step
        optimizer.step(clippedGrads)

        // Materialize updated parameters
        for param in model.parameters() { param.value.markForMaterialization() }
        LazyTensorBarrier(on: device)

        // LR scheduler step
        scheduler.step()

        let stepTime = Date().timeIntervalSince(stepStart)

        // Logging
        if step % trainCfg.logInterval == 0 || step == trainCfg.numSteps - 1 {
            gradNorm.markForMaterialization()
            LazyTensorBarrier(on: device)
            let normVal = gradNorm.scalars()[0]
            let elapsed = Date().timeIntervalSince(startTime)
            print(String(format: "step %4d | loss %.4f | lr %.2e | grad_norm %.2f | %.1fms/step | %.0fs elapsed",
                         step, lossVal, currentLR, normVal, stepTime * 1000, elapsed))
        }

        // Validation
        if step > 0 && step % trainCfg.evalInterval == 0 {
            let valLoss = evaluateValidation(
                model: model, config: modelConfig, dataset: valDataset,
                causalMask: causalMask, numBatches: trainCfg.evalBatches,
                batchSize: trainCfg.batchSize, device: device
            )
            let marker = valLoss < bestValLoss ? " *" : ""
            if valLoss < bestValLoss { bestValLoss = valLoss }
            print(String(format: "         val_loss %.4f%@", valLoss, marker))
        }

        // Text generation sample
        if step > 0 && step % trainCfg.generateInterval == 0 {
            print("\n--- Sample (step \(step)) ---")
            let sample = generateSample(model: model, meta: meta, config: modelConfig,
                                         device: device, numTokens: trainCfg.generateTokens)
            print(sample)
            print("--- End sample ---\n")
        }
    }

    // Final evaluation
    print("\n" + String(repeating: "=", count: 60))
    print("Training complete!")
    let totalTime = Date().timeIntervalSince(startTime)
    print(String(format: "Total time: %.1f seconds (%.1f min)", totalTime, totalTime / 60))

    let finalValLoss = evaluateValidation(
        model: model, config: modelConfig, dataset: valDataset,
        causalMask: causalMask, numBatches: trainCfg.evalBatches * 2,
        batchSize: trainCfg.batchSize, device: device
    )
    print(String(format: "Final val loss: %.4f (best: %.4f)", finalValLoss, bestValLoss))

    // Final generation sample
    print("\n--- Final Sample ---")
    let finalSample = generateSample(model: model, meta: meta, config: modelConfig,
                                      device: device, numTokens: 500, temperature: 0.8)
    print(finalSample)
    print("--- End ---")
}

// MARK: - Main

print("╔════════════════════════════════════════════════════════════╗")
print("║        GPT-2 Shakespeare - Magma / MetalHLO              ║")
print("╚════════════════════════════════════════════════════════════╝\n")

trainShakespeare()

print("\nDone!")

#else
print("ERROR: MetalHLO not available")
exit(1)
#endif
