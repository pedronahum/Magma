// GPT-2 Shakespeare Training Example
// Trains a small GPT-2 model on Shakespeare text using Magma
//
// Usage:
//   1. First, tokenize Shakespeare with Python (see prepare_data.py)
//   2. swift run -c release GPT2Shakespeare
//
// This demonstrates:
// - Building a GPT-2 model from scratch with Magma layers
// - Causal self-attention with proper masking
// - Training loop with loss computation

import Foundation
import Magma
import LazyTensor
import XLARuntime

#if canImport(MetalHLO)
import MetalHLO

// MARK: - GPT-2 Configuration

/// Configuration for GPT-2 model
struct GPT2Config {
    let vocabSize: Int      // Vocabulary size
    let blockSize: Int      // Maximum sequence length (context window)
    let nLayer: Int         // Number of transformer blocks
    let nHead: Int          // Number of attention heads
    let nEmbd: Int          // Embedding dimension
    let dropout: Float      // Dropout probability
    let bias: Bool          // Use bias in Linear layers and LayerNorm

    /// GPT-2 Small (124M params) - use for full training
    static let gpt2Small = GPT2Config(
        vocabSize: 50304,   // Padded for efficiency (50257 + padding)
        blockSize: 1024,
        nLayer: 12,
        nHead: 12,
        nEmbd: 768,
        dropout: 0.0,
        bias: true
    )

    /// Tiny model for Shakespeare (faster training)
    static let shakespeareTiny = GPT2Config(
        vocabSize: 65,      // Character-level: ~65 unique characters
        blockSize: 256,
        nLayer: 6,
        nHead: 6,
        nEmbd: 384,
        dropout: 0.0,
        bias: true
    )

    /// Very small for quick testing
    static let test = GPT2Config(
        vocabSize: 65,
        blockSize: 64,
        nLayer: 2,
        nHead: 2,
        nEmbd: 64,
        dropout: 0.0,
        bias: true
    )
}

// MARK: - Causal Self Attention

/// Multi-head causal self-attention for GPT-2.
///
/// Uses separate Q, K, V projections with causal masking to prevent
/// attending to future tokens.
struct CausalSelfAttention {
    let config: GPT2Config
    let headDim: Int
    let scale: Float

    var wQ: nn.Linear  // Query projection
    var wK: nn.Linear  // Key projection
    var wV: nn.Linear  // Value projection
    var wO: nn.Linear  // Output projection

    init(config: GPT2Config, device: Device = .default) {
        self.config = config
        self.headDim = config.nEmbd / config.nHead
        self.scale = 1.0 / Float(headDim).squareRoot()

        self.wQ = nn.Linear(inputSize: config.nEmbd, outputSize: config.nEmbd, bias: config.bias, device: device)
        self.wK = nn.Linear(inputSize: config.nEmbd, outputSize: config.nEmbd, bias: config.bias, device: device)
        self.wV = nn.Linear(inputSize: config.nEmbd, outputSize: config.nEmbd, bias: config.bias, device: device)
        self.wO = nn.Linear(inputSize: config.nEmbd, outputSize: config.nEmbd, bias: config.bias, device: device)
    }

    /// Computes multi-head causal self-attention.
    ///
    /// - Parameter x: Input tensor of shape [batch, seqLen, nEmbd]
    /// - Returns: Output tensor of shape [batch, seqLen, nEmbd]
    func forward(_ x: Tensor<Float>) -> Tensor<Float> {
        let B = x.shape[0]
        let T = x.shape[1]
        let nHead = config.nHead

        // Project Q, K, V using 3D-native Linear: [B, T, C] -> [B, T, C]
        let qProj = wQ(x)
        let kProj = wK(x)
        let vProj = wV(x)

        // Materialize projections to establish shapes for lazy graph
        qProj.markForMaterialization()
        kProj.markForMaterialization()
        vProj.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        // Reshape to multi-head format: [B, T, C] -> [B, nHead, T, headDim]
        let q = qProj.reshape([B, T, nHead, headDim]).transpose(1, 2)
        let k = kProj.reshape([B, T, nHead, headDim]).transpose(1, 2)
        let v = vProj.reshape([B, T, nHead, headDim]).transpose(1, 2)

        q.markForMaterialization()
        k.markForMaterialization()
        v.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        // Scaled dot-product attention: QK^T / sqrt(d)
        let kT = k.transpose(-1, -2)
        let scores = q.batchedMatmul(kT) * Tensor<Float>.full([], scale, on: x.device)

        scores.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        // Causal mask: additive mask with 0 for valid, -1e9 for future positions
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

        // Softmax over keys and apply to values
        let attnWeights = maskedScores.softmax(dim: -1)
        let out = attnWeights.batchedMatmul(v)

        out.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        // Concatenate heads and project: [B, nHead, T, headDim] -> [B, T, C]
        let outConcat = out.transpose(1, 2).reshape([B, T, config.nEmbd])
        return wO(outConcat)
    }

    func parameters() -> [Parameter] {
        wQ.parameters() + wK.parameters() + wV.parameters() + wO.parameters()
    }
}

// MARK: - GPT-2 MLP

/// Position-wise feed-forward network: Linear(C -> 4C) -> GELU -> Linear(4C -> C)
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

/// Single transformer decoder block with pre-normalization.
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
        // Attention with residual
        let normed1 = ln1(x)
        normed1.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        let attnOut = attn.forward(normed1)
        attnOut.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        let h = x + attnOut
        h.markForMaterialization()
        LazyTensorBarrier(on: x.device)

        // MLP with residual
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

/// Complete GPT-2 decoder-only transformer language model.
struct GPT2 {
    let config: GPT2Config

    var wte: nn.Embedding    // Token embeddings
    var wpe: nn.Embedding    // Position embeddings
    var blocks: [GPT2Block]  // Transformer blocks
    var lnF: nn.LayerNorm    // Final layer norm
    var lmHead: nn.Linear    // LM head

    init(config: GPT2Config, device: Device = .default) {
        self.config = config

        self.wte = nn.Embedding(numEmbeddings: config.vocabSize, embeddingDim: config.nEmbd, device: device)
        self.wpe = nn.Embedding(numEmbeddings: config.blockSize, embeddingDim: config.nEmbd, device: device)
        self.blocks = (0..<config.nLayer).map { _ in GPT2Block(config: config, device: device) }
        self.lnF = nn.LayerNorm(normalizedShape: [config.nEmbd], device: device)
        self.lmHead = nn.Linear(inputSize: config.nEmbd, outputSize: config.vocabSize, bias: false, device: device)
    }

    func forward(_ idx: Tensor<Float>, targets: Tensor<Float>? = nil) -> (logits: Tensor<Float>, loss: Tensor<Float>?) {
        let B = idx.shape[0]
        let T = idx.shape[1]

        // Embed tokens and positions
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

        // Apply transformer blocks
        for block in blocks {
            x = block.forward(x)
            x.markForMaterialization()
            LazyTensorBarrier(on: idx.device)
        }

        // Final norm and project to logits
        let normed = lnF(x)
        normed.markForMaterialization()
        LazyTensorBarrier(on: idx.device)

        let logits = lmHead(normed)
        logits.markForMaterialization()
        LazyTensorBarrier(on: idx.device)

        // Compute loss if targets provided
        var loss: Tensor<Float>? = nil
        if let targets = targets {
            let logitsFlat = logits.reshape([B * T, config.vocabSize])
            let targetsFlat = targets.reshape([B * T])
            loss = nn.functional.crossEntropy(logitsFlat, targetsFlat)
            loss?.markForMaterialization()
            LazyTensorBarrier(on: idx.device)
        }

        return (logits, loss)
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

/// Weights for one transformer block (all raw tensors, no Parameter wrappers)
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

/// All GPT-2 model weights in a single differentiable struct
struct GPT2Weights: Differentiable {
    var wte: Tensor<Float>    // Token embeddings [vocabSize, nEmbd]
    var wpe: Tensor<Float>    // Position embeddings [blockSize, nEmbd]
    var blocks: [BlockWeights]
    var lnFW: Tensor<Float>
    var lnFB: Tensor<Float>
    var lmHeadW: Tensor<Float>  // [nEmbd, vocabSize]
}

// MARK: - Differentiable Primitives

/// Differentiable linear projection: input @ weight^T + bias
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

/// Differentiable layer normalization
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

/// Differentiable embedding lookup
@differentiable(reverse, wrt: table)
func embedding(_ table: Tensor<Float>, indices: Tensor<Float>) -> Tensor<Float> {
    let inputShape = withoutDerivative(at: indices.shape)
    let batchSize = withoutDerivative(at: inputShape.reduce(1, *))
    let flatIndices = withoutDerivative(at: indices.reshape([batchSize]))
    let gathered = table.gather(indices: flatIndices, axis: 0)
    let embDim = withoutDerivative(at: table.shape[1])
    return gathered.reshape(inputShape + [embDim])
}

/// Differentiable cross-entropy loss (logits: [N, C], targets as indices: [N])
@differentiable(reverse, wrt: logits)
func crossEntropyLoss(_ logits: Tensor<Float>, targets: Tensor<Float>, numClasses: Int, device: Device) -> Tensor<Float> {
    // Inline logSoftmax: softmax then log (both have VJPs)
    let logProbs = logits.softmax(dim: -1).log()
    let oneHot = withoutDerivative(at: Tensor<Float>.oneHot(targets, numClasses: numClasses, on: device))
    let targetLogProbs = (logProbs * oneHot).sum(dims: [1], keepDims: false)
    return (-targetLogProbs).mean()
}

// MARK: - Differentiable Forward Pass

/// Full GPT-2 forward pass as a differentiable function.
/// Returns scalar loss suitable for gradient computation.
@differentiable(reverse, wrt: weights)
func gpt2Loss(
    weights: GPT2Weights,
    input: Tensor<Float>,
    targets: Tensor<Float>,
    causalMask: Tensor<Float>,
    config: GPT2Config
) -> Tensor<Float> {
    let B = withoutDerivative(at: input.shape[0])
    let T = withoutDerivative(at: input.shape[1])
    let nHead = withoutDerivative(at: config.nHead)
    let headDim = withoutDerivative(at: config.nEmbd / config.nHead)
    let nEmbd = withoutDerivative(at: config.nEmbd)
    let scale = withoutDerivative(at: 1.0 / Float(headDim).squareRoot())
    let vocabSize = withoutDerivative(at: config.vocabSize)
    let device = withoutDerivative(at: input.device)

    // Token + position embeddings
    let tokEmb = embedding(weights.wte, indices: input)
    let posIndices = withoutDerivative(at: Tensor<Float>.arange(T, on: device).reshape([1, T]))
    let posEmb = embedding(weights.wpe, indices: posIndices)
    let tokEmbShape = withoutDerivative(at: tokEmb.shape)
    var x = tokEmb + posEmb.broadcast(to: tokEmbShape)

    // Transformer blocks
    let blockCount = withoutDerivative(at: weights.blocks.count)
    for i in withoutDerivative(at: 0..<blockCount) {
        let block = weights.blocks[i]
        // Pre-norm attention
        let normed1 = layerNorm(x, weight: block.ln1W, bias: block.ln1B)

        // Q, K, V projections -> reshape to multi-head
        let q4d = linear(normed1, weight: block.wQW, bias: block.wQB)
            .reshape([B, T, nHead, headDim]).transpose(1, 2)
        let k4d = linear(normed1, weight: block.wKW, bias: block.wKB)
            .reshape([B, T, nHead, headDim]).transpose(1, 2)
        let v4d = linear(normed1, weight: block.wVW, bias: block.wVB)
            .reshape([B, T, nHead, headDim]).transpose(1, 2)

        // Scaled dot-product attention with causal mask
        let scaleTensor = withoutDerivative(at: Tensor<Float>.full([], scale, on: device))
        let scores = q4d.batchedMatmul(k4d.transpose(-1, -2)) * scaleTensor
        let scoresShape = withoutDerivative(at: scores.shape)
        let maskedScores = scores + causalMask.broadcast(to: scoresShape)
        let attnW = maskedScores.softmax(dim: -1)
        let attnOut = attnW.batchedMatmul(v4d)

        // Concatenate heads + output projection
        let concat = attnOut.transpose(1, 2).reshape([B, T, nEmbd])
        let projected = linear(concat, weight: block.wOW, bias: block.wOB)
        let h = x + projected

        // Pre-norm MLP
        let normed2 = layerNorm(h, weight: block.ln2W, bias: block.ln2B)
        let hidden = linear(normed2, weight: block.fcW, bias: block.fcB).gelu()
        let mlpOut = linear(hidden, weight: block.projW, bias: block.projB)
        x = h + mlpOut
    }

    // Final layer norm + LM head (no bias for LM head)
    let normed = layerNorm(x, weight: weights.lnFW, bias: weights.lnFB)
    let zeroBias = withoutDerivative(at: Tensor<Float>.zeros([vocabSize], on: device))
    let logits = linear(normed, weight: weights.lmHeadW, bias: zeroBias)
    let logitsFlat = logits.reshape([B * T, vocabSize])
    let targetsFlat = withoutDerivative(at: targets.reshape([B * T]))

    return crossEntropyLoss(logitsFlat, targets: targetsFlat, numClasses: vocabSize, device: device)
}

// MARK: - Weight Extraction Helpers

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

/// Copy updated weights back into model Parameters
func applyWeights(_ weights: GPT2Weights, to model: inout GPT2) {
    model.wte.parameters()[0].value = weights.wte
    model.wpe.parameters()[0].value = weights.wpe
    for (i, block) in model.blocks.enumerated() {
        let bw = weights.blocks[i]
        block.ln1.parameters()[0].value = bw.ln1W
        block.ln1.parameters()[1].value = bw.ln1B
        block.attn.wQ.parameters()[0].value = bw.wQW
        block.attn.wQ.parameters()[1].value = bw.wQB
        block.attn.wK.parameters()[0].value = bw.wKW
        block.attn.wK.parameters()[1].value = bw.wKB
        block.attn.wV.parameters()[0].value = bw.wVW
        block.attn.wV.parameters()[1].value = bw.wVB
        block.attn.wO.parameters()[0].value = bw.wOW
        block.attn.wO.parameters()[1].value = bw.wOB
        block.ln2.parameters()[0].value = bw.ln2W
        block.ln2.parameters()[1].value = bw.ln2B
        block.mlp.cFc.parameters()[0].value = bw.fcW
        block.mlp.cFc.parameters()[1].value = bw.fcB
        block.mlp.cProj.parameters()[0].value = bw.projW
        block.mlp.cProj.parameters()[1].value = bw.projB
    }
    model.lnF.parameters()[0].value = weights.lnFW
    model.lnF.parameters()[1].value = weights.lnFB
    model.lmHead.parameters()[0].value = weights.lmHeadW
}

/// Flatten GPT2Weights gradient into array matching model.parameters() order
func flattenGradients(_ grad: GPT2Weights.TangentVector) -> [Tensor<Float>] {
    var grads: [Tensor<Float>] = []
    // wte, wpe
    grads.append(grad.wte)
    grads.append(grad.wpe)
    // blocks - .base accesses the underlying array from DifferentiableView
    for bg in grad.blocks.base {
        grads.append(contentsOf: [
            bg.ln1W, bg.ln1B,
            bg.wQW, bg.wQB, bg.wKW, bg.wKB, bg.wVW, bg.wVB, bg.wOW, bg.wOB,
            bg.ln2W, bg.ln2B,
            bg.fcW, bg.fcB, bg.projW, bg.projB
        ])
    }
    // lnF, lmHead
    grads.append(contentsOf: [grad.lnFW, grad.lnFB, grad.lmHeadW])
    return grads
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

    let config = GPT2Config.test
    print("Model config:")
    print("  - Vocab size: \(config.vocabSize)")
    print("  - Block size: \(config.blockSize)")
    print("  - Layers: \(config.nLayer)")
    print("  - Heads: \(config.nHead)")
    print("  - Embedding dim: \(config.nEmbd)\n")

    print("Creating model...")
    let model = GPT2(config: config, device: device)
    let paramCount = model.parameterCount()
    print("Total parameters: \(paramCount) (\(String(format: "%.2f", Float(paramCount) / 1_000_000))M)")

    // Materialize all model weights
    print("Materializing weights...")
    for param in model.parameters() {
        param.value.markForMaterialization()
        LazyTensorBarrier(on: device)
    }
    print("Weights materialized.\n")

    // Build causal mask (constant, not differentiated)
    let seqLen = config.blockSize
    var maskData = [Float](repeating: -1e9, count: seqLen * seqLen)
    for i in 0..<seqLen {
        for j in 0...i {
            maskData[i * seqLen + j] = 0.0
        }
    }
    let causalMask = Tensor<Float>(maskData, shape: [1, 1, seqLen, seqLen], on: device)
    causalMask.markForMaterialization()
    LazyTensorBarrier(on: device)

    // Create sample training data
    print("Creating sample data...")
    let batchSize = 4

    var inputData: [Float] = []
    var targetData: [Float] = []
    for _ in 0..<(batchSize * seqLen) {
        let token = Float(Int.random(in: 0..<config.vocabSize))
        inputData.append(token)
        targetData.append(Float((Int(token) + 1) % config.vocabSize))
    }

    let inputs = Tensor<Float>(inputData, shape: [batchSize, seqLen], on: device)
    let targets = Tensor<Float>(targetData, shape: [batchSize, seqLen], on: device)
    inputs.markForMaterialization()
    targets.markForMaterialization()
    LazyTensorBarrier(on: device)

    print("Input shape: \(inputs.shape)")
    print("Target shape: \(targets.shape)")
    print("Expected initial loss (random): ~\(String(format: "%.2f", log(Float(config.vocabSize))))\n")

    setbuf(stdout, nil)  // Disable output buffering

    // Training loop
    let numSteps = 10
    let lr: Float = 1e-3
    var optimizer = optim.Adam(parameters: model.parameters(), lr: lr)
    var losses: [Float] = []

    print("Training for \(numSteps) steps (lr=\(lr))...\n")

    for step in 0..<numSteps {
        // Extract current weights
        let weights = extractWeights(from: model, config: config)

        // Forward + backward
        let (loss, grad) = valueWithGradient(at: weights) { w -> Tensor<Float> in
            gpt2Loss(weights: w, input: inputs, targets: targets,
                     causalMask: causalMask, config: config)
        }

        // Materialize loss
        loss.markForMaterialization()
        LazyTensorBarrier(on: device)
        let lossVal = loss.scalars()[0]
        losses.append(lossVal)

        // Flatten gradients and materialize
        let grads = flattenGradients(grad)
        for g in grads {
            g.markForMaterialization()
        }
        LazyTensorBarrier(on: device)

        // Gradient clipping
        let (clippedGrads, _) = optim.clipGradNorm(grads, maxNorm: 1.0)
        for g in clippedGrads {
            g.markForMaterialization()
        }
        LazyTensorBarrier(on: device)

        // Optimizer step
        optimizer.step(clippedGrads)

        // Materialize updated parameters
        for param in model.parameters() {
            param.value.markForMaterialization()
        }
        LazyTensorBarrier(on: device)

        print("Step \(step + 1)/\(numSteps) | Loss: \(String(format: "%.4f", lossVal))")
    }

    // Summary
    print("\n--- Training Summary ---")
    print("Initial loss: \(String(format: "%.4f", losses.first ?? 0))")
    print("Final loss:   \(String(format: "%.4f", losses.last ?? 0))")
    if let first = losses.first, let last = losses.last {
        let decreased = last < first
        print("Loss decreased: \(decreased ? "YES" : "NO") (\(String(format: "%.4f", first - last)) reduction)")
    }
    print("\nTraining complete!")
}

// MARK: - Main

print("╔════════════════════════════════════════════════════════════╗")
print("║          GPT-2 Shakespeare - Magma Example                 ║")
print("╚════════════════════════════════════════════════════════════╝\n")

trainShakespeare()

print("\nDone!")

#else
print("ERROR: MetalHLO not available")
exit(1)
#endif
