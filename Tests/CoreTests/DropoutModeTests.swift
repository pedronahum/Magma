// Magma - Dropout honoured by attention, transformer, positional-encoding and
// recurrent layers.
// These layers used to accept a dropout rate and ignore it. Each test pairs a
// layer built with p > 0 against an identical-weights twin built with p = 0:
// in eval mode the two must agree exactly (and be deterministic); in training
// mode the p > 0 layer must differ. eval() is also checked through nn.Sequential,
// so setTraining must propagate through the type-erased container.
//
// .serialized: materialization drives the PJRT backend.

import Testing
@testable import Magma

/// Deterministic, varied, mixed-sign fill (no RNG).
private func fill(_ shape: [Int], _ scale: Float = 0.1, _ bias: Float = -0.3) -> Tensor<Float> {
    let n = shape.reduce(1, *)
    return Tensor<Float>((0..<n).map { Float($0 % 7) * scale + bias }, shape: shape)
}

/// Makes `dst` share `src`'s weights (same parameter order by construction).
private func copyWeights<A: Module, B: Module>(from src: A, to dst: B) {
    let s = src.parameters(), d = dst.parameters()
    precondition(s.count == d.count)
    for (a, b) in zip(s, d) { b.value = a.value }
}

private func maxAbsDiff(_ a: Tensor<Float>, _ b: Tensor<Float>) -> Float {
    let x = a.scalars(), y = b.scalars()
    precondition(x.count == y.count && !x.isEmpty, "failed to materialize")
    return zip(x, y).map { abs($0 - $1) }.max()!
}

@Suite("Dropout train/eval in composite layers", .serialized, .enabled(if: PluginAvailability.cpu, "CPU PJRT plugin not available"))
struct DropoutModeTests {

    @Test("scaledDotProductAttention applies dropout when dropout > 0")
    func sdpaDropout() {
        let q = fill([2, 2, 5, 4]), k = fill([2, 2, 5, 4], 0.07, -0.2), v = fill([2, 2, 5, 4], 0.13, -0.4)
        let plain = nn.scaledDotProductAttention(query: q, key: k, value: v)
        let dropped = nn.scaledDotProductAttention(query: q, key: k, value: v, dropout: 0.5)
        #expect(maxAbsDiff(plain.output, dropped.output) > 1e-3)
        // Returned weights are the pre-dropout probabilities.
        #expect(maxAbsDiff(plain.weights, dropped.weights) < 1e-6)
    }

    @Test("MultiheadAttention: eval disables dropout, train applies it")
    func multiheadAttention() {
        var m = nn.MultiheadAttention(embedDim: 8, numHeads: 2, dropout: 0.5)
        let ref = nn.MultiheadAttention(embedDim: 8, numHeads: 2, dropout: 0)
        copyWeights(from: m, to: ref)
        let x = fill([2, 6, 8])

        m.eval()
        #expect(!m.training)
        #expect(maxAbsDiff(m(x), ref(x)) < 1e-6)

        m.train()
        #expect(maxAbsDiff(m(x), ref(x)) > 1e-3)
    }

    @Test("TransformerEncoderLayer: eval is deterministic and equals dropout=0; train differs",
          arguments: [false, true])
    func encoderLayer(normFirst: Bool) {
        var layer = nn.TransformerEncoderLayer(dModel: 8, nHead: 2, dimFeedforward: 16,
                                               dropout: 0.5, normFirst: normFirst)
        let ref = nn.TransformerEncoderLayer(dModel: 8, nHead: 2, dimFeedforward: 16,
                                             dropout: 0, normFirst: normFirst)
        copyWeights(from: layer, to: ref)
        let x = fill([2, 5, 8])

        // Default is training mode, as in PyTorch.
        #expect(layer.training && layer.selfAttn.training)
        #expect(maxAbsDiff(layer(x), ref(x)) > 1e-3)

        layer.eval()
        #expect(!layer.selfAttn.training)
        let a = layer(x), b = layer(x)
        #expect(maxAbsDiff(a, b) == 0)
        #expect(maxAbsDiff(a, ref(x)) < 1e-5)
    }

    @Test("TransformerDecoderLayer: eval equals dropout=0; train differs")
    func decoderLayer() {
        var layer = nn.TransformerDecoderLayer(dModel: 8, nHead: 2, dimFeedforward: 16, dropout: 0.5)
        let ref = nn.TransformerDecoderLayer(dModel: 8, nHead: 2, dimFeedforward: 16, dropout: 0)
        copyWeights(from: layer, to: ref)
        let tgt = fill([2, 4, 8]), memory = fill([2, 6, 8], 0.05, -0.1)

        #expect(maxAbsDiff(layer(tgt: tgt, memory: memory), ref(tgt: tgt, memory: memory)) > 1e-3)
        layer.eval()
        #expect(!layer.selfAttn.training && !layer.crossAttn.training)
        #expect(maxAbsDiff(layer(tgt: tgt, memory: memory), ref(tgt: tgt, memory: memory)) < 1e-5)
    }

    @Test("SinusoidalPositionalEncoding: eval equals dropout=0; train differs")
    func positionalEncoding() {
        var pe = nn.SinusoidalPositionalEncoding(dModel: 8, maxLen: 16, dropout: 0.5)
        let ref = nn.SinusoidalPositionalEncoding(dModel: 8, maxLen: 16, dropout: 0)
        let x = fill([2, 10, 8])
        #expect(maxAbsDiff(pe(x), ref(x)) > 1e-3)
        pe.eval()
        #expect(maxAbsDiff(pe(x), ref(x)) < 1e-6)
    }

    @Test("LSTM / GRU / RNN inter-layer dropout honours train/eval")
    func recurrentDropout() {
        let x = fill([2, 4, 3])

        var lstm = nn.LSTM(inputSize: 3, hiddenSize: 5, numLayers: 2, dropout: 0.5)
        let lstmRef = nn.LSTM(inputSize: 3, hiddenSize: 5, numLayers: 2, dropout: 0)
        copyWeights(from: lstm, to: lstmRef)
        #expect(maxAbsDiff(lstm(x).0, lstmRef(x).0) > 1e-3)
        lstm.eval()
        #expect(maxAbsDiff(lstm(x).0, lstmRef(x).0) < 1e-6)

        var gru = nn.GRU(inputSize: 3, hiddenSize: 5, numLayers: 2, dropout: 0.5)
        let gruRef = nn.GRU(inputSize: 3, hiddenSize: 5, numLayers: 2, dropout: 0)
        copyWeights(from: gru, to: gruRef)
        #expect(maxAbsDiff(gru(x).0, gruRef(x).0) > 1e-3)
        gru.eval()
        #expect(maxAbsDiff(gru(x).0, gruRef(x).0) < 1e-6)

        // nn.RNN already applied its dropout, but eval() never reached it.
        var rnn = nn.RNN(inputSize: 3, hiddenSize: 5, numLayers: 2, dropout: 0.5)
        let rnnRef = nn.RNN(inputSize: 3, hiddenSize: 5, numLayers: 2, dropout: 0)
        copyWeights(from: rnn, to: rnnRef)
        #expect(maxAbsDiff(rnn(x).0, rnnRef(x).0) > 1e-3)
        rnn.eval()
        #expect(maxAbsDiff(rnn(x).0, rnnRef(x).0) < 1e-6)
    }

    @Test("eval() on nn.Sequential reaches a nested transformer layer's dropout")
    func sequentialPropagation() {
        let layer = nn.TransformerEncoderLayer(dModel: 8, nHead: 2, dimFeedforward: 16, dropout: 0.5)
        let ref = nn.TransformerEncoderLayer(dModel: 8, nHead: 2, dimFeedforward: 16, dropout: 0)
        copyWeights(from: layer, to: ref)
        var model = nn.sequential { layer }
        let x = fill([1, 5, 8])

        #expect(maxAbsDiff(model(x), ref(x)) > 1e-3)
        model.eval()
        #expect(maxAbsDiff(model(x), ref(x)) < 1e-5)
    }
}
