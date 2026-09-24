// Magma - Module.to(device:) coverage
// The protocol default used to be a no-op, so nn.Sequential and most
// parameterized layers (LayerNorm, MultiheadAttention, BatchNorm1d incl. its
// running stats, RNN/LSTM/GRU, Transformer layers, ...) silently stayed on the
// original device. These tests only relabel lazy (unmaterialized) tensors onto a
// second CPU device index, so no device buffers or clients are created.

import Testing
@testable import Magma
import XLARuntime

@Suite("Module.to(device:)")
struct ModuleDeviceTests {

    private let target = Device(backend: .cpu, index: 1)

    /// Moves `module` and checks every parameter and buffer landed on `target`.
    private func expectMoved<M: Module>(_ module: M, _ label: String,
                                        sourceLocation: SourceLocation = #_sourceLocation) {
        var m = module
        let state = m.parameters() + m.buffers()
        #expect(!state.isEmpty, "\(label) has no parameters to move", sourceLocation: sourceLocation)
        #expect(state.allSatisfy { $0.value.device == .default }, "\(label) did not start on the default device",
                sourceLocation: sourceLocation)
        m.to(device: target)
        for p in state {
            #expect(p.value.device == target, "\(label): \(p.name ?? "param") stayed on \(p.value.device)",
                    sourceLocation: sourceLocation)
        }
    }

    @Test("nn.Sequential forwards to(device:) to every child, including buffers")
    func sequentialMovesChildren() {
        let bn = nn.BatchNorm1d(numFeatures: 4)
        var model = nn.sequential {
            nn.Linear(inputSize: 4, outputSize: 4)
            nn.LayerNorm(4)
            bn
            nn.PReLU()
        }
        model.to(device: target)
        let state = model.parameters() + model.buffers()
        #expect(state.count == 2 + 2 + 2 + 2 + 1)
        #expect(state.allSatisfy { $0.value.device == target })
        // The buffers are the BatchNorm running statistics.
        #expect(bn.runningMean.value.device == target)
        #expect(bn.runningVar.value.device == target)
    }

    @Test("parameterized layers without a custom override move all their state")
    func layersMove() {
        expectMoved(nn.LayerNorm(8), "LayerNorm")
        expectMoved(nn.BatchNorm1d(numFeatures: 3), "BatchNorm1d")
        expectMoved(nn.GroupNorm(numGroups: 2, numChannels: 4), "GroupNorm")
        expectMoved(nn.InstanceNorm2d(numFeatures: 3, affine: true), "InstanceNorm2d")
        expectMoved(nn.PReLU(numParameters: 3), "PReLU")
        expectMoved(nn.MultiheadAttention(embedDim: 8, numHeads: 2), "MultiheadAttention")
        expectMoved(nn.TransformerEncoderLayer(dModel: 8, nHead: 2, dimFeedforward: 16), "TransformerEncoderLayer")
        expectMoved(nn.TransformerDecoderLayer(dModel: 8, nHead: 2, dimFeedforward: 16), "TransformerDecoderLayer")
        expectMoved(nn.LearnedPositionalEmbedding(dModel: 4, maxLen: 6), "LearnedPositionalEmbedding")
        expectMoved(nn.RNNCell(inputSize: 3, hiddenSize: 4), "RNNCell")
        expectMoved(nn.LSTMCell(inputSize: 3, hiddenSize: 4), "LSTMCell")
        expectMoved(nn.GRUCell(inputSize: 3, hiddenSize: 4), "GRUCell")
        expectMoved(nn.RNN(inputSize: 3, hiddenSize: 4, numLayers: 2, bidirectional: true), "RNN")
        expectMoved(nn.LSTM(inputSize: 3, hiddenSize: 4, numLayers: 2, bidirectional: true), "LSTM")
        expectMoved(nn.GRU(inputSize: 3, hiddenSize: 4, numLayers: 2), "GRU")
    }

    @Test("SinusoidalPositionalEncoding moves its (non-Parameter) encoding table")
    func positionalEncodingMoves() {
        var pe = nn.SinusoidalPositionalEncoding(dModel: 4, maxLen: 8)
        pe.to(device: target)
        #expect(pe.pe.device == target)
    }
}
