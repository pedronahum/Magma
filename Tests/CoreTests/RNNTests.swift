// Magma - Recurrent Neural Network Tests
// Tests for RNNCell, LSTMCell, GRUCell, RNN, LSTM, and GRU layers

import Testing
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - RNNCell Tests

@Suite("RNNCell Tests")
struct RNNCellTests {

    @Test("RNNCell creation")
    func rnnCellCreation() {
        let cell = nn.RNNCell(inputSize: 64, hiddenSize: 128)
        #expect(cell.inputSize == 64)
        #expect(cell.hiddenSize == 128)
        #expect(cell.bias)
        #expect(cell.nonlinearity == .tanh)
    }

    @Test("RNNCell creation ReLU")
    func rnnCellCreationReLU() {
        let cell = nn.RNNCell(inputSize: 32, hiddenSize: 64, nonlinearity: .relu)
        #expect(cell.nonlinearity == .relu)
    }

    @Test("RNNCell no bias")
    func rnnCellNoBias() {
        let cell = nn.RNNCell(inputSize: 32, hiddenSize: 64, bias: false)
        #expect(!cell.bias)
        let params = cell.parameters()
        #expect(params.count == 2)  // Only weightIH and weightHH
    }

    @Test("RNNCell parameters")
    func rnnCellParameters() {
        let cell = nn.RNNCell(inputSize: 64, hiddenSize: 128)
        let params = cell.parameters()
        #expect(params.count == 4)  // weightIH, weightHH, biasIH, biasHH
        #expect(params[0].shape == [128, 64])   // weightIH
        #expect(params[1].shape == [128, 128])  // weightHH
        #expect(params[2].shape == [128])       // biasIH
        #expect(params[3].shape == [128])       // biasHH
    }

    @Test("RNNCell forward")
    func rnnCellForward() {
        let cell = nn.RNNCell(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([8, 32])       // [batch, input]
        let h = Tensor<Float>.zeros([8, 64])       // [batch, hidden]
        let hNew = cell.forward(x: x, hidden: h)
        #expect(hNew.shape == [8, 64])
    }

    @Test("RNNCell callable")
    func rnnCellCallable() {
        let cell = nn.RNNCell(inputSize: 16, hiddenSize: 32)
        let x = Tensor<Float>.randn([4, 16])
        let h = Tensor<Float>.randn([4, 32])
        let hNew = cell((x, h))
        #expect(hNew.shape == [4, 32])
    }
}

// MARK: - LSTMCell Tests

@Suite("LSTMCell Tests")
struct LSTMCellTests {

    @Test("LSTMCell creation")
    func lstmCellCreation() {
        let cell = nn.LSTMCell(inputSize: 64, hiddenSize: 128)
        #expect(cell.inputSize == 64)
        #expect(cell.hiddenSize == 128)
        #expect(cell.bias)
    }

    @Test("LSTMCell no bias")
    func lstmCellNoBias() {
        let cell = nn.LSTMCell(inputSize: 32, hiddenSize: 64, bias: false)
        #expect(!cell.bias)
        let params = cell.parameters()
        #expect(params.count == 2)  // Only weights
    }

    @Test("LSTMCell parameters")
    func lstmCellParameters() {
        let cell = nn.LSTMCell(inputSize: 64, hiddenSize: 128)
        let params = cell.parameters()
        #expect(params.count == 4)
        // LSTM has 4 gates, so weights are 4x larger
        #expect(params[0].shape == [512, 64])   // weightIH: [4*hidden, input]
        #expect(params[1].shape == [512, 128])  // weightHH: [4*hidden, hidden]
        #expect(params[2].shape == [512])       // biasIH
        #expect(params[3].shape == [512])       // biasHH
    }

    @Test("LSTMCell forward")
    func lstmCellForward() {
        let cell = nn.LSTMCell(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([8, 32])       // [batch, input]
        let h = Tensor<Float>.zeros([8, 64])       // [batch, hidden]
        let c = Tensor<Float>.zeros([8, 64])       // [batch, cell]
        let (hNew, cNew) = cell.forward(x: x, hidden: h, cell: c)
        #expect(hNew.shape == [8, 64])
        #expect(cNew.shape == [8, 64])
    }

    @Test("LSTMCell callable")
    func lstmCellCallable() {
        let cell = nn.LSTMCell(inputSize: 16, hiddenSize: 32)
        let x = Tensor<Float>.randn([4, 16])
        let h = Tensor<Float>.randn([4, 32])
        let c = Tensor<Float>.randn([4, 32])
        let (hNew, cNew) = cell((x, (h, c)))
        #expect(hNew.shape == [4, 32])
        #expect(cNew.shape == [4, 32])
    }
}

// MARK: - GRUCell Tests

@Suite("GRUCell Tests")
struct GRUCellTests {

    @Test("GRUCell creation")
    func gruCellCreation() {
        let cell = nn.GRUCell(inputSize: 64, hiddenSize: 128)
        #expect(cell.inputSize == 64)
        #expect(cell.hiddenSize == 128)
        #expect(cell.bias)
    }

    @Test("GRUCell no bias")
    func gruCellNoBias() {
        let cell = nn.GRUCell(inputSize: 32, hiddenSize: 64, bias: false)
        #expect(!cell.bias)
        let params = cell.parameters()
        #expect(params.count == 2)
    }

    @Test("GRUCell parameters")
    func gruCellParameters() {
        let cell = nn.GRUCell(inputSize: 64, hiddenSize: 128)
        let params = cell.parameters()
        #expect(params.count == 4)
        // GRU has 3 gates
        #expect(params[0].shape == [384, 64])   // weightIH: [3*hidden, input]
        #expect(params[1].shape == [384, 128])  // weightHH: [3*hidden, hidden]
        #expect(params[2].shape == [384])       // biasIH
        #expect(params[3].shape == [384])       // biasHH
    }

    @Test("GRUCell forward")
    func gruCellForward() {
        let cell = nn.GRUCell(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([8, 32])
        let h = Tensor<Float>.zeros([8, 64])
        let hNew = cell.forward(x: x, hidden: h)
        #expect(hNew.shape == [8, 64])
    }

    @Test("GRUCell callable")
    func gruCellCallable() {
        let cell = nn.GRUCell(inputSize: 16, hiddenSize: 32)
        let x = Tensor<Float>.randn([4, 16])
        let h = Tensor<Float>.randn([4, 32])
        let hNew = cell((x, h))
        #expect(hNew.shape == [4, 32])
    }
}

// MARK: - RNN Layer Tests

@Suite("RNN Layer Tests")
struct RNNLayerTests {

    @Test("RNN creation")
    func rnnCreation() {
        let rnn = nn.RNN(inputSize: 64, hiddenSize: 128)
        #expect(rnn.inputSize == 64)
        #expect(rnn.hiddenSize == 128)
        #expect(rnn.numLayers == 1)
        #expect(!rnn.bidirectional)
        #expect(rnn.batchFirst)
    }

    @Test("RNN multi layer")
    func rnnMultiLayer() {
        let rnn = nn.RNN(inputSize: 64, hiddenSize: 128, numLayers: 3)
        #expect(rnn.numLayers == 3)
        #expect(rnn.cells.count == 3)
        // First layer: input -> hidden
        #expect(rnn.cells[0].inputSize == 64)
        #expect(rnn.cells[0].hiddenSize == 128)
        // Subsequent layers: hidden -> hidden
        #expect(rnn.cells[1].inputSize == 128)
        #expect(rnn.cells[2].inputSize == 128)
    }

    @Test("RNN parameters")
    func rnnParameters() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let params = rnn.parameters()
        // Each cell has 4 params (weightIH, weightHH, biasIH, biasHH)
        #expect(params.count == 8)  // 2 layers * 4 params
    }

    @Test("RNN forward")
    func rnnForward() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([4, 10, 32])  // [batch, seq, input]
        let (output, hn) = rnn(x)
        #expect(output.shape == [4, 10, 64])  // [batch, seq, hidden]
        #expect(hn.shape == [1, 4, 64])       // [layers, batch, hidden]
    }

    @Test("RNN forward multi layer")
    func rnnForwardMultiLayer() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, numLayers: 3)
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, hn) = rnn(x)
        #expect(output.shape == [4, 10, 64])
        #expect(hn.shape == [3, 4, 64])  // 3 layers
    }

    @Test("RNN with initial hidden")
    func rnnWithInitialHidden() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let x = Tensor<Float>.randn([4, 10, 32])
        let h0 = Tensor<Float>.randn([2, 4, 64])
        let (output, hn) = rnn.forward(x, hidden: h0)
        #expect(output.shape == [4, 10, 64])
        #expect(hn.shape == [2, 4, 64])
    }

    @Test("RNN seq first")
    func rnnSeqFirst() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, batchFirst: false)
        let x = Tensor<Float>.randn([10, 4, 32])  // [seq, batch, input]
        let (output, hn) = rnn(x)
        #expect(output.shape == [10, 4, 64])  // [seq, batch, hidden]
        #expect(hn.shape == [1, 4, 64])
    }
}

// MARK: - LSTM Layer Tests

@Suite("LSTM Layer Tests")
struct LSTMLayerTests {

    @Test("LSTM creation")
    func lstmCreation() {
        let lstm = nn.LSTM(inputSize: 64, hiddenSize: 128)
        #expect(lstm.inputSize == 64)
        #expect(lstm.hiddenSize == 128)
        #expect(lstm.numLayers == 1)
        #expect(!lstm.bidirectional)
    }

    @Test("LSTM multi layer")
    func lstmMultiLayer() {
        let lstm = nn.LSTM(inputSize: 64, hiddenSize: 128, numLayers: 4)
        #expect(lstm.numLayers == 4)
        #expect(lstm.cells.count == 4)
    }

    @Test("LSTM parameters")
    func lstmParameters() {
        let lstm = nn.LSTM(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let params = lstm.parameters()
        #expect(params.count == 8)  // 2 layers * 4 params each
    }

    @Test("LSTM forward")
    func lstmForward() {
        let lstm = nn.LSTM(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, (hn, cn)) = lstm(x)
        #expect(output.shape == [4, 10, 64])
        #expect(hn.shape == [1, 4, 64])
        #expect(cn.shape == [1, 4, 64])
    }

    @Test("LSTM forward multi layer")
    func lstmForwardMultiLayer() {
        let lstm = nn.LSTM(inputSize: 32, hiddenSize: 64, numLayers: 3)
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, (hn, cn)) = lstm(x)
        #expect(output.shape == [4, 10, 64])
        #expect(hn.shape == [3, 4, 64])
        #expect(cn.shape == [3, 4, 64])
    }

    @Test("LSTM with initial state")
    func lstmWithInitialState() {
        let lstm = nn.LSTM(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let x = Tensor<Float>.randn([4, 10, 32])
        let h0 = Tensor<Float>.randn([2, 4, 64])
        let c0 = Tensor<Float>.randn([2, 4, 64])
        let (output, (hn, cn)) = lstm.forward(x, hidden: (h0, c0))
        #expect(output.shape == [4, 10, 64])
        #expect(hn.shape == [2, 4, 64])
        #expect(cn.shape == [2, 4, 64])
    }

    @Test("LSTM seq first")
    func lstmSeqFirst() {
        let lstm = nn.LSTM(inputSize: 32, hiddenSize: 64, batchFirst: false)
        let x = Tensor<Float>.randn([10, 4, 32])  // [seq, batch, input]
        let (output, (hn, cn)) = lstm(x)
        #expect(output.shape == [10, 4, 64])
        #expect(hn.shape == [1, 4, 64])
        #expect(cn.shape == [1, 4, 64])
    }
}

// MARK: - GRU Layer Tests

@Suite("GRU Layer Tests")
struct GRULayerTests {

    @Test("GRU creation")
    func gruCreation() {
        let gru = nn.GRU(inputSize: 64, hiddenSize: 128)
        #expect(gru.inputSize == 64)
        #expect(gru.hiddenSize == 128)
        #expect(gru.numLayers == 1)
        #expect(!gru.bidirectional)
    }

    @Test("GRU multi layer")
    func gruMultiLayer() {
        let gru = nn.GRU(inputSize: 64, hiddenSize: 128, numLayers: 3)
        #expect(gru.numLayers == 3)
        #expect(gru.cells.count == 3)
    }

    @Test("GRU parameters")
    func gruParameters() {
        let gru = nn.GRU(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let params = gru.parameters()
        #expect(params.count == 8)
    }

    @Test("GRU forward")
    func gruForward() {
        let gru = nn.GRU(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, hn) = gru(x)
        #expect(output.shape == [4, 10, 64])
        #expect(hn.shape == [1, 4, 64])
    }

    @Test("GRU forward multi layer")
    func gruForwardMultiLayer() {
        let gru = nn.GRU(inputSize: 32, hiddenSize: 64, numLayers: 3)
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, hn) = gru(x)
        #expect(output.shape == [4, 10, 64])
        #expect(hn.shape == [3, 4, 64])
    }

    @Test("GRU with initial hidden")
    func gruWithInitialHidden() {
        let gru = nn.GRU(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let x = Tensor<Float>.randn([4, 10, 32])
        let h0 = Tensor<Float>.randn([2, 4, 64])
        let (output, hn) = gru.forward(x, hidden: h0)
        #expect(output.shape == [4, 10, 64])
        #expect(hn.shape == [2, 4, 64])
    }

    @Test("GRU seq first")
    func gruSeqFirst() {
        let gru = nn.GRU(inputSize: 32, hiddenSize: 64, batchFirst: false)
        let x = Tensor<Float>.randn([10, 4, 32])
        let (output, hn) = gru(x)
        #expect(output.shape == [10, 4, 64])
        #expect(hn.shape == [1, 4, 64])
    }
}

// MARK: - RNN Integration Tests

@Suite("RNN Integration Tests")
struct RNNIntegrationTests {

    @Test("RNNCell sequence processing")
    func rnnCellSequenceProcessing() {
        // Manually process a sequence using RNNCell
        let cell = nn.RNNCell(inputSize: 16, hiddenSize: 32)
        let seqLen = 5
        let batch = 4

        var h = Tensor<Float>.zeros([batch, 32])
        for _ in 0..<seqLen {
            let x = Tensor<Float>.randn([batch, 16])
            h = cell.forward(x: x, hidden: h)
        }
        #expect(h.shape == [batch, 32])
    }

    @Test("LSTMCell sequence processing")
    func lstmCellSequenceProcessing() {
        let cell = nn.LSTMCell(inputSize: 16, hiddenSize: 32)
        let seqLen = 5
        let batch = 4

        var h = Tensor<Float>.zeros([batch, 32])
        var c = Tensor<Float>.zeros([batch, 32])
        for _ in 0..<seqLen {
            let x = Tensor<Float>.randn([batch, 16])
            (h, c) = cell.forward(x: x, hidden: h, cell: c)
        }
        #expect(h.shape == [batch, 32])
        #expect(c.shape == [batch, 32])
    }

    @Test("GRUCell sequence processing")
    func gruCellSequenceProcessing() {
        let cell = nn.GRUCell(inputSize: 16, hiddenSize: 32)
        let seqLen = 5
        let batch = 4

        var h = Tensor<Float>.zeros([batch, 32])
        for _ in 0..<seqLen {
            let x = Tensor<Float>.randn([batch, 16])
            h = cell.forward(x: x, hidden: h)
        }
        #expect(h.shape == [batch, 32])
    }

    @Test("Stacked LSTM")
    func stackedLSTM() {
        // Create a deep LSTM
        let lstm = nn.LSTM(inputSize: 64, hiddenSize: 128, numLayers: 4)
        let x = Tensor<Float>.randn([8, 20, 64])  // [batch, seq, input]
        let (output, (hn, cn)) = lstm(x)

        #expect(output.shape == [8, 20, 128])
        #expect(hn.shape == [4, 8, 128])  // 4 layers
        #expect(cn.shape == [4, 8, 128])
    }

    @Test("RNN parameter count")
    func rnnParameterCount() {
        // Verify parameter counts match expected
        let inputSize = 32
        let hiddenSize = 64
        let numLayers = 2

        let rnn = nn.RNN(inputSize: inputSize, hiddenSize: hiddenSize, numLayers: numLayers)
        let params = rnn.parameters()

        var totalParams = 0
        for p in params {
            totalParams += p.elementCount
        }

        // Layer 0: inputSize * hiddenSize + hiddenSize * hiddenSize + 2 * hiddenSize
        // Layer 1+: hiddenSize * hiddenSize + hiddenSize * hiddenSize + 2 * hiddenSize
        let layer0Params = inputSize * hiddenSize + hiddenSize * hiddenSize + 2 * hiddenSize
        let layer1Params = hiddenSize * hiddenSize + hiddenSize * hiddenSize + 2 * hiddenSize
        let expected = layer0Params + layer1Params

        #expect(totalParams == expected)
    }

    @Test("LSTM parameter count")
    func lstmParameterCount() {
        let inputSize = 32
        let hiddenSize = 64

        let lstm = nn.LSTM(inputSize: inputSize, hiddenSize: hiddenSize, numLayers: 1)
        let params = lstm.parameters()

        var totalParams = 0
        for p in params {
            totalParams += p.elementCount
        }

        // LSTM has 4 gates, each with input and hidden weights plus biases
        // 4 * (inputSize * hiddenSize) + 4 * (hiddenSize * hiddenSize) + 4 * 2 * hiddenSize
        let expected = 4 * inputSize * hiddenSize + 4 * hiddenSize * hiddenSize + 8 * hiddenSize

        #expect(totalParams == expected)
    }

    @Test("GRU parameter count")
    func gruParameterCount() {
        let inputSize = 32
        let hiddenSize = 64

        let gru = nn.GRU(inputSize: inputSize, hiddenSize: hiddenSize, numLayers: 1)
        let params = gru.parameters()

        var totalParams = 0
        for p in params {
            totalParams += p.elementCount
        }

        // GRU has 3 gates
        let expected = 3 * inputSize * hiddenSize + 3 * hiddenSize * hiddenSize + 6 * hiddenSize

        #expect(totalParams == expected)
    }
}

// MARK: - Bidirectional RNN Tests

@Suite("Bidirectional RNN Tests")
struct BidirectionalRNNTests {

    @Test("Bidirectional RNN creation")
    func bidirectionalRNNCreation() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, bidirectional: true)
        #expect(rnn.bidirectional)
        #expect(rnn.numDirections == 2)
        #expect(rnn.cells.count == 2)  // Forward and backward cells
    }

    @Test("Bidirectional RNN forward")
    func bidirectionalRNNForward() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, bidirectional: true)
        let x = Tensor<Float>.randn([4, 10, 32])  // [batch, seq, input]
        let (output, hn) = rnn(x)
        // Output should have 2*hiddenSize features (forward + backward concatenated)
        #expect(output.shape == [4, 10, 128])
        // Hidden should have 2 layers (forward and backward)
        #expect(hn.shape == [2, 4, 64])
    }

    @Test("Bidirectional RNN multi layer")
    func bidirectionalRNNMultiLayer() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, numLayers: 2, bidirectional: true)
        #expect(rnn.cells.count == 4)  // 2 layers * 2 directions
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, hn) = rnn(x)
        #expect(output.shape == [4, 10, 128])  // 2*hiddenSize
        #expect(hn.shape == [4, 4, 64])  // 2 layers * 2 directions
    }

    @Test("Bidirectional RNN parameters")
    func bidirectionalRNNParameters() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, bidirectional: true)
        let params = rnn.parameters()
        // 2 cells, each with 4 params (weightIH, weightHH, biasIH, biasHH)
        #expect(params.count == 8)
    }
}
