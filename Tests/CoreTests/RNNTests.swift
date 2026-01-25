// Magma - Recurrent Neural Network Tests
// Tests for RNNCell, LSTMCell, GRUCell, RNN, LSTM, and GRU layers

import XCTest
import _Differentiation
@testable import Magma
@testable import LazyTensor

// MARK: - RNNCell Tests

final class RNNCellTests: XCTestCase {

    func testRNNCellCreation() {
        let cell = nn.RNNCell(inputSize: 64, hiddenSize: 128)
        XCTAssertEqual(cell.inputSize, 64)
        XCTAssertEqual(cell.hiddenSize, 128)
        XCTAssertTrue(cell.bias)
        XCTAssertEqual(cell.nonlinearity, .tanh)
    }

    func testRNNCellCreationReLU() {
        let cell = nn.RNNCell(inputSize: 32, hiddenSize: 64, nonlinearity: .relu)
        XCTAssertEqual(cell.nonlinearity, .relu)
    }

    func testRNNCellNoBias() {
        let cell = nn.RNNCell(inputSize: 32, hiddenSize: 64, bias: false)
        XCTAssertFalse(cell.bias)
        let params = cell.parameters()
        XCTAssertEqual(params.count, 2)  // Only weightIH and weightHH
    }

    func testRNNCellParameters() {
        let cell = nn.RNNCell(inputSize: 64, hiddenSize: 128)
        let params = cell.parameters()
        XCTAssertEqual(params.count, 4)  // weightIH, weightHH, biasIH, biasHH
        XCTAssertEqual(params[0].shape, [128, 64])   // weightIH
        XCTAssertEqual(params[1].shape, [128, 128])  // weightHH
        XCTAssertEqual(params[2].shape, [128])       // biasIH
        XCTAssertEqual(params[3].shape, [128])       // biasHH
    }

    func testRNNCellForward() {
        let cell = nn.RNNCell(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([8, 32])       // [batch, input]
        let h = Tensor<Float>.zeros([8, 64])       // [batch, hidden]
        let hNew = cell.forward(x: x, hidden: h)
        XCTAssertEqual(hNew.shape, [8, 64])
    }

    func testRNNCellCallable() {
        let cell = nn.RNNCell(inputSize: 16, hiddenSize: 32)
        let x = Tensor<Float>.randn([4, 16])
        let h = Tensor<Float>.randn([4, 32])
        let hNew = cell((x, h))
        XCTAssertEqual(hNew.shape, [4, 32])
    }
}

// MARK: - LSTMCell Tests

final class LSTMCellTests: XCTestCase {

    func testLSTMCellCreation() {
        let cell = nn.LSTMCell(inputSize: 64, hiddenSize: 128)
        XCTAssertEqual(cell.inputSize, 64)
        XCTAssertEqual(cell.hiddenSize, 128)
        XCTAssertTrue(cell.bias)
    }

    func testLSTMCellNoBias() {
        let cell = nn.LSTMCell(inputSize: 32, hiddenSize: 64, bias: false)
        XCTAssertFalse(cell.bias)
        let params = cell.parameters()
        XCTAssertEqual(params.count, 2)  // Only weights
    }

    func testLSTMCellParameters() {
        let cell = nn.LSTMCell(inputSize: 64, hiddenSize: 128)
        let params = cell.parameters()
        XCTAssertEqual(params.count, 4)
        // LSTM has 4 gates, so weights are 4x larger
        XCTAssertEqual(params[0].shape, [512, 64])   // weightIH: [4*hidden, input]
        XCTAssertEqual(params[1].shape, [512, 128])  // weightHH: [4*hidden, hidden]
        XCTAssertEqual(params[2].shape, [512])       // biasIH
        XCTAssertEqual(params[3].shape, [512])       // biasHH
    }

    func testLSTMCellForward() {
        let cell = nn.LSTMCell(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([8, 32])       // [batch, input]
        let h = Tensor<Float>.zeros([8, 64])       // [batch, hidden]
        let c = Tensor<Float>.zeros([8, 64])       // [batch, cell]
        let (hNew, cNew) = cell.forward(x: x, hidden: h, cell: c)
        XCTAssertEqual(hNew.shape, [8, 64])
        XCTAssertEqual(cNew.shape, [8, 64])
    }

    func testLSTMCellCallable() {
        let cell = nn.LSTMCell(inputSize: 16, hiddenSize: 32)
        let x = Tensor<Float>.randn([4, 16])
        let h = Tensor<Float>.randn([4, 32])
        let c = Tensor<Float>.randn([4, 32])
        let (hNew, cNew) = cell((x, (h, c)))
        XCTAssertEqual(hNew.shape, [4, 32])
        XCTAssertEqual(cNew.shape, [4, 32])
    }
}

// MARK: - GRUCell Tests

final class GRUCellTests: XCTestCase {

    func testGRUCellCreation() {
        let cell = nn.GRUCell(inputSize: 64, hiddenSize: 128)
        XCTAssertEqual(cell.inputSize, 64)
        XCTAssertEqual(cell.hiddenSize, 128)
        XCTAssertTrue(cell.bias)
    }

    func testGRUCellNoBias() {
        let cell = nn.GRUCell(inputSize: 32, hiddenSize: 64, bias: false)
        XCTAssertFalse(cell.bias)
        let params = cell.parameters()
        XCTAssertEqual(params.count, 2)
    }

    func testGRUCellParameters() {
        let cell = nn.GRUCell(inputSize: 64, hiddenSize: 128)
        let params = cell.parameters()
        XCTAssertEqual(params.count, 4)
        // GRU has 3 gates
        XCTAssertEqual(params[0].shape, [384, 64])   // weightIH: [3*hidden, input]
        XCTAssertEqual(params[1].shape, [384, 128])  // weightHH: [3*hidden, hidden]
        XCTAssertEqual(params[2].shape, [384])       // biasIH
        XCTAssertEqual(params[3].shape, [384])       // biasHH
    }

    func testGRUCellForward() {
        let cell = nn.GRUCell(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([8, 32])
        let h = Tensor<Float>.zeros([8, 64])
        let hNew = cell.forward(x: x, hidden: h)
        XCTAssertEqual(hNew.shape, [8, 64])
    }

    func testGRUCellCallable() {
        let cell = nn.GRUCell(inputSize: 16, hiddenSize: 32)
        let x = Tensor<Float>.randn([4, 16])
        let h = Tensor<Float>.randn([4, 32])
        let hNew = cell((x, h))
        XCTAssertEqual(hNew.shape, [4, 32])
    }
}

// MARK: - RNN Layer Tests

final class RNNLayerTests: XCTestCase {

    func testRNNCreation() {
        let rnn = nn.RNN(inputSize: 64, hiddenSize: 128)
        XCTAssertEqual(rnn.inputSize, 64)
        XCTAssertEqual(rnn.hiddenSize, 128)
        XCTAssertEqual(rnn.numLayers, 1)
        XCTAssertFalse(rnn.bidirectional)
        XCTAssertTrue(rnn.batchFirst)
    }

    func testRNNMultiLayer() {
        let rnn = nn.RNN(inputSize: 64, hiddenSize: 128, numLayers: 3)
        XCTAssertEqual(rnn.numLayers, 3)
        XCTAssertEqual(rnn.cells.count, 3)
        // First layer: input -> hidden
        XCTAssertEqual(rnn.cells[0].inputSize, 64)
        XCTAssertEqual(rnn.cells[0].hiddenSize, 128)
        // Subsequent layers: hidden -> hidden
        XCTAssertEqual(rnn.cells[1].inputSize, 128)
        XCTAssertEqual(rnn.cells[2].inputSize, 128)
    }

    func testRNNParameters() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let params = rnn.parameters()
        // Each cell has 4 params (weightIH, weightHH, biasIH, biasHH)
        XCTAssertEqual(params.count, 8)  // 2 layers * 4 params
    }

    func testRNNForward() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([4, 10, 32])  // [batch, seq, input]
        let (output, hn) = rnn(x)
        XCTAssertEqual(output.shape, [4, 10, 64])  // [batch, seq, hidden]
        XCTAssertEqual(hn.shape, [1, 4, 64])       // [layers, batch, hidden]
    }

    func testRNNForwardMultiLayer() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, numLayers: 3)
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, hn) = rnn(x)
        XCTAssertEqual(output.shape, [4, 10, 64])
        XCTAssertEqual(hn.shape, [3, 4, 64])  // 3 layers
    }

    func testRNNWithInitialHidden() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let x = Tensor<Float>.randn([4, 10, 32])
        let h0 = Tensor<Float>.randn([2, 4, 64])
        let (output, hn) = rnn.forward(x, hidden: h0)
        XCTAssertEqual(output.shape, [4, 10, 64])
        XCTAssertEqual(hn.shape, [2, 4, 64])
    }

    func testRNNSeqFirst() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, batchFirst: false)
        let x = Tensor<Float>.randn([10, 4, 32])  // [seq, batch, input]
        let (output, hn) = rnn(x)
        XCTAssertEqual(output.shape, [10, 4, 64])  // [seq, batch, hidden]
        XCTAssertEqual(hn.shape, [1, 4, 64])
    }
}

// MARK: - LSTM Layer Tests

final class LSTMLayerTests: XCTestCase {

    func testLSTMCreation() {
        let lstm = nn.LSTM(inputSize: 64, hiddenSize: 128)
        XCTAssertEqual(lstm.inputSize, 64)
        XCTAssertEqual(lstm.hiddenSize, 128)
        XCTAssertEqual(lstm.numLayers, 1)
        XCTAssertFalse(lstm.bidirectional)
    }

    func testLSTMMultiLayer() {
        let lstm = nn.LSTM(inputSize: 64, hiddenSize: 128, numLayers: 4)
        XCTAssertEqual(lstm.numLayers, 4)
        XCTAssertEqual(lstm.cells.count, 4)
    }

    func testLSTMParameters() {
        let lstm = nn.LSTM(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let params = lstm.parameters()
        XCTAssertEqual(params.count, 8)  // 2 layers * 4 params each
    }

    func testLSTMForward() {
        let lstm = nn.LSTM(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, (hn, cn)) = lstm(x)
        XCTAssertEqual(output.shape, [4, 10, 64])
        XCTAssertEqual(hn.shape, [1, 4, 64])
        XCTAssertEqual(cn.shape, [1, 4, 64])
    }

    func testLSTMForwardMultiLayer() {
        let lstm = nn.LSTM(inputSize: 32, hiddenSize: 64, numLayers: 3)
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, (hn, cn)) = lstm(x)
        XCTAssertEqual(output.shape, [4, 10, 64])
        XCTAssertEqual(hn.shape, [3, 4, 64])
        XCTAssertEqual(cn.shape, [3, 4, 64])
    }

    func testLSTMWithInitialState() {
        let lstm = nn.LSTM(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let x = Tensor<Float>.randn([4, 10, 32])
        let h0 = Tensor<Float>.randn([2, 4, 64])
        let c0 = Tensor<Float>.randn([2, 4, 64])
        let (output, (hn, cn)) = lstm.forward(x, hidden: (h0, c0))
        XCTAssertEqual(output.shape, [4, 10, 64])
        XCTAssertEqual(hn.shape, [2, 4, 64])
        XCTAssertEqual(cn.shape, [2, 4, 64])
    }

    func testLSTMSeqFirst() {
        let lstm = nn.LSTM(inputSize: 32, hiddenSize: 64, batchFirst: false)
        let x = Tensor<Float>.randn([10, 4, 32])  // [seq, batch, input]
        let (output, (hn, cn)) = lstm(x)
        XCTAssertEqual(output.shape, [10, 4, 64])
        XCTAssertEqual(hn.shape, [1, 4, 64])
        XCTAssertEqual(cn.shape, [1, 4, 64])
    }
}

// MARK: - GRU Layer Tests

final class GRULayerTests: XCTestCase {

    func testGRUCreation() {
        let gru = nn.GRU(inputSize: 64, hiddenSize: 128)
        XCTAssertEqual(gru.inputSize, 64)
        XCTAssertEqual(gru.hiddenSize, 128)
        XCTAssertEqual(gru.numLayers, 1)
        XCTAssertFalse(gru.bidirectional)
    }

    func testGRUMultiLayer() {
        let gru = nn.GRU(inputSize: 64, hiddenSize: 128, numLayers: 3)
        XCTAssertEqual(gru.numLayers, 3)
        XCTAssertEqual(gru.cells.count, 3)
    }

    func testGRUParameters() {
        let gru = nn.GRU(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let params = gru.parameters()
        XCTAssertEqual(params.count, 8)
    }

    func testGRUForward() {
        let gru = nn.GRU(inputSize: 32, hiddenSize: 64)
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, hn) = gru(x)
        XCTAssertEqual(output.shape, [4, 10, 64])
        XCTAssertEqual(hn.shape, [1, 4, 64])
    }

    func testGRUForwardMultiLayer() {
        let gru = nn.GRU(inputSize: 32, hiddenSize: 64, numLayers: 3)
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, hn) = gru(x)
        XCTAssertEqual(output.shape, [4, 10, 64])
        XCTAssertEqual(hn.shape, [3, 4, 64])
    }

    func testGRUWithInitialHidden() {
        let gru = nn.GRU(inputSize: 32, hiddenSize: 64, numLayers: 2)
        let x = Tensor<Float>.randn([4, 10, 32])
        let h0 = Tensor<Float>.randn([2, 4, 64])
        let (output, hn) = gru.forward(x, hidden: h0)
        XCTAssertEqual(output.shape, [4, 10, 64])
        XCTAssertEqual(hn.shape, [2, 4, 64])
    }

    func testGRUSeqFirst() {
        let gru = nn.GRU(inputSize: 32, hiddenSize: 64, batchFirst: false)
        let x = Tensor<Float>.randn([10, 4, 32])
        let (output, hn) = gru(x)
        XCTAssertEqual(output.shape, [10, 4, 64])
        XCTAssertEqual(hn.shape, [1, 4, 64])
    }
}

// MARK: - RNN Integration Tests

final class RNNIntegrationTests: XCTestCase {

    func testRNNCellSequenceProcessing() {
        // Manually process a sequence using RNNCell
        let cell = nn.RNNCell(inputSize: 16, hiddenSize: 32)
        let seqLen = 5
        let batch = 4

        var h = Tensor<Float>.zeros([batch, 32])
        for _ in 0..<seqLen {
            let x = Tensor<Float>.randn([batch, 16])
            h = cell.forward(x: x, hidden: h)
        }
        XCTAssertEqual(h.shape, [batch, 32])
    }

    func testLSTMCellSequenceProcessing() {
        let cell = nn.LSTMCell(inputSize: 16, hiddenSize: 32)
        let seqLen = 5
        let batch = 4

        var h = Tensor<Float>.zeros([batch, 32])
        var c = Tensor<Float>.zeros([batch, 32])
        for _ in 0..<seqLen {
            let x = Tensor<Float>.randn([batch, 16])
            (h, c) = cell.forward(x: x, hidden: h, cell: c)
        }
        XCTAssertEqual(h.shape, [batch, 32])
        XCTAssertEqual(c.shape, [batch, 32])
    }

    func testGRUCellSequenceProcessing() {
        let cell = nn.GRUCell(inputSize: 16, hiddenSize: 32)
        let seqLen = 5
        let batch = 4

        var h = Tensor<Float>.zeros([batch, 32])
        for _ in 0..<seqLen {
            let x = Tensor<Float>.randn([batch, 16])
            h = cell.forward(x: x, hidden: h)
        }
        XCTAssertEqual(h.shape, [batch, 32])
    }

    func testStackedLSTM() {
        // Create a deep LSTM
        let lstm = nn.LSTM(inputSize: 64, hiddenSize: 128, numLayers: 4)
        let x = Tensor<Float>.randn([8, 20, 64])  // [batch, seq, input]
        let (output, (hn, cn)) = lstm(x)

        XCTAssertEqual(output.shape, [8, 20, 128])
        XCTAssertEqual(hn.shape, [4, 8, 128])  // 4 layers
        XCTAssertEqual(cn.shape, [4, 8, 128])
    }

    func testRNNParameterCount() {
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

        XCTAssertEqual(totalParams, expected)
    }

    func testLSTMParameterCount() {
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

        XCTAssertEqual(totalParams, expected)
    }

    func testGRUParameterCount() {
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

        XCTAssertEqual(totalParams, expected)
    }
}

// MARK: - Bidirectional RNN Tests

final class BidirectionalRNNTests: XCTestCase {

    func testBidirectionalRNNCreation() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, bidirectional: true)
        XCTAssertTrue(rnn.bidirectional)
        XCTAssertEqual(rnn.numDirections, 2)
        XCTAssertEqual(rnn.cells.count, 2)  // Forward and backward cells
    }

    func testBidirectionalRNNForward() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, bidirectional: true)
        let x = Tensor<Float>.randn([4, 10, 32])  // [batch, seq, input]
        let (output, hn) = rnn(x)
        // Output should have 2*hiddenSize features (forward + backward concatenated)
        XCTAssertEqual(output.shape, [4, 10, 128])
        // Hidden should have 2 layers (forward and backward)
        XCTAssertEqual(hn.shape, [2, 4, 64])
    }

    func testBidirectionalRNNMultiLayer() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, numLayers: 2, bidirectional: true)
        XCTAssertEqual(rnn.cells.count, 4)  // 2 layers * 2 directions
        let x = Tensor<Float>.randn([4, 10, 32])
        let (output, hn) = rnn(x)
        XCTAssertEqual(output.shape, [4, 10, 128])  // 2*hiddenSize
        XCTAssertEqual(hn.shape, [4, 4, 64])  // 2 layers * 2 directions
    }

    func testBidirectionalRNNParameters() {
        let rnn = nn.RNN(inputSize: 32, hiddenSize: 64, bidirectional: true)
        let params = rnn.parameters()
        // 2 cells, each with 4 params (weightIH, weightHH, biasIH, biasHH)
        XCTAssertEqual(params.count, 8)
    }
}
