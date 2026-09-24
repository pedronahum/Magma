// Magma - Value-level tests for recurrent cells and (bidirectional) LSTM/GRU.
// The existing RNN tests only checked shapes. These compare materialized
// outputs against a plain host implementation of the PyTorch equations:
//   - one LSTMCell / GRUCell step with batch > 1 (the gate split used to mix
//     rows across the batch once batch > 1);
//   - multi-layer bidirectional LSTM/GRU, whose reverse direction used to be
//     skipped entirely (wrong output width; crash for numLayers >= 2).
//
// .serialized: materialization drives the PJRT backend.

import Foundation
import Testing
@testable import Magma

// MARK: - Host reference

private func sigmoid(_ x: Float) -> Float { 1 / (1 + Foundation.exp(-x)) }

/// y[b, o] = sum_i x[b, i] * W[o, i] + bias[o]
private func affine(_ x: [Float], batch: Int, inSize: Int, _ w: [Float], outSize: Int, _ b: [Float]) -> [Float] {
    var y = [Float](repeating: 0, count: batch * outSize)
    for bi in 0..<batch {
        for o in 0..<outSize {
            var acc = b[o]
            for i in 0..<inSize { acc += x[bi * inSize + i] * w[o * inSize + i] }
            y[bi * outSize + o] = acc
        }
    }
    return y
}

private struct HostCell {
    let wIH: [Float], wHH: [Float], bIH: [Float], bHH: [Float]
    let inputSize: Int, hiddenSize: Int

    init(lstm c: nn.LSTMCell) {
        wIH = c.weightIH.value.scalars(); wHH = c.weightHH.value.scalars()
        bIH = c.biasIH.value.scalars(); bHH = c.biasHH.value.scalars()
        inputSize = c.inputSize; hiddenSize = c.hiddenSize
    }

    init(gru c: nn.GRUCell) {
        wIH = c.weightIH.value.scalars(); wHH = c.weightHH.value.scalars()
        bIH = c.biasIH.value.scalars(); bHH = c.biasHH.value.scalars()
        inputSize = c.inputSize; hiddenSize = c.hiddenSize
    }

    func lstmStep(x: [Float], h: [Float], c: [Float], batch: Int) -> (h: [Float], c: [Float]) {
        let hs = hiddenSize
        let gx = affine(x, batch: batch, inSize: inputSize, wIH, outSize: 4 * hs, bIH)
        let gh = affine(h, batch: batch, inSize: hs, wHH, outSize: 4 * hs, bHH)
        var newH = [Float](repeating: 0, count: batch * hs), newC = newH
        for b in 0..<batch {
            for j in 0..<hs {
                func g(_ k: Int) -> Float { gx[b * 4 * hs + k * hs + j] + gh[b * 4 * hs + k * hs + j] }
                let i = sigmoid(g(0)), f = sigmoid(g(1)), gg = Foundation.tanh(g(2)), o = sigmoid(g(3))
                let cNew = f * c[b * hs + j] + i * gg
                newC[b * hs + j] = cNew
                newH[b * hs + j] = o * Foundation.tanh(cNew)
            }
        }
        return (newH, newC)
    }

    func gruStep(x: [Float], h: [Float], batch: Int) -> [Float] {
        let hs = hiddenSize
        let gx = affine(x, batch: batch, inSize: inputSize, wIH, outSize: 3 * hs, bIH)
        let gh = affine(h, batch: batch, inSize: hs, wHH, outSize: 3 * hs, bHH)
        var newH = [Float](repeating: 0, count: batch * hs)
        for b in 0..<batch {
            for j in 0..<hs {
                func at(_ a: [Float], _ k: Int) -> Float { a[b * 3 * hs + k * hs + j] }
                let r = sigmoid(at(gx, 0) + at(gh, 0))
                let z = sigmoid(at(gx, 1) + at(gh, 1))
                let n = Foundation.tanh(at(gx, 2) + r * at(gh, 2))
                newH[b * hs + j] = (1 - z) * n + z * h[b * hs + j]
            }
        }
        return newH
    }
}

/// Multi-layer (optionally bidirectional) recurrent reference over a
/// batch-first [batch, seq, features] input with zero initial state.
/// Returns (output [batch, seq, dirs*H], final h [layers*dirs, batch, H], final c).
private func hostRecurrent(_ cells: [HostCell], lstm: Bool, x: [Float], batch: Int, seq: Int,
                           inputSize: Int, hidden: Int, layers: Int, dirs: Int)
    -> (out: [Float], h: [Float], c: [Float]) {
    var layerIn = x, features = inputSize
    var finalH: [Float] = [], finalC: [Float] = []
    for layer in 0..<layers {
        var outs = [[Float]](repeating: [], count: dirs)  // per direction: [seq][batch*H] flattened
        for d in 0..<dirs {
            let cell = cells[layer * dirs + d]
            var h = [Float](repeating: 0, count: batch * hidden), c = h
            var perT = [[Float]](repeating: [], count: seq)
            for t in (d == 0 ? Array(0..<seq) : Array((0..<seq).reversed())) {
                var xt = [Float](repeating: 0, count: batch * features)
                for b in 0..<batch {
                    for f in 0..<features { xt[b * features + f] = layerIn[(b * seq + t) * features + f] }
                }
                if lstm { (h, c) = cell.lstmStep(x: xt, h: h, c: c, batch: batch) }
                else { h = cell.gruStep(x: xt, h: h, batch: batch) }
                perT[t] = h
            }
            finalH += h; finalC += c
            outs[d] = perT.flatMap { $0 }
        }
        let outF = dirs * hidden
        var next = [Float](repeating: 0, count: batch * seq * outF)
        for b in 0..<batch { for t in 0..<seq { for d in 0..<dirs { for j in 0..<hidden {
            next[(b * seq + t) * outF + d * hidden + j] = outs[d][t * batch * hidden + b * hidden + j]
        } } } }
        layerIn = next; features = outF
    }
    return (layerIn, finalH, finalC)
}

private func fill(_ count: Int, _ scale: Float = 0.1, _ bias: Float = -0.3, mod: Int = 7) -> [Float] {
    (0..<count).map { Float($0 % mod) * scale + bias }
}

private func expectClose(_ actual: [Float], _ expected: [Float], tol: Float = 1e-4,
                         sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(actual.count == expected.count, "count \(actual.count) != \(expected.count)",
            sourceLocation: sourceLocation)
    guard actual.count == expected.count else { return }
    let maxDiff = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    #expect(maxDiff <= tol, "max abs diff \(maxDiff)", sourceLocation: sourceLocation)
}

@Suite("Recurrent layer numerics (materialized)", .serialized, .enabled(if: PluginAvailability.cpu, "CPU PJRT plugin not available"))
struct RecurrentNumericTests {

    // MARK: Cells

    @Test("LSTMCell step matches the PyTorch equations (batch 2, non-zero biases)")
    func lstmCellStep() {
        let (batch, inSize, hs) = (2, 3, 2)
        let cell = nn.LSTMCell(inputSize: inSize, hiddenSize: hs)
        cell.weightIH.value = Tensor<Float>(fill(4 * hs * inSize, 0.09, -0.3), shape: [4 * hs, inSize])
        cell.weightHH.value = Tensor<Float>(fill(4 * hs * hs, 0.11, -0.25, mod: 5), shape: [4 * hs, hs])
        cell.biasIH.value = Tensor<Float>(fill(4 * hs, 0.05, -0.2, mod: 8), shape: [4 * hs])
        cell.biasHH.value = Tensor<Float>(fill(4 * hs, -0.04, 0.1, mod: 3), shape: [4 * hs])

        let x = fill(batch * inSize, 0.3, -0.5)
        let h = [0.2, -0.1, 0.4, 0.3] as [Float]
        let c = [-0.3, 0.5, 0.1, -0.2] as [Float]
        let (hNew, cNew) = cell.forward(x: Tensor<Float>(x, shape: [batch, inSize]),
                                        hidden: Tensor<Float>(h, shape: [batch, hs]),
                                        cell: Tensor<Float>(c, shape: [batch, hs]))

        let ref = HostCell(lstm: cell).lstmStep(x: x, h: h, c: c, batch: batch)
        expectClose(hNew.scalars(), ref.h)
        expectClose(cNew.scalars(), ref.c)
    }

    @Test("GRUCell step matches the PyTorch equations (batch 2, non-zero biases)")
    func gruCellStep() {
        let (batch, inSize, hs) = (2, 3, 2)
        let cell = nn.GRUCell(inputSize: inSize, hiddenSize: hs)
        cell.weightIH.value = Tensor<Float>(fill(3 * hs * inSize, 0.09, -0.3), shape: [3 * hs, inSize])
        cell.weightHH.value = Tensor<Float>(fill(3 * hs * hs, 0.11, -0.25, mod: 5), shape: [3 * hs, hs])
        cell.biasIH.value = Tensor<Float>(fill(3 * hs, 0.05, -0.2, mod: 4), shape: [3 * hs])
        cell.biasHH.value = Tensor<Float>(fill(3 * hs, -0.04, 0.1, mod: 3), shape: [3 * hs])

        let x = fill(batch * inSize, 0.3, -0.5)
        let h = [0.2, -0.1, 0.4, 0.3] as [Float]
        let hNew = cell.forward(x: Tensor<Float>(x, shape: [batch, inSize]), hidden: Tensor<Float>(h, shape: [batch, hs]))

        expectClose(hNew.scalars(), HostCell(gru: cell).gruStep(x: x, h: h, batch: batch))
    }

    // MARK: Bidirectional LSTM / GRU

    @Test("bidirectional LSTM: shapes and values match a host reference", arguments: [1, 2])
    func bidirectionalLSTM(numLayers: Int) {
        let (batch, seq, inSize, hs) = (2, 4, 3, 2)
        let lstm = nn.LSTM(inputSize: inSize, hiddenSize: hs, numLayers: numLayers, bidirectional: true)
        let x = fill(batch * seq * inSize, 0.2, -0.6, mod: 9)
        let (out, (h, c)) = lstm(Tensor<Float>(x, shape: [batch, seq, inSize]))

        #expect(out.shape == [batch, seq, 2 * hs])
        #expect(h.shape == [numLayers * 2, batch, hs])
        #expect(c.shape == [numLayers * 2, batch, hs])

        let ref = hostRecurrent(lstm.cells.map { HostCell(lstm: $0) }, lstm: true, x: x, batch: batch,
                                seq: seq, inputSize: inSize, hidden: hs, layers: numLayers, dirs: 2)
        expectClose(out.scalars(), ref.out)
        expectClose(h.scalars(), ref.h)
        expectClose(c.scalars(), ref.c)
    }

    @Test("bidirectional GRU: shapes and values match a host reference", arguments: [1, 2])
    func bidirectionalGRU(numLayers: Int) {
        let (batch, seq, inSize, hs) = (2, 4, 3, 2)
        let gru = nn.GRU(inputSize: inSize, hiddenSize: hs, numLayers: numLayers, bidirectional: true)
        let x = fill(batch * seq * inSize, 0.2, -0.6, mod: 9)
        let (out, h) = gru(Tensor<Float>(x, shape: [batch, seq, inSize]))

        #expect(out.shape == [batch, seq, 2 * hs])
        #expect(h.shape == [numLayers * 2, batch, hs])

        let ref = hostRecurrent(gru.cells.map { HostCell(gru: $0) }, lstm: false, x: x, batch: batch,
                                seq: seq, inputSize: inSize, hidden: hs, layers: numLayers, dirs: 2)
        expectClose(out.scalars(), ref.out)
        expectClose(h.scalars(), ref.h)
    }

    @Test("the reverse direction sees the future: changing the last step changes output at t = 0")
    func reverseDirectionMatters() {
        let (batch, seq, inSize, hs) = (1, 3, 2, 3)
        let lstm = nn.LSTM(inputSize: inSize, hiddenSize: hs, bidirectional: true)
        var x = fill(batch * seq * inSize, 0.2, -0.3)
        let a = lstm(Tensor<Float>(x, shape: [batch, seq, inSize])).0.scalars()
        x[x.count - 1] += 1.0  // perturb only the last timestep
        let b = lstm(Tensor<Float>(x, shape: [batch, seq, inSize])).0.scalars()

        // t = 0 row: [forward(hs) | backward(hs)]
        let fwd0 = zip(a[0..<hs], b[0..<hs]).map { abs($0 - $1) }.max()!
        let bwd0 = zip(a[hs..<(2 * hs)], b[hs..<(2 * hs)]).map { abs($0 - $1) }.max()!
        #expect(fwd0 == 0)       // forward state at t=0 cannot see t=2
        #expect(bwd0 > 1e-4)     // backward state at t=0 has consumed t=2
    }
}
