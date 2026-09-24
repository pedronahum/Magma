// Magma - Value-level tests for nn layers that used to lower to ops the StableHLO
// emitter could not handle (Conv1d, ConvTranspose2d, GroupNorm, InstanceNorm2d).
// The older PortedLayersTests only checked shapes, so materializing any of these
// crashed the process while the suite stayed green. Every test here materializes
// the output and compares it against a straightforward host-side reference.
//
// .serialized: materialization drives the PJRT backend.

import Testing
import _Differentiation
@testable import Magma

/// Deterministic, varied, mixed-sign fill (no RNG).
private func fill(_ count: Int, _ scale: Float = 0.1, _ bias: Float = -0.3, mod: Int = 7) -> [Float] {
    (0..<count).map { Float($0 % mod) * scale + bias }
}

private func expectClose(_ actual: [Float], _ expected: [Float], tol: Float = 1e-4,
                         sourceLocation: SourceLocation = #_sourceLocation) {
    #expect(actual.count == expected.count, "count \(actual.count) != \(expected.count)",
            sourceLocation: sourceLocation)
    guard actual.count == expected.count else { return }
    let maxDiff = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    #expect(maxDiff <= tol, "max abs diff \(maxDiff)\nactual:   \(actual)\nexpected: \(expected)",
            sourceLocation: sourceLocation)
}

// MARK: - Host references

/// NLC conv1d; weight [k, Cin, Cout].
private func refConv1d(_ x: [Float], n: Int, l: Int, cin: Int,
                       _ w: [Float], k: Int, cout: Int, b: [Float],
                       stride: Int, pad: Int) -> [Float] {
    let ol = (l + 2 * pad - k) / stride + 1
    var out = [Float](repeating: 0, count: n * ol * cout)
    for bi in 0..<n {
        for o in 0..<ol {
            for co in 0..<cout {
                var acc = b[co]
                for u in 0..<k {
                    let pos = o * stride - pad + u
                    guard pos >= 0 && pos < l else { continue }
                    for ci in 0..<cin {
                        acc += x[(bi * l + pos) * cin + ci] * w[(u * cin + ci) * cout + co]
                    }
                }
                out[(bi * ol + o) * cout + co] = acc
            }
        }
    }
    return out
}

/// NHWC transposed conv (PyTorch scatter semantics); weight [kH, kW, Cout, Cin].
private func refConvTranspose2d(_ x: [Float], n: Int, h: Int, w: Int, cin: Int,
                                _ wt: [Float], kh: Int, kw: Int, cout: Int, b: [Float],
                                stride: (Int, Int), pad: (Int, Int), outPad: (Int, Int)) -> [Float] {
    let oh = (h - 1) * stride.0 - 2 * pad.0 + kh + outPad.0
    let ow = (w - 1) * stride.1 - 2 * pad.1 + kw + outPad.1
    var out = [Float](repeating: 0, count: n * oh * ow * cout)
    for bi in 0..<n {
        for i in 0..<oh { for j in 0..<ow { for co in 0..<cout {
            out[((bi * oh + i) * ow + j) * cout + co] = b[co]
        } } }
        for i in 0..<h { for j in 0..<w { for u in 0..<kh { for v in 0..<kw {
            let oi = i * stride.0 - pad.0 + u
            let oj = j * stride.1 - pad.1 + v
            guard oi >= 0 && oi < oh && oj >= 0 && oj < ow else { continue }
            for ci in 0..<cin { for co in 0..<cout {
                out[((bi * oh + oi) * ow + oj) * cout + co] +=
                    x[((bi * h + i) * w + j) * cin + ci] * wt[((u * kw + v) * cout + co) * cin + ci]
            } }
        } } } }
    }
    return out
}

/// NHWC group norm (biased variance); weight/bias per channel.
private func refGroupNorm(_ x: [Float], n: Int, h: Int, w: Int, c: Int, groups: Int,
                          gamma: [Float], beta: [Float], eps: Float) -> [Float] {
    let cpg = c / groups
    var out = x
    for bi in 0..<n {
        for g in 0..<groups {
            var idx: [Int] = []
            for i in 0..<h { for j in 0..<w { for cc in 0..<cpg {
                idx.append(((bi * h + i) * w + j) * c + g * cpg + cc)
            } } }
            let mean = idx.map { x[$0] }.reduce(0, +) / Float(idx.count)
            let variance = idx.map { (x[$0] - mean) * (x[$0] - mean) }.reduce(0, +) / Float(idx.count)
            let denom = (variance + eps).squareRoot()
            for k in idx {
                let ch = k % c
                out[k] = (x[k] - mean) / denom * gamma[ch] + beta[ch]
            }
        }
    }
    return out
}

@Suite("nn layer numerics (materialized)", .serialized, .enabled(if: PluginAvailability.cpu, "CPU PJRT plugin not available"))
struct NNLayerNumericTests {

    // MARK: Conv1d

    @Test("Conv1d matches a host reference (stride 1, no padding)")
    func conv1dValid() {
        let conv = nn.Conv1d(inChannels: 2, outChannels: 3, kernelSize: 3)
        let w = fill(3 * 2 * 3, 0.07, -0.2)
        let b: [Float] = [0.1, -0.2, 0.3]
        conv.weight.value = Tensor<Float>(w, shape: [3, 2, 3])
        conv.bias.value = Tensor<Float>(b, shape: [3])

        let x = fill(2 * 5 * 2)
        let y = conv(Tensor<Float>(x, shape: [2, 5, 2]))
        #expect(y.shape == [2, 3, 3])
        expectClose(y.scalars(),
                    refConv1d(x, n: 2, l: 5, cin: 2, w, k: 3, cout: 3, b: b, stride: 1, pad: 0))
    }

    @Test("Conv1d matches a host reference (stride 2, padding 1)")
    func conv1dStridedPadded() {
        let conv = nn.Conv1d(inChannels: 1, outChannels: 2, kernelSize: 3, stride: 2, padding: 1)
        let w = fill(3 * 1 * 2, 0.11, -0.25)
        let b: [Float] = [0.5, -0.5]
        conv.weight.value = Tensor<Float>(w, shape: [3, 1, 2])
        conv.bias.value = Tensor<Float>(b, shape: [2])

        let x = fill(1 * 7 * 1, 0.2, -0.6)
        let y = conv(Tensor<Float>(x, shape: [1, 7, 1]))
        #expect(y.shape == [1, 4, 2])
        expectClose(y.scalars(),
                    refConv1d(x, n: 1, l: 7, cin: 1, w, k: 3, cout: 2, b: b, stride: 2, pad: 1))
    }

    @Test("Conv1d forward is differentiable w.r.t. its input (gradcheck)")
    func conv1dGradcheck() {
        // The layer lowers to reshape + differentiable conv2d; check the same lowering.
        let k = Tensor<Float>(fill(3 * 2 * 2, 0.07, -0.2), shape: [3, 2, 2])
        let x = Tensor<Float>(fill(1 * 6 * 2), shape: [1, 6, 2])
        let result = gradcheck({ (t: Tensor<Float>) -> Tensor<Float> in
            t.reshape([1, 1, 6, 2])
                .conv2d(k.reshape([1, 3, 2, 2]), strides: [1, 2], padding: [[0, 0], [1, 1]])
                .sum()
        }, input: x, eps: 1e-2, atol: 1e-2, rtol: 1e-2)
        #expect(result.passed, "max abs diff \(result.maxAbsDiff)")
    }

    // MARK: ConvTranspose2d

    @Test("ConvTranspose2d stride 2 with a ones kernel expands each pixel into a block")
    func convTransposeBlocks() {
        let convT = nn.ConvTranspose2d(inChannels: 1, outChannels: 1, kernelSize: 2, stride: 2, bias: false)
        convT.weight.value = Tensor<Float>([1, 1, 1, 1], shape: [2, 2, 1, 1])
        let y = convT(Tensor<Float>([1, 2, 3, 4], shape: [1, 2, 2, 1]))
        #expect(y.shape == [1, 4, 4, 1])
        expectClose(y.scalars(), [1, 1, 2, 2,
                                  1, 1, 2, 2,
                                  3, 3, 4, 4,
                                  3, 3, 4, 4])
    }

    @Test("ConvTranspose2d matches a host reference", arguments: [
        // (stride, padding, outputPadding, kernel)
        (1, 0, 0, 3),
        (2, 1, 1, 3),
        (2, 0, 0, 2),
        (3, 1, 2, 4),
    ])
    func convTransposeReference(_ cfg: (Int, Int, Int, Int)) {
        let (s, p, op, k) = cfg
        let (n, h, w, cin, cout) = (2, 3, 2, 2, 3)
        let convT = nn.ConvTranspose2d(inChannels: cin, outChannels: cout, kernelSize: k,
                                       stride: s, padding: p, outputPadding: op)
        let wt = fill(k * k * cout * cin, 0.05, -0.15, mod: 11)
        let b: [Float] = [0.1, 0.0, -0.1]
        convT.weight.value = Tensor<Float>(wt, shape: [k, k, cout, cin])
        convT.bias.value = Tensor<Float>(b, shape: [cout])

        let x = fill(n * h * w * cin, 0.1, -0.35)
        let y = convT(Tensor<Float>(x, shape: [n, h, w, cin]))
        let expected = refConvTranspose2d(x, n: n, h: h, w: w, cin: cin, wt, kh: k, kw: k, cout: cout,
                                          b: b, stride: (s, s), pad: (p, p), outPad: (op, op))
        let oh = (h - 1) * s - 2 * p + k + op
        let ow = (w - 1) * s - 2 * p + k + op
        #expect(y.shape == [n, oh, ow, cout])
        expectClose(y.scalars(), expected)
    }

    @Test("convTranspose2d gradcheck w.r.t. input and kernel", arguments: [
        // (stride, padding, outputPadding)
        (1, 0, 0),
        (2, 1, 1),
        (2, 0, 1),
    ])
    func convTransposeGradcheck(_ cfg: (Int, Int, Int)) {
        let (s, p, op) = cfg
        let x = Tensor<Float>(fill(1 * 3 * 3 * 2), shape: [1, 3, 3, 2])
        let k = Tensor<Float>(fill(3 * 3 * 2 * 2, 0.07, -0.2), shape: [3, 3, 2, 2])
        let strides = [s, s], padding = [[p, p], [p, p]], outPad = [op, op]

        let gInput = gradcheck({ $0.convTranspose2d(k, strides: strides, padding: padding,
                                                    outputPadding: outPad).sum() },
                               input: x, eps: 1e-2, atol: 1e-2, rtol: 1e-2)
        #expect(gInput.passed, "input grad max abs diff \(gInput.maxAbsDiff)")

        // A weighted sum so the kernel gradient is not symmetric in the taps.
        let outShape = x.convTranspose2d(k, strides: strides, padding: padding, outputPadding: outPad).shape
        let weights = Tensor<Float>(fill(outShape.reduce(1, *), 0.13, -0.4, mod: 5), shape: outShape)
        let gKernel = gradcheck({ (x.convTranspose2d($0, strides: strides, padding: padding,
                                                     outputPadding: outPad) * weights).sum() },
                                input: k, eps: 1e-2, atol: 1e-2, rtol: 1e-2)
        #expect(gKernel.passed, "kernel grad max abs diff \(gKernel.maxAbsDiff)")
    }

    // MARK: GroupNorm / InstanceNorm2d

    @Test("GroupNorm matches a host reference (with affine)")
    func groupNormReference() {
        let (n, h, w, c, g) = (2, 2, 3, 4, 2)
        let gn = nn.GroupNorm(numGroups: g, numChannels: c)
        let gamma: [Float] = [1.0, 0.5, -1.0, 2.0]
        let beta: [Float] = [0.0, 0.1, 0.2, -0.3]
        gn.weight.value = Tensor<Float>(gamma, shape: [c])
        gn.bias.value = Tensor<Float>(beta, shape: [c])

        let x = fill(n * h * w * c, 0.3, -1.0, mod: 13)
        let y = gn(Tensor<Float>(x, shape: [n, h, w, c]))
        #expect(y.shape == [n, h, w, c])
        expectClose(y.scalars(),
                    refGroupNorm(x, n: n, h: h, w: w, c: c, groups: g, gamma: gamma, beta: beta, eps: 1e-5),
                    tol: 1e-3)
    }

    @Test("InstanceNorm2d normalizes each channel of each sample independently")
    func instanceNormReference() {
        let (n, h, w, c) = (2, 3, 2, 3)
        let inorm = nn.InstanceNorm2d(numFeatures: c)
        let x = fill(n * h * w * c, 0.25, -0.8, mod: 11)
        let y = inorm(Tensor<Float>(x, shape: [n, h, w, c]))
        let ones = [Float](repeating: 1, count: c), zeros = [Float](repeating: 0, count: c)
        // InstanceNorm == GroupNorm with one channel per group.
        expectClose(y.scalars(),
                    refGroupNorm(x, n: n, h: h, w: w, c: c, groups: c, gamma: ones, beta: zeros, eps: 1e-5),
                    tol: 1e-3)
    }

    @Test("InstanceNorm2d with affine applies per-channel scale and shift")
    func instanceNormAffine() {
        let (n, h, w, c) = (1, 2, 2, 2)
        let inorm = nn.InstanceNorm2d(numFeatures: c, affine: true)
        let gamma: [Float] = [2.0, -0.5]
        let beta: [Float] = [1.0, 0.25]
        inorm.weight.value = Tensor<Float>(gamma, shape: [c])
        inorm.bias.value = Tensor<Float>(beta, shape: [c])
        let x: [Float] = [1, 10, 2, 20, 3, 30, 4, 40]
        let y = inorm(Tensor<Float>(x, shape: [n, h, w, c]))
        expectClose(y.scalars(),
                    refGroupNorm(x, n: n, h: h, w: w, c: c, groups: c, gamma: gamma, beta: beta, eps: 1e-5),
                    tol: 1e-3)
    }
}
