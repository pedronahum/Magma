// Magma - Differentiable conv2d verification
// Forward parity with the existing conv primitive, plus numerical gradient checks
// of the hand-written VJP w.r.t. both input and kernel, across padding and stride.
// The VJP expresses both gradients as forward convolutions (dilation + window
// reversal); gradcheck is what proves those window parameters are correct.
//
// .serialized: materialization drives the PJRT backend (see Conv2dNumericTests).

import Testing
import _Differentiation
@testable import Magma

@Suite("Differentiable conv2d", .serialized)
struct ConvDifferentiableTests {

    // Deterministic, varied, mixed-sign fill (no RNG -> no gradcheck flakiness).
    private func fill(_ shape: [Int], _ scale: Float = 0.1, _ bias: Float = -0.3) -> Tensor<Float> {
        let n = shape.reduce(1, *)
        let v = (0..<n).map { Float($0 % 7) * scale + bias }
        return Tensor<Float>(v, shape: shape)
    }

    @Test("forward matches the conv primitive (2x2 ones kernel sums windows)")
    func forwardParity() {
        let input = Tensor<Float>([1, 2, 3, 4, 5, 6, 7, 8, 9], shape: [1, 3, 3, 1])
        let kernel = Tensor<Float>([1, 1, 1, 1], shape: [2, 2, 1, 1])
        let out = input.conv2d(kernel).scalars()
        #expect(out == [12, 16, 24, 28])
    }

    @Test("gradcheck wrt input and kernel — stride 1, no padding")
    func gradStride1Valid() {
        let x = fill([1, 3, 3, 2])
        let k = fill([2, 2, 2, 3], 0.07, -0.2)

        let gInput = gradcheck({ $0.conv2d(k).sum() }, input: x,
                               eps: 1e-2, atol: 1e-2, rtol: 1e-2)
        #expect(gInput.passed, "input grad max abs diff \(gInput.maxAbsDiff)")

        let gKernel = gradcheck({ x.conv2d($0).sum() }, input: k,
                                eps: 1e-2, atol: 1e-2, rtol: 1e-2)
        #expect(gKernel.passed, "kernel grad max abs diff \(gKernel.maxAbsDiff)")
    }

    @Test("gradcheck wrt input and kernel — stride 1, padding 1 (same)")
    func gradStride1Padded() {
        let x = fill([1, 3, 3, 1])
        let k = fill([3, 3, 1, 2], 0.05, -0.15)
        let pad = [[1, 1], [1, 1]]

        let gInput = gradcheck({ $0.conv2d(k, padding: pad).sum() }, input: x,
                               eps: 1e-2, atol: 1e-2, rtol: 1e-2)
        #expect(gInput.passed, "input grad max abs diff \(gInput.maxAbsDiff)")

        let gKernel = gradcheck({ x.conv2d($0, padding: pad).sum() }, input: k,
                                eps: 1e-2, atol: 1e-2, rtol: 1e-2)
        #expect(gKernel.passed, "kernel grad max abs diff \(gKernel.maxAbsDiff)")
    }

    @Test("gradcheck wrt input and kernel — stride 2")
    func gradStride2() {
        let x = fill([1, 4, 4, 1])
        let k = fill([2, 2, 1, 1], 0.09, -0.25)
        let stride = [2, 2]

        let gInput = gradcheck({ $0.conv2d(k, strides: stride).sum() }, input: x,
                               eps: 1e-2, atol: 1e-2, rtol: 1e-2)
        #expect(gInput.passed, "input grad max abs diff \(gInput.maxAbsDiff)")

        let gKernel = gradcheck({ x.conv2d($0, strides: stride).sum() }, input: k,
                                eps: 1e-2, atol: 1e-2, rtol: 1e-2)
        #expect(gKernel.passed, "kernel grad max abs diff \(gKernel.maxAbsDiff)")
    }
}
