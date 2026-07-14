// Magma - Differentiable 2D convolution
//
// A `@differentiable` `conv2d` on `Tensor`, so value-semantic conv layers train
// through Swift autodiff (see ValueLayers `Conv2d`). The forward reuses the same
// `stablehlo.convolution` primitive as `nn.Conv2d`; the novelty is the VJP.
//
// Both gradients of a convolution are themselves convolutions, so the pullback is
// written entirely in terms of the *forward* conv primitive over transposed /
// dilated / reversed operands — no separate transposed-conv or dim-number
// machinery. This keeps the backward path on the exact op the forward is tested
// against (Conv2dNumericTests) and is verified end to end by gradient checking
// (ConvDifferentiableTests).
//
// Layout: input NHWC `[N, H, W, Cin]`, kernel HWIO `[kH, kW, Cin, Cout]`, output
// NHWC `[N, oH, oW, Cout]`.

import _Differentiation
import LazyTensor
import StableHLO

extension Tensor {

    /// Low-level convolution exposing the full StableHLO window config (base/window
    /// dilation and spatial kernel reversal). `self` is the NHWC input. NOT
    /// differentiable — the differentiable `conv2d` wraps it and its VJP calls this
    /// again with backprop-specific window parameters.
    func _conv2dRaw(
        kernel: Tensor,
        strides: [Int],
        padding: [[Int]],
        lhsDilation: [Int]? = nil,
        rhsDilation: [Int]? = nil,
        reverseKernel: [Bool]? = nil,
        outputShape: [Int]
    ) -> Tensor {
        var attributes: [String: Any] = [
            "strides": strides,
            "padding": padding,
        ]
        if let lhsDilation { attributes["lhsDilation"] = lhsDilation }
        if let rhsDilation { attributes["rhsDilation"] = rhsDilation }
        if let reverseKernel { attributes["reverseKernel"] = reverseKernel }

        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(id: id, shape: outputShape, dtype: dtype, device: device)
        handle.irNode = .operation(op: .conv2d, inputs: [self.handle, kernel.handle], attributes: attributes)
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// NHWC output shape of a forward conv (no dilation).
    static func _conv2dOutputShape(
        input: [Int], kernel: [Int], strides: [Int], padding: [[Int]]
    ) -> [Int] {
        let n = input[0], h = input[1], w = input[2]
        let kh = kernel[0], kw = kernel[1], cout = kernel[3]
        let oh = (h + padding[0][0] + padding[0][1] - kh) / strides[0] + 1
        let ow = (w + padding[1][0] + padding[1][1] - kw) / strides[1] + 1
        return [n, oh, ow, cout]
    }

    /// 2D convolution: NHWC input (`self`) ⋆ HWIO `kernel` → NHWC output.
    /// Differentiable with respect to both the input and the kernel.
    ///
    /// - Parameters:
    ///   - kernel: HWIO kernel `[kH, kW, Cin, Cout]`.
    ///   - strides: `[strideH, strideW]`.
    ///   - padding: `[[padTop, padBottom], [padLeft, padRight]]`.
    /// Differentiable w.r.t. input and kernel via `vjpConv2d` (registered below).
    public func conv2d(
        _ kernel: Tensor,
        strides: [Int] = [1, 1],
        padding: [[Int]] = [[0, 0], [0, 0]]
    ) -> Tensor {
        _conv2dRaw(
            kernel: kernel,
            strides: strides,
            padding: padding,
            outputShape: Tensor._conv2dOutputShape(
                input: shape, kernel: kernel.shape, strides: strides, padding: padding))
    }
}

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {

    /// VJP for `conv2d`. Both gradients are expressed as forward convolutions:
    ///
    /// - Input gradient (`dX`) is a *transposed* convolution — the incoming
    ///   gradient `dY` dilated by the stride (`lhs_dilate`), convolved with the
    ///   kernel spatially reversed (`reverse`) and its in/out channels swapped.
    /// - Filter gradient (`dW`) is a convolution of the input by `dY`, with the
    ///   input channel acting as the batch dim and the batch acting as the feature
    ///   dim (achieved by transposing the operands), `dY` dilated by the stride
    ///   (`rhs_dilate`).
    ///
    /// Padding on each backward conv is chosen so the result matches `X`/`W`
    /// exactly, including the `stride`-remainder terms for non-unit strides.
    @derivative(of: conv2d)
    public func vjpConv2d(
        _ kernel: Tensor,
        strides: [Int],
        padding: [[Int]]
    ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
        let value = self.conv2d(kernel, strides: strides, padding: padding)

        let xShape = self.shape          // [N, H, W, Cin]
        let kShape = kernel.shape        // [kH, kW, Cin, Cout]
        let sH = strides[0], sW = strides[1]
        let pTop = padding[0][0], pBottom = padding[0][1]
        let pLeft = padding[1][0], pRight = padding[1][1]
        let (h, w, cin) = (xShape[1], xShape[2], xShape[3])
        let (kH, kW, cout) = (kShape[0], kShape[1], kShape[3])

        // Stride remainders: how much the strided window overshoots the input.
        let remH = (h + pTop + pBottom - kH) % sH
        let remW = (w + pLeft + pRight - kW) % sW

        let input = self
        return (value, { dY in
            // dY: [N, oH, oW, Cout]

            // --- Gradient w.r.t. input: transposed convolution ---
            // Swap the kernel's in/out channels ([kH,kW,Cin,Cout] -> [kH,kW,Cout,Cin])
            // and let the window `reverse` flip its spatial taps.
            let kSwapped = kernel.transpose(2, 3)
            let dataPad = [
                [kH - 1 - pTop,  kH - 1 - pBottom + remH],
                [kW - 1 - pLeft, kW - 1 - pRight + remW],
            ]
            let dX = dY._conv2dRaw(
                kernel: kSwapped,
                strides: [1, 1],
                padding: dataPad,
                lhsDilation: [sH, sW],
                reverseKernel: [true, true],
                outputShape: xShape)

            // --- Gradient w.r.t. kernel ---
            // input  -> [Cin, H, W, N]     (Cin becomes batch, N becomes feature)
            // kernel -> [oH, oW, N, Cout]   (dY, dilated by the stride)
            // output -> [Cin, kH, kW, Cout] -> permute back to [kH, kW, Cin, Cout]
            let xPerm = input.transpose(0, 3)                // [Cin, H, W, N]
            let dYPerm = dY.transpose(0, 1).transpose(1, 2)  // [oH, oW, N, Cout]
            let filterPad = [
                [pTop,  pBottom - remH],
                [pLeft, pRight - remW],
            ]
            let dWPerm = xPerm._conv2dRaw(
                kernel: dYPerm,
                strides: [1, 1],
                padding: filterPad,
                rhsDilation: [sH, sW],
                outputShape: [cin, kH, kW, cout])
            let dW = dWPerm.transpose(0, 1).transpose(1, 2)  // [kH, kW, Cin, Cout]

            return (dX, dW)
        })
    }
}
