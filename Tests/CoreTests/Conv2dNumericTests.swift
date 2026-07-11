// Magma - conv2d numerical materialization test (scratch verification)
// Verifies stablehlo.convolution emission produces correct values on the backend.
//
// .serialized: materialization creates a GPU PJRT client (see XLAGPUSmokeTests).

import Testing
@testable import Magma
@testable import LazyTensor

@Suite("Conv2d Numeric Tests", .serialized)
struct Conv2dNumericTests {

    @Test("3x3 input, 2x2 all-ones kernel, valid conv sums windows")
    func sumWindows() {
        // Input NHWC [1,3,3,1]: 1..9 row-major.
        let input = Tensor<Float>([1, 2, 3, 4, 5, 6, 7, 8, 9], shape: [1, 3, 3, 1])
        var conv = nn.Conv2d(inChannels: 1, outChannels: 1, kernelSize: 2, bias: false)
        // Kernel HWIO [2,2,1,1] all ones -> each output is the sum of a 2x2 window.
        conv.weight.value = Tensor<Float>([1, 1, 1, 1], shape: [2, 2, 1, 1])

        let out = conv(input).scalars()
        // out[0,0]=1+2+4+5=12, out[0,1]=2+3+5+6=16, out[1,0]=4+5+7+8=24, out[1,1]=5+6+8+9=28
        #expect(out == [12, 16, 24, 28])
    }
}
