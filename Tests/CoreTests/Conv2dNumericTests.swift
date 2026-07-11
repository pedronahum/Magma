// Magma - conv/pool numerical materialization tests
// Verifies stablehlo.convolution and stablehlo.reduce_window (max/avg pool)
// emission produce correct values on the backend.
//
// .serialized: materialization creates a GPU PJRT client (see XLAGPUSmokeTests).

import Testing
@testable import Magma
@testable import LazyTensor

@Suite("Conv/Pool Numeric Tests", .serialized)
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

    @Test("maxPool2d 2x2 stride 2 takes window maxima")
    func maxPool() {
        // Input NHWC [1,4,4,1]: 1..16 row-major.
        let input = Tensor<Float>(Array((1...16).map(Float.init)), shape: [1, 4, 4, 1])
        let pool = nn.MaxPool2d(kernelSize: 2, stride: 2)
        let out = pool(input).scalars()
        // 2x2 windows: [1,2,5,6]->6, [3,4,7,8]->8, [9,10,13,14]->14, [11,12,15,16]->16
        #expect(out == [6, 8, 14, 16])
    }

    @Test("avgPool2d 2x2 stride 2 averages windows")
    func avgPool() {
        let input = Tensor<Float>(Array((1...16).map(Float.init)), shape: [1, 4, 4, 1])
        let pool = nn.AvgPool2d(kernelSize: 2, stride: 2)
        let out = pool(input).scalars()
        // Means: (1+2+5+6)/4=3.5, (3+4+7+8)/4=5.5, (9+10+13+14)/4=11.5, (11+12+15+16)/4=13.5
        #expect(out == [3.5, 5.5, 11.5, 13.5])
    }
}
