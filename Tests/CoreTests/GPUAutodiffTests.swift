// Magma - GPU autodiff tests
// End-to-end check that Swift-native automatic differentiation executes on the
// GPU: both the forward pass and the reverse pass run through the CUDA PJRT
// plugin, and the materialized gradient values are correct.
//
// This suite only runs in a *GPU-only* process — one where the CPU plugin is
// absent from MAGMA_XLA_PATH. In that case the high-level default device (CPU)
// resolves, via `resolveExecutionBackend`, to the GPU plugin, so the whole
// autodiff graph lands on the GPU without the one-plugin-per-process clash that
// a mixed CPU/GPU run would hit. In a normal (CPU-plugin-present) run every test
// here is skipped.
//
// To exercise it:
//   MAGMA_XLA_PATH=<dir with only pjrt_c_api_gpu_plugin.so> \
//     swift test --no-parallel --filter GPUAutodiffTests

import Foundation
import Testing
import _Differentiation
@testable import Magma
@testable import LazyTensor
@testable import XLARuntime

/// True only when the CPU plugin is unavailable but the GPU plugin is, i.e. the
/// default device will fall back to the GPU. Used to *skip* (not fail) this suite
/// in ordinary CPU runs — it only runs in a GPU-only process.
private func isGPUOnlyMode() -> Bool {
    !Backend.cpu.isAvailable && Backend.gpu.isAvailable
}

@Suite("GPU Autodiff Tests", .serialized, .enabled(if: isGPUOnlyMode()))
struct GPUAutodiffTests {

    @Test("gradient of sum(x*x) is 2x, executed on GPU")
    func squareGradient() throws {
        let x = Tensor<Float>([1, 2, 3, 4], shape: [4])
        let grad = gradient(at: x) { t in (t * t).sum() }

        // d/dx sum(x*x) = 2x
        let g = grad.scalars()   // forces compile + execute on the GPU
        #expect(g.count == 4)
        for (i, expected) in [Float(2), 4, 6, 8].enumerated() where i < g.count {
            #expect(abs(g[i] - expected) < 1e-4, "grad[\(i)] = \(g[i]), want \(expected)")
        }
    }

    @Test("gradient of a small linear layer, executed on GPU")
    func linearGradient() throws {
        // f(w) = sum(x @ w); df/dw = column-sums of x broadcast across w's columns.
        // x is [2x3], w is [3x2] -> each dw[j,k] = sum_i x[i,j].
        let x = Tensor<Float>([1, 2, 3, 4, 5, 6], shape: [2, 3])
        let w = Tensor<Float>([0, 0, 0, 0, 0, 0], shape: [3, 2])
        let grad = gradient(at: w) { weights in x.matmul(weights).sum() }

        let g = grad.scalars()
        #expect(g.count == 6)
        // column sums of x: col0 = 1+4 = 5, col1 = 2+5 = 7, col2 = 3+6 = 9
        // dw rows are [colSum, colSum]: [5,5, 7,7, 9,9]
        let expected: [Float] = [5, 5, 7, 7, 9, 9]
        for (i, e) in expected.enumerated() where i < g.count {
            #expect(abs(g[i] - e) < 1e-4, "grad[\(i)] = \(g[i]), want \(e)")
        }
    }
}
