// Magma - GPU numerics tests
// Broader end-to-end numeric checks on the CUDA PJRT plugin, beyond the
// elementwise/GEMM smoke tests: reductions, transpose, broadcast, transcendental
// ops, a full softmax composition, and a 2-D convolution. Each result is checked
// against hand-computed expected values.
//
// A single shared GPU client is used for the whole suite: on unified-memory
// boxes (e.g. NVIDIA GB10) each CUDA client reserves ~75-80% of the pool, so we
// hold exactly one reservation for the suite's lifetime instead of spinning up a
// fresh client per test. `.serialized` keeps tests one-at-a-time regardless.

import Foundation
import Testing
@testable import XLARuntime

@Suite("GPU Numerics Tests", .serialized)
struct GPUNumericsTests {
    /// One GPU client shared by every test in this suite. nil when no GPU plugin
    /// is present, in which case each test skips via `#require`.
    static let client: PJRTClient? = try? PJRTClient.create(backend: .gpu)

    // MARK: helpers

    /// Compile `mlir`, upload each (data, shape) input as an f32 buffer, execute,
    /// and return the first output as a flat Float array.
    private func runF32(
        _ client: PJRTClient,
        _ mlir: String,
        _ inputs: [(data: [Float], shape: [Int])]
    ) throws -> [Float] {
        let exe = try client.compile(mlir)
        let buffers = try inputs.map {
            try client.createBuffer($0.data, shape: $0.shape, elementType: .float32)
        }
        let outputs = try exe.execute(buffers)
        return try outputs[0].toFloatArray()
    }

    /// Elementwise approximate-equality over two Float arrays.
    private func expectClose(_ got: [Float], _ want: [Float], tol: Float = 1e-4) {
        #expect(got.count == want.count)
        guard got.count == want.count else { return }
        for i in 0..<want.count {
            #expect(abs(got[i] - want[i]) < tol, "index \(i): got \(got[i]), want \(want[i])")
        }
    }

    // MARK: reductions

    @Test("reduce sum over an axis")
    func reduceSum() throws {
        let client = try #require(Self.client, "GPU PJRT plugin not available")
        let mlir = """
        module @reduce_sum {
          func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2xf32> {
            %c = stablehlo.constant dense<0.0> : tensor<f32>
            %0 = stablehlo.reduce(%arg0 init: %c) applies stablehlo.add across dimensions = [1]
              : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            return %0 : tensor<2xf32>
          }
        }
        """
        // [[1,2,3],[4,5,6]] summed over axis 1 -> [6, 15]
        let r = try runF32(client, mlir, [(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])])
        expectClose(r, [6, 15])
    }

    // MARK: shape ops

    @Test("transpose")
    func transpose() throws {
        let client = try #require(Self.client, "GPU PJRT plugin not available")
        let mlir = """
        module @transpose {
          func.func @main(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
            %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
            return %0 : tensor<3x2xf32>
          }
        }
        """
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let r = try runF32(client, mlir, [(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])])
        expectClose(r, [1, 4, 2, 5, 3, 6])
    }

    @Test("broadcast_in_dim")
    func broadcast() throws {
        let client = try #require(Self.client, "GPU PJRT plugin not available")
        let mlir = """
        module @bcast {
          func.func @main(%arg0: tensor<3xf32>) -> tensor<2x3xf32> {
            %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
            return %0 : tensor<2x3xf32>
          }
        }
        """
        // [10,20,30] broadcast to two rows
        let r = try runF32(client, mlir, [(data: [10, 20, 30], shape: [3])])
        expectClose(r, [10, 20, 30, 10, 20, 30])
    }

    // MARK: transcendental

    @Test("tanh")
    func tanhOp() throws {
        let client = try #require(Self.client, "GPU PJRT plugin not available")
        let mlir = """
        module @tanhm {
          func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
            %0 = stablehlo.tanh %arg0 : tensor<3xf32>
            return %0 : tensor<3xf32>
          }
        }
        """
        let r = try runF32(client, mlir, [(data: [0, 1, -1], shape: [3])])
        expectClose(r, [0, 0.7615942, -0.7615942])
    }

    // MARK: softmax composition (reduce + broadcast + exp + divide)

    @Test("softmax over last axis")
    func softmax() throws {
        let client = try #require(Self.client, "GPU PJRT plugin not available")
        let mlir = """
        module @softmax {
          func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
            %ninf = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %max = stablehlo.reduce(%arg0 init: %ninf) applies stablehlo.maximum across dimensions = [1]
              : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            %maxb = stablehlo.broadcast_in_dim %max, dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>
            %shift = stablehlo.subtract %arg0, %maxb : tensor<2x3xf32>
            %exp = stablehlo.exponential %shift : tensor<2x3xf32>
            %zero = stablehlo.constant dense<0.0> : tensor<f32>
            %sum = stablehlo.reduce(%exp init: %zero) applies stablehlo.add across dimensions = [1]
              : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
            %sumb = stablehlo.broadcast_in_dim %sum, dims = [0] : (tensor<2xf32>) -> tensor<2x3xf32>
            %out = stablehlo.divide %exp, %sumb : tensor<2x3xf32>
            return %out : tensor<2x3xf32>
          }
        }
        """
        // Row 0 uniform -> [1/3,1/3,1/3]; row 1 = softmax([0,1,2]).
        let r = try runF32(client, mlir, [(data: [0, 0, 0, 0, 1, 2], shape: [2, 3])])
        let e0: Float = 1.0 / 3.0
        // softmax([0,1,2]): exp([-2,-1,0]) / sum
        let denom: Float = exp(-2) + exp(-1) + 1
        expectClose(r, [e0, e0, e0, exp(-2) / denom, exp(-1) / denom, 1 / denom])
        // Each row must sum to 1.
        #expect(abs((r[0] + r[1] + r[2]) - 1) < 1e-4)
        #expect(abs((r[3] + r[4] + r[5]) - 1) < 1e-4)
    }

    // MARK: convolution

    @Test("2-D convolution")
    func convolution() throws {
        let client = try #require(Self.client, "GPU PJRT plugin not available")
        // 1x1x4x4 input, 1x1x2x2 kernel, VALID, stride 1 -> 1x1x3x3.
        let mlir = """
        module @conv {
          func.func @main(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x1x2x2xf32>) -> tensor<1x1x3x3xf32> {
            %0 = stablehlo.convolution(%arg0, %arg1)
              dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
              window = {stride = [1, 1]}
              {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
              : (tensor<1x1x4x4xf32>, tensor<1x1x2x2xf32>) -> tensor<1x1x3x3xf32>
            return %0 : tensor<1x1x3x3xf32>
          }
        }
        """
        let ones16 = [Float](repeating: 1, count: 16)
        let ones4 = [Float](repeating: 1, count: 4)
        // Every 2x2 window of ones summed with an all-ones kernel -> 4.
        let r = try runF32(client, mlir, [
            (data: ones16, shape: [1, 1, 4, 4]),
            (data: ones4, shape: [1, 1, 2, 2]),
        ])
        expectClose(r, [Float](repeating: 4, count: 9))
    }
}
