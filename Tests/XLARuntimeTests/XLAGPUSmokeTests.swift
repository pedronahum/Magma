// Magma - XLA GPU Smoke Tests
// Minimal end-to-end checks that the CUDA PJRT plugin (e.g. JAX's
// xla_cuda_plugin.so, symlinked as pjrt_c_api_gpu_plugin.so) loads,
// compiles, and executes on the GPU.
//
// Gated on GPU availability so it is a no-op on machines without a GPU plugin.

import Testing
@testable import XLARuntime

// `.serialized` is important on unified-memory machines (e.g. NVIDIA GB10):
// the CUDA PJRT plugin reserves ~75-80% of the unified pool per client, so
// running these tests in parallel spins up several clients at once and OOMs
// the whole box. Keep them one-at-a-time.
@Suite("XLA GPU Smoke Tests", .serialized)
struct XLAGPUSmokeTests {

    /// Whether a GPU PJRT client can be created on this machine.
    static let gpuAvailable: Bool = {
        do {
            _ = try PJRTClient.create(backend: .gpu)
            return true
        } catch {
            print("GPU not available: \(error)")
            return false
        }
    }()

    @Test("GPU client creation")
    func gpuClientCreation() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")

        let client = try PJRTClient.create(backend: .gpu)
        #expect(client.backend == .gpu)
        #expect(!client.devices.isEmpty, "GPU client should have at least one device")
        #expect(!client.platformName.isEmpty, "Platform name should not be empty")
        print("GPU platform: \(client.platformName), devices: \(client.devices.count)")
        if let device = client.defaultDevice {
            print("GPU device kind: \(device.kind)")
        }
    }

    @Test("GPU buffer round trip")
    func gpuBufferRoundTrip() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")

        let client = try PJRTClient.create(backend: .gpu)
        let originalData: [Float] = [1.5, 2.5, 3.5, 4.5]

        let buffer = try client.createBuffer(originalData, shape: [2, 2], elementType: .float32)
        let retrievedData = try buffer.toFloatArray()

        #expect(retrievedData.count == originalData.count)
        for i in 0..<originalData.count {
            #expect(abs(retrievedData[i] - originalData[i]) < 1e-6)
        }
    }

    @Test("GPU simple add")
    func gpuSimpleAdd() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")

        let client = try PJRTClient.create(backend: .gpu)

        let mlir = """
        module @simple_add {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let x = try client.createBuffer([1.0, 2.0, 3.0, 4.0] as [Float], shape: [4], elementType: .float32)
        let y = try client.createBuffer([10.0, 20.0, 30.0, 40.0] as [Float], shape: [4], elementType: .float32)

        let outputs = try executable.execute([x, y])
        #expect(outputs.count == 1)

        let result = try outputs[0].toFloatArray()
        let expected: [Float] = [11.0, 22.0, 33.0, 44.0]

        #expect(result.count == expected.count)
        for i in 0..<expected.count {
            #expect(abs(result[i] - expected[i]) < 1e-5)
        }
    }

    @Test("GPU matmul")
    func gpuMatmul() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")

        let client = try PJRTClient.create(backend: .gpu)

        // [2x3] @ [3x2] = [2x2]
        let mlir = """
        module @matmul {
          func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
            %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let a = try client.createBuffer([1, 2, 3, 4, 5, 6] as [Float], shape: [2, 3], elementType: .float32)
        let b = try client.createBuffer([7, 8, 9, 10, 11, 12] as [Float], shape: [3, 2], elementType: .float32)

        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()

        // Row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // Row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        let expected: [Float] = [58, 64, 139, 154]
        #expect(result.count == 4)
        for i in 0..<expected.count {
            #expect(abs(result[i] - expected[i]) < 1e-4)
        }
    }

    /// A meaningfully-sized cuBLAS GEMM on a single client. Uses all-ones
    /// inputs so every output element is deterministically `n` (= inner dim),
    /// which is trivial to verify without recomputing the product on CPU.
    @Test("GPU large matmul (512x512)")
    func gpuLargeMatmul() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")

        let client = try PJRTClient.create(backend: .gpu)

        let n = 512
        let mlir = """
        module @big_matmul {
          func.func @main(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>) -> tensor<512x512xf32> {
            %0 = stablehlo.dot %arg0, %arg1 : (tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
            return %0 : tensor<512x512xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let ones = [Float](repeating: 1.0, count: n * n)
        let a = try client.createBuffer(ones, shape: [n, n], elementType: .float32)
        let b = try client.createBuffer(ones, shape: [n, n], elementType: .float32)

        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()

        // ones[n,n] @ ones[n,n] => every element == n
        #expect(result.count == n * n)
        #expect(abs(result[0] - Float(n)) < 1e-2)
        #expect(abs(result[n * n - 1] - Float(n)) < 1e-2)
        #expect(abs(result[n * n / 2] - Float(n)) < 1e-2)
    }
}
