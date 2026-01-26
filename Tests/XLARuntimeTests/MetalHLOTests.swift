// MetalHLOTests.swift
// Tests for MetalHLO integration

import Testing
@testable import XLARuntime
@testable import StableHLO
@testable import LazyTensor

#if os(macOS) && canImport(MetalHLO)
import MetalHLO

@Suite("MetalHLO Integration Tests")
struct MetalHLOTests {

    // MARK: - Backend Detection Tests

    @Test("Metal backend is available on macOS")
    func metalBackendIsAvailable() {
        // Metal should be available on macOS with Apple Silicon or discrete GPU
        #expect(MetalBackend.isAvailable, "Metal backend should be available on macOS")
    }

    @Test("Metal backend has a device name")
    func metalBackendDeviceName() {
        let deviceName = MetalBackend.deviceName
        #expect(deviceName != nil, "Should have a Metal device name")
        print("Metal device: \(deviceName ?? "unknown")")
    }

    @Test("Backend enum includes Metal")
    func backendEnumIncludesMetal() {
        let availableBackends = Backend.availableBackends
        #expect(availableBackends.contains(.metal), "Available backends should include .metal on macOS")
    }

    @Test("Best available prefers Metal over CPU")
    func bestAvailablePrefersMetal() {
        // On macOS, if Metal is available, it should be preferred over CPU
        if Backend.metal.isAvailable && !Backend.tpu.isAvailable {
            #expect(Backend.bestAvailable == .metal, "Best available should be .metal on macOS when TPU not present")
        }
    }

    // MARK: - Client Tests

    @Test("Create MetalHLO client")
    func createMetalHLOClient() throws {
        let client = try MetalHLOClient.create()
        #expect(!client.deviceName.isEmpty, "Client should have a device name")
        print("Created MetalHLO client for device: \(client.deviceName)")
    }

    // MARK: - Compilation Tests

    @Test("Compile simple addition")
    func compileSimpleAddition() throws {
        let client = try MetalHLOClient.create()

        let mlir = """
        module @test_add {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        #expect(executable.inputCount == 2, "Should have 2 inputs")
        #expect(executable.outputCount == 1, "Should have 1 output")
    }

    @Test("Compile matrix multiplication")
    func compileMatmul() throws {
        let client = try MetalHLOClient.create()

        let mlir = """
        module @test_matmul {
          func.func @main(%arg0: tensor<32x64xf32>, %arg1: tensor<64x16xf32>) -> (tensor<32x16xf32>) {
            %0 = stablehlo.dot %arg0, %arg1 : (tensor<32x64xf32>, tensor<64x16xf32>) -> tensor<32x16xf32>
            return %0 : tensor<32x16xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        #expect(executable.inputCount == 2, "Should have 2 inputs")
        #expect(executable.outputCount == 1, "Should have 1 output")
    }

    // MARK: - Buffer Tests

    @Test("Create buffer")
    func createBuffer() throws {
        let client = try MetalHLOClient.create()

        let data: [Float] = [1.0, 2.0, 3.0, 4.0]
        let buffer = try client.createBuffer(data, shape: [4])

        #expect(buffer.shape == [4], "Shape should match")
        #expect(buffer.elementCount == 4, "Element count should be 4")
    }

    @Test("Buffer round trip")
    func bufferRoundTrip() throws {
        let client = try MetalHLOClient.create()

        let originalData: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let buffer = try client.createBuffer(originalData, shape: [2, 3])

        let retrievedData = try buffer.toFloatArray()

        #expect(retrievedData.count == originalData.count, "Data count should match")
        for (original, retrieved) in zip(originalData, retrievedData) {
            #expect(abs(original - retrieved) < 1e-6, "Values should match")
        }
    }

    // MARK: - Execution Tests

    @Test("Execute addition")
    func executeAddition() throws {
        let client = try MetalHLOClient.create()

        let mlir = """
        module @exec_add {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let a: [Float] = [1.0, 2.0, 3.0, 4.0]
        let b: [Float] = [0.5, 0.5, 0.5, 0.5]

        let bufferA = try client.createBuffer(a, shape: [4])
        let bufferB = try client.createBuffer(b, shape: [4])

        let outputs = try executable.execute([bufferA, bufferB])

        #expect(outputs.count == 1, "Should have 1 output")

        let result = try outputs[0].toFloatArray()
        let expected: [Float] = [1.5, 2.5, 3.5, 4.5]

        #expect(result.count == expected.count, "Result count should match")
        for (r, e) in zip(result, expected) {
            #expect(abs(r - e) < 1e-6, "Values should match")
        }
    }

    @Test("Execute multiply")
    func executeMultiply() throws {
        let client = try MetalHLOClient.create()

        let mlir = """
        module @exec_mul {
          func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> (tensor<3xf32>) {
            %0 = stablehlo.multiply %arg0, %arg1 : tensor<3xf32>
            return %0 : tensor<3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let a: [Float] = [2.0, 3.0, 4.0]
        let b: [Float] = [3.0, 4.0, 5.0]

        let bufferA = try client.createBuffer(a, shape: [3])
        let bufferB = try client.createBuffer(b, shape: [3])

        let outputs = try executable.execute([bufferA, bufferB])

        let result = try outputs[0].toFloatArray()
        let expected: [Float] = [6.0, 12.0, 20.0]

        for (r, e) in zip(result, expected) {
            #expect(abs(r - e) < 1e-6, "Values should match")
        }
    }

    @Test("Execute ReLU")
    func executeRelu() throws {
        let client = try MetalHLOClient.create()

        let mlir = """
        module @exec_relu {
          func.func @main(%arg0: tensor<6xf32>) -> (tensor<6xf32>) {
            %0 = stablehlo.constant dense<0.0> : tensor<6xf32>
            %1 = stablehlo.maximum %arg0, %0 : tensor<6xf32>
            return %1 : tensor<6xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let input: [Float] = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
        let buffer = try client.createBuffer(input, shape: [6])

        let outputs = try executable.execute([buffer])

        let result = try outputs[0].toFloatArray()
        let expected: [Float] = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0]

        for (r, e) in zip(result, expected) {
            #expect(abs(r - e) < 1e-6, "ReLU values should match")
        }
    }

    @Test("Execute matrix multiplication")
    func executeMatmul() throws {
        let client = try MetalHLOClient.create()

        let mlir = """
        module @exec_matmul {
          func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> (tensor<2x2xf32>) {
            %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // A = [[1, 2, 3], [4, 5, 6]]
        let a: [Float] = [1, 2, 3, 4, 5, 6]
        // B = [[1, 2], [3, 4], [5, 6]]
        let b: [Float] = [1, 2, 3, 4, 5, 6]

        let bufferA = try client.createBuffer(a, shape: [2, 3])
        let bufferB = try client.createBuffer(b, shape: [3, 2])

        let outputs = try executable.execute([bufferA, bufferB])

        let result = try outputs[0].toFloatArray()
        // Expected: [[22, 28], [49, 64]]
        // (1*1 + 2*3 + 3*5) = 22, (1*2 + 2*4 + 3*6) = 28
        // (4*1 + 5*3 + 6*5) = 49, (4*2 + 5*4 + 6*6) = 64
        let expected: [Float] = [22, 28, 49, 64]

        #expect(result.count == expected.count, "Result count should match")
        for (r, e) in zip(result, expected) {
            #expect(abs(r - e) < 1e-5, "Matmul values should match")
        }
    }

    // MARK: - Timing Tests

    @Test("Execute with timing")
    func executeWithTiming() throws {
        let client = try MetalHLOClient.create()

        let size = 256
        let mlir = """
        module @timing_test {
          func.func @main(%arg0: tensor<\(size)x\(size)xf32>, %arg1: tensor<\(size)x\(size)xf32>) -> (tensor<\(size)x\(size)xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<\(size)x\(size)xf32>
            return %0 : tensor<\(size)x\(size)xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let data = [Float](repeating: 1.0, count: size * size)
        let bufferA = try client.createBuffer(data, shape: [size, size])
        let bufferB = try client.createBuffer(data, shape: [size, size])

        let (outputs, timing) = try executable.executeWithTiming([bufferA, bufferB])

        #expect(outputs.count == 1, "Should have 1 output")
        #expect(timing.totalTimeNs > 0, "Total time should be > 0")

        print("Metal execution timing:")
        print("  GPU time: \(timing.gpuMs) ms")
        print("  Total time: \(timing.totalMs) ms")
    }

    // MARK: - Cache Tests

    @Test("Metal compilation cache")
    func metalCompilationCache() throws {
        let client = try MetalHLOClient.create()

        let mlir = """
        module @cache_test {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %0 = stablehlo.negate %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        // Clear cache first
        MetalCompilationCache.shared.clear()

        // First compilation - should be a miss
        let _ = try client.compile(mlir)
        // Note: The client.compile doesn't use the cache directly -
        // the cache is used by LazyTensorBarrier. This test just verifies
        // the cache API works.

        // Manually test cache
        let hash = "test_hash_123"
        let executable = try client.compile(mlir)
        MetalCompilationCache.shared.put(hash: hash, executable: executable)

        let cached = MetalCompilationCache.shared.get(hash: hash)
        #expect(cached != nil, "Should retrieve cached executable")
    }

    // MARK: - Transpose Tests

    @Test("Execute 4D transpose for attention pattern")
    func execute4DTranspose() throws {
        let client = try MetalHLOClient.create()

        // Test transpose swapping last two dimensions: [2, 2, 3, 4] -> [2, 2, 4, 3]
        // This is the pattern used in attention: K^T = K.transpose(dim1: 2, dim2: 3)
        let mlir = """
        module @exec_transpose {
          func.func @main(%arg0: tensor<2x2x3x4xf32>) -> (tensor<2x2x4x3xf32>) {
            %0 = stablehlo.transpose %arg0, dims = [0, 1, 3, 2] : (tensor<2x2x3x4xf32>) -> tensor<2x2x4x3xf32>
            return %0 : tensor<2x2x4x3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Input shape: [2, 2, 3, 4] = 48 elements
        // Create test data where we can verify the transpose is correct
        // Using indices as values to make verification easier
        var input = [Float](repeating: 0, count: 48)
        for i in 0..<48 { input[i] = Float(i) }

        let buffer = try client.createBuffer(input, shape: [2, 2, 3, 4])
        let outputs = try executable.execute([buffer])

        let result = try outputs[0].toFloatArray()

        // Verify output shape implicitly through count (2*2*4*3 = 48)
        #expect(result.count == 48, "Output should have 48 elements")

        // Verify transpose correctness by checking specific values
        // Input[b, h, i, j] at linear index = b*24 + h*12 + i*4 + j
        // Output[b, h, j, i] at linear index = b*24 + h*12 + j*3 + i
        // So output[0,0,0,0] = input[0,0,0,0] = 0
        // output[0,0,1,0] = input[0,0,0,1] = 1
        // output[0,0,0,1] = input[0,0,1,0] = 4
        #expect(abs(result[0] - 0.0) < 1e-6, "result[0,0,0,0] should be input[0,0,0,0]=0")
        #expect(abs(result[1] - 4.0) < 1e-6, "result[0,0,0,1] should be input[0,0,1,0]=4")
        #expect(abs(result[2] - 8.0) < 1e-6, "result[0,0,0,2] should be input[0,0,2,0]=8")
        #expect(abs(result[3] - 1.0) < 1e-6, "result[0,0,1,0] should be input[0,0,0,1]=1")

        print("4D transpose test passed!")
    }

    @Test("Execute batched matmul with transposed K")
    func executeBatchedMatmulWithTranspose() throws {
        let client = try MetalHLOClient.create()

        // Test the attention pattern: Q @ K^T
        // Q: [1, 2, 4, 3] (batch=1, heads=2, seq=4, headDim=3)
        // K: [1, 2, 4, 3] (same shape)
        // K^T: [1, 2, 3, 4] (transpose last two dims)
        // Result: [1, 2, 4, 4]
        let mlir = """
        module @attention_matmul {
          func.func @main(%q: tensor<1x2x4x3xf32>, %k: tensor<1x2x4x3xf32>) -> (tensor<1x2x4x4xf32>) {
            %kT = stablehlo.transpose %k, dims = [0, 1, 3, 2] : (tensor<1x2x4x3xf32>) -> tensor<1x2x3x4xf32>
            %result = stablehlo.dot_general %q, %kT, #stablehlo.dot<lhs_batching_dimensions = [0, 1], rhs_batching_dimensions = [0, 1], lhs_contracting_dimensions = [3], rhs_contracting_dimensions = [2]> : (tensor<1x2x4x3xf32>, tensor<1x2x3x4xf32>) -> tensor<1x2x4x4xf32>
            return %result : tensor<1x2x4x4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Q = all 1s for simplicity, K = all 1s
        let q = [Float](repeating: 1.0, count: 1*2*4*3)
        let k = [Float](repeating: 1.0, count: 1*2*4*3)

        let bufferQ = try client.createBuffer(q, shape: [1, 2, 4, 3])
        let bufferK = try client.createBuffer(k, shape: [1, 2, 4, 3])

        let outputs = try executable.execute([bufferQ, bufferK])
        let result = try outputs[0].toFloatArray()

        // Q @ K^T with all 1s should give 3.0 for each element (sum of 3 ones)
        #expect(result.count == 1*2*4*4, "Output should have 32 elements")
        for i in 0..<result.count {
            #expect(abs(result[i] - 3.0) < 1e-5, "All values should be 3.0, got \(result[i]) at index \(i)")
        }

        print("Batched matmul with transpose test passed!")
    }
}

#else

// Empty test suite for non-macOS platforms
@Suite("MetalHLO Tests (Non-macOS)")
struct MetalHLOTests {
    @Test("Metal not available on this platform")
    func metalNotAvailable() {
        // Metal is only available on macOS
        #expect(!Backend.metal.isAvailable, "Metal should not be available on this platform")
    }
}

#endif
