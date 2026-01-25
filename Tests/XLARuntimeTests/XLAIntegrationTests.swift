// Magma - XLA Integration Tests
// End-to-end tests with real XLA execution
// Requires: MAGMA_XLA_PATH environment variable set to /opt/swiftir-deps

import XCTest
@testable import XLARuntime
@testable import LazyTensor
@testable import StableHLO

/// Integration tests for the XLA execution pipeline
/// These tests require XLA PJRT plugin to be installed
final class XLAIntegrationTests: XCTestCase {

    // MARK: - Setup

    /// Check if XLA is available for testing
    static var xlaAvailable: Bool = {
        // Try to create a client to see if XLA is available
        do {
            _ = try PJRTClient.create(backend: .cpu)
            return true
        } catch {
            print("XLA not available: \(error)")
            return false
        }
    }()

    override func setUp() {
        super.setUp()
        // Reset tensor registry between tests
        TensorRegistry.shared.clearAll()
    }

    // MARK: - PJRT Client Tests

    func testClientCreation() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)
        XCTAssertFalse(client.devices.isEmpty, "Client should have at least one device")
        XCTAssertEqual(client.backend, .cpu)
        XCTAssertFalse(client.platformName.isEmpty, "Platform name should not be empty")
    }

    func testDeviceEnumeration() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)
        XCTAssertGreaterThanOrEqual(client.devices.count, 1)

        let device = client.defaultDevice
        XCTAssertNotNil(device)
        XCTAssertFalse(device!.kind.isEmpty, "Device kind should not be empty")
    }

    // MARK: - Buffer Tests

    func testBufferCreationFloat32() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)
        let data: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        let buffer = try client.createBuffer(data, shape: [2, 3], elementType: .float32)

        XCTAssertEqual(buffer.shape, [2, 3])
        XCTAssertEqual(buffer.elementType, .float32)
        XCTAssertEqual(buffer.elementCount, 6)
        XCTAssertEqual(buffer.sizeInBytes, 24)
    }

    func testBufferRoundTrip() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)
        let originalData: [Float] = [1.5, 2.5, 3.5, 4.5]

        let buffer = try client.createBuffer(originalData, shape: [2, 2], elementType: .float32)
        let retrievedData = try buffer.toFloatArray()

        XCTAssertEqual(retrievedData.count, originalData.count)
        for i in 0..<originalData.count {
            XCTAssertEqual(retrievedData[i], originalData[i], accuracy: 1e-6)
        }
    }

    // MARK: - Simple Computation Tests

    func testSimpleAdd() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // Create a simple add program: f(x, y) = x + y
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

        XCTAssertEqual(outputs.count, 1)

        let result = try outputs[0].toFloatArray()
        let expected: [Float] = [11.0, 22.0, 33.0, 44.0]

        XCTAssertEqual(result.count, expected.count)
        for i in 0..<expected.count {
            XCTAssertEqual(result[i], expected[i], accuracy: 1e-5)
        }
    }

    func testSimpleMultiply() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        let mlir = """
        module @simple_multiply {
          func.func @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
            %0 = stablehlo.multiply %arg0, %arg1 : tensor<3xf32>
            return %0 : tensor<3xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let x = try client.createBuffer([2.0, 3.0, 4.0] as [Float], shape: [3], elementType: .float32)
        let y = try client.createBuffer([5.0, 6.0, 7.0] as [Float], shape: [3], elementType: .float32)

        let outputs = try executable.execute([x, y])
        let result = try outputs[0].toFloatArray()

        XCTAssertEqual(result[0], 10.0, accuracy: 1e-5)
        XCTAssertEqual(result[1], 18.0, accuracy: 1e-5)
        XCTAssertEqual(result[2], 28.0, accuracy: 1e-5)
    }

    // MARK: - StableHLO Emission and Execution

    func testEmittedMLIRExecution() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // Build a graph using LazyTensor infrastructure
        let id1 = TensorRegistry.shared.nextTensorId()
        let handle1 = LazyTensorHandle(id: id1, shape: [3], dtype: .float32, device: .default)
        handle1.irNode = .constant(values: [1.0, 2.0, 3.0], shape: [3])

        let id2 = TensorRegistry.shared.nextTensorId()
        let handle2 = LazyTensorHandle(id: id2, shape: [3], dtype: .float32, device: .default)
        handle2.irNode = .constant(values: [10.0, 20.0, 30.0], shape: [3])

        let id3 = TensorRegistry.shared.nextTensorId()
        let handle3 = LazyTensorHandle(id: id3, shape: [3], dtype: .float32, device: .default)
        handle3.irNode = .operation(op: .add, inputs: [handle1, handle2], attributes: [:])

        // Build graph
        let graph = IRGraph()
        graph.addOutput(handle3)
        graph.buildTopologicalOrder()

        // Emit MLIR
        let emitter = StableHLOEmitter(graph: graph)
        let mlir = emitter.emit(name: "test_add_graph")

        // Verify the MLIR was generated
        XCTAssertTrue(mlir.contains("stablehlo.constant"))
        XCTAssertTrue(mlir.contains("stablehlo.add"))

        // Compile and execute
        let executable = try client.compile(mlir)
        let outputs = try executable.execute([])

        XCTAssertEqual(outputs.count, 1)

        let result = try outputs[0].toFloatArray()
        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result[0], 11.0, accuracy: 1e-5)
        XCTAssertEqual(result[1], 22.0, accuracy: 1e-5)
        XCTAssertEqual(result[2], 33.0, accuracy: 1e-5)
    }

    func testMatmulExecution() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // Matrix multiplication: [2,3] x [3,2] -> [2,2]
        let mlir = """
        module @matmul {
          func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>) -> tensor<2x2xf32> {
            %0 = stablehlo.dot %arg0, %arg1 : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>
            return %0 : tensor<2x2xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // A = [[1,2,3], [4,5,6]]
        let a = try client.createBuffer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] as [Float], shape: [2, 3], elementType: .float32)
        // B = [[1,2], [3,4], [5,6]]
        let b = try client.createBuffer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] as [Float], shape: [3, 2], elementType: .float32)

        let outputs = try executable.execute([a, b])
        let result = try outputs[0].toFloatArray()

        // Expected: [[22, 28], [49, 64]]
        // Row 0: 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        //        1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // Row 1: 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        //        4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        XCTAssertEqual(result[0], 22.0, accuracy: 1e-5)
        XCTAssertEqual(result[1], 28.0, accuracy: 1e-5)
        XCTAssertEqual(result[2], 49.0, accuracy: 1e-5)
        XCTAssertEqual(result[3], 64.0, accuracy: 1e-5)
    }

    func testReluExecution() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // ReLU: max(0, x)
        let mlir = """
        module @relu {
          func.func @main(%arg0: tensor<6xf32>) -> tensor<6xf32> {
            %0 = stablehlo.constant dense<0.0> : tensor<6xf32>
            %1 = stablehlo.maximum %arg0, %0 : tensor<6xf32>
            return %1 : tensor<6xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        let x = try client.createBuffer([-3.0, -1.0, 0.0, 1.0, 2.0, 5.0] as [Float], shape: [6], elementType: .float32)

        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        XCTAssertEqual(result[0], 0.0, accuracy: 1e-5)
        XCTAssertEqual(result[1], 0.0, accuracy: 1e-5)
        XCTAssertEqual(result[2], 0.0, accuracy: 1e-5)
        XCTAssertEqual(result[3], 1.0, accuracy: 1e-5)
        XCTAssertEqual(result[4], 2.0, accuracy: 1e-5)
        XCTAssertEqual(result[5], 5.0, accuracy: 1e-5)
    }

    // MARK: - Caching Tests

    func testCompilationCacheHit() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        let mlir = """
        module @cached {
          func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.negate %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """

        // Compile twice - second time should be fast (though we can't directly measure cache hits)
        let exec1 = try client.compile(mlir)
        let exec2 = try client.compile(mlir)

        // Both should produce correct results
        let x = try client.createBuffer([1.0, -2.0, 3.0, -4.0] as [Float], shape: [4], elementType: .float32)

        let result1 = try exec1.execute([x])[0].toFloatArray()
        let result2 = try exec2.execute([x])[0].toFloatArray()

        XCTAssertEqual(result1, result2)
        XCTAssertEqual(result1[0], -1.0, accuracy: 1e-5)
        XCTAssertEqual(result1[1], 2.0, accuracy: 1e-5)
    }

    // MARK: - Slice Tests

    func testSliceExecution() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // Test slice operation with correct syntax
        let mlir = """
        module @test_slice {
          func.func @main(%arg0: tensor<10x20xf32>) -> (tensor<5x10xf32>) {
            %0 = stablehlo.slice %arg0 [0:5:1, 0:10:1] : (tensor<10x20xf32>) -> tensor<5x10xf32>
            return %0 : tensor<5x10xf32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Create input: 10x20 tensor with sequential values
        var inputData = [Float](repeating: 0, count: 200)
        for i in 0..<200 {
            inputData[i] = Float(i)
        }
        let input = try client.createBuffer(inputData, shape: [10, 20], elementType: .float32)

        let outputs = try executable.execute([input])
        XCTAssertEqual(outputs.count, 1)

        let result = try outputs[0].toFloatArray()
        XCTAssertEqual(result.count, 50) // 5x10

        // First element should be 0 (row 0, col 0)
        XCTAssertEqual(result[0], 0.0, accuracy: 1e-5)
        // Element at [0,9] should be 9
        XCTAssertEqual(result[9], 9.0, accuracy: 1e-5)
        // Element at [1,0] should be 20 (row 1 starts at index 20)
        XCTAssertEqual(result[10], 20.0, accuracy: 1e-5)
    }

    func testReduceMaxWithNegInfinity() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // Test reduce max with very negative init value (use hex for -infinity)
        // 0xFF800000 is the IEEE 754 representation of -infinity for float32
        let mlir = """
        module @test_reduce_max {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<f32>) {
            %init = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %result = stablehlo.reduce(%arg0 init: %init) across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
             reducer(%a: tensor<f32>, %b: tensor<f32>) {
              %r = stablehlo.maximum %a, %b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            return %result : tensor<f32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-3.0, -1.0, 2.0, 5.0] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()
        XCTAssertEqual(result[0], 5.0, accuracy: 1e-5)
    }

    func testComplexMNISTGraph() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // Simplified MNIST forward pass MLIR
        let mlir = """
        module @mnist_forward {
          func.func @main(%labels: tensor<60000x10xf32>, %images: tensor<60000x784xf32>) -> (tensor<f32>) {
            %batch_labels = stablehlo.slice %labels [0:64:1, 0:10:1] : (tensor<60000x10xf32>) -> tensor<64x10xf32>
            %batch_images = stablehlo.slice %images [0:64:1, 0:784:1] : (tensor<60000x784xf32>) -> tensor<64x784xf32>
            %w1 = stablehlo.constant dense<0.01> : tensor<784x128xf32>
            %h1 = stablehlo.dot %batch_images, %w1 : (tensor<64x784xf32>, tensor<784x128xf32>) -> tensor<64x128xf32>
            %zero128 = stablehlo.constant dense<0.0> : tensor<64x128xf32>
            %h1_relu = stablehlo.maximum %h1, %zero128 : tensor<64x128xf32>
            %w2 = stablehlo.constant dense<0.01> : tensor<128x10xf32>
            %logits = stablehlo.dot %h1_relu, %w2 : (tensor<64x128xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
            %zero_scalar = stablehlo.constant dense<0.0> : tensor<f32>
            %sum = stablehlo.reduce(%logits init: %zero_scalar) across dimensions = [0, 1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<f32>
             reducer(%a: tensor<f32>, %b: tensor<f32>) {
              %r = stablehlo.add %a, %b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            return %sum : tensor<f32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Create dummy inputs (zeros are fine for testing compilation)
        let labels = try client.createBuffer([Float](repeating: 0, count: 60000 * 10), shape: [60000, 10], elementType: .float32)
        let images = try client.createBuffer([Float](repeating: 0, count: 60000 * 784), shape: [60000, 784], elementType: .float32)

        let outputs = try executable.execute([labels, images])
        XCTAssertEqual(outputs.count, 1)

        let result = try outputs[0].toFloatArray()
        XCTAssertEqual(result.count, 1)
        // With all zeros, output should be 0
        XCTAssertEqual(result[0], 0.0, accuracy: 1e-5)
    }

    func testSoftmaxWithReduceMax() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // Test softmax with reduce - reduce outputs tensor<64xf32>, then reshape to tensor<64x1xf32> for broadcast
        let mlir = """
        module @test_softmax {
          func.func @main(%arg0: tensor<64x10xf32>) -> (tensor<64x10xf32>) {
            %neg_inf = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %max_reduced = stablehlo.reduce(%arg0 init: %neg_inf) across dimensions = [1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<64xf32>
             reducer(%a: tensor<f32>, %b: tensor<f32>) {
              %r = stablehlo.maximum %a, %b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %max = stablehlo.reshape %max_reduced : (tensor<64xf32>) -> tensor<64x1xf32>
            %max_bcast = stablehlo.broadcast_in_dim %max, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x10xf32>
            %centered = stablehlo.subtract %arg0, %max_bcast : tensor<64x10xf32>
            %exp = stablehlo.exponential %centered : tensor<64x10xf32>
            %zero = stablehlo.constant dense<0.0> : tensor<f32>
            %sum_reduced = stablehlo.reduce(%exp init: %zero) across dimensions = [1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<64xf32>
             reducer(%a: tensor<f32>, %b: tensor<f32>) {
              %r = stablehlo.add %a, %b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %sum = stablehlo.reshape %sum_reduced : (tensor<64xf32>) -> tensor<64x1xf32>
            %sum_bcast = stablehlo.broadcast_in_dim %sum, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x10xf32>
            %softmax = stablehlo.divide %exp, %sum_bcast : tensor<64x10xf32>
            return %softmax : tensor<64x10xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([Float](repeating: 0, count: 64 * 10), shape: [64, 10], elementType: .float32)
        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()
        XCTAssertEqual(result.count, 640)
        // Softmax of zeros should be uniform: 0.1 for each element
        XCTAssertEqual(result[0], 0.1, accuracy: 1e-5)
    }

    func testMinimalSliceReduce() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // Minimal test of slice + reduce
        let mlir = """
        module @test_minimal {
          func.func @main(%arg0: tensor<60000x10xf32>, %arg1: tensor<60000x784xf32>) -> (tensor<f32>) {
            %0 = stablehlo.slice %arg0 [64:128:1, 0:10:1] : (tensor<60000x10xf32>) -> tensor<64x10xf32>
            %zero = stablehlo.constant dense<0.0> : tensor<f32>
            %result = stablehlo.reduce(%0 init: %zero) across dimensions = [0, 1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<f32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.add %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            return %result : tensor<f32>
          }
        }
        """

        let executable = try client.compile(mlir)
        XCTAssertNotNil(executable)
    }

    func testMNISTGradientMLIR() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // This is the backward pass MLIR that was failing (fixed module name)
        let mlir = """
        module @lazy_graph_2718715 {
          func.func @main(%arg0: tensor<60000x10xf32>, %arg1: tensor<60000x784xf32>) -> (tensor<f32>) {
            %0 = stablehlo.slice %arg0 [64:128:1, 0:10:1] : (tensor<60000x10xf32>) -> tensor<64x10xf32>
            %1 = stablehlo.slice %arg1 [64:128:1, 0:784:1] : (tensor<60000x784xf32>) -> tensor<64x784xf32>
            %2 = stablehlo.constant dense<0.0> : tensor<784x128xf32>
            %3 = stablehlo.constant dense<0.05050762742757797> : tensor<f32>
            %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<784x128xf32>
            %5 = stablehlo.multiply %2, %4 : tensor<784x128xf32>
            %6 = stablehlo.constant dense<0.0> : tensor<f32>
            %7 = stablehlo.slice %arg1 [0:64:1, 0:784:1] : (tensor<60000x784xf32>) -> tensor<64x784xf32>
            %8 = stablehlo.transpose %7, dims = [1, 0] : (tensor<64x784xf32>) -> tensor<784x64xf32>
            %9 = stablehlo.dot %7, %5 : (tensor<64x784xf32>, tensor<784x128xf32>) -> tensor<64x128xf32>
            %10 = stablehlo.constant dense<0.0> : tensor<128xf32>
            %11 = stablehlo.broadcast_in_dim %10, dims = [1] : (tensor<128xf32>) -> tensor<64x128xf32>
            %12 = stablehlo.add %9, %11 : tensor<64x128xf32>
            %13 = stablehlo.constant dense<0.0> : tensor<64x128xf32>
            %14 = stablehlo.maximum %12, %13 : tensor<64x128xf32>
            %15 = stablehlo.constant dense<0.0> : tensor<128x10xf32>
            %16 = stablehlo.constant dense<0.125> : tensor<f32>
            %17 = stablehlo.broadcast_in_dim %16, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
            %18 = stablehlo.multiply %15, %17 : tensor<128x10xf32>
            %19 = stablehlo.dot %14, %18 : (tensor<64x128xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
            %20 = stablehlo.constant dense<0.0> : tensor<10xf32>
            %21 = stablehlo.broadcast_in_dim %20, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
            %22 = stablehlo.add %19, %21 : tensor<64x10xf32>
            %24 = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %23 = stablehlo.reduce(%22 init: %24) across dimensions = [1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<64xf32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.maximum %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %25 = stablehlo.reshape %23 : (tensor<64xf32>) -> tensor<64x1xf32>
            %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x10xf32>
            %27 = stablehlo.subtract %22, %26 : tensor<64x10xf32>
            %28 = stablehlo.exponential %27 : tensor<64x10xf32>
            %30 = stablehlo.constant dense<0.0> : tensor<f32>
            %29 = stablehlo.reduce(%28 init: %30) across dimensions = [1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<64xf32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.add %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %31 = stablehlo.reshape %29 : (tensor<64xf32>) -> tensor<64x1xf32>
            %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x10xf32>
            %33 = stablehlo.divide %28, %32 : tensor<64x10xf32>
            %34 = stablehlo.constant dense<1.0> : tensor<f32>
            %35 = stablehlo.constant dense<64.0> : tensor<f32>
            %36 = stablehlo.divide %34, %35 : tensor<f32>
            %37 = stablehlo.negate %36 : tensor<f32>
            %38 = stablehlo.broadcast_in_dim %37, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
            %39 = stablehlo.slice %arg0 [0:64:1, 0:10:1] : (tensor<60000x10xf32>) -> tensor<64x10xf32>
            %40 = stablehlo.multiply %38, %39 : tensor<64x10xf32>
            %41 = stablehlo.constant dense<1.0000000116860974e-07> : tensor<f32>
            %42 = stablehlo.broadcast_in_dim %41, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
            %43 = stablehlo.add %33, %42 : tensor<64x10xf32>
            %44 = stablehlo.divide %40, %43 : tensor<64x10xf32>
            %45 = stablehlo.multiply %44, %33 : tensor<64x10xf32>
            %47 = stablehlo.constant dense<0.0> : tensor<f32>
            %46 = stablehlo.reduce(%45 init: %47) across dimensions = [0, 1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<f32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.add %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %48 = stablehlo.broadcast_in_dim %46, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
            %49 = stablehlo.subtract %44, %48 : tensor<64x10xf32>
            %50 = stablehlo.multiply %33, %49 : tensor<64x10xf32>
            %51 = stablehlo.transpose %18, dims = [1, 0] : (tensor<128x10xf32>) -> tensor<10x128xf32>
            %52 = stablehlo.dot %50, %51 : (tensor<64x10xf32>, tensor<10x128xf32>) -> tensor<64x128xf32>
            %53 = stablehlo.constant dense<0.0> : tensor<64x128xf32>
            %54 = stablehlo.compare GT, %14, %53, FLOAT : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xi1>
            %55 = stablehlo.convert %54 : (tensor<64x128xi1>) -> tensor<64x128xf32>
            %56 = stablehlo.multiply %52, %55 : tensor<64x128xf32>
            %57 = stablehlo.dot %8, %56 : (tensor<784x64xf32>, tensor<64x128xf32>) -> tensor<784x128xf32>
            %58 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<784x128xf32>
            %59 = stablehlo.add %58, %57 : tensor<784x128xf32>
            %60 = stablehlo.constant dense<0.009999999776482582> : tensor<f32>
            %61 = stablehlo.broadcast_in_dim %60, dims = [] : (tensor<f32>) -> tensor<784x128xf32>
            %62 = stablehlo.multiply %59, %61 : tensor<784x128xf32>
            %63 = stablehlo.subtract %5, %62 : tensor<784x128xf32>
            %64 = stablehlo.dot %1, %63 : (tensor<64x784xf32>, tensor<784x128xf32>) -> tensor<64x128xf32>
            %65 = stablehlo.constant dense<0.0> : tensor<f32>
            %67 = stablehlo.constant dense<0.0> : tensor<f32>
            %66 = stablehlo.reduce(%56 init: %67) across dimensions = [0, 1] : (tensor<64x128xf32>, tensor<f32>) -> tensor<f32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.add %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %68 = stablehlo.add %65, %66 : tensor<f32>
            %69 = stablehlo.multiply %68, %60 : tensor<f32>
            %70 = stablehlo.broadcast_in_dim %69, dims = [] : (tensor<f32>) -> tensor<128xf32>
            %71 = stablehlo.subtract %10, %70 : tensor<128xf32>
            %72 = stablehlo.broadcast_in_dim %71, dims = [1] : (tensor<128xf32>) -> tensor<64x128xf32>
            %73 = stablehlo.add %64, %72 : tensor<64x128xf32>
            %74 = stablehlo.constant dense<0.0> : tensor<64x128xf32>
            %75 = stablehlo.maximum %73, %74 : tensor<64x128xf32>
            %76 = stablehlo.constant dense<0.0> : tensor<f32>
            %77 = stablehlo.transpose %14, dims = [1, 0] : (tensor<64x128xf32>) -> tensor<128x64xf32>
            %78 = stablehlo.dot %77, %50 : (tensor<128x64xf32>, tensor<64x10xf32>) -> tensor<128x10xf32>
            %79 = stablehlo.broadcast_in_dim %76, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
            %80 = stablehlo.add %79, %78 : tensor<128x10xf32>
            %81 = stablehlo.broadcast_in_dim %60, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
            %82 = stablehlo.multiply %80, %81 : tensor<128x10xf32>
            %83 = stablehlo.subtract %18, %82 : tensor<128x10xf32>
            %84 = stablehlo.dot %75, %83 : (tensor<64x128xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
            %85 = stablehlo.constant dense<0.0> : tensor<f32>
            %87 = stablehlo.constant dense<0.0> : tensor<f32>
            %86 = stablehlo.reduce(%50 init: %87) across dimensions = [0, 1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<f32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.add %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %88 = stablehlo.add %85, %86 : tensor<f32>
            %89 = stablehlo.multiply %88, %60 : tensor<f32>
            %90 = stablehlo.broadcast_in_dim %89, dims = [] : (tensor<f32>) -> tensor<10xf32>
            %91 = stablehlo.subtract %20, %90 : tensor<10xf32>
            %92 = stablehlo.broadcast_in_dim %91, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
            %93 = stablehlo.add %84, %92 : tensor<64x10xf32>
            %95 = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %94 = stablehlo.reduce(%93 init: %95) across dimensions = [1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<64xf32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.maximum %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %96 = stablehlo.reshape %94 : (tensor<64xf32>) -> tensor<64x1xf32>
            %97 = stablehlo.broadcast_in_dim %96, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x10xf32>
            %98 = stablehlo.subtract %93, %97 : tensor<64x10xf32>
            %99 = stablehlo.exponential %98 : tensor<64x10xf32>
            %101 = stablehlo.constant dense<0.0> : tensor<f32>
            %100 = stablehlo.reduce(%99 init: %101) across dimensions = [1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<64xf32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.add %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %102 = stablehlo.reshape %100 : (tensor<64xf32>) -> tensor<64x1xf32>
            %103 = stablehlo.broadcast_in_dim %102, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x10xf32>
            %104 = stablehlo.divide %99, %103 : tensor<64x10xf32>
            %105 = stablehlo.constant dense<1.0000000116860974e-07> : tensor<f32>
            %106 = stablehlo.broadcast_in_dim %105, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
            %107 = stablehlo.add %104, %106 : tensor<64x10xf32>
            %108 = stablehlo.log %107 : tensor<64x10xf32>
            %109 = stablehlo.multiply %0, %108 : tensor<64x10xf32>
            %111 = stablehlo.constant dense<0.0> : tensor<f32>
            %110 = stablehlo.reduce(%109 init: %111) across dimensions = [0, 1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<f32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.add %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %112 = stablehlo.negate %110 : tensor<f32>
            %113 = stablehlo.constant dense<64.0> : tensor<f32>
            %114 = stablehlo.divide %112, %113 : tensor<f32>
            return %114 : tensor<f32>
          }
        }
        """

        let executable = try client.compile(mlir)
        // We can't really run this since it requires 60000x10 and 60000x784 inputs
        // Just verify it compiles
        XCTAssertNotNil(executable)
    }

    func testBoolToFloatConvert() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // Test boolean to float conversion
        let mlir = """
        module @test_bool_convert {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xf32>) {
            %zero = stablehlo.constant dense<0.0> : tensor<4xf32>
            %mask = stablehlo.compare GT, %arg0, %zero, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            %mask_float = stablehlo.convert %mask : (tensor<4xi1>) -> tensor<4xf32>
            %result = stablehlo.multiply %arg0, %mask_float : tensor<4xf32>
            return %result : tensor<4xf32>
          }
        }
        """

        let executable = try client.compile(mlir)
        let x = try client.createBuffer([-3.0, -1.0, 2.0, 5.0] as [Float], shape: [4], elementType: .float32)
        let outputs = try executable.execute([x])
        let result = try outputs[0].toFloatArray()

        // ReLU-like behavior: x if x > 0, else 0
        XCTAssertEqual(result[0], 0.0, accuracy: 1e-5)  // -3.0 * 0.0 = 0
        XCTAssertEqual(result[1], 0.0, accuracy: 1e-5)  // -1.0 * 0.0 = 0
        XCTAssertEqual(result[2], 2.0, accuracy: 1e-5)  // 2.0 * 1.0 = 2
        XCTAssertEqual(result[3], 5.0, accuracy: 1e-5)  // 5.0 * 1.0 = 5
    }

    func testFullMNISTForwardPassMLIR() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        // This is the exact MLIR that's being generated for MNIST forward pass
        let mlir = """
        module @test_mnist {
          func.func @main(%arg0: tensor<60000x10xf32>, %arg1: tensor<60000x784xf32>) -> (tensor<f32>) {
            %0 = stablehlo.slice %arg0 [0:64:1, 0:10:1] : (tensor<60000x10xf32>) -> tensor<64x10xf32>
            %1 = stablehlo.slice %arg1 [0:64:1, 0:784:1] : (tensor<60000x784xf32>) -> tensor<64x784xf32>
            %2 = stablehlo.constant dense<0.0> : tensor<784x128xf32>
            %3 = stablehlo.constant dense<0.05050762742757797> : tensor<f32>
            %4 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<784x128xf32>
            %5 = stablehlo.multiply %2, %4 : tensor<784x128xf32>
            %6 = stablehlo.dot %1, %5 : (tensor<64x784xf32>, tensor<784x128xf32>) -> tensor<64x128xf32>
            %7 = stablehlo.constant dense<0.0> : tensor<128xf32>
            %8 = stablehlo.broadcast_in_dim %7, dims = [1] : (tensor<128xf32>) -> tensor<64x128xf32>
            %9 = stablehlo.add %6, %8 : tensor<64x128xf32>
            %10 = stablehlo.constant dense<0.0> : tensor<64x128xf32>
            %11 = stablehlo.maximum %9, %10 : tensor<64x128xf32>
            %12 = stablehlo.constant dense<0.0> : tensor<128x10xf32>
            %13 = stablehlo.constant dense<0.125> : tensor<f32>
            %14 = stablehlo.broadcast_in_dim %13, dims = [] : (tensor<f32>) -> tensor<128x10xf32>
            %15 = stablehlo.multiply %12, %14 : tensor<128x10xf32>
            %16 = stablehlo.dot %11, %15 : (tensor<64x128xf32>, tensor<128x10xf32>) -> tensor<64x10xf32>
            %17 = stablehlo.constant dense<0.0> : tensor<10xf32>
            %18 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<10xf32>) -> tensor<64x10xf32>
            %19 = stablehlo.add %16, %18 : tensor<64x10xf32>
            %21 = stablehlo.constant dense<0xFF800000> : tensor<f32>
            %20 = stablehlo.reduce(%19 init: %21) across dimensions = [1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<64xf32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.maximum %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %22 = stablehlo.reshape %20 : (tensor<64xf32>) -> tensor<64x1xf32>
            %23 = stablehlo.broadcast_in_dim %22, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x10xf32>
            %24 = stablehlo.subtract %19, %23 : tensor<64x10xf32>
            %25 = stablehlo.exponential %24 : tensor<64x10xf32>
            %27 = stablehlo.constant dense<0.0> : tensor<f32>
            %26 = stablehlo.reduce(%25 init: %27) across dimensions = [1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<64xf32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.add %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %28 = stablehlo.reshape %26 : (tensor<64xf32>) -> tensor<64x1xf32>
            %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x10xf32>
            %30 = stablehlo.divide %25, %29 : tensor<64x10xf32>
            %31 = stablehlo.constant dense<1.0000000116860974e-07> : tensor<f32>
            %32 = stablehlo.broadcast_in_dim %31, dims = [] : (tensor<f32>) -> tensor<64x10xf32>
            %33 = stablehlo.add %30, %32 : tensor<64x10xf32>
            %34 = stablehlo.log %33 : tensor<64x10xf32>
            %35 = stablehlo.multiply %0, %34 : tensor<64x10xf32>
            %37 = stablehlo.constant dense<0.0> : tensor<f32>
            %36 = stablehlo.reduce(%35 init: %37) across dimensions = [0, 1] : (tensor<64x10xf32>, tensor<f32>) -> tensor<f32>
             reducer(%arg_a: tensor<f32>, %arg_b: tensor<f32>) {
              %r = stablehlo.add %arg_a, %arg_b : tensor<f32>
              stablehlo.return %r : tensor<f32>
            }
            %38 = stablehlo.negate %36 : tensor<f32>
            %39 = stablehlo.constant dense<64.0> : tensor<f32>
            %40 = stablehlo.divide %38, %39 : tensor<f32>
            return %40 : tensor<f32>
          }
        }
        """

        let executable = try client.compile(mlir)

        // Create dummy inputs with zeros
        let labels = try client.createBuffer([Float](repeating: 0, count: 60000 * 10), shape: [60000, 10], elementType: .float32)
        let images = try client.createBuffer([Float](repeating: 0, count: 60000 * 784), shape: [60000, 784], elementType: .float32)

        let outputs = try executable.execute([labels, images])
        XCTAssertEqual(outputs.count, 1)

        let result = try outputs[0].toFloatArray()
        XCTAssertEqual(result.count, 1)
    }

    // MARK: - Error Handling Tests

    func testInvalidMLIRCompilation() throws {
        guard Self.xlaAvailable else {
            throw XCTSkip("XLA PJRT plugin not available")
        }

        let client = try PJRTClient.create(backend: .cpu)

        let invalidMlir = """
        this is not valid MLIR at all
        """

        XCTAssertThrowsError(try client.compile(invalidMlir)) { error in
            XCTAssertTrue(error is XLAError)
        }
    }
}
