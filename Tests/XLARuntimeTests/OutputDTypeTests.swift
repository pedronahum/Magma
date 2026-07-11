// Magma - Output DType Tests
// Regression for the PJRT execute() path, which used to hardcode every output
// buffer's element type to .float32. It must report the executable's real
// output element type.
//
// .serialized: creating a GPU PJRT client reserves most of unified memory, so
// concurrent clients OOM the machine (see XLAGPUSmokeTests).

import Testing
@testable import XLARuntime

@Suite("Output DType Tests", .serialized)
struct OutputDTypeTests {
    static let gpuAvailable: Bool = { (try? PJRTClient.create(backend: .gpu)) != nil }()

    @Test("int32 output reports .int32")
    func int32Output() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .gpu)
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<4xf32>) -> tensor<4xi32> {
            %0 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xi32>
            return %0 : tensor<4xi32>
          }
        }
        """
        let exe = try client.compile(mlir)
        let x = try client.createBuffer([1.0, 2.0, 3.0, 4.0] as [Float], shape: [4], elementType: .float32)
        let outs = try exe.execute([x])
        #expect(outs.count == 1)
        #expect(outs[0].elementType == .int32)
    }

    @Test("bool output reports .bool")
    func boolOutput() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .gpu)
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
            %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            return %0 : tensor<4xi1>
          }
        }
        """
        let exe = try client.compile(mlir)
        let a = try client.createBuffer([1.0, 5.0, 3.0, 9.0] as [Float], shape: [4], elementType: .float32)
        let b = try client.createBuffer([2.0, 2.0, 2.0, 2.0] as [Float], shape: [4], elementType: .float32)
        let outs = try exe.execute([a, b])
        #expect(outs[0].elementType == .bool)
    }

    @Test("float32 output still reports .float32")
    func float32Output() throws {
        try #require(Self.gpuAvailable, "GPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .gpu)
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        let exe = try client.compile(mlir)
        let x = try client.createBuffer([1.0, 2.0, 3.0, 4.0] as [Float], shape: [4], elementType: .float32)
        let outs = try exe.execute([x])
        #expect(outs[0].elementType == .float32)
    }
}
