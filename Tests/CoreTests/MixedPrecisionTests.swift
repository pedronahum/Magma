// Magma - Mixed Precision Tests
// Tests for bfloat16 mixed precision support

import Testing
@testable import Magma
@testable import LazyTensor
@testable import StableHLO
@testable import XLARuntime

// MARK: - Type Conversion Tests

@Suite("Type Conversion Tests")
struct TypeConversionTests {

    @Test("toReducedPrecision dtype")
    func toReducedPrecisionDtype() {
        let tensor = Tensor<Float>.ones([2, 3])
        let bf16Tensor = tensor.toReducedPrecision()

        // Verify the dtype changed to bfloat16
        #expect(bf16Tensor.dtype == .bfloat16)
        #expect(bf16Tensor.shape == [2, 3])
    }

    @Test("toFullPrecision dtype")
    func toFullPrecisionDtype() {
        let tensor = Tensor<Float>.ones([2, 3])
        let bf16Tensor = tensor.toReducedPrecision()
        let fp32Tensor = bf16Tensor.toFullPrecision()

        // Verify dtype is back to float32
        #expect(fp32Tensor.dtype == .float32)
        #expect(fp32Tensor.shape == [2, 3])
    }

    @Test("toFullPrecision no-op for float32")
    func toFullPrecisionNoOpForFloat32() {
        let tensor = Tensor<Float>.ones([2, 3])

        // toFullPrecision on float32 should be a no-op (same handle)
        let result = tensor.toFullPrecision()

        // Shape and dtype should be the same
        #expect(result.shape == tensor.shape)
        #expect(result.dtype == .float32)
    }

    @Test("toDtype conversion")
    func toDtypeConversion() {
        let tensor = Tensor<Float>.ones([3, 3])

        // Convert to bfloat16
        let bf16 = tensor.to(.bfloat16)
        #expect(bf16.dtype == .bfloat16)

        // Convert to float64
        let fp64 = tensor.to(.float64)
        #expect(fp64.dtype == .float64)

        // No-op conversion
        let fp32 = tensor.to(.float32)
        #expect(fp32.dtype == .float32)
    }

    @Test("Conversion preserves shape")
    func conversionPreservesShape() {
        let shapes: [[Int]] = [
            [],           // Scalar
            [5],          // 1D
            [3, 4],       // 2D
            [2, 3, 4],    // 3D
            [2, 2, 2, 2], // 4D
        ]

        for shape in shapes {
            let tensor = Tensor<Float>.randn(shape)
            let bf16 = tensor.toReducedPrecision()
            let fp32 = bf16.toFullPrecision()

            #expect(bf16.shape == shape, "Shape mismatch for bfloat16 conversion with shape \(shape)")
            #expect(fp32.shape == shape, "Shape mismatch for float32 conversion with shape \(shape)")
        }
    }
}

// MARK: - Mixed Precision Roundtrip Tests

@Suite("Mixed Precision Roundtrip Tests")
struct MixedPrecisionRoundtripTests {

    @Test("Simple roundtrip")
    func simpleRoundtrip() {
        let original = Tensor<Float>([1.0, 2.0, 3.0, 4.0], shape: [4])

        // Convert to bfloat16 and back
        let bf16 = original.toReducedPrecision()
        let recovered = bf16.toFullPrecision()

        // The values should be approximately equal (bfloat16 has limited precision)
        let recoveredValues = recovered.scalars()
        #expect(recoveredValues.count == 4)

        // Note: bfloat16 has ~3 decimal digits of precision
        for (i, val) in recoveredValues.enumerated() {
            let expected = Float(i + 1)
            #expect(abs(val - expected) < 0.01, "Value at index \(i) doesn't match")
        }
    }

    @Test("Matrix roundtrip")
    func matrixRoundtrip() {
        let original = Tensor<Float>([
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ], shape: [2, 3])

        let bf16 = original.toReducedPrecision()
        let recovered = bf16.toFullPrecision()
        let values = recovered.scalars()

        #expect(values.count == 6)
        for i in 0..<6 {
            #expect(abs(values[i] - Float(i + 1)) < 0.01)
        }
    }
}

// MARK: - Mixed Precision Operation Tests

@Suite("Mixed Precision Operation Tests")
struct MixedPrecisionOperationTests {

    @Test("BF16 addition")
    func bf16Addition() {
        let a = Tensor<Float>.ones([2, 2]).toReducedPrecision()
        let b = Tensor<Float>.ones([2, 2]).toReducedPrecision()

        // Addition in bfloat16
        let c = a + b
        #expect(c.dtype == .bfloat16)
        #expect(c.shape == [2, 2])

        // Convert back and verify
        let result = c.toFullPrecision()
        let values = result.scalars()
        #expect(values.count == 4)
        for val in values {
            #expect(abs(val - 2.0) < 0.01)
        }
    }

    @Test("BF16 multiplication")
    func bf16Multiplication() {
        let a = Tensor<Float>.full([2, 2], 2.0).toReducedPrecision()
        let b = Tensor<Float>.full([2, 2], 3.0).toReducedPrecision()

        let c = a * b
        #expect(c.dtype == .bfloat16)

        let result = c.toFullPrecision()
        let values = result.scalars()
        for val in values {
            #expect(abs(val - 6.0) < 0.01)
        }
    }

    @Test("BF16 matmul")
    func bf16MatMul() {
        // Matrix multiplication in reduced precision
        let a = Tensor<Float>.ones([2, 3]).toReducedPrecision()
        let b = Tensor<Float>.ones([3, 4]).toReducedPrecision()

        let c = a.matmul(b)
        #expect(c.dtype == .bfloat16)
        #expect(c.shape == [2, 4])

        let result = c.toFullPrecision()
        let values = result.scalars()
        #expect(values.count == 8)
        // Each element should be sum of 3 ones = 3
        for val in values {
            #expect(abs(val - 3.0) < 0.1)
        }
    }
}

// MARK: - Mixed Precision Utility Tests

@Suite("Mixed Precision Utility Tests")
struct MixedPrecisionUtilityTests {

    @Test("Is recommended for TPU")
    func isRecommendedForTPU() {
        let tpuDevice = Device(backend: .tpu, index: 0)
        #expect(MixedPrecision.isRecommended(on: tpuDevice))
    }

    @Test("Is recommended for GPU")
    func isRecommendedForGPU() {
        let gpuDevice = Device(backend: .gpu, index: 0)
        #expect(MixedPrecision.isRecommended(on: gpuDevice))
    }

    @Test("Is not recommended for CPU")
    func isNotRecommendedForCPU() {
        let cpuDevice = Device(backend: .cpu, index: 0)
        #expect(!MixedPrecision.isRecommended(on: cpuDevice))
    }

    @Test("Autocast")
    func autocast() {
        let a = Tensor<Float>.ones([2, 2])
        let b = Tensor<Float>.ones([2, 2])

        let result = MixedPrecision.autocast(inputs: [a, b]) { inputs in
            // This computation happens in bfloat16
            inputs[0] + inputs[1]
        }

        // Result should be in float32
        #expect(result.dtype == .float32)

        let values = result.scalars()
        for val in values {
            #expect(abs(val - 2.0) < 0.01)
        }
    }
}

// MARK: - DType Tests

@Suite("DType Mixed Precision Tests")
struct DTypeMixedPrecisionTests {

    @Test("BFloat16 properties")
    func bFloat16Properties() {
        let bf16 = DType.bfloat16

        #expect(bf16.byteSize == 2)
        #expect(bf16.isFloatingPoint)
        #expect(!bf16.isSignedInteger)
        #expect(!bf16.isUnsignedInteger)
        #expect(bf16.mlirName == "bf16")
    }

    @Test("Float32 properties")
    func float32Properties() {
        let fp32 = DType.float32

        #expect(fp32.byteSize == 4)
        #expect(fp32.isFloatingPoint)
        #expect(fp32.mlirName == "f32")
    }

    @Test("Float16 properties")
    func float16Properties() {
        let fp16 = DType.float16

        #expect(fp16.byteSize == 2)
        #expect(fp16.isFloatingPoint)
        #expect(fp16.mlirName == "f16")
    }
}

// MARK: - Numerical Precision Tests

@Suite("Numerical Precision Tests")
struct NumericalPrecisionTests {

    @Test("BFloat16 precision limits")
    func bFloat16PrecisionLimits() {
        // bfloat16 has ~3 decimal digits of precision
        // Test that small differences are lost
        let a = Tensor<Float>([1.0], shape: [1])
        let b = Tensor<Float>([1.001], shape: [1])

        let aBf16 = a.toReducedPrecision().toFullPrecision()
        let bBf16 = b.toReducedPrecision().toFullPrecision()

        // Both should round to approximately 1.0 in bfloat16
        let aVal = aBf16.scalars()[0]
        let bVal = bBf16.scalars()[0]

        #expect(abs(aVal - 1.0) < 0.01)
        #expect(abs(bVal - 1.0) < 0.01)
    }

    @Test("BFloat16 large values")
    func bFloat16LargeValues() {
        // bfloat16 has the same exponent range as float32
        let large = Tensor<Float>([1e30], shape: [1])

        let bf16 = large.toReducedPrecision()
        let recovered = bf16.toFullPrecision()

        let val = recovered.scalars()[0]
        // Should preserve order of magnitude
        #expect(val > 1e29 && val < 1e31, "Large value not preserved: \(val)")
    }

    @Test("BFloat16 small values")
    func bFloat16SmallValues() {
        // Test subnormal handling
        let small = Tensor<Float>([1e-30], shape: [1])

        let bf16 = small.toReducedPrecision()
        let recovered = bf16.toFullPrecision()

        let val = recovered.scalars()[0]
        // Should be close to original (within bfloat16 precision)
        #expect(val > 0 && val < 1e-20, "Small value not preserved: \(val)")
    }
}

// MARK: - OpKind Tests

@Suite("Convert OpKind Tests")
struct ConvertOpKindTests {

    @Test("Convert OpKind exists")
    func convertOpKindExists() {
        let op = OpKind.convert
        #expect(op.rawValue == "convert")
    }
}
