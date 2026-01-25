// Magma - Mixed Precision Tests
// Tests for bfloat16 mixed precision support

import XCTest
@testable import Magma
@testable import LazyTensor
@testable import StableHLO
@testable import XLARuntime

// MARK: - Type Conversion Tests

final class TypeConversionTests: XCTestCase {

    func testToReducedPrecisionDtype() {
        let tensor = Tensor<Float>.ones([2, 3])
        let bf16Tensor = tensor.toReducedPrecision()

        // Verify the dtype changed to bfloat16
        XCTAssertEqual(bf16Tensor.dtype, .bfloat16)
        XCTAssertEqual(bf16Tensor.shape, [2, 3])
    }

    func testToFullPrecisionDtype() {
        let tensor = Tensor<Float>.ones([2, 3])
        let bf16Tensor = tensor.toReducedPrecision()
        let fp32Tensor = bf16Tensor.toFullPrecision()

        // Verify dtype is back to float32
        XCTAssertEqual(fp32Tensor.dtype, .float32)
        XCTAssertEqual(fp32Tensor.shape, [2, 3])
    }

    func testToFullPrecisionNoOpForFloat32() {
        let tensor = Tensor<Float>.ones([2, 3])

        // toFullPrecision on float32 should be a no-op (same handle)
        let result = tensor.toFullPrecision()

        // Shape and dtype should be the same
        XCTAssertEqual(result.shape, tensor.shape)
        XCTAssertEqual(result.dtype, .float32)
    }

    func testToDtypeConversion() {
        let tensor = Tensor<Float>.ones([3, 3])

        // Convert to bfloat16
        let bf16 = tensor.to(.bfloat16)
        XCTAssertEqual(bf16.dtype, .bfloat16)

        // Convert to float64
        let fp64 = tensor.to(.float64)
        XCTAssertEqual(fp64.dtype, .float64)

        // No-op conversion
        let fp32 = tensor.to(.float32)
        XCTAssertEqual(fp32.dtype, .float32)
    }

    func testConversionPreservesShape() {
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

            XCTAssertEqual(bf16.shape, shape, "Shape mismatch for bfloat16 conversion with shape \(shape)")
            XCTAssertEqual(fp32.shape, shape, "Shape mismatch for float32 conversion with shape \(shape)")
        }
    }
}

// MARK: - Mixed Precision Roundtrip Tests

final class MixedPrecisionRoundtripTests: XCTestCase {

    func testSimpleRoundtrip() {
        let original = Tensor<Float>([1.0, 2.0, 3.0, 4.0], shape: [4])

        // Convert to bfloat16 and back
        let bf16 = original.toReducedPrecision()
        let recovered = bf16.toFullPrecision()

        // The values should be approximately equal (bfloat16 has limited precision)
        let recoveredValues = recovered.scalars()
        XCTAssertEqual(recoveredValues.count, 4)

        // Note: bfloat16 has ~3 decimal digits of precision
        for (i, val) in recoveredValues.enumerated() {
            let expected = Float(i + 1)
            XCTAssertEqual(val, expected, accuracy: 0.01, "Value at index \(i) doesn't match")
        }
    }

    func testMatrixRoundtrip() {
        let original = Tensor<Float>([
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ], shape: [2, 3])

        let bf16 = original.toReducedPrecision()
        let recovered = bf16.toFullPrecision()
        let values = recovered.scalars()

        XCTAssertEqual(values.count, 6)
        for i in 0..<6 {
            XCTAssertEqual(values[i], Float(i + 1), accuracy: 0.01)
        }
    }
}

// MARK: - Mixed Precision Operation Tests

final class MixedPrecisionOperationTests: XCTestCase {

    func testBF16Addition() {
        let a = Tensor<Float>.ones([2, 2]).toReducedPrecision()
        let b = Tensor<Float>.ones([2, 2]).toReducedPrecision()

        // Addition in bfloat16
        let c = a + b
        XCTAssertEqual(c.dtype, .bfloat16)
        XCTAssertEqual(c.shape, [2, 2])

        // Convert back and verify
        let result = c.toFullPrecision()
        let values = result.scalars()
        XCTAssertEqual(values.count, 4)
        for val in values {
            XCTAssertEqual(val, 2.0, accuracy: 0.01)
        }
    }

    func testBF16Multiplication() {
        let a = Tensor<Float>.full([2, 2], 2.0).toReducedPrecision()
        let b = Tensor<Float>.full([2, 2], 3.0).toReducedPrecision()

        let c = a * b
        XCTAssertEqual(c.dtype, .bfloat16)

        let result = c.toFullPrecision()
        let values = result.scalars()
        for val in values {
            XCTAssertEqual(val, 6.0, accuracy: 0.01)
        }
    }

    func testBF16MatMul() {
        // Matrix multiplication in reduced precision
        let a = Tensor<Float>.ones([2, 3]).toReducedPrecision()
        let b = Tensor<Float>.ones([3, 4]).toReducedPrecision()

        let c = a.matmul(b)
        XCTAssertEqual(c.dtype, .bfloat16)
        XCTAssertEqual(c.shape, [2, 4])

        let result = c.toFullPrecision()
        let values = result.scalars()
        XCTAssertEqual(values.count, 8)
        // Each element should be sum of 3 ones = 3
        for val in values {
            XCTAssertEqual(val, 3.0, accuracy: 0.1)
        }
    }
}

// MARK: - Mixed Precision Utility Tests

final class MixedPrecisionUtilityTests: XCTestCase {

    func testIsRecommendedForTPU() {
        let tpuDevice = Device(backend: .tpu, index: 0)
        XCTAssertTrue(MixedPrecision.isRecommended(on: tpuDevice))
    }

    func testIsRecommendedForGPU() {
        let gpuDevice = Device(backend: .gpu, index: 0)
        XCTAssertTrue(MixedPrecision.isRecommended(on: gpuDevice))
    }

    func testIsNotRecommendedForCPU() {
        let cpuDevice = Device(backend: .cpu, index: 0)
        XCTAssertFalse(MixedPrecision.isRecommended(on: cpuDevice))
    }

    func testAutocast() {
        let a = Tensor<Float>.ones([2, 2])
        let b = Tensor<Float>.ones([2, 2])

        let result = MixedPrecision.autocast(inputs: [a, b]) { inputs in
            // This computation happens in bfloat16
            inputs[0] + inputs[1]
        }

        // Result should be in float32
        XCTAssertEqual(result.dtype, .float32)

        let values = result.scalars()
        for val in values {
            XCTAssertEqual(val, 2.0, accuracy: 0.01)
        }
    }
}

// MARK: - DType Tests

final class DTypeTests: XCTestCase {

    func testBFloat16Properties() {
        let bf16 = DType.bfloat16

        XCTAssertEqual(bf16.byteSize, 2)
        XCTAssertTrue(bf16.isFloatingPoint)
        XCTAssertFalse(bf16.isSignedInteger)
        XCTAssertFalse(bf16.isUnsignedInteger)
        XCTAssertEqual(bf16.mlirName, "bf16")
    }

    func testFloat32Properties() {
        let fp32 = DType.float32

        XCTAssertEqual(fp32.byteSize, 4)
        XCTAssertTrue(fp32.isFloatingPoint)
        XCTAssertEqual(fp32.mlirName, "f32")
    }

    func testFloat16Properties() {
        let fp16 = DType.float16

        XCTAssertEqual(fp16.byteSize, 2)
        XCTAssertTrue(fp16.isFloatingPoint)
        XCTAssertEqual(fp16.mlirName, "f16")
    }
}

// MARK: - Numerical Precision Tests

final class NumericalPrecisionTests: XCTestCase {

    func testBFloat16PrecisionLimits() {
        // bfloat16 has ~3 decimal digits of precision
        // Test that small differences are lost
        let a = Tensor<Float>([1.0], shape: [1])
        let b = Tensor<Float>([1.001], shape: [1])

        let aBf16 = a.toReducedPrecision().toFullPrecision()
        let bBf16 = b.toReducedPrecision().toFullPrecision()

        // Both should round to approximately 1.0 in bfloat16
        let aVal = aBf16.scalars()[0]
        let bVal = bBf16.scalars()[0]

        XCTAssertEqual(aVal, 1.0, accuracy: 0.01)
        XCTAssertEqual(bVal, 1.0, accuracy: 0.01)
    }

    func testBFloat16LargeValues() {
        // bfloat16 has the same exponent range as float32
        let large = Tensor<Float>([1e30], shape: [1])

        let bf16 = large.toReducedPrecision()
        let recovered = bf16.toFullPrecision()

        let val = recovered.scalars()[0]
        // Should preserve order of magnitude
        XCTAssertTrue(val > 1e29 && val < 1e31, "Large value not preserved: \(val)")
    }

    func testBFloat16SmallValues() {
        // Test subnormal handling
        let small = Tensor<Float>([1e-30], shape: [1])

        let bf16 = small.toReducedPrecision()
        let recovered = bf16.toFullPrecision()

        let val = recovered.scalars()[0]
        // Should be close to original (within bfloat16 precision)
        XCTAssertTrue(val > 0 && val < 1e-20, "Small value not preserved: \(val)")
    }
}

// MARK: - OpKind Tests

final class ConvertOpKindTests: XCTestCase {

    func testConvertOpKindExists() {
        let op = OpKind.convert
        XCTAssertEqual(op.rawValue, "convert")
    }
}
