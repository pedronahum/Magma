// Magma - Gradient Checking Tests
// Tests for numerical gradient verification

import Foundation
import Testing
@testable import Magma
@testable import LazyTensor

/// Deterministic, well-conditioned input for gradient checks. float32
/// finite-difference gradchecks occasionally grazed tolerance on unlucky
/// `randn` draws — a tiny gradient inflates the relative error, and shifts such
/// as `randn + 1` can drop a value into a non-differentiable neighbourhood
/// (abs/relu at 0). These fixed values have magnitude in ~[0.7, 1.5], stay away
/// from zero, and mix sign, so every gradcheck is deterministic (green once ⇒
/// green always) while still exercising both branches of piecewise ops.
private func detGradTensor(_ shape: [Int]) -> Tensor<Float> {
    let count = shape.isEmpty ? 1 : shape.reduce(1, *)
    let values = (0..<count).map { i -> Float in
        let mag = 0.7 + 0.8 * Foundation.sin(Float(i) * 0.9 + 0.35).magnitude   // [0.7, 1.5]
        return (i % 3 == 0) ? -mag : mag
    }
    return Tensor<Float>(values, shape: shape, on: .default)
}

// MARK: - Basic Gradient Check Tests

@Suite("Gradcheck Basic Tests")
struct GradcheckBasicTests {

    // Use reasonable tolerances for float32 numerical precision
    let atol: Float = 1e-3
    let rtol: Float = 1e-2

    @Test("Gradcheck sum")
    func gradcheckSum() {
        // Sum has gradient of all ones
        let x = detGradTensor([2, 3])
        let result = gradcheck({ $0.sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for sum. MaxAbsDiff: \(result.maxAbsDiff)")
        #expect(result.maxAbsDiff < 0.01)  // Reasonable for float32
    }

    @Test("Gradcheck mean")
    func gradcheckMean() {
        // Mean has gradient of 1/n for all elements
        let x = detGradTensor([3, 4])
        let result = gradcheck({ $0.mean() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for mean. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck addition")
    func gradcheckAddition() {
        let x = detGradTensor([2, 2])
        let result = gradcheck({ ($0 + $0).sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for addition. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck subtraction")
    func gradcheckSubtraction() {
        let x = detGradTensor([2, 2])
        let result = gradcheck({ ($0 - $0 * Tensor<Float>.full([2, 2], 0.5, on: .default)).sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for subtraction. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck multiplication")
    func gradcheckMultiplication() {
        // d(sum(x^2))/dx = 2x. Use deterministic inputs bounded away from 0:
        // near x=0 the analytic gradient is ~0, so finite-difference relative
        // error blows up (a numerical artifact, not a VJP error) and randn
        // occasionally drew such a value, making this test flaky.
        let x = Tensor<Float>([1.5, 2.0, -1.8, 2.2, -2.5, 1.3], shape: [2, 3])
        let result = gradcheck({ ($0 * $0).sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for multiplication. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck division")
    func gradcheckDivision() {
        // Use positive values to avoid division issues
        let x = detGradTensor([2, 3]).abs() + Tensor<Float>.full([2, 3], 1.0, on: .default)
        let ones = Tensor<Float>.ones([2, 3])
        let result = gradcheck({ (ones / $0).sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for division. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck negation")
    func gradcheckNegation() {
        let x = detGradTensor([3, 3])
        let result = gradcheck({ (-$0).sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for negation. MaxAbsDiff: \(result.maxAbsDiff)")
    }
}

// MARK: - Activation Function Gradient Tests

@Suite("Gradcheck Activation Tests")
struct GradcheckActivationTests {

    // Use reasonable tolerances for float32 numerical precision
    let atol: Float = 1e-3
    let rtol: Float = 1e-2

    @Test("Gradcheck ReLU")
    func gradcheckRelu() {
        // Use values away from 0 where ReLU is non-differentiable
        let x = detGradTensor([3, 4])
        // Shift to avoid values near zero
        let shifted = x + Tensor<Float>.full([3, 4], 0.5, on: .default)

        let result = gradcheck({ $0.relu().sum() }, input: shifted, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for ReLU. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck sigmoid")
    func gradcheckSigmoid() {
        let x = detGradTensor([2, 3])
        let result = gradcheck({ $0.sigmoid().sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for sigmoid. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck GELU")
    func gradcheckGelu() {
        // Guards that vjpGelu is the derivative of the tanh approximation the
        // forward actually emits (the old sigmoid-approx pullback failed this).
        let x = detGradTensor([2, 3])
        let result = gradcheck({ $0.gelu().sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for GELU. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck tanh")
    func gradcheckTanh() {
        let x = detGradTensor([2, 3])
        let result = gradcheck({ $0.tanh().sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for tanh. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck exp")
    func gradcheckExp() {
        // Use small values to avoid exp overflow
        let x = detGradTensor([2, 3]) * Tensor<Float>.full([2, 3], 0.5, on: .default)
        let result = gradcheck({ $0.exp().sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for exp. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck log")
    func gradcheckLog() {
        // Use positive values for log
        let x = detGradTensor([2, 3]).abs() + Tensor<Float>.full([2, 3], 0.5, on: .default)
        let result = gradcheck({ $0.log().sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for log. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck abs")
    func gradcheckAbs() {
        // Values clearly away from zero (|x| >= 1.2), where abs is differentiable.
        let x = detGradTensor([3, 3]).abs() + Tensor<Float>.full([3, 3], 0.5, on: .default)
        let result = gradcheck({ $0.abs().sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for abs. MaxAbsDiff: \(result.maxAbsDiff)")
    }
}

// MARK: - Matrix Operation Gradient Tests

@Suite("Gradcheck Matrix Tests")
struct GradcheckMatrixTests {

    // Use reasonable tolerances for float32 numerical precision
    let atol: Float = 1e-3
    let rtol: Float = 1e-2

    @Test("Gradcheck matmul")
    func gradcheckMatmul() {
        let a = detGradTensor([2, 3])
        let b = detGradTensor([3, 4])

        // Check gradient w.r.t. first input
        let result1 = gradcheck({ $0.matmul(b).sum() }, input: a, atol: atol, rtol: rtol)
        #expect(result1.passed, "Gradcheck failed for matmul (first input). MaxAbsDiff: \(result1.maxAbsDiff)")

        // Check gradient w.r.t. second input
        let result2 = gradcheck({ a.matmul($0).sum() }, input: b, atol: atol, rtol: rtol)
        #expect(result2.passed, "Gradcheck failed for matmul (second input). MaxAbsDiff: \(result2.maxAbsDiff)")
    }

    @Test("Gradcheck transpose")
    func gradcheckTranspose() {
        let x = detGradTensor([3, 4])
        let result = gradcheck({ $0.transpose().sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for transpose. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck reshape")
    func gradcheckReshape() {
        let x = detGradTensor([2, 6])
        let result = gradcheck({ $0.reshape([3, 4]).sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for reshape. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    // Note: Broadcast VJP has known issues with shape handling
    // This test is disabled until the VJP is fixed
    // func testGradcheckBroadcast() { ... }
}

// MARK: - Multi-Input Gradient Tests

@Suite("Gradcheck Multi-Input Tests")
struct GradcheckMultiInputTests {

    // Use reasonable tolerances for float32 numerical precision
    let atol: Float = 1e-3
    let rtol: Float = 1e-2

    @Test("Gradcheck two-input add")
    func gradcheckTwoInputAdd() {
        let a = detGradTensor([2, 3])
        let b = detGradTensor([2, 3])

        let (result1, result2) = gradcheck({ x, y in (x + y).sum() }, input1: a, input2: b, atol: atol, rtol: rtol)

        #expect(result1.passed, "Gradcheck failed for add (first input). MaxAbsDiff: \(result1.maxAbsDiff)")
        #expect(result2.passed, "Gradcheck failed for add (second input). MaxAbsDiff: \(result2.maxAbsDiff)")
    }

    @Test("Gradcheck two-input mul")
    func gradcheckTwoInputMul() {
        let a = detGradTensor([2, 3])
        let b = detGradTensor([2, 3])

        let (result1, result2) = gradcheck({ x, y in (x * y).sum() }, input1: a, input2: b, atol: atol, rtol: rtol)

        #expect(result1.passed, "Gradcheck failed for mul (first input). MaxAbsDiff: \(result1.maxAbsDiff)")
        #expect(result2.passed, "Gradcheck failed for mul (second input). MaxAbsDiff: \(result2.maxAbsDiff)")
    }

    @Test("Gradcheck two-input matmul")
    func gradcheckTwoInputMatmul() {
        let a = detGradTensor([2, 3])
        let b = detGradTensor([3, 2])

        let (result1, result2) = gradcheck({ x, y in x.matmul(y).sum() }, input1: a, input2: b, atol: atol, rtol: rtol)

        #expect(result1.passed, "Gradcheck failed for matmul (first input). MaxAbsDiff: \(result1.maxAbsDiff)")
        #expect(result2.passed, "Gradcheck failed for matmul (second input). MaxAbsDiff: \(result2.maxAbsDiff)")
    }
}

// MARK: - Numerical Gradient Tests

@Suite("Numerical Gradient Tests")
struct NumericalGradientTests {

    @Test("Numerical gradient sum")
    func numericalGradientSum() {
        let x = Tensor<Float>.ones([2, 3])
        let grad = numericalGradient(of: { $0.sum() }, at: x)

        // scalars() triggers evaluation internally
        let values = grad.scalars()

        // Gradient of sum should be all ones (with numerical precision)
        for v in values {
            #expect(abs(v - 1.0) < 0.01)
        }
    }

    @Test("Numerical gradient square")
    func numericalGradientSquare() {
        let x = Tensor<Float>([Float(1.0), Float(2.0), Float(3.0)], shape: [3], on: .default)
        let grad = numericalGradient(of: { ($0 * $0).sum() }, at: x)

        let values = grad.scalars()

        // Gradient of x^2 is 2x (with numerical precision)
        #expect(abs(values[0] - 2.0) < 0.01)
        #expect(abs(values[1] - 4.0) < 0.01)
        #expect(abs(values[2] - 6.0) < 0.01)
    }

    @Test("Numerical gradient forward")
    func testNumericalGradientForward() {
        let x = Tensor<Float>.ones([2, 2])
        let grad = Magma.numericalGradientForward(of: { $0.sum() }, at: x)

        let values = grad.scalars()

        for v in values {
            #expect(abs(v - 1.0) < 0.01)  // Less accurate than centered
        }
    }
}

// MARK: - Jacobian Tests

@Suite("Jacobian Tests")
struct JacobianTests {

    @Test("Jacobian identity")
    func jacobianIdentity() {
        let x = detGradTensor([3])
        let jacobian = numericalJacobian(of: { $0 }, at: x)

        #expect(jacobian.shape == [3, 3])

        let values = jacobian.scalars()

        // Should be identity matrix (with numerical tolerance)
        for i in 0..<3 {
            for j in 0..<3 {
                let expected: Float = (i == j) ? 1.0 : 0.0
                #expect(abs(values[i * 3 + j] - expected) < 0.01)
            }
        }
    }

    @Test("Jacobian linear")
    func jacobianLinear() {
        // f(x) = Ax where A is 2x3 matrix
        let aData: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let A = Tensor<Float>(aData, shape: [2, 3], on: .default)
        let x = detGradTensor([3])

        let jacobian = numericalJacobian(of: { input in
            A.matmul(input.reshape([3, 1])).reshape([2])
        }, at: x)

        #expect(jacobian.shape == [2, 3])

        let jacobianValues = jacobian.scalars()
        let aValues = A.scalars()

        // Jacobian should equal A (with numerical tolerance)
        for i in 0..<6 {
            #expect(abs(jacobianValues[i] - aValues[i]) < 0.1)
        }
    }
}

// MARK: - Convenience Function Tests

@Suite("Gradcheck Convenience Tests")
struct GradcheckConvenienceTests {

    // Use reasonable tolerances for float32 numerical precision
    let atol: Float = 1e-3
    let rtol: Float = 1e-2

    @Test("Gradcheck passes")
    func testGradcheckPasses() {
        let x = detGradTensor([2, 3])
        let passes = Magma.gradcheckPasses({ $0.sum() }, input: x, atol: atol, rtol: rtol)
        #expect(passes)
    }

    @Test("Assert gradcheck success")
    func assertGradcheckSuccess() throws {
        let x = detGradTensor([2, 3])
        try assertGradcheck({ $0.sum() }, input: x, atol: atol, rtol: rtol)
    }

    @Test("Gradcheck verbose")
    func gradcheckVerbose() {
        let x = detGradTensor([2, 2])
        let result = gradcheck({ $0.sum() }, input: x, atol: atol, rtol: rtol, verbose: true)

        #expect(result.details != nil)
        #expect(result.details?.count == 4)  // 2x2 = 4 elements
    }
}

// MARK: - Complex Function Gradient Tests

@Suite("Gradcheck Complex Tests")
struct GradcheckComplexTests {

    // Use reasonable tolerances for float32 numerical precision
    let atol: Float = 1e-3
    let rtol: Float = 1e-2

    @Test("Gradcheck chained operations")
    func gradcheckChainedOperations() {
        let x = detGradTensor([2, 3])
        // Chained operations need slightly higher tolerance due to error accumulation
        let result = gradcheck({
            let y = $0 * $0  // square
            let z = y.sigmoid()  // sigmoid
            return z.sum()  // reduce
        }, input: x, atol: 2e-3, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for chained ops. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck matmul chain")
    func gradcheckMatmulChain() {
        let x = detGradTensor([2, 3])
        let w1 = detGradTensor([3, 4])
        let w2 = detGradTensor([4, 2])

        let result = gradcheck({
            let h = $0.matmul(w1).relu()
            let out = h.matmul(w2)
            return out.sum()
        }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for matmul chain. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    // Note: Softmax VJP has known issues with dimension handling
    // This test is disabled until the VJP is fixed
    // func testGradcheckSoftmaxCrossEntropy() { ... }
}

// MARK: - Edge Case Tests

@Suite("Gradcheck Edge Case Tests")
struct GradcheckEdgeCaseTests {

    // Use reasonable tolerances for float32 numerical precision
    let atol: Float = 1e-3
    let rtol: Float = 1e-2

    @Test("Gradcheck scalar")
    func gradcheckScalar() {
        let x = Tensor<Float>([Float(2.0)], shape: [], on: .default)
        let result = gradcheck({ $0 * $0 }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for scalar. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck 1D")
    func gradcheck1D() {
        let x = detGradTensor([5])
        let result = gradcheck({ $0.sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for 1D. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    @Test("Gradcheck 4D")
    func gradcheck4D() {
        // Smaller size for speed
        let x = detGradTensor([2, 2, 2, 2])
        let result = gradcheck({ $0.sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck failed for 4D. MaxAbsDiff: \(result.maxAbsDiff)")
    }

    // Note: Very small eps can cause numerical instability
    // This is expected behavior, not a bug
    @Test("Gradcheck default eps")
    func gradcheckDefaultEps() {
        let x = detGradTensor([2, 2])
        let result = gradcheck({ $0.sum() }, input: x, atol: atol, rtol: rtol)

        #expect(result.passed, "Gradcheck with default eps failed. MaxAbsDiff: \(result.maxAbsDiff)")
    }
}

// MARK: - Linear Index Conversion Tests

@Suite("Index Conversion Tests")
struct IndexConversionTests {

    // Use reasonable tolerances for float32 numerical precision
    let atol: Float = 1e-3
    let rtol: Float = 1e-2

    @Test("Linear to multi-index")
    func linearToMultiIndex() {
        // Test via verbose gradcheck which uses this function
        let x = detGradTensor([2, 3])
        let result = gradcheck({ $0.sum() }, input: x, atol: atol, rtol: rtol, verbose: true)

        #expect(result.details != nil)

        if let details = result.details {
            // Check that indices are correctly computed
            #expect(details[0].multiIndex == [0, 0])
            #expect(details[1].multiIndex == [0, 1])
            #expect(details[2].multiIndex == [0, 2])
            #expect(details[3].multiIndex == [1, 0])
            #expect(details[4].multiIndex == [1, 1])
            #expect(details[5].multiIndex == [1, 2])
        }
    }
}
