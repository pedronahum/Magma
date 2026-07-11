// Magma - Broadcast Gradient Tests
// Regression tests for VJPs of the arithmetic operators under broadcasting.
//
// The pullbacks for + - * / must reduce the incoming cotangent back to each
// operand's original shape (summing over the axes that were broadcast). If they
// don't, an implicit broadcast like `x[2,3] + bias[3]` yields a bias gradient of
// shape [2,3] instead of [3] — silently wrong for any code that relies on
// implicit broadcasting in a differentiated expression.
//
// `.serialized` because materializing gradients spins up a GPU PJRT client, and
// concurrent clients OOM a unified-memory machine (see XLAGPUSmokeTests).

import Testing
import _Differentiation
@testable import Magma
@testable import LazyTensor

@Suite("Broadcast Gradient Tests", .serialized)
struct BroadcastGradientTests {

    @Test("add: bias[N] broadcast over [B,N] reduces gradient to [N]")
    func addBiasGrad() {
        let x = Tensor<Float>.ones([2, 3])            // treated as constant
        let bias = Tensor<Float>([10, 20, 30], shape: [3])

        let grad = gradient(at: bias) { b in (x + b).sum() }

        // Each bias_j is added to B=2 rows; d(sum)/d(bias_j) = 2.
        #expect(grad.shape == [3])
        #expect(grad.scalars() == [2, 2, 2])
    }

    @Test("multiply: scalar-broadcast gradient reduces to operand shape")
    func mulBroadcastGrad() {
        let x = Tensor<Float>([1, 2, 3, 4], shape: [2, 2])   // constant
        let s = Tensor<Float>([5], shape: [1])               // broadcast scalar

        let grad = gradient(at: s) { v in (x * v).sum() }

        // d(sum(x*s))/ds = sum(x) = 1+2+3+4 = 10, reduced to shape [1].
        #expect(grad.shape == [1])
        #expect(grad.scalars() == [10])
    }

    @Test("subtract: rhs bias gradient reduces and negates")
    func subBiasGrad() {
        let x = Tensor<Float>.ones([2, 3])
        let bias = Tensor<Float>([1, 1, 1], shape: [3])

        let grad = gradient(at: bias) { b in (x - b).sum() }

        // d(sum(x - b))/d(b_j) = -B = -2, reduced to [3].
        #expect(grad.shape == [3])
        #expect(grad.scalars() == [-2, -2, -2])
    }

    @Test("same-shape add gradient is unaffected (regression guard)")
    func sameShapeAddGrad() {
        let x = Tensor<Float>.ones([2, 3])
        let y = Tensor<Float>.ones([2, 3])

        let (gx, gy) = gradient(at: x, y) { a, b in (a + b).sum() }

        #expect(gx.shape == [2, 3])
        #expect(gy.shape == [2, 3])
        #expect(gx.scalars() == [1, 1, 1, 1, 1, 1])
        #expect(gy.scalars() == [1, 1, 1, 1, 1, 1])
    }
}
