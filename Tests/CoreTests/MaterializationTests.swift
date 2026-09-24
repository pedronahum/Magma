// Magma - Materialization tests
// A failed barrier must surface its real error (thrown by the checked APIs,
// recorded on the handle) instead of leaving scalars() to return [].
//
// .serialized: these suites share the global registry and compilation cache.

import Foundation
import Testing
@testable import Magma
@testable import LazyTensor
@testable import XLARuntime

/// A lazy op whose declared output shape disagrees with its input (a reshape
/// of 2 elements to 3), which graph validation rejects before any compilation.
private func invalidReshape(of input: Tensor<Float>) -> Tensor<Float> {
    let handle = LazyTensorHandle(
        id: TensorRegistry.shared.nextTensorId(), shape: [3], dtype: .float32, device: .default)
    handle.irNode = .operation(op: .reshape, inputs: [input.handle], attributes: ["shape": [3]])
    return Tensor(handle: handle)
}

/// A slice whose limit runs past its (device-resident) input: it passes graph
/// validation but the backend's verifier rejects the program.
private func outOfBoundsSlice(of input: Tensor<Float>) -> Tensor<Float> {
    let handle = LazyTensorHandle(
        id: TensorRegistry.shared.nextTensorId(), shape: [5], dtype: .float32, device: .default)
    handle.irNode = .operation(
        op: .slice, inputs: [input.handle],
        attributes: ["start": [0], "limit": [5], "strides": [1]])
    return Tensor(handle: handle)
}

@Suite("Materialization Error Tests", .serialized, .enabled(if: PluginAvailability.cpu))
struct MaterializationErrorTests {

    @Test("constant reads need no barrier and no backend")
    func constantReadsNeedNoBackend() throws {
        let t = Tensor<Float>([1, 2, 3], shape: [3])
        #expect(try t.fetchScalars() == [1, 2, 3])
        #expect(!t.handle.isMaterialized, "reading a constant must not execute anything")
        #expect(Tensor<Float>([4], shape: []).item() == 4)
    }

    @Test("validation failure is thrown and recorded on the handle")
    func validationFailureIsThrown() {
        let bad = invalidReshape(of: Tensor<Float>([1, 2], shape: [2]))
        let error = #expect(throws: MaterializationError.self) { try bad.fetchScalars() }
        #expect(error?.stage == .validation)
        #expect(bad.handle.materializationError?.stage == .validation)
        #expect(!bad.handle.isMaterialized)
    }

    @Test("bare barrier does not throw but records the error")
    func bareBarrierRecordsError() throws {
        let bad = invalidReshape(of: Tensor<Float>([1, 2], shape: [2]))
        bad.markForMaterialization()
        LazyTensorBarrier()
        #expect(bad.handle.materializationError?.stage == .validation)
        // The failed tensor is no longer pending: a later barrier is a no-op.
        try LazyTensorBarrierThrowing()
    }

    @Test("throwing barrier reports the failure to every pending tensor")
    func throwingBarrierReportsAllOutputs() {
        let good = Tensor<Float>([1, 2], shape: [2]) * Tensor<Float>([3, 4], shape: [2])
        let bad = invalidReshape(of: Tensor<Float>([1, 2], shape: [2]))
        good.markForMaterialization()
        bad.markForMaterialization()
        #expect(throws: MaterializationError.self) { try LazyTensorBarrierThrowing() }
        #expect(good.handle.materializationError?.stage == .validation)
        #expect(bad.handle.materializationError?.stage == .validation)

        // Retrying the valid tensor on its own succeeds and clears its error.
        #expect(good.scalars() == [3, 8])
        #expect(good.handle.materializationError == nil)
    }

    @Test("compilation failure carries the backend's error")
    func compilationFailureIsThrown() {
        let input = Tensor<Float>([1, 2], shape: [2]).materialize()
        let bad = outOfBoundsSlice(of: input)
        let error = #expect(throws: MaterializationError.self) { try bad.fetchScalars() }
        #expect(error?.stage == .compilation)
        #expect(error?.underlying != nil)
        #expect(error.map { "\($0)".contains("MAGMA_DEBUG") || $0.mlirDumpPath != nil } == true)
    }

    @Test("materialized tensors read without another barrier")
    func materializedReads() throws {
        let t = (Tensor<Float>([1, 2], shape: [2]) + Tensor<Float>([10, 20], shape: [2])).materialize()
        #expect(t.handle.materializedBuffer != nil)
        #expect(try t.fetchScalars() == [11, 22])
        let e = try Tensor<Float>([2], shape: []).exp().fetchItem()
        #expect(abs(e - Float(exp(2.0))) < 1e-5)
    }
}
