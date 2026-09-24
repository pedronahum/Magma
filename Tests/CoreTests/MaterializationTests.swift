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
import StableHLO

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
        let error = materializationError { _ = try bad.fetchScalars() }
        #expect(error != nil, "expected a MaterializationError")
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
        let error = materializationError { _ = try bad.fetchScalars() }
        #expect(error != nil, "expected a MaterializationError")
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

// Constants are held as Float in the IR and promoted to program inputs; the
// upload must use the element type the program declares (i32/f64/i1), not f32.
@Suite("Promoted Constant DType Tests", .serialized, .enabled(if: PluginAvailability.cpu))
struct PromotedConstantDTypeTests {

    @Test("Int32 arithmetic on constants")
    func int32Constants() {
        let sum = Tensor<Int32>([1, 2, 3], shape: [3]) + Tensor<Int32>([10, 20, -30], shape: [3])
        #expect(sum.scalars() == [11, 22, -27])
    }

    @Test("Int32 arithmetic mixing device data and a promoted constant")
    func int32DataAndConstant() {
        let a = Tensor<Int32>([5, -7, 9], shape: [3]).materialize()
        #expect(a.handle.materializedBuffer?.elementType == .int32)
        #expect((a * Tensor<Int32>([2, 3, -4], shape: [3])).scalars() == [10, -21, -36])
        #expect((a - Tensor<Int32>([1, 1, 1], shape: [3])).scalars() == [4, -8, 8])
    }

    @Test("Double arithmetic mixing device data and a promoted constant")
    func doubleDataAndConstant() {
        let a = Tensor<Double>([1.5, 2.25], shape: [2]).materialize()
        #expect(a.handle.materializedBuffer?.elementType == .float64)
        #expect((a + Tensor<Double>([0.25, 0.5], shape: [2])).scalars() == [1.75, 2.75])
        #expect((a * Tensor<Double>([2, -4], shape: [2])).scalars() == [3.0, -9.0])
    }

    @Test("Bool logic mixing device data and a promoted constant")
    func boolDataAndConstant() {
        let a = Tensor<Bool>([true, false, true, false], shape: [4]).materialize()
        #expect(a.handle.materializedBuffer?.elementType == .bool)
        let rhs = Tensor<Bool>([true, true, false, false], shape: [4])
        // StableHLO multiply/add on i1 are logical and/or.
        #expect((a * rhs).scalars() == [true, false, false, false])
        #expect((a + rhs).scalars() == [true, true, true, false])
    }

    @Test("reduced-precision results are converted on read")
    func reducedPrecisionReads() {
        let x = Tensor<Float>([1.5, -2.0, 0.25], shape: [3])
        #expect(x.to(.bfloat16).scalars() == [1.5, -2.0, 0.25])
        #expect(x.to(.float16).scalars() == [1.5, -2.0, 0.25])
    }

    @Test("reduced-precision constants upload with their own element type")
    func reducedPrecisionConstants() {
        // A constant typed bf16/f16 must be uploaded as 16-bit data.
        for dtype in [DType.bfloat16, .float16] {
            let handle = LazyTensorHandle(
                id: TensorRegistry.shared.nextTensorId(), shape: [2], dtype: dtype, device: .default)
            handle.irNode = .constant(values: [3.0, -0.5], shape: [2])
            let up = Tensor<Float>(handle: handle).to(.float32)
            #expect(up.scalars() == [3.0, -0.5], "dtype \(dtype)")
        }
    }

    @Test("half-precision bit conversion rounds to nearest even")
    func halfPrecisionBitConversion() {
        #expect(halfPrecisionBits(1.0) == 0x3C00)
        #expect(halfPrecisionBits(-2.0) == 0xC000)
        #expect(halfPrecisionBits(65504) == 0x7BFF)
        #expect(halfPrecisionBits(1e6) == 0x7C00)                    // overflow → inf
        #expect(halfPrecisionBits(5.960464477539063e-8) == 0x0001)   // smallest subnormal
        #expect(halfPrecisionBits(0) == 0)
        #expect(bfloat16Bits(1.0) == 0x3F80)
        #expect(bfloat16Bits(-2.0) == 0xC000)
    }
}

// executeGraph shares the executable cache with the barrier: its key must
// distinguish every attribute and constant value the program depends on.
@Suite("executeGraph Cache Tests", .serialized, .enabled(if: PluginAvailability.cpu))
struct ExecuteGraphCacheTests {

    private func constant(_ values: [Float]) -> LazyTensorHandle {
        let handle = LazyTensorHandle(
            id: TensorRegistry.shared.nextTensorId(), shape: [values.count],
            dtype: .float32, device: .default)
        handle.irNode = .constant(values: values, shape: [values.count])
        return handle
    }

    private func run(_ output: LazyTensorHandle) throws -> [Float] {
        let graph = IRGraph()
        graph.addOutput(output)
        let buffers = try executeGraph(graph)
        try #require(buffers.count == 1)
        return try buffers[0].toFloatArray()
    }

    private func slice(_ input: LazyTensorHandle, start: Int, limit: Int) -> LazyTensorHandle {
        let handle = LazyTensorHandle(
            id: TensorRegistry.shared.nextTensorId(), shape: [limit - start],
            dtype: .float32, device: .default)
        handle.irNode = .operation(
            op: .slice, inputs: [input],
            attributes: ["start": [start], "limit": [limit], "strides": [1]])
        return handle
    }

    @Test("graphs differing only in slice offsets get different executables")
    func sliceOffsetsAreKeyed() throws {
        let first = try run(slice(constant([1, 2, 3, 4]), start: 0, limit: 2))
        let second = try run(slice(constant([1, 2, 3, 4]), start: 2, limit: 4))
        #expect(first == [1, 2])
        #expect(second == [3, 4])
    }

    @Test("graphs differing only in later constant values give different results")
    func laterConstantValuesAreUsed() throws {
        let first = try run(slice(constant([1, 2, 3, 4, 5, 6]), start: 3, limit: 6))
        let second = try run(slice(constant([1, 2, 3, 4, 50, 60]), start: 3, limit: 6))
        #expect(first == [4, 5, 6])
        #expect(second == [4, 50, 60])
    }
}

// The trace (fast-path) cache keys on constant values and pins promoted
// constants on the device, so a loop over fresh data must not grow it forever.
@Suite("Compilation Cache Bound Tests", .serialized, .enabled(if: PluginAvailability.cpu))
struct CompilationCacheBoundTests {

    /// Run `body` with small cache limits, restoring the previous ones after.
    private func withLimits(
        executables: Int, traceEntries: Int, tracePinnedBytes: Int, _ body: () throws -> Void
    ) rethrows {
        let cache = CompilationCache.shared
        let previous = cache.setLimits(
            maxExecutables: executables, maxTraceEntries: traceEntries,
            maxTracePinnedBytes: tracePinnedBytes)
        defer {
            cache.setLimits(
                maxExecutables: previous.maxExecutables, maxTraceEntries: previous.maxTraceEntries,
                maxTracePinnedBytes: previous.maxTracePinnedBytes)
        }
        try body()
    }

    @Test("fresh data every step keeps the trace cache bounded")
    func freshDataStaysBounded() {
        withLimits(executables: 64, traceEntries: 8, tracePinnedBytes: 4096) {
            for step in 0..<40 {
                let batch = (0..<16).map { Float(step * 16 + $0) }
                let y = Tensor<Float>(batch, shape: [16]).materialize() * Tensor<Float>(
                    Array(repeating: Float(step), count: 16), shape: [16])
                let values = y.scalars()
                #expect(values.count == 16)
                #expect(values == batch.map { $0 * Float(step) }, "step \(step)")

                let usage = CompilationCache.shared.traceCacheUsage
                #expect(usage.entries <= 8)
                #expect(usage.pinnedBytes <= 4096)
            }
        }
    }

    @Test("an entry pinning more than the byte bound is not cached")
    func largePromotedConstantsAreNotPinned() {
        withLimits(executables: 64, traceEntries: 8, tracePinnedBytes: 1024) {
            let before = CompilationCache.shared.traceCacheUsage
            let big = Tensor<Float>((0..<1024).map(Float.init), shape: [1024])   // 4 KiB
            let values = (big + big).scalars()
            #expect(values.count == 1024)
            #expect(values[1023] == 2046)
            let after = CompilationCache.shared.traceCacheUsage
            #expect(after.pinnedBytes <= 1024)
            #expect(after.entries <= before.entries + 1)
        }
    }

    @Test("evicted executables are recompiled and stay correct")
    func executableEviction() {
        withLimits(executables: 2, traceEntries: 0, tracePinnedBytes: 0) {
            // Five structurally different graphs through a two-entry cache, twice.
            for round in 0..<2 {
                for length in 1...5 {
                    let x = Tensor<Float>((0..<length).map(Float.init), shape: [length])
                    let values = (x + x).scalars()
                    #expect(values.count == length, "round \(round)")
                    #expect(values == (0..<length).map { Float(2 * $0) }, "round \(round)")
                    #expect(CompilationCache.shared.executableCount <= 2)
                }
            }
        }
    }
}

// Moving a materialized tensor copies its buffer through the host; the copy
// must keep the element type rather than assume f32.
@Suite("Device Transfer Tests", .serialized, .enabled(if: PluginAvailability.cpu))
struct DeviceTransferTests {

    private let target = Device(backend: .cpu, index: 1)

    @Test("materialized Float, Double and Int32 tensors keep their values")
    func transferKeepsValues() {
        let f = Tensor<Float>([1.5, -2.5], shape: [2]).materialize().to(device: target)
        #expect(f.device == target)
        #expect(f.handle.materializedBuffer?.elementType == .float32)
        #expect(f.scalars() == [1.5, -2.5])

        let d = Tensor<Double>([0.5, -3.25], shape: [2]).materialize().to(device: target)
        #expect(d.handle.materializedBuffer?.elementType == .float64)
        #expect(d.scalars() == [0.5, -3.25])

        let i = Tensor<Int32>([7, -9, 123456], shape: [3]).materialize().to(device: target)
        #expect(i.handle.materializedBuffer?.elementType == .int32)
        #expect(i.scalars() == [7, -9, 123456])
    }
}

@Suite("LRU Map Tests", .enabled(if: PluginAvailability.cpu, "CPU PJRT plugin not available"))
struct LRUMapTests {

    @Test("evicts the least recently used entry by count")
    func evictsByCount() {
        var map = LRUMap<String, Int>(maxEntries: 2)
        map.set("a", 1)
        map.set("b", 2)
        #expect(map.get("a") == 1)       // "b" is now the least recently used
        map.set("c", 3)
        #expect(map.get("b") == nil)
        #expect(map.get("a") == 1)
        #expect(map.get("c") == 3)
        #expect(map.count == 2)
    }

    @Test("evicts by total cost and rejects an oversized entry")
    func evictsByCost() {
        var map = LRUMap<Int, String>(maxEntries: 10, maxCost: 100)
        map.set(1, "one", cost: 60)
        map.set(2, "two", cost: 30)
        map.set(3, "three", cost: 30)    // 120 > 100: evicts 1
        #expect(map.get(1) == nil)
        #expect(map.totalCost == 60)
        map.set(4, "four", cost: 101)    // larger than the bound: not kept
        #expect(map.get(4) == nil)
        #expect(map.totalCost == 60)
        map.set(2, "two'", cost: 10)     // replacing updates the cost
        #expect(map.totalCost == 40)
    }
}

// Optimization passes run on every barrier; a pass that changes a value,
// shape or dtype silently corrupts results.
@Suite("Optimization Correctness Tests", .serialized, .enabled(if: PluginAvailability.cpu))
struct OptimizationCorrectnessTests {

    @Test("CSE does not merge large constants that differ in unsampled positions")
    func cseKeepsDistinctLargeConstants() {
        var sparse = [Float](repeating: 0, count: 1000)
        sparse[1] = 5
        let a = Tensor<Float>(sparse, shape: [1000])
        let b = Tensor<Float>([Float](repeating: 0, count: 1000), shape: [1000])
        let values = (a + b).scalars()
        #expect(values.count == 1000)
        #expect(values[1] == 5)
        #expect(values.reduce(0, +) == 5)
    }

    @Test("CSE does not merge constants of different dtypes")
    func cseKeepsConstantDTypes() {
        let x = Tensor<Float>([1.5, 2.5], shape: [2])
        let y = Tensor<Double>([1.5, 2.5], shape: [2])
        let sx = x.materialize()
        let sy = y.materialize()
        #expect(sx.handle.materializedBuffer?.elementType == .float32)
        #expect(sy.handle.materializedBuffer?.elementType == .float64)
    }

    @Test("CSE does not merge conversions to different dtypes")
    func cseKeepsConvertTargets() {
        let x = Tensor<Float>([1.5, -2.25], shape: [2]).materialize()
        let wide = x.to(.float64)
        let narrow = x.to(.bfloat16)
        wide.markForMaterialization()
        narrow.markForMaterialization()
        LazyTensorBarrier()
        #expect(wide.handle.materializedBuffer?.elementType == .float64)
        #expect(narrow.handle.materializedBuffer?.elementType == .bfloat16)
        #expect(wide.scalars() == [1.5, -2.25])
        #expect(narrow.scalars() == [1.5, -2.25])
    }

    @Test("x + 0 keeps the broadcast shape")
    func addZeroKeepsBroadcastShape() {
        let x = Tensor<Float>([5], shape: []).materialize()
        let y = x + Tensor<Float>.zeros([3])
        #expect(y.scalars() == [5, 5, 5])
    }

    @Test("x * 1 keeps the broadcast shape")
    func multiplyOneKeepsBroadcastShape() {
        let x = Tensor<Float>([2, 3], shape: [2]).materialize()
        let y = x * Tensor<Float>.ones([2, 2])
        #expect(y.scalars() == [2, 3, 2, 3])
    }

    @Test("float x * 0 keeps IEEE semantics")
    func multiplyZeroKeepsNaN() {
        let x = Tensor<Float>([.infinity, 1], shape: [2]).materialize()
        let y = (x * Tensor<Float>.zeros([2])).scalars()
        #expect(y.count == 2)
        #expect(y[0].isNaN, "inf * 0 is NaN")
        #expect(y[1] == 0)
    }
}

// The trace (fast-path) cache key must distinguish constants bit-exactly:
// Float hashing equates -0.0 and +0.0, so a value-based key would reuse the
// first graph's pinned constants for the second.
@Suite("Trace Cache Key Tests", .serialized, .enabled(if: PluginAvailability.cpu))
struct TraceCacheKeyTests {

    @Test("constants differing only in the sign of zero are not conflated")
    func signedZeroConstantsAreKeyed() {
        let x = Tensor<Float>([1, 1, 1], shape: [3]).materialize()
        let positive = (x / Tensor<Float>([0, 0, 0], shape: [3])).scalars()
        let negative = (x / Tensor<Float>([-0.0, -0.0, -0.0], shape: [3])).scalars()
        #expect(positive == [.infinity, .infinity, .infinity])
        #expect(negative == [-.infinity, -.infinity, -.infinity])
    }
}

@Suite("Concurrent Materialization Tests", .enabled(if: PluginAvailability.cpu))
struct ConcurrentMaterializationTests {
    @Test("tensors read from many threads at once all materialize")
    func concurrentReads() {
        let threads = 8
        let rounds = 20
        let results = LockedArray(count: threads)
        DispatchQueue.concurrentPerform(iterations: threads) { t in
            var ok = true
            for r in 0..<rounds {
                let scale = Float(t * rounds + r)
                let x = Tensor<Float>([1, 2, 3], shape: [3]) * Tensor<Float>([scale], shape: [1]).broadcast(to: [3])
                ok = ok && x.sum().item() == 6 * scale
            }
            results.set(t, ok)
        }
        #expect(results.values == Array(repeating: true, count: threads))
    }

    @Test("one thread's failing tensor does not fail other threads' reads")
    func failuresStayWithTheirTensor() {
        let threads = 6
        let rounds = 20
        let results = LockedArray(count: threads)
        DispatchQueue.concurrentPerform(iterations: threads) { t in
            var ok = true
            for r in 0..<rounds {
                if t == 0 {
                    // Keeps putting an invalid tensor into shared barriers.
                    let bad = invalidReshape(of: Tensor<Float>([1, 2], shape: [2]))
                    ok = ok && (try? bad.fetchScalars()) == nil
                } else {
                    let scale = Float(t * rounds + r)
                    let x = Tensor<Float>([1, 2], shape: [2]) * Tensor<Float>([scale], shape: [1]).broadcast(to: [2])
                    ok = ok && (try? x.sum().fetchItem()) == 3 * scale
                }
            }
            results.set(t, ok)
        }
        #expect(results.values == Array(repeating: true, count: threads))
    }
}

/// A fixed-size array of Bools that threads can write to safely.
private final class LockedArray: @unchecked Sendable {
    private let lock = Foundation.NSLock()
    private var storage: [Bool]
    init(count: Int) { storage = Array(repeating: false, count: count) }
    func set(_ index: Int, _ value: Bool) { lock.lock(); storage[index] = value; lock.unlock() }
    var values: [Bool] { lock.lock(); defer { lock.unlock() }; return storage }
}

/// The `MaterializationError` thrown by `body`, or nil if it did not throw one.
/// (`#expect(throws:)` only returns the error on Swift 6.1+ toolchains.)
private func materializationError(_ body: () throws -> Void) -> MaterializationError? {
    do {
        try body()
        return nil
    } catch let error as MaterializationError {
        return error
    } catch {
        return nil
    }
}
