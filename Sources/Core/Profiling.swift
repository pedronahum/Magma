// Magma - Profiling Utilities
// Performance profiling and benchmarking with XLA integration
//
// This module provides:
// - Profiler: High-level profiling API compatible with XLA xprof
// - Benchmark: Utilities for benchmarking tensor operations
// - ExecutionTimer: Fine-grained timing for execution phases
//
// Usage:
// ```swift
// // Simple timing
// let (result, timing) = Profiler.timed {
//     model.forward(input)
// }
// print("Forward pass: \(timing.milliseconds)ms")
//
// // Detailed profiling
// try Profiler.profile(to: "/tmp/profile") {
//     for epoch in 0..<10 {
//         Profiler.step("train", epoch) {
//             train(model, data)
//         }
//     }
// }
//
// // Benchmarking
// let stats = Benchmark.measure(iterations: 100) {
//     tensor.matmul(weights)
// }
// print(stats)
// ```

import Foundation
import XLARuntime
import CXLARuntime
import LazyTensor
import StableHLO

// MARK: - Timing Utilities

/// High-resolution timing for operation profiling
public struct Timing: Sendable, CustomStringConvertible {
    /// Duration in nanoseconds
    public let nanoseconds: UInt64

    /// Duration in microseconds
    public var microseconds: Double {
        Double(nanoseconds) / 1_000.0
    }

    /// Duration in milliseconds
    public var milliseconds: Double {
        Double(nanoseconds) / 1_000_000.0
    }

    /// Duration in seconds
    public var seconds: Double {
        Double(nanoseconds) / 1_000_000_000.0
    }

    public var description: String {
        if milliseconds < 1 {
            return String(format: "%.2f µs", microseconds)
        } else if seconds < 1 {
            return String(format: "%.2f ms", milliseconds)
        } else {
            return String(format: "%.3f s", seconds)
        }
    }

    public init(nanoseconds: UInt64) {
        self.nanoseconds = nanoseconds
    }

    public static let zero = Timing(nanoseconds: 0)

    public static func + (lhs: Timing, rhs: Timing) -> Timing {
        Timing(nanoseconds: lhs.nanoseconds + rhs.nanoseconds)
    }
}

/// Detailed execution timing breakdown from PJRT
public struct ExecutionProfile: Sendable, CustomStringConvertible {
    /// Time to create input buffers (H2D transfer)
    public let h2dTransfer: Timing

    /// Time for kernel execution
    public let execution: Timing

    /// Time to initiate D2H transfers
    public let d2hInitiate: Timing

    /// Time waiting for D2H completion
    public let d2hAwait: Timing

    /// Time for buffer cleanup
    public let cleanup: Timing

    /// Total end-to-end time
    public let total: Timing

    /// Number of input tensors
    public let inputCount: Int

    /// Number of output tensors
    public let outputCount: Int

    public var description: String {
        var lines: [String] = []
        lines.append("ExecutionProfile:")
        lines.append("  H2D Transfer:  \(h2dTransfer)")
        lines.append("  Execution:     \(execution)")
        lines.append("  D2H Initiate:  \(d2hInitiate)")
        lines.append("  D2H Await:     \(d2hAwait)")
        lines.append("  Cleanup:       \(cleanup)")
        lines.append("  ─────────────────")
        lines.append("  Total:         \(total)")
        lines.append("  Inputs:        \(inputCount)")
        lines.append("  Outputs:       \(outputCount)")
        return lines.joined(separator: "\n")
    }

    /// Create from PJRT ExecutionTiming
    internal init(from timing: ExecutionTiming) {
        self.h2dTransfer = Timing(nanoseconds: timing.h2dCreateNs)
        self.execution = Timing(nanoseconds: timing.executeNs)
        self.d2hInitiate = Timing(nanoseconds: timing.d2hInitiateNs)
        self.d2hAwait = Timing(nanoseconds: timing.d2hAwaitNs)
        self.cleanup = Timing(nanoseconds: timing.bufferDestroyNs)
        self.total = Timing(nanoseconds: timing.totalNs)
        self.inputCount = timing.numInputs
        self.outputCount = timing.numOutputs
    }

    public init(
        h2dTransfer: Timing = .zero,
        execution: Timing = .zero,
        d2hInitiate: Timing = .zero,
        d2hAwait: Timing = .zero,
        cleanup: Timing = .zero,
        total: Timing = .zero,
        inputCount: Int = 0,
        outputCount: Int = 0
    ) {
        self.h2dTransfer = h2dTransfer
        self.execution = execution
        self.d2hInitiate = d2hInitiate
        self.d2hAwait = d2hAwait
        self.cleanup = cleanup
        self.total = total
        self.inputCount = inputCount
        self.outputCount = outputCount
    }
}

// MARK: - Benchmark Statistics

/// Statistics from a benchmark run
public struct BenchmarkStats: Sendable, CustomStringConvertible {
    /// Number of iterations run
    public let iterations: Int

    /// Total time for all iterations
    public let totalTime: Timing

    /// Mean time per iteration
    public let mean: Timing

    /// Minimum time
    public let min: Timing

    /// Maximum time
    public let max: Timing

    /// Standard deviation
    public let stdDev: Timing

    /// Median time
    public let median: Timing

    /// 95th percentile
    public let p95: Timing

    /// 99th percentile
    public let p99: Timing

    /// Throughput in operations per second
    public var opsPerSecond: Double {
        guard mean.nanoseconds > 0 else { return 0 }
        return 1_000_000_000.0 / Double(mean.nanoseconds)
    }

    public var description: String {
        var lines: [String] = []
        lines.append("BenchmarkStats (\(iterations) iterations):")
        lines.append("  Mean:     \(mean)")
        lines.append("  Min:      \(min)")
        lines.append("  Max:      \(max)")
        lines.append("  Std Dev:  \(stdDev)")
        lines.append("  Median:   \(median)")
        lines.append("  P95:      \(p95)")
        lines.append("  P99:      \(p99)")
        lines.append("  Throughput: \(String(format: "%.2f", opsPerSecond)) ops/sec")
        return lines.joined(separator: "\n")
    }

    internal init(timings: [UInt64]) {
        let sorted = timings.sorted()
        self.iterations = timings.count

        let total = timings.reduce(0, +)
        self.totalTime = Timing(nanoseconds: total)

        let meanNs = timings.count > 0 ? total / UInt64(timings.count) : 0
        self.mean = Timing(nanoseconds: meanNs)

        self.min = Timing(nanoseconds: sorted.first ?? 0)
        self.max = Timing(nanoseconds: sorted.last ?? 0)

        self.median = Timing(nanoseconds: sorted.count > 0 ? sorted[sorted.count / 2] : 0)

        let p95Index = Int(Double(sorted.count) * 0.95)
        self.p95 = Timing(nanoseconds: sorted.count > p95Index ? sorted[p95Index] : sorted.last ?? 0)

        let p99Index = Int(Double(sorted.count) * 0.99)
        self.p99 = Timing(nanoseconds: sorted.count > p99Index ? sorted[p99Index] : sorted.last ?? 0)

        // Calculate standard deviation
        if timings.count > 1 {
            let meanDouble = Double(meanNs)
            let variance = timings.map { pow(Double($0) - meanDouble, 2) }.reduce(0, +) / Double(timings.count - 1)
            self.stdDev = Timing(nanoseconds: UInt64(Foundation.sqrt(variance)))
        } else {
            self.stdDev = .zero
        }
    }
}

// MARK: - Profiler

/// High-level profiling API for Magma operations
///
/// Profiler provides utilities for timing operations, profiling to xprof-compatible
/// output, and benchmarking tensor operations.
public enum Profiler {

    // MARK: - Simple Timing

    /// Time a block of code and return both result and timing
    ///
    /// Example:
    /// ```swift
    /// let (result, timing) = Profiler.timed {
    ///     tensor.matmul(weights).relu()
    /// }
    /// print("Operation took: \(timing)")
    /// ```
    @discardableResult
    public static func timed<T>(_ block: () throws -> T) rethrows -> (result: T, timing: Timing) {
        let start = DispatchTime.now()
        let result = try block()
        let end = DispatchTime.now()
        let ns = end.uptimeNanoseconds - start.uptimeNanoseconds
        return (result, Timing(nanoseconds: ns))
    }

    /// Time a block of code, forcing tensor materialization
    ///
    /// This ensures lazy operations are executed and timed.
    /// Pass tensors to materialize after the block executes.
    @discardableResult
    public static func timedWithBarrier<T>(
        _ block: () throws -> T,
        materialize tensors: () -> [Tensor<Float>]
    ) rethrows -> (result: T, timing: Timing) {
        let start = DispatchTime.now()
        let result = try block()

        // Force materialization
        for tensor in tensors() {
            _ = tensor.scalars()
        }

        let end = DispatchTime.now()
        let ns = end.uptimeNanoseconds - start.uptimeNanoseconds
        return (result, Timing(nanoseconds: ns))
    }

    // MARK: - Async Timing

    /// Time an async block of code
    @available(macOS 10.15, iOS 13.0, watchOS 6.0, tvOS 13.0, *)
    @discardableResult
    public static func timed<T>(_ block: () async throws -> T) async rethrows -> (result: T, timing: Timing) {
        let start = DispatchTime.now()
        let result = try await block()
        let end = DispatchTime.now()
        let ns = end.uptimeNanoseconds - start.uptimeNanoseconds
        return (result, Timing(nanoseconds: ns))
    }

    // MARK: - Step Markers

    /// Mark a training step for profiling
    ///
    /// This creates trace markers that appear in xprof/TensorBoard.
    ///
    /// Example:
    /// ```swift
    /// for epoch in 0..<epochs {
    ///     Profiler.step("train", epoch) {
    ///         train(model, data)
    ///     }
    /// }
    /// ```
    @discardableResult
    public static func step<T>(_ name: String, _ stepNum: Int, _ block: () throws -> T) rethrows -> T {
        // Use TraceMe API if available
        let activityId = PJRT_TraceMeStart("\(name)#step_num=\(stepNum),_r=1#", 1)
        defer { PJRT_TraceMeStop(activityId) }
        return try block()
    }

    /// Mark a named scope in the trace
    @discardableResult
    public static func scope<T>(_ name: String, level: Int32 = 1, _ block: () throws -> T) rethrows -> T {
        let activityId = PJRT_TraceMeStart(name, level)
        defer { PJRT_TraceMeStop(activityId) }
        return try block()
    }

    // MARK: - Profile Session

    /// Check if XLA profiling extension is available
    public static var isProfilingAvailable: Bool {
        PJRT_HasProfilerExtension()
    }

    /// Check if TraceMe API is available
    public static var isTracingAvailable: Bool {
        PJRT_HasTraceMeApi()
    }

    /// Check if tracing is active at the given level
    public static func isTracingActive(level: Int32 = 1) -> Bool {
        PJRT_TraceMeActive(level)
    }

    /// Record an instant event (point in time, no duration)
    public static func instant(_ name: String, level: Int32 = 1) {
        PJRT_TraceMeInstant(name, level)
    }
}

// MARK: - Benchmark

/// Benchmarking utilities for tensor operations
public enum Benchmark {

    /// Run a benchmark with multiple iterations
    ///
    /// Example:
    /// ```swift
    /// let stats = Benchmark.measure(iterations: 100, warmup: 5) {
    ///     tensor.matmul(weights)
    /// }
    /// print(stats)
    /// ```
    public static func measure(
        iterations: Int = 100,
        warmup: Int = 5,
        _ block: () throws -> Void
    ) rethrows -> BenchmarkStats {
        // Warmup runs
        for _ in 0..<warmup {
            try block()
        }

        // Timed runs
        var timings: [UInt64] = []
        timings.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let start = DispatchTime.now()
            try block()
            let end = DispatchTime.now()
            timings.append(end.uptimeNanoseconds - start.uptimeNanoseconds)
        }

        return BenchmarkStats(timings: timings)
    }

    /// Run a benchmark that returns a result, forcing materialization
    ///
    /// The result function is called to get the tensor that should be materialized.
    /// This ensures lazy operations are fully executed.
    public static func measureWithBarrier<T>(
        iterations: Int = 100,
        warmup: Int = 5,
        _ block: () throws -> T,
        result: (T) -> Tensor<Float>
    ) rethrows -> BenchmarkStats {
        // Warmup runs
        for _ in 0..<warmup {
            let r = try block()
            _ = result(r).scalars()
        }

        // Timed runs
        var timings: [UInt64] = []
        timings.reserveCapacity(iterations)

        for _ in 0..<iterations {
            let start = DispatchTime.now()
            let r = try block()
            _ = result(r).scalars()  // Force materialization
            let end = DispatchTime.now()
            timings.append(end.uptimeNanoseconds - start.uptimeNanoseconds)
        }

        return BenchmarkStats(timings: timings)
    }

    /// Compare multiple implementations
    ///
    /// Example:
    /// ```swift
    /// let results = Benchmark.compare(
    ///     iterations: 100,
    ///     implementations: [
    ///         ("matmul", { tensor.matmul(weights) }),
    ///         ("einsum", { einsum("ij,jk->ik", tensor, weights) })
    ///     ]
    /// )
    /// for (name, stats) in results {
    ///     print("\(name): \(stats.mean)")
    /// }
    /// ```
    public static func compare(
        iterations: Int = 100,
        warmup: Int = 5,
        implementations: [(name: String, block: () throws -> Void)]
    ) throws -> [(name: String, stats: BenchmarkStats)] {
        var results: [(name: String, stats: BenchmarkStats)] = []

        for (name, block) in implementations {
            let stats = try measure(iterations: iterations, warmup: warmup, block)
            results.append((name, stats))
        }

        return results
    }

    /// Print a comparison table
    public static func printComparison(_ results: [(name: String, stats: BenchmarkStats)]) {
        guard let baseline = results.first else { return }

        print("\nBenchmark Comparison:")
        print("────────────────────────────────────────────────────────")
        print(String(format: "%-20s %12s %12s %12s", "Name", "Mean", "Std Dev", "vs Baseline"))
        print("────────────────────────────────────────────────────────")

        for (name, stats) in results {
            let ratio = Double(stats.mean.nanoseconds) / Double(baseline.stats.mean.nanoseconds)
            let ratioStr = String(format: "%.2fx", ratio)
            print(String(format: "%-20s %12s %12s %12s", name, "\(stats.mean)", "\(stats.stdDev)", ratioStr))
        }
        print("────────────────────────────────────────────────────────")
    }
}

// MARK: - Memory Profiling

/// Memory usage statistics
public struct MemoryStats: Sendable, CustomStringConvertible {
    /// Total bytes allocated for tensors
    public let tensorBytes: Int

    /// Number of active tensors
    public let tensorCount: Int

    /// Peak memory usage (if tracked)
    public let peakBytes: Int?

    public var tensorMegabytes: Double {
        Double(tensorBytes) / (1024 * 1024)
    }

    public var description: String {
        var lines: [String] = []
        lines.append("MemoryStats:")
        lines.append("  Tensors:     \(tensorCount)")
        lines.append("  Usage:       \(String(format: "%.2f", tensorMegabytes)) MB")
        if let peak = peakBytes {
            lines.append("  Peak:        \(String(format: "%.2f", Double(peak) / (1024*1024))) MB")
        }
        return lines.joined(separator: "\n")
    }

    public init(tensorBytes: Int, tensorCount: Int, peakBytes: Int? = nil) {
        self.tensorBytes = tensorBytes
        self.tensorCount = tensorCount
        self.peakBytes = peakBytes
    }
}

/// Memory profiling utilities
public enum MemoryProfiler {

    /// Get current memory statistics from tensor registry
    public static func currentStats() -> MemoryStats {
        let registry = TensorRegistry.shared
        let count = registry.pendingCount
        // Estimate bytes based on pending tensors (we don't have exact tracking)
        // This is a rough estimate
        return MemoryStats(tensorBytes: 0, tensorCount: count, peakBytes: nil)
    }

    /// Profile memory usage of a block
    public static func profile<T>(_ block: () throws -> T) rethrows -> (result: T, before: MemoryStats, after: MemoryStats) {
        let before = currentStats()
        let result = try block()
        let after = currentStats()
        return (result, before, after)
    }
}

// MARK: - FLOPS Estimation

/// FLOPS (Floating Point Operations Per Second) calculation utilities
public enum FLOPSEstimator {

    /// Estimate FLOPS for a matrix multiplication
    public static func matmul(m: Int, n: Int, k: Int) -> Int {
        // Each output element requires k multiplications and k-1 additions
        // Total: m * n * (2k - 1) ≈ 2 * m * n * k for large k
        return 2 * m * n * k
    }

    /// Estimate FLOPS for a convolution
    public static func conv2d(
        batchSize: Int,
        inputChannels: Int,
        outputChannels: Int,
        inputHeight: Int,
        inputWidth: Int,
        kernelHeight: Int,
        kernelWidth: Int,
        stride: Int = 1,
        padding: Int = 0
    ) -> Int {
        let outputHeight = (inputHeight + 2 * padding - kernelHeight) / stride + 1
        let outputWidth = (inputWidth + 2 * padding - kernelWidth) / stride + 1

        // Each output element requires kernelH * kernelW * inputC multiplications
        // and same number of additions (minus 1)
        let flopsPerOutput = 2 * kernelHeight * kernelWidth * inputChannels
        return batchSize * outputChannels * outputHeight * outputWidth * flopsPerOutput
    }

    /// Calculate effective GFLOPS given operation count and timing
    public static func gflops(operations: Int, timing: Timing) -> Double {
        guard timing.nanoseconds > 0 else { return 0 }
        let flopsPerSecond = Double(operations) * 1_000_000_000.0 / Double(timing.nanoseconds)
        return flopsPerSecond / 1_000_000_000.0  // Convert to GFLOPS
    }

    /// Benchmark and report GFLOPS for matrix multiplication
    public static func benchmarkMatmul(m: Int, n: Int, k: Int, iterations: Int = 100) -> (stats: BenchmarkStats, gflops: Double) {
        let a = Tensor<Float>.randn([m, k])
        let b = Tensor<Float>.randn([k, n])

        let stats = Benchmark.measureWithBarrier(iterations: iterations, warmup: 5, {
            a.matmul(b)
        }, result: { $0 })

        let ops = matmul(m: m, n: n, k: k)
        let effectiveGflops = gflops(operations: ops, timing: stats.mean)

        return (stats, effectiveGflops)
    }
}

// MARK: - Convenience Extensions

extension Tensor {

    /// Time the materialization of this tensor
    public func timedScalars() -> (values: [Scalar], timing: Timing) {
        let (result, timing) = Profiler.timed {
            self.scalars()
        }
        return (values: result, timing: timing)
    }
}
