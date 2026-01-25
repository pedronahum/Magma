// Magma - Profiling Tests
// Tests for profiling and benchmarking utilities

import XCTest
@testable import Magma
@testable import LazyTensor

// MARK: - Timing Tests

final class TimingTests: XCTestCase {

    func testTimingCreation() {
        let timing = Timing(nanoseconds: 1_000_000_000)  // 1 second

        XCTAssertEqual(timing.nanoseconds, 1_000_000_000)
        XCTAssertEqual(timing.microseconds, 1_000_000, accuracy: 1)
        XCTAssertEqual(timing.milliseconds, 1_000, accuracy: 0.001)
        XCTAssertEqual(timing.seconds, 1.0, accuracy: 0.001)
    }

    func testTimingDescription() {
        let microsecondTiming = Timing(nanoseconds: 500_000)  // 0.5 ms = 500 µs
        XCTAssertTrue(microsecondTiming.description.contains("µs"))

        let millisecondTiming = Timing(nanoseconds: 50_000_000)  // 50 ms
        XCTAssertTrue(millisecondTiming.description.contains("ms"))

        let secondTiming = Timing(nanoseconds: 2_000_000_000)  // 2 seconds
        XCTAssertTrue(secondTiming.description.contains("s"))
    }

    func testTimingAddition() {
        let a = Timing(nanoseconds: 100)
        let b = Timing(nanoseconds: 200)
        let sum = a + b

        XCTAssertEqual(sum.nanoseconds, 300)
    }

    func testTimingZero() {
        let zero = Timing.zero
        XCTAssertEqual(zero.nanoseconds, 0)
        XCTAssertEqual(zero.milliseconds, 0)
    }
}

// MARK: - Profiler Timing Tests

final class ProfilerTimingTests: XCTestCase {

    func testTimedBlock() {
        let (result, timing) = Profiler.timed {
            // Simple computation
            var sum = 0
            for i in 0..<1000 {
                sum += i
            }
            return sum
        }

        XCTAssertEqual(result, 499500)  // Sum of 0..999
        XCTAssertTrue(timing.nanoseconds > 0, "Timing should be positive")
    }

    func testTimedTensorOperation() {
        let (result, timing) = Profiler.timed {
            let a = Tensor<Float>.ones([10, 10])
            let b = Tensor<Float>.ones([10, 10])
            return (a + b).scalars()
        }

        XCTAssertEqual(result.count, 100)
        XCTAssertEqual(result[0], 2.0, accuracy: 1e-5)
        XCTAssertTrue(timing.nanoseconds > 0)
    }

    func testTimedWithBarrier() {
        var outputTensor: Tensor<Float>!

        let (_, timing) = Profiler.timedWithBarrier({
            let a = Tensor<Float>.randn([100, 100])
            let b = Tensor<Float>.randn([100, 100])
            outputTensor = a.matmul(b)
        }, materialize: {
            [outputTensor!]
        })

        XCTAssertTrue(timing.nanoseconds > 0)
        XCTAssertEqual(outputTensor.shape, [100, 100])
    }
}

// MARK: - Benchmark Tests

final class BenchmarkTests: XCTestCase {

    func testBasicBenchmark() {
        let stats = Benchmark.measure(iterations: 10, warmup: 2) {
            // Simple work
            var sum = 0.0
            for i in 0..<100 {
                sum += Double(i)
            }
        }

        XCTAssertEqual(stats.iterations, 10)
        XCTAssertTrue(stats.mean.nanoseconds > 0)
        XCTAssertTrue(stats.min.nanoseconds <= stats.mean.nanoseconds)
        XCTAssertTrue(stats.max.nanoseconds >= stats.mean.nanoseconds)
        XCTAssertTrue(stats.median.nanoseconds > 0)
    }

    func testBenchmarkWithBarrier() {
        let stats = Benchmark.measureWithBarrier(iterations: 5, warmup: 1, {
            let a = Tensor<Float>.ones([50, 50])
            let b = Tensor<Float>.ones([50, 50])
            return a + b
        }, result: { $0 })

        XCTAssertEqual(stats.iterations, 5)
        XCTAssertTrue(stats.mean.nanoseconds > 0)
    }

    func testBenchmarkStatistics() {
        // Create synthetic timing data
        let timings: [UInt64] = [100, 200, 150, 300, 250]

        let stats = BenchmarkStats(timings: timings)

        XCTAssertEqual(stats.iterations, 5)
        XCTAssertEqual(stats.min.nanoseconds, 100)
        XCTAssertEqual(stats.max.nanoseconds, 300)

        // Mean should be 200
        XCTAssertEqual(stats.mean.nanoseconds, 200)

        // Median of sorted [100, 150, 200, 250, 300] is 200
        XCTAssertEqual(stats.median.nanoseconds, 200)
    }

    func testBenchmarkOpsPerSecond() {
        // If mean is 1ms, ops/sec should be 1000
        let timings: [UInt64] = [1_000_000]  // 1ms
        let stats = BenchmarkStats(timings: timings)

        XCTAssertEqual(stats.opsPerSecond, 1000, accuracy: 1)
    }

    func testCompare() throws {
        let results = try Benchmark.compare(
            iterations: 5,
            warmup: 1,
            implementations: [
                ("add", {
                    let a = Tensor<Float>.ones([10, 10])
                    let b = Tensor<Float>.ones([10, 10])
                    _ = (a + b).scalars()
                }),
                ("multiply", {
                    let a = Tensor<Float>.ones([10, 10])
                    let b = Tensor<Float>.ones([10, 10])
                    _ = (a * b).scalars()
                })
            ]
        )

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].name, "add")
        XCTAssertEqual(results[1].name, "multiply")
    }
}

// MARK: - ExecutionProfile Tests

final class ExecutionProfileTests: XCTestCase {

    func testExecutionProfileDescription() {
        let profile = ExecutionProfile(
            h2dTransfer: Timing(nanoseconds: 100_000),
            execution: Timing(nanoseconds: 500_000),
            d2hInitiate: Timing(nanoseconds: 50_000),
            d2hAwait: Timing(nanoseconds: 100_000),
            cleanup: Timing(nanoseconds: 25_000),
            total: Timing(nanoseconds: 775_000),
            inputCount: 2,
            outputCount: 1
        )

        let desc = profile.description
        XCTAssertTrue(desc.contains("ExecutionProfile"))
        XCTAssertTrue(desc.contains("H2D Transfer"))
        XCTAssertTrue(desc.contains("Execution"))
        XCTAssertTrue(desc.contains("Total"))
        XCTAssertTrue(desc.contains("Inputs"))
        XCTAssertTrue(desc.contains("Outputs"))
    }
}

// MARK: - FLOPS Estimator Tests

final class FLOPSEstimatorTests: XCTestCase {

    func testMatmulFLOPS() {
        // For a [100, 200] x [200, 300] matmul
        // FLOPS = 2 * 100 * 300 * 200 = 12,000,000
        let flops = FLOPSEstimator.matmul(m: 100, n: 300, k: 200)
        XCTAssertEqual(flops, 12_000_000)
    }

    func testConv2dFLOPS() {
        // Simple convolution: batch=1, in_c=3, out_c=64, 224x224 input, 3x3 kernel
        let flops = FLOPSEstimator.conv2d(
            batchSize: 1,
            inputChannels: 3,
            outputChannels: 64,
            inputHeight: 224,
            inputWidth: 224,
            kernelHeight: 3,
            kernelWidth: 3
        )

        // Output size: 222x222 (no padding, stride=1)
        // FLOPS per output: 2 * kernelH * kernelW * inputC = 2 * 3 * 3 * 3 = 54
        // Total: 1 * 64 * 222 * 222 * 54 = 170,325,504
        XCTAssertEqual(flops, 170_325_504)
    }

    func testGFLOPSCalculation() {
        // 1 billion ops in 1 second = 1 GFLOPS
        let gflops = FLOPSEstimator.gflops(
            operations: 1_000_000_000,
            timing: Timing(nanoseconds: 1_000_000_000)
        )
        XCTAssertEqual(gflops, 1.0, accuracy: 0.001)

        // 1 billion ops in 0.5 seconds = 2 GFLOPS
        let gflops2 = FLOPSEstimator.gflops(
            operations: 1_000_000_000,
            timing: Timing(nanoseconds: 500_000_000)
        )
        XCTAssertEqual(gflops2, 2.0, accuracy: 0.001)
    }
}

// MARK: - Memory Stats Tests

final class MemoryStatsTests: XCTestCase {

    func testMemoryStatsDescription() {
        let stats = MemoryStats(
            tensorBytes: 1024 * 1024,  // 1 MB
            tensorCount: 10,
            peakBytes: 2 * 1024 * 1024
        )

        XCTAssertEqual(stats.tensorMegabytes, 1.0, accuracy: 0.01)

        let desc = stats.description
        XCTAssertTrue(desc.contains("MemoryStats"))
        XCTAssertTrue(desc.contains("Tensors"))
        XCTAssertTrue(desc.contains("Usage"))
    }

    func testMemoryProfiler() {
        let (result, before, after) = MemoryProfiler.profile {
            let tensor = Tensor<Float>.randn([100, 100])
            return tensor.sum().scalars()[0]
        }

        // Result should be a valid float
        XCTAssertFalse(result.isNaN)
        XCTAssertFalse(result.isInfinite)

        // Just verify we got stats back
        XCTAssertTrue(before.tensorCount >= 0)
        XCTAssertTrue(after.tensorCount >= 0)
    }
}

// MARK: - Profiler API Tests

final class ProfilerAPITests: XCTestCase {

    func testProfilerAvailability() {
        // These should not crash, even if profiling isn't available
        _ = Profiler.isProfilingAvailable
        _ = Profiler.isTracingAvailable
    }

    func testProfilerScope() {
        // Test that scope executes the block correctly
        let result = Profiler.scope("test_scope") {
            42
        }
        XCTAssertEqual(result, 42)
    }

    func testProfilerStep() {
        // Test that step executes the block correctly
        var counter = 0
        Profiler.step("train", 0) {
            counter += 1
        }
        XCTAssertEqual(counter, 1)
    }

    func testProfilerInstant() {
        // Just verify this doesn't crash
        Profiler.instant("test_instant")
    }
}

// MARK: - Tensor Extension Tests

final class TensorProfilingExtensionTests: XCTestCase {

    func testTimedScalars() {
        let tensor = Tensor<Float>.ones([10, 10])
        let (values, timing) = tensor.timedScalars()

        XCTAssertEqual(values.count, 100)
        XCTAssertEqual(values[0], 1.0, accuracy: 1e-5)
        XCTAssertTrue(timing.nanoseconds > 0)
    }
}

// MARK: - Benchmark Stats Edge Cases

final class BenchmarkStatsEdgeCasesTests: XCTestCase {

    func testEmptyTimings() {
        let stats = BenchmarkStats(timings: [])

        XCTAssertEqual(stats.iterations, 0)
        XCTAssertEqual(stats.mean.nanoseconds, 0)
        XCTAssertEqual(stats.stdDev.nanoseconds, 0)
    }

    func testSingleTiming() {
        let stats = BenchmarkStats(timings: [1000])

        XCTAssertEqual(stats.iterations, 1)
        XCTAssertEqual(stats.mean.nanoseconds, 1000)
        XCTAssertEqual(stats.min.nanoseconds, 1000)
        XCTAssertEqual(stats.max.nanoseconds, 1000)
        XCTAssertEqual(stats.stdDev.nanoseconds, 0)  // Can't compute stddev with 1 sample
    }

    func testLargeTimings() {
        // Test with microsecond-scale timings (more realistic)
        let timings: [UInt64] = (0..<100).map { _ in
            UInt64.random(in: 1_000_000...10_000_000)  // 1-10ms
        }

        let stats = BenchmarkStats(timings: timings)

        XCTAssertEqual(stats.iterations, 100)
        XCTAssertTrue(stats.min.nanoseconds >= 1_000_000)
        XCTAssertTrue(stats.max.nanoseconds <= 10_000_000)
        XCTAssertTrue(stats.p95.nanoseconds >= stats.median.nanoseconds)
        XCTAssertTrue(stats.p99.nanoseconds >= stats.p95.nanoseconds)
    }
}
