// Magma - Profiling Tests
// Tests for profiling and benchmarking utilities

import Testing
@testable import Magma
@testable import LazyTensor

// MARK: - Timing Tests

@Suite("Timing Tests")
struct TimingTests {

    @Test("Timing creation")
    func timingCreation() {
        let timing = Timing(nanoseconds: 1_000_000_000)  // 1 second

        #expect(timing.nanoseconds == 1_000_000_000)
        #expect(abs(timing.microseconds - 1_000_000) < 1)
        #expect(abs(timing.milliseconds - 1_000) < 0.001)
        #expect(abs(timing.seconds - 1.0) < 0.001)
    }

    @Test("Timing description")
    func timingDescription() {
        let microsecondTiming = Timing(nanoseconds: 500_000)  // 0.5 ms = 500 µs
        #expect(microsecondTiming.description.contains("µs"))

        let millisecondTiming = Timing(nanoseconds: 50_000_000)  // 50 ms
        #expect(millisecondTiming.description.contains("ms"))

        let secondTiming = Timing(nanoseconds: 2_000_000_000)  // 2 seconds
        #expect(secondTiming.description.contains("s"))
    }

    @Test("Timing addition")
    func timingAddition() {
        let a = Timing(nanoseconds: 100)
        let b = Timing(nanoseconds: 200)
        let sum = a + b

        #expect(sum.nanoseconds == 300)
    }

    @Test("Timing zero")
    func timingZero() {
        let zero = Timing.zero
        #expect(zero.nanoseconds == 0)
        #expect(zero.milliseconds == 0)
    }
}

// MARK: - Profiler Timing Tests

@Suite("Profiler Timing Tests")
struct ProfilerTimingTests {

    @Test("Timed block")
    func timedBlock() {
        let (result, timing) = Profiler.timed {
            // Simple computation
            var sum = 0
            for i in 0..<1000 {
                sum += i
            }
            return sum
        }

        #expect(result == 499500)  // Sum of 0..999
        #expect(timing.nanoseconds > 0)
    }

    @Test("Timed tensor operation")
    func timedTensorOperation() {
        let (result, timing) = Profiler.timed {
            let a = Tensor<Float>.ones([10, 10])
            let b = Tensor<Float>.ones([10, 10])
            return (a + b).scalars()
        }

        #expect(result.count == 100)
        #expect(abs(result[0] - 2.0) < 1e-5)
        #expect(timing.nanoseconds > 0)
    }

    @Test("Timed with barrier")
    func timedWithBarrier() {
        var outputTensor: Tensor<Float>!

        let (_, timing) = Profiler.timedWithBarrier({
            let a = Tensor<Float>.randn([100, 100])
            let b = Tensor<Float>.randn([100, 100])
            outputTensor = a.matmul(b)
        }, materialize: {
            [outputTensor!]
        })

        #expect(timing.nanoseconds > 0)
        #expect(outputTensor.shape == [100, 100])
    }
}

// MARK: - Benchmark Tests

@Suite("Benchmark Tests")
struct BenchmarkTests {

    @Test("Basic benchmark")
    func basicBenchmark() {
        let stats = Benchmark.measure(iterations: 10, warmup: 2) {
            // Simple work
            var sum = 0.0
            for i in 0..<100 {
                sum += Double(i)
            }
        }

        #expect(stats.iterations == 10)
        #expect(stats.mean.nanoseconds > 0)
        #expect(stats.min.nanoseconds <= stats.mean.nanoseconds)
        #expect(stats.max.nanoseconds >= stats.mean.nanoseconds)
        #expect(stats.median.nanoseconds > 0)
    }

    @Test("Benchmark with barrier")
    func benchmarkWithBarrier() {
        let stats = Benchmark.measureWithBarrier(iterations: 5, warmup: 1, {
            let a = Tensor<Float>.ones([50, 50])
            let b = Tensor<Float>.ones([50, 50])
            return a + b
        }, result: { $0 })

        #expect(stats.iterations == 5)
        #expect(stats.mean.nanoseconds > 0)
    }

    @Test("Benchmark statistics")
    func benchmarkStatistics() {
        // Create synthetic timing data
        let timings: [UInt64] = [100, 200, 150, 300, 250]

        let stats = BenchmarkStats(timings: timings)

        #expect(stats.iterations == 5)
        #expect(stats.min.nanoseconds == 100)
        #expect(stats.max.nanoseconds == 300)

        // Mean should be 200
        #expect(stats.mean.nanoseconds == 200)

        // Median of sorted [100, 150, 200, 250, 300] is 200
        #expect(stats.median.nanoseconds == 200)
    }

    @Test("Benchmark ops per second")
    func benchmarkOpsPerSecond() {
        // If mean is 1ms, ops/sec should be 1000
        let timings: [UInt64] = [1_000_000]  // 1ms
        let stats = BenchmarkStats(timings: timings)

        #expect(abs(stats.opsPerSecond - 1000) < 1)
    }

    @Test("Compare")
    func compare() throws {
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

        #expect(results.count == 2)
        #expect(results[0].name == "add")
        #expect(results[1].name == "multiply")
    }
}

// MARK: - ExecutionProfile Tests

@Suite("ExecutionProfile Tests")
struct ExecutionProfileTests {

    @Test("Execution profile description")
    func executionProfileDescription() {
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
        #expect(desc.contains("ExecutionProfile"))
        #expect(desc.contains("H2D Transfer"))
        #expect(desc.contains("Execution"))
        #expect(desc.contains("Total"))
        #expect(desc.contains("Inputs"))
        #expect(desc.contains("Outputs"))
    }
}

// MARK: - FLOPS Estimator Tests

@Suite("FLOPS Estimator Tests")
struct FLOPSEstimatorTests {

    @Test("Matmul FLOPS")
    func matmulFLOPS() {
        // For a [100, 200] x [200, 300] matmul
        // FLOPS = 2 * 100 * 300 * 200 = 12,000,000
        let flops = FLOPSEstimator.matmul(m: 100, n: 300, k: 200)
        #expect(flops == 12_000_000)
    }

    @Test("Conv2d FLOPS")
    func conv2dFLOPS() {
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
        #expect(flops == 170_325_504)
    }

    @Test("GFLOPS calculation")
    func gflopsCalculation() {
        // 1 billion ops in 1 second = 1 GFLOPS
        let gflops = FLOPSEstimator.gflops(
            operations: 1_000_000_000,
            timing: Timing(nanoseconds: 1_000_000_000)
        )
        #expect(abs(gflops - 1.0) < 0.001)

        // 1 billion ops in 0.5 seconds = 2 GFLOPS
        let gflops2 = FLOPSEstimator.gflops(
            operations: 1_000_000_000,
            timing: Timing(nanoseconds: 500_000_000)
        )
        #expect(abs(gflops2 - 2.0) < 0.001)
    }
}

// MARK: - Memory Stats Tests

@Suite("Memory Stats Tests")
struct MemoryStatsTests {

    @Test("Memory stats description")
    func memoryStatsDescription() {
        let stats = MemoryStats(
            tensorBytes: 1024 * 1024,  // 1 MB
            tensorCount: 10,
            peakBytes: 2 * 1024 * 1024
        )

        #expect(abs(stats.tensorMegabytes - 1.0) < 0.01)

        let desc = stats.description
        #expect(desc.contains("MemoryStats"))
        #expect(desc.contains("Tensors"))
        #expect(desc.contains("Usage"))
    }

    @Test("Memory profiler")
    func memoryProfiler() {
        let (result, before, after) = MemoryProfiler.profile {
            let tensor = Tensor<Float>.randn([100, 100])
            return tensor.sum().scalars()[0]
        }

        // Result should be a valid float
        #expect(!result.isNaN)
        #expect(!result.isInfinite)

        // Just verify we got stats back
        #expect(before.tensorCount >= 0)
        #expect(after.tensorCount >= 0)
    }
}

// MARK: - Profiler API Tests

@Suite("Profiler API Tests")
struct ProfilerAPITests {

    @Test("Profiler availability")
    func profilerAvailability() {
        // These should not crash, even if profiling isn't available
        _ = Profiler.isProfilingAvailable
        _ = Profiler.isTracingAvailable
    }

    @Test("Profiler scope")
    func profilerScope() {
        // Test that scope executes the block correctly
        let result = Profiler.scope("test_scope") {
            42
        }
        #expect(result == 42)
    }

    @Test("Profiler step")
    func profilerStep() {
        // Test that step executes the block correctly
        var counter = 0
        Profiler.step("train", 0) {
            counter += 1
        }
        #expect(counter == 1)
    }

    @Test("Profiler instant")
    func profilerInstant() {
        // Just verify this doesn't crash
        Profiler.instant("test_instant")
    }
}

// MARK: - Tensor Extension Tests

@Suite("Tensor Profiling Extension Tests")
struct TensorProfilingExtensionTests {

    @Test("Timed scalars")
    func timedScalars() {
        let tensor = Tensor<Float>.ones([10, 10])
        let (values, timing) = tensor.timedScalars()

        #expect(values.count == 100)
        #expect(abs(values[0] - 1.0) < 1e-5)
        #expect(timing.nanoseconds > 0)
    }
}

// MARK: - Benchmark Stats Edge Cases

@Suite("Benchmark Stats Edge Cases Tests")
struct BenchmarkStatsEdgeCasesTests {

    @Test("Empty timings")
    func emptyTimings() {
        let stats = BenchmarkStats(timings: [])

        #expect(stats.iterations == 0)
        #expect(stats.mean.nanoseconds == 0)
        #expect(stats.stdDev.nanoseconds == 0)
    }

    @Test("Single timing")
    func singleTiming() {
        let stats = BenchmarkStats(timings: [1000])

        #expect(stats.iterations == 1)
        #expect(stats.mean.nanoseconds == 1000)
        #expect(stats.min.nanoseconds == 1000)
        #expect(stats.max.nanoseconds == 1000)
        #expect(stats.stdDev.nanoseconds == 0)  // Can't compute stddev with 1 sample
    }

    @Test("Large timings")
    func largeTimings() {
        // Test with microsecond-scale timings (more realistic)
        let timings: [UInt64] = (0..<100).map { _ in
            UInt64.random(in: 1_000_000...10_000_000)  // 1-10ms
        }

        let stats = BenchmarkStats(timings: timings)

        #expect(stats.iterations == 100)
        #expect(stats.min.nanoseconds >= 1_000_000)
        #expect(stats.max.nanoseconds <= 10_000_000)
        #expect(stats.p95.nanoseconds >= stats.median.nanoseconds)
        #expect(stats.p99.nanoseconds >= stats.p95.nanoseconds)
    }
}
