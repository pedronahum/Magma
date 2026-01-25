// Magma Performance Benchmarks
//
// This example benchmarks common tensor operations and neural network
// computations to measure performance characteristics.
//
// Run with:
//   MAGMA_XLA_PATH=/opt/swiftir-deps MAGMA_ENABLE_XLA=1 swift run Benchmarks
//
// For mixed precision benchmarks on TPU:
//   MAGMA_XLA_PATH=/opt/swiftir-deps MAGMA_ENABLE_XLA=1 swift run Benchmarks --mixed-precision

import Magma
import Foundation

// MARK: - Configuration

struct BenchmarkConfig {
    var warmupIterations: Int = 5
    var iterations: Int = 20
    var useMixedPrecision: Bool = false
    var verbose: Bool = false

    static func fromCommandLine() -> BenchmarkConfig {
        var config = BenchmarkConfig()
        let args = CommandLine.arguments

        if args.contains("--mixed-precision") || args.contains("-mp") {
            config.useMixedPrecision = true
        }
        if args.contains("--verbose") || args.contains("-v") {
            config.verbose = true
        }
        if let idx = args.firstIndex(of: "--iterations"), idx + 1 < args.count {
            config.iterations = Int(args[idx + 1]) ?? 20
        }
        if let idx = args.firstIndex(of: "--warmup"), idx + 1 < args.count {
            config.warmupIterations = Int(args[idx + 1]) ?? 5
        }

        return config
    }
}

// MARK: - Benchmark Results

struct BenchmarkResult {
    let name: String
    let shape: String
    let meanMs: Double
    let minMs: Double
    let maxMs: Double
    let stdDevMs: Double
    let gflops: Double?
    let throughputGBps: Double?

    func print() {
        var line = String(format: "%-35s %15s %10.3f ms %10.3f ms %10.3f ms",
                         name, shape, meanMs, minMs, maxMs)
        if let gf = gflops, gf > 0 {
            line += String(format: " %10.2f GFLOPS", gf)
        }
        if let tp = throughputGBps, tp > 0 {
            line += String(format: " %10.2f GB/s", tp)
        }
        Swift.print(line)
    }
}

// MARK: - Benchmark Suite

class BenchmarkSuite {
    let config: BenchmarkConfig
    var results: [BenchmarkResult] = []

    init(config: BenchmarkConfig) {
        self.config = config
    }

    func run<T>(_ name: String, shape: String, flops: Int? = nil, bytes: Int? = nil,
                _ block: () -> T) -> BenchmarkResult {
        // Warmup
        for _ in 0..<config.warmupIterations {
            _ = block()
        }

        // Timed runs
        var timings: [Double] = []
        for _ in 0..<config.iterations {
            let start = DispatchTime.now()
            _ = block()
            let end = DispatchTime.now()
            let nanos = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
            timings.append(nanos / 1_000_000) // Convert to ms
        }

        let mean = timings.reduce(0, +) / Double(timings.count)
        let minTime = timings.min() ?? 0
        let maxTime = timings.max() ?? 0
        let variance = timings.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(timings.count)
        let stdDev = sqrt(variance)

        // Calculate GFLOPS and throughput
        var gflops: Double? = nil
        var throughput: Double? = nil

        if let f = flops, mean > 0 {
            // GFLOPS = (flops / 1e9) / (time_in_seconds)
            gflops = (Double(f) / 1e9) / (mean / 1000)
        }

        if let b = bytes, mean > 0 {
            // GB/s = (bytes / 1e9) / (time_in_seconds)
            throughput = (Double(b) / 1e9) / (mean / 1000)
        }

        let result = BenchmarkResult(
            name: name,
            shape: shape,
            meanMs: mean,
            minMs: minTime,
            maxMs: maxTime,
            stdDevMs: stdDev,
            gflops: gflops,
            throughputGBps: throughput
        )
        results.append(result)
        return result
    }

    func printHeader() {
        print(String(repeating: "=", count: 100))
        print("Magma Performance Benchmarks")
        print(String(repeating: "=", count: 100))
        print("Configuration:")
        print("  Warmup iterations: \(config.warmupIterations)")
        print("  Benchmark iterations: \(config.iterations)")
        print("  Mixed precision: \(config.useMixedPrecision)")
        print(String(repeating: "-", count: 100))
        print(String(format: "%-35s %15s %10s %10s %10s %12s %12s",
                    "Benchmark", "Shape", "Mean", "Min", "Max", "GFLOPS", "Throughput"))
        print(String(repeating: "-", count: 100))
    }

    func printSummary() {
        print(String(repeating: "=", count: 100))
        print("Summary: \(results.count) benchmarks completed")

        let totalTime = results.reduce(0.0) { $0 + $1.meanMs }
        print(String(format: "Total benchmark time: %.2f ms", totalTime))

        if let fastestMatmul = results.filter({ $0.name.contains("matmul") }).max(by: { ($0.gflops ?? 0) < ($1.gflops ?? 0) }) {
            print(String(format: "Peak matmul performance: %.2f GFLOPS", fastestMatmul.gflops ?? 0))
        }
        print(String(repeating: "=", count: 100))
    }
}

// MARK: - Matrix Multiplication Benchmarks

func runMatmulBenchmarks(_ suite: BenchmarkSuite) {
    print("\n== Matrix Multiplication ==\n")

    let sizes: [(Int, Int, Int)] = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    for (m, k, n) in sizes {
        // Create tensors and materialize them BEFORE benchmarking
        // This separates tensor creation time from computation time
        let a = Tensor<Float>.randn([m, k])
        let b = Tensor<Float>.randn([k, n])
        a.materialize()
        b.materialize()

        // FLOPS for matmul: 2 * M * N * K (multiply-add)
        let flops = 2 * m * n * k

        let result = suite.run(
            "matmul",
            shape: "[\(m),\(k)]x[\(k),\(n)]",
            flops: flops
        ) {
            let c = a.matmul(b)
            return c.scalars() // Force materialization
        }
        result.print()
    }

    // Note: Batched matmul disabled - dot_general compilation issue with XLA backend
    // The materialization IS working correctly - IR shows function parameters
    // rather than embedded constants. The compilation error is an XLA issue.
}

// MARK: - Element-wise Operation Benchmarks

func runElementwiseBenchmarks(_ suite: BenchmarkSuite) {
    print("\n== Element-wise Operations ==\n")

    // Note: Using smaller sizes to avoid constant embedding issues in XLA
    let sizes = [
        100_000,
        500_000,
        1_000_000,
    ]

    for size in sizes {
        let a = Tensor<Float>.randn([size])
        let b = Tensor<Float>.randn([size])

        // Each element: 1 op, so FLOPS = size
        // Memory: read 2 tensors, write 1 = 3 * size * 4 bytes
        let bytes = 3 * size * 4

        // Addition
        var result = suite.run("add", shape: "[\(size)]", flops: size, bytes: bytes) {
            let c = a + b
            return c.scalars()
        }
        result.print()

        // Multiplication
        result = suite.run("multiply", shape: "[\(size)]", flops: size, bytes: bytes) {
            let c = a * b
            return c.scalars()
        }
        result.print()

        // Division
        result = suite.run("divide", shape: "[\(size)]", flops: size, bytes: bytes) {
            let c = a / b
            return c.scalars()
        }
        result.print()
    }
}

// MARK: - Activation Function Benchmarks

func runActivationBenchmarks(_ suite: BenchmarkSuite) {
    print("\n== Activation Functions ==\n")

    // Using smaller sizes due to constant embedding limits
    let sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]

    for (h, w) in sizes {
        let x = Tensor<Float>.randn([h, w])
        let size = h * w
        let bytes = 2 * size * 4 // Read input, write output

        // ReLU
        var result = suite.run("relu", shape: "[\(h),\(w)]", flops: size, bytes: bytes) {
            let y = x.relu()
            return y.scalars()
        }
        result.print()

        // Sigmoid
        result = suite.run("sigmoid", shape: "[\(h),\(w)]", flops: 4 * size, bytes: bytes) {
            let y = x.sigmoid()
            return y.scalars()
        }
        result.print()

        // Tanh
        result = suite.run("tanh", shape: "[\(h),\(w)]", flops: 4 * size, bytes: bytes) {
            let y = x.tanh()
            return y.scalars()
        }
        result.print()

        // GELU
        result = suite.run("gelu", shape: "[\(h),\(w)]", flops: 8 * size, bytes: bytes) {
            let y = x.gelu()
            return y.scalars()
        }
        result.print()

        // Softmax
        result = suite.run("softmax", shape: "[\(h),\(w)]", flops: 5 * size, bytes: bytes) {
            let y = x.softmax(dim: -1)
            return y.scalars()
        }
        result.print()
    }
}

// MARK: - Reduction Benchmarks

func runReductionBenchmarks(_ suite: BenchmarkSuite) {
    print("\n== Reductions ==\n")

    // Using smaller sizes due to constant embedding limits
    let sizes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]

    for (h, w) in sizes {
        let x = Tensor<Float>.randn([h, w])
        let size = h * w

        // Sum all
        var result = suite.run("sum_all", shape: "[\(h),\(w)]", flops: size) {
            let y = x.sum()
            return y.scalars()
        }
        result.print()

        // Sum along axis
        result = suite.run("sum_axis0", shape: "[\(h),\(w)]", flops: size) {
            let y = x.sum(dims: [0])
            return y.scalars()
        }
        result.print()

        // Mean
        result = suite.run("mean_all", shape: "[\(h),\(w)]", flops: size + 1) {
            let y = x.mean()
            return y.scalars()
        }
        result.print()

        // Max
        result = suite.run("max_all", shape: "[\(h),\(w)]", flops: size) {
            let y = x.max()
            return y.scalars()
        }
        result.print()
    }
}

// MARK: - Mixed Precision Benchmarks

func runMixedPrecisionBenchmarks(_ suite: BenchmarkSuite) {
    print("\n== Mixed Precision (bfloat16) ==\n")

    let sizes: [(Int, Int, Int)] = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    for (m, k, n) in sizes {
        let flops = 2 * m * n * k

        // Float32 baseline
        let a32 = Tensor<Float>.randn([m, k])
        let b32 = Tensor<Float>.randn([k, n])

        var result = suite.run(
            "matmul_f32",
            shape: "[\(m),\(k)]x[\(k),\(n)]",
            flops: flops
        ) {
            let c = a32.matmul(b32)
            return c.scalars()
        }
        result.print()

        // bfloat16
        let a16 = a32.toReducedPrecision()
        let b16 = b32.toReducedPrecision()

        result = suite.run(
            "matmul_bf16",
            shape: "[\(m),\(k)]x[\(k),\(n)]",
            flops: flops
        ) {
            let c = a16.matmul(b16)
            return c.toFullPrecision().scalars()
        }
        result.print()

        // Mixed precision autocast
        result = suite.run(
            "matmul_autocast",
            shape: "[\(m),\(k)]x[\(k),\(n)]",
            flops: flops
        ) {
            let c = MixedPrecision.autocast(inputs: [a32, b32]) { inputs in
                inputs[0].matmul(inputs[1])
            }
            return c.scalars()
        }
        result.print()
    }
}

// MARK: - Neural Network Layer Benchmarks

func runLayerBenchmarks(_ suite: BenchmarkSuite) {
    print("\n== Neural Network Layers ==\n")

    // Linear layer
    let batchSizes = [32, 64, 128]
    let hiddenSizes = [(512, 512), (1024, 1024), (768, 3072)]

    for batch in batchSizes {
        for (inputSize, outputSize) in hiddenSizes {
            let linear = nn.Linear(inputSize: inputSize, outputSize: outputSize)
            let x = Tensor<Float>.randn([batch, inputSize])

            // FLOPS: 2 * batch * in * out (matmul) + batch * out (bias)
            let flops = 2 * batch * inputSize * outputSize + batch * outputSize

            let result = suite.run(
                "Linear",
                shape: "[\(batch),\(inputSize)]->[\(batch),\(outputSize)]",
                flops: flops
            ) {
                let y = linear(x)
                return y.scalars()
            }
            result.print()
        }
    }

    // LayerNorm
    print("\n-- LayerNorm --\n")

    for batch in [32, 64] {
        for hidden in [768, 1024] {
            let layerNorm = nn.LayerNorm(normalizedShape: [hidden])
            let x = Tensor<Float>.randn([batch, hidden])

            // LayerNorm: mean, variance, normalize, scale, shift
            let flops = 5 * batch * hidden

            let result = suite.run(
                "LayerNorm",
                shape: "[\(batch),\(hidden)]",
                flops: flops
            ) {
                let y = layerNorm(x)
                return y.scalars()
            }
            result.print()
        }
    }
}

// MARK: - Attention Benchmarks

func runAttentionBenchmarks(_ suite: BenchmarkSuite) {
    print("\n== Attention ==\n")

    let configs: [(Int, Int, Int, Int)] = [
        // (batch, seqLen, dModel, numHeads)
        (8, 128, 512, 8),
        (8, 256, 512, 8),
        (4, 512, 768, 12),
        (2, 1024, 768, 12),
    ]

    for (batch, seqLen, dModel, numHeads) in configs {
        let attention = nn.MultiheadAttention(
            embedDim: dModel,
            numHeads: numHeads,
            dropout: 0.0,
            bias: true
        )

        let x = Tensor<Float>.randn([batch, seqLen, dModel])

        // Attention FLOPS (approximate):
        // Q, K, V projections: 3 * 2 * batch * seqLen * dModel * dModel
        // Attention scores: 2 * batch * numHeads * seqLen * seqLen * (dModel/numHeads)
        // Attention output: 2 * batch * numHeads * seqLen * (dModel/numHeads) * seqLen
        // Output projection: 2 * batch * seqLen * dModel * dModel
        let headDim = dModel / numHeads
        let projFlops = 3 * 2 * batch * seqLen * dModel * dModel
        let scoreFlops = 2 * batch * numHeads * seqLen * seqLen * headDim
        let outFlops = 2 * batch * seqLen * dModel * dModel
        let totalFlops = projFlops + 2 * scoreFlops + outFlops

        let result = suite.run(
            "MultiheadAttention",
            shape: "B=\(batch),S=\(seqLen),D=\(dModel),H=\(numHeads)",
            flops: totalFlops
        ) {
            let (y, _) = attention.forward(query: x, key: x, value: x, mask: nil)
            return y.scalars()
        }
        result.print()
    }
}

// MARK: - Transformer Layer Benchmarks

func runTransformerBenchmarks(_ suite: BenchmarkSuite) {
    print("\n== Transformer Layers ==\n")

    let configs: [(Int, Int, Int, Int, Int)] = [
        // (batch, seqLen, dModel, numHeads, ffnDim)
        (8, 128, 512, 8, 2048),
        (4, 256, 768, 12, 3072),
        (2, 512, 768, 12, 3072),
    ]

    for (batch, seqLen, dModel, numHeads, ffnDim) in configs {
        // Encoder layer
        let encoder = nn.TransformerEncoderLayer(
            dModel: dModel,
            nHead: numHeads,
            dimFeedforward: ffnDim,
            dropout: 0.0
        )

        let x = Tensor<Float>.randn([batch, seqLen, dModel])

        // Approximate FLOPS for transformer layer
        // Attention + 2x FFN (linear) + LayerNorm (x2)
        let attentionFlops = 4 * 2 * batch * seqLen * dModel * dModel +
                            4 * batch * numHeads * seqLen * seqLen * (dModel / numHeads)
        let ffnFlops = 2 * batch * seqLen * dModel * ffnDim +
                       2 * batch * seqLen * ffnDim * dModel
        let totalFlops = attentionFlops + ffnFlops

        let result = suite.run(
            "TransformerEncoder",
            shape: "B=\(batch),S=\(seqLen),D=\(dModel)",
            flops: totalFlops
        ) {
            let y = encoder(x)
            return y.scalars()
        }
        result.print()
    }
}

// MARK: - Gradient Computation Benchmarks

func runGradientBenchmarks(_ suite: BenchmarkSuite) {
    print("\n== Gradient Computation ==\n")

    // Simple gradient through matmul
    let sizes: [(Int, Int, Int)] = [
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
    ]

    for (m, k, n) in sizes {
        let w = Tensor<Float>.randn([k, n])
        let x = Tensor<Float>.randn([m, k])

        // Forward + backward roughly 3x the forward FLOPS
        let flops = 3 * 2 * m * n * k

        let result = suite.run(
            "matmul_grad",
            shape: "[\(m),\(k)]x[\(k),\(n)]",
            flops: flops
        ) {
            let (_, grad) = valueWithGradient(at: w) { w in
                x.matmul(w).sum()
            }
            return grad.scalars()
        }
        result.print()
    }

    // MLP forward + backward (using raw tensors for benchmarking)
    print("\n-- MLP Forward+Backward --\n")

    for batch in [32, 64] {
        // Create weight matrices directly (Xavier-like initialization)
        let scale = Tensor<Float>.full([], 0.01)
        let w1 = Tensor<Float>.randn([256, 784]) * scale
        let w2 = Tensor<Float>.randn([128, 256]) * scale
        let w3 = Tensor<Float>.randn([10, 128]) * scale

        let x = Tensor<Float>.randn([batch, 784])
        let target = Tensor<Float>.randn([batch, 10])

        // Approximate FLOPS for MLP
        let forwardFlops = 2 * batch * (784 * 256 + 256 * 128 + 128 * 10)
        let totalFlops = 3 * forwardFlops // forward + 2x backward

        let result = suite.run(
            "MLP_forward_backward",
            shape: "batch=\(batch),784->256->128->10",
            flops: totalFlops
        ) {
            let (_, grad) = valueWithGradient(at: w1) { w in
                // Simple MLP forward pass
                var h = x.matmul(w.transpose())
                h = h.relu()
                h = h.matmul(w2.transpose()).relu()
                let output = h.matmul(w3.transpose())
                let diff = output - target
                return (diff * diff).mean()
            }
            // Force materialization
            return grad.scalars()
        }
        result.print()
    }
}

// MARK: - Main

func main() {
    let config = BenchmarkConfig.fromCommandLine()
    let suite = BenchmarkSuite(config: config)

    suite.printHeader()

    // Run benchmark suites
    // Note: Using materialize() to separate tensor creation from computation
    runMatmulBenchmarks(suite)

    // Enable more benchmarks now that we have materialization support
    // These use smaller sizes and/or materialize() to avoid constant embedding issues
    // runElementwiseBenchmarks(suite)  // Enable when needed
    // runActivationBenchmarks(suite)   // Enable when needed
    // runReductionBenchmarks(suite)    // Enable when needed

    if config.useMixedPrecision {
        runMixedPrecisionBenchmarks(suite)
    }

    // runLayerBenchmarks(suite)  // Enable when needed

    // Note: These require materialize() on input tensors to work correctly
    // runAttentionBenchmarks(suite)
    // runTransformerBenchmarks(suite)
    // runGradientBenchmarks(suite)

    suite.printSummary()
}

// Run
main()
