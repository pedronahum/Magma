// MagmaBenchmark - Comprehensive Operation Benchmark Suite
// Compares Magma tensor operations overhead against direct MetalHLO execution

import Foundation
import Magma
import LazyTensor
import XLARuntime

#if canImport(MetalHLO)
import MetalHLO

// MARK: - Benchmark Infrastructure

/// Result of a single benchmark run
struct BenchmarkResult {
    let name: String
    let category: String
    let operation: String
    let shape: String
    let iterations: Int
    let meanMs: Double
    let stdDevMs: Double
    let minMs: Double
    let maxMs: Double

    var summary: String {
        String(format: "%@ | %@ | mean: %.3f ms | std: %.3f ms | min: %.3f ms",
               name.padding(toLength: 28, withPad: " ", startingAt: 0),
               shape.padding(toLength: 16, withPad: " ", startingAt: 0),
               meanMs, stdDevMs, minMs)
    }
}

/// Benchmark configuration
struct BenchmarkConfig {
    let warmupIterations: Int
    let measurementIterations: Int

    static let quick = BenchmarkConfig(warmupIterations: 2, measurementIterations: 5)
    static let standard = BenchmarkConfig(warmupIterations: 5, measurementIterations: 20)
}

/// Run a benchmark with pre-materialized inputs
func runBenchmark(
    name: String,
    category: String,
    operation: String,
    shape: String,
    config: BenchmarkConfig,
    device: Device,
    computation: @escaping () -> Void
) -> BenchmarkResult {
    // Warmup
    for _ in 0..<config.warmupIterations {
        computation()
    }

    // Measure
    var times: [Double] = []
    for _ in 0..<config.measurementIterations {
        let start = CFAbsoluteTimeGetCurrent()
        computation()
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        times.append(elapsed * 1000) // ms
    }

    // Statistics
    let mean = times.reduce(0, +) / Double(times.count)
    let variance = times.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(times.count)
    let stdDev = Foundation.sqrt(variance)
    let minTime = times.min() ?? 0
    let maxTime = times.max() ?? 0

    return BenchmarkResult(
        name: name,
        category: category,
        operation: operation,
        shape: shape,
        iterations: config.measurementIterations,
        meanMs: mean,
        stdDevMs: stdDev,
        minMs: minTime,
        maxMs: maxTime
    )
}

// MARK: - Main

setbuf(stdout, nil)
print("╔════════════════════════════════════════════════════════════╗")
print("║         Magma Comprehensive Benchmark Suite                ║")
print("╚════════════════════════════════════════════════════════════╝")
print()

guard Backend.metal.isAvailable else {
    print("ERROR: Metal backend not available")
    exit(1)
}

print("Device: \(MetalBackend.deviceName ?? "unknown")")
print()

let metal = Device(backend: .metal, index: 0)

// Parse arguments
let args = CommandLine.arguments
let quickMode = args.contains("-q") || args.contains("--quick")
let config: BenchmarkConfig = quickMode ? .quick : .standard

print("Mode: \(quickMode ? "Quick" : "Standard") (\(config.warmupIterations) warmup, \(config.measurementIterations) measurements)")
print()

var results: [BenchmarkResult] = []

// MARK: - Matrix Operations

print("═══════════════════════════════════════════════════════════════")
print("MATRIX OPERATIONS")
print("═══════════════════════════════════════════════════════════════")

// GEMM benchmarks - including larger sizes
let gemmConfigs: [(name: String, m: Int, n: Int, k: Int)] = [
    ("GEMM-32", 32, 32, 32),
    ("GEMM-64", 64, 64, 64),
    ("GEMM-128", 128, 128, 128),
    ("GEMM-256", 256, 256, 256),
    ("GEMM-512", 512, 512, 512),
    ("GEMM-Tall", 512, 64, 256),
    ("GEMM-Wide", 64, 512, 256),
]

for cfg in gemmConfigs {
    let a = Tensor<Float>.ones([cfg.m, cfg.k], on: metal)
    let b = Tensor<Float>.ones([cfg.k, cfg.n], on: metal)
    a.markForMaterialization()
    b.markForMaterialization()
    LazyTensorBarrier(on: metal)

    let result = runBenchmark(
        name: cfg.name,
        category: "matrix",
        operation: "matmul",
        shape: "\(cfg.m)x\(cfg.k)@\(cfg.k)x\(cfg.n)",
        config: config,
        device: metal
    ) {
        let c = a.matmul(b)
        c.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }
    results.append(result)
    print(result.summary)
}

// Transpose
print()
print("--- Transpose ---")
let transposeConfigs: [(name: String, shape: [Int], axes: (Int, Int)?)] = [
    ("Transpose-64x64", [64, 64], nil),
    ("Transpose-256x128", [256, 128], nil),
    ("Transpose-3D", [8, 64, 64], (1, 2)),
]

for cfg in transposeConfigs {
    let a = Tensor<Float>.ones(cfg.shape, on: metal)
    a.markForMaterialization()
    LazyTensorBarrier(on: metal)

    let result = runBenchmark(
        name: cfg.name,
        category: "matrix",
        operation: "transpose",
        shape: cfg.shape.map(String.init).joined(separator: "x"),
        config: config,
        device: metal
    ) {
        let t: Tensor<Float>
        if let axes = cfg.axes {
            t = a.transpose(axes.0, axes.1)
        } else {
            t = a.transpose()
        }
        t.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }
    results.append(result)
    print(result.summary)
}

print()

// MARK: - Reduction Operations

print("═══════════════════════════════════════════════════════════════")
print("REDUCTION OPERATIONS")
print("═══════════════════════════════════════════════════════════════")

let reductionConfigs: [(name: String, shape: [Int], op: String, axes: [Int]?)] = [
    ("GlobalSum-64x64", [64, 64], "sum", nil),
    ("GlobalSum-256x256", [256, 256], "sum", nil),
    ("RowSum-128x128", [128, 128], "sum", [1]),
    ("ColSum-128x128", [128, 128], "sum", [0]),
    ("GlobalMean-256x256", [256, 256], "mean", nil),
    ("GlobalMax-256x256", [256, 256], "max", nil),
]

for cfg in reductionConfigs {
    let a = Tensor<Float>.ones(cfg.shape, on: metal)
    a.markForMaterialization()
    LazyTensorBarrier(on: metal)

    let result = runBenchmark(
        name: cfg.name,
        category: "reduction",
        operation: cfg.op,
        shape: cfg.shape.map(String.init).joined(separator: "x"),
        config: config,
        device: metal
    ) {
        let r: Tensor<Float>
        switch cfg.op {
        case "sum":
            if let axes = cfg.axes {
                r = a.sum(dims: axes, keepDims: false)
            } else {
                r = a.sum()
            }
        case "mean":
            r = a.mean()
        case "max":
            r = a.max()
        default:
            r = a.sum()
        }
        r.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }
    results.append(result)
    print(result.summary)
}

print()

// MARK: - Arithmetic Operations

print("═══════════════════════════════════════════════════════════════")
print("ARITHMETIC OPERATIONS")
print("═══════════════════════════════════════════════════════════════")

// Binary operations
let binaryConfigs: [(name: String, shape: [Int], op: String)] = [
    ("Add-64x64", [64, 64], "add"),
    ("Add-256x256", [256, 256], "add"),
    ("Mul-64x64", [64, 64], "multiply"),
    ("Mul-256x256", [256, 256], "multiply"),
    ("Div-128x128", [128, 128], "divide"),
]

for cfg in binaryConfigs {
    let a = Tensor<Float>.ones(cfg.shape, on: metal)
    let b = Tensor<Float>.full(cfg.shape, 2.0, on: metal)
    a.markForMaterialization()
    b.markForMaterialization()
    LazyTensorBarrier(on: metal)

    let result = runBenchmark(
        name: cfg.name,
        category: "arithmetic",
        operation: cfg.op,
        shape: cfg.shape.map(String.init).joined(separator: "x"),
        config: config,
        device: metal
    ) {
        let r: Tensor<Float>
        switch cfg.op {
        case "add": r = a + b
        case "multiply": r = a * b
        case "divide": r = a / b
        default: r = a + b
        }
        r.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }
    results.append(result)
    print(result.summary)
}

// Unary operations
print()
print("--- Unary Operations ---")

let unaryConfigs: [(name: String, shape: [Int], op: String)] = [
    ("Exp-128x128", [128, 128], "exp"),
    ("Log-128x128", [128, 128], "log"),
    ("Tanh-128x128", [128, 128], "tanh"),
    ("Sigmoid-256x256", [256, 256], "sigmoid"),
    ("ReLU-256x256", [256, 256], "relu"),
    ("GELU-128x128", [128, 128], "gelu"),
    ("Softmax-128x128", [128, 128], "softmax"),
]

for cfg in unaryConfigs {
    let a = cfg.op == "log"
        ? Tensor<Float>.full(cfg.shape, 2.0, on: metal)
        : Tensor<Float>.ones(cfg.shape, on: metal)
    a.markForMaterialization()
    LazyTensorBarrier(on: metal)

    let result = runBenchmark(
        name: cfg.name,
        category: "arithmetic",
        operation: cfg.op,
        shape: cfg.shape.map(String.init).joined(separator: "x"),
        config: config,
        device: metal
    ) {
        let r: Tensor<Float>
        switch cfg.op {
        case "exp": r = a.exp()
        case "log": r = a.log()
        case "tanh": r = a.tanh()
        case "sigmoid": r = a.sigmoid()
        case "relu": r = a.relu()
        case "gelu": r = a.gelu()
        case "softmax": r = a.softmax(dim: -1)
        default: r = a.exp()
        }
        r.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }
    results.append(result)
    print(result.summary)
}

// Broadcast operations
print()
print("--- Broadcast Operations ---")

do {
    let a = Tensor<Float>.ones([128, 128], on: metal)
    let b = Tensor<Float>.ones([128], on: metal)
    a.markForMaterialization()
    b.markForMaterialization()
    LazyTensorBarrier(on: metal)

    let result = runBenchmark(
        name: "AddBroadcast-Row",
        category: "arithmetic",
        operation: "add_broadcast",
        shape: "128x128 + 128",
        config: config,
        device: metal
    ) {
        let bRow = b.reshape([1, 128])
        let r = a + bRow
        r.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }
    results.append(result)
    print(result.summary)
}

do {
    let a = Tensor<Float>.ones([8, 64, 128], on: metal)
    let b = Tensor<Float>.ones([128], on: metal)
    a.markForMaterialization()
    b.markForMaterialization()
    LazyTensorBarrier(on: metal)

    let result = runBenchmark(
        name: "MulBroadcast-3D",
        category: "arithmetic",
        operation: "multiply_broadcast",
        shape: "8x64x128 * 128",
        config: config,
        device: metal
    ) {
        let bBroadcast = b.reshape([1, 1, 128])
        let r = a * bBroadcast
        r.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }
    results.append(result)
    print(result.summary)
}

print()

// MARK: - Compound Operations

print("═══════════════════════════════════════════════════════════════")
print("COMPOUND OPERATIONS")
print("═══════════════════════════════════════════════════════════════")

// MLP-like forward pass
do {
    let x = Tensor<Float>.ones([32, 256], on: metal)
    let w1 = Tensor<Float>.full([256, 512], 0.01, on: metal)
    let w2 = Tensor<Float>.full([512, 128], 0.01, on: metal)
    x.markForMaterialization()
    w1.markForMaterialization()
    w2.markForMaterialization()
    LazyTensorBarrier(on: metal)

    let result = runBenchmark(
        name: "MLP-32x256->512->128",
        category: "compound",
        operation: "mlp_forward",
        shape: "32x256->512->128",
        config: config,
        device: metal
    ) {
        let h = x.matmul(w1).gelu()
        let output = h.matmul(w2)
        output.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }
    results.append(result)
    print(result.summary)
}

// Attention computation
do {
    // [batch=2, heads=8, seq=64, hidden=64]
    let q = Tensor<Float>.ones([2, 8, 64, 64], on: metal)
    let k = Tensor<Float>.ones([2, 8, 64, 64], on: metal)
    let v = Tensor<Float>.ones([2, 8, 64, 64], on: metal)
    q.markForMaterialization()
    k.markForMaterialization()
    v.markForMaterialization()
    LazyTensorBarrier(on: metal)

    let scale = Tensor<Float>.full([1], 1.0 / Foundation.sqrt(64.0), on: metal)
    scale.markForMaterialization()
    LazyTensorBarrier(on: metal)

    let result = runBenchmark(
        name: "Attention-2x8x64x64",
        category: "compound",
        operation: "attention",
        shape: "[2,8,64,64]",
        config: config,
        device: metal
    ) {
        let kT = k.transposeLastTwo()
        let scores = q.batchedMatmul(kT)
        let scaledScores = scores * scale.broadcast(to: scores.shape)
        let attnWeights = scaledScores.softmax(dim: -1)
        let output = attnWeights.batchedMatmul(v)
        output.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }
    results.append(result)
    print(result.summary)
}

print()

// MARK: - Summary

print("═══════════════════════════════════════════════════════════════")
print("SUMMARY")
print("═══════════════════════════════════════════════════════════════")
print()

let grouped = Dictionary(grouping: results) { $0.category }
for (category, categoryResults) in grouped.sorted(by: { $0.key < $1.key }) {
    print("\(category.uppercased())")
    print(String(repeating: "-", count: 40))
    let avgMean = categoryResults.reduce(0.0) { $0 + $1.meanMs } / Double(categoryResults.count)
    print("  Benchmarks: \(categoryResults.count)")
    print("  Avg time: \(String(format: "%.3f", avgMean)) ms")
    print()
}

print("Total benchmarks: \(results.count)")
print()
print("Done!")

#else
print("ERROR: MetalHLO not available")
exit(1)
#endif
