// MLXComparison - Magma vs MLX Performance Benchmark
// Compares identical operations across both frameworks

import Foundation
import Magma
import LazyTensor
import XLARuntime
import MLX
import MLXNN
import _Differentiation

// MARK: - Benchmark Infrastructure

struct BenchmarkResult {
    let name: String
    let category: String
    let shape: String
    let magmaMeanMs: Double
    let magmaMinMs: Double
    let mlxMeanMs: Double
    let mlxMinMs: Double

    var speedup: Double {
        mlxMeanMs / magmaMeanMs
    }

    var summary: String {
        let speedupStr: String
        if speedup > 1.0 {
            speedupStr = String(format: "Magma %.2fx faster", speedup)
        } else if speedup < 1.0 {
            speedupStr = String(format: "MLX %.2fx faster", 1.0 / speedup)
        } else {
            speedupStr = "Equal"
        }
        return String(format: "%@ | %@ | Magma: %.3f ms | MLX: %.3f ms | %@",
                      name.padding(toLength: 24, withPad: " ", startingAt: 0),
                      shape.padding(toLength: 16, withPad: " ", startingAt: 0),
                      magmaMeanMs, mlxMeanMs, speedupStr)
    }
}

struct BenchmarkConfig {
    let warmupIterations: Int
    let measurementIterations: Int

    static let quick = BenchmarkConfig(warmupIterations: 2, measurementIterations: 5)
    static let standard = BenchmarkConfig(warmupIterations: 5, measurementIterations: 20)
}

func measureTime(iterations: Int, warmup: Int, operation: () -> Void) -> (mean: Double, min: Double) {
    // Warmup
    for _ in 0..<warmup {
        operation()
    }

    // Measure
    var times: [Double] = []
    for _ in 0..<iterations {
        let start = CFAbsoluteTimeGetCurrent()
        operation()
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        times.append(elapsed * 1000)
    }

    let mean = times.reduce(0, +) / Double(times.count)
    let minTime = times.min() ?? 0
    return (mean, minTime)
}

// MARK: - Main

setbuf(stdout, nil)
print("╔════════════════════════════════════════════════════════════════════════╗")
print("║              Magma vs MLX Performance Comparison                       ║")
print("╚════════════════════════════════════════════════════════════════════════╝")
print()

#if canImport(MetalHLO)
import MetalHLO

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

print("═══════════════════════════════════════════════════════════════════════════")
print("MATRIX OPERATIONS (GEMM)")
print("═══════════════════════════════════════════════════════════════════════════")

let gemmConfigs: [(name: String, m: Int, n: Int, k: Int)] = [
    ("GEMM-128", 128, 128, 128),
    ("GEMM-256", 256, 256, 256),
    ("GEMM-512", 512, 512, 512),
    ("GEMM-1024", 1024, 1024, 1024),
    ("GEMM-2048", 2048, 2048, 2048),
    ("GEMM-4096", 4096, 4096, 4096),
]

for cfg in gemmConfigs {
    // Magma setup
    let magmaA = Tensor<Float>.ones([cfg.m, cfg.k], on: metal)
    let magmaB = Tensor<Float>.ones([cfg.k, cfg.n], on: metal)
    magmaA.markForMaterialization()
    magmaB.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxA = MLXArray.ones([cfg.m, cfg.k])
    let mlxB = MLXArray.ones([cfg.k, cfg.n])
    eval(mlxA, mlxB)

    // Benchmark Magma
    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let c = magmaA.matmul(magmaB)
        c.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    // Benchmark MLX
    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let c = matmul(mlxA, mlxB)
        eval(c)
    }

    let result = BenchmarkResult(
        name: cfg.name,
        category: "matrix",
        shape: "\(cfg.m)x\(cfg.k)@\(cfg.k)x\(cfg.n)",
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

print()

// MARK: - Transpose + Matmul (forces actual data movement, unlike bare transpose which is a no-op in MLX)

print("═══════════════════════════════════════════════════════════════════════════")
print("TRANSPOSE + MATMUL (A^T @ A)")
print("═══════════════════════════════════════════════════════════════════════════")

let transMatConfigs: [(name: String, m: Int, n: Int)] = [
    ("TransMatmul-256", 256, 256),
    ("TransMatmul-512", 512, 512),
    ("TransMatmul-1024", 1024, 1024),
    ("TransMatmul-2048", 2048, 2048),
]

for cfg in transMatConfigs {
    // Magma setup
    let magmaA = Tensor<Float>.ones([cfg.m, cfg.n], on: metal)
    magmaA.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxA = MLXArray.ones([cfg.m, cfg.n])
    eval(mlxA)

    // Benchmark Magma: transpose then matmul
    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let t = magmaA.transpose()
        let r = t.matmul(magmaA)
        r.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    // Benchmark MLX: transpose then matmul
    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let t = mlxA.T
        let r = matmul(t, mlxA)
        eval(r)
    }

    let result = BenchmarkResult(
        name: cfg.name,
        category: "matrix",
        shape: "\(cfg.m)x\(cfg.n)",
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

print()

// MARK: - Reduction Operations

print("═══════════════════════════════════════════════════════════════════════════")
print("REDUCTION OPERATIONS")
print("═══════════════════════════════════════════════════════════════════════════")

let reductionConfigs: [(name: String, shape: [Int], axis: Int?)] = [
    ("GlobalSum-512x512", [512, 512], nil),
    ("GlobalSum-2048x2048", [2048, 2048], nil),
    ("RowSum-1024x1024", [1024, 1024], 1),
    ("ColSum-1024x1024", [1024, 1024], 0),
    ("GlobalSum-4096x4096", [4096, 4096], nil),
]

for cfg in reductionConfigs {
    // Magma setup
    let magmaA = Tensor<Float>.ones(cfg.shape, on: metal)
    magmaA.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxA = MLXArray.ones(cfg.shape)
    eval(mlxA)

    // Benchmark Magma
    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let r: Tensor<Float>
        if let axis = cfg.axis {
            r = magmaA.sum(dims: [axis], keepDims: false)
        } else {
            r = magmaA.sum()
        }
        r.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    // Benchmark MLX
    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let r: MLXArray
        if let axis = cfg.axis {
            r = mlxA.sum(axis: axis)
        } else {
            r = mlxA.sum()
        }
        eval(r)
    }

    let result = BenchmarkResult(
        name: cfg.name,
        category: "reduction",
        shape: cfg.shape.map(String.init).joined(separator: "x"),
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

print()

// MARK: - Arithmetic Operations

print("═══════════════════════════════════════════════════════════════════════════")
print("ELEMENT-WISE ARITHMETIC")
print("═══════════════════════════════════════════════════════════════════════════")

let arithmeticConfigs: [(name: String, shape: [Int], op: String)] = [
    ("Add-512x512", [512, 512], "add"),
    ("Add-2048x2048", [2048, 2048], "add"),
    ("Mul-512x512", [512, 512], "multiply"),
    ("Mul-2048x2048", [2048, 2048], "multiply"),
    ("Add-4096x4096", [4096, 4096], "add"),
]

for cfg in arithmeticConfigs {
    // Magma setup
    let magmaA = Tensor<Float>.ones(cfg.shape, on: metal)
    let magmaB = Tensor<Float>.full(cfg.shape, 2.0, on: metal)
    magmaA.markForMaterialization()
    magmaB.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxA = MLXArray.ones(cfg.shape)
    let mlxB = MLXArray.ones(cfg.shape) * Float(2.0)
    eval(mlxA, mlxB)

    // Benchmark Magma
    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let r: Tensor<Float> = cfg.op == "add" ? magmaA + magmaB : magmaA * magmaB
        r.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    // Benchmark MLX
    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let r: MLXArray = cfg.op == "add" ? mlxA + mlxB : mlxA * mlxB
        eval(r)
    }

    let result = BenchmarkResult(
        name: cfg.name,
        category: "arithmetic",
        shape: cfg.shape.map(String.init).joined(separator: "x"),
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

print()

// MARK: - Unary Operations

print("═══════════════════════════════════════════════════════════════════════════")
print("UNARY OPERATIONS")
print("═══════════════════════════════════════════════════════════════════════════")

let unaryConfigs: [(name: String, shape: [Int], op: String)] = [
    ("Exp-1024x1024", [1024, 1024], "exp"),
    ("Tanh-1024x1024", [1024, 1024], "tanh"),
    ("Sigmoid-1024x1024", [1024, 1024], "sigmoid"),
    ("ReLU-2048x2048", [2048, 2048], "relu"),
    ("GELU-1024x1024", [1024, 1024], "gelu"),
    ("Softmax-1024x1024", [1024, 1024], "softmax"),
]

for cfg in unaryConfigs {
    // Magma setup
    let magmaA = Tensor<Float>.ones(cfg.shape, on: metal)
    magmaA.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxA = MLXArray.ones(cfg.shape)
    eval(mlxA)

    // Benchmark Magma
    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let r: Tensor<Float>
        switch cfg.op {
        case "exp": r = magmaA.exp()
        case "tanh": r = magmaA.tanh()
        case "sigmoid": r = magmaA.sigmoid()
        case "relu": r = magmaA.relu()
        case "gelu": r = magmaA.gelu()
        case "softmax": r = magmaA.softmax(dim: -1)
        default: r = magmaA.exp()
        }
        r.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    // Benchmark MLX
    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let r: MLXArray
        switch cfg.op {
        case "exp": r = exp(mlxA)
        case "tanh": r = tanh(mlxA)
        case "sigmoid": r = sigmoid(mlxA)
        case "relu": r = maximum(mlxA, MLXArray(Float(0)))
        case "gelu": r = MLXNN.gelu(mlxA)
        case "softmax": r = softmax(mlxA, axis: -1)
        default: r = exp(mlxA)
        }
        eval(r)
    }

    let result = BenchmarkResult(
        name: cfg.name,
        category: "unary",
        shape: cfg.shape.map(String.init).joined(separator: "x"),
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

print()

// MARK: - Compound Operations

print("═══════════════════════════════════════════════════════════════════════════")
print("COMPOUND OPERATIONS")
print("═══════════════════════════════════════════════════════════════════════════")

// MLP Forward Pass - Small
do {
    let batchSize = 32
    let inputDim = 512
    let hiddenDim = 1024
    let outputDim = 256

    // Magma setup
    let magmaX = Tensor<Float>.ones([batchSize, inputDim], on: metal)
    let magmaW1 = Tensor<Float>.full([inputDim, hiddenDim], 0.01, on: metal)
    let magmaW2 = Tensor<Float>.full([hiddenDim, outputDim], 0.01, on: metal)
    magmaX.markForMaterialization()
    magmaW1.markForMaterialization()
    magmaW2.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxX = MLXArray.ones([batchSize, inputDim])
    let mlxW1 = MLXArray.ones([inputDim, hiddenDim]) * Float(0.01)
    let mlxW2 = MLXArray.ones([hiddenDim, outputDim]) * Float(0.01)
    eval(mlxX, mlxW1, mlxW2)

    // Benchmark Magma
    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let h = magmaX.matmul(magmaW1).gelu()
        let output = h.matmul(magmaW2)
        output.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    // Benchmark MLX
    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let h = MLXNN.gelu(matmul(mlxX, mlxW1))
        let output = matmul(h, mlxW2)
        eval(output)
    }

    let result = BenchmarkResult(
        name: "MLP-Small",
        category: "compound",
        shape: "32x512->1024->256",
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

// MLP Forward Pass - Large
do {
    let batchSize = 128
    let inputDim = 1024
    let hiddenDim = 4096
    let outputDim = 1024

    // Magma setup
    let magmaX = Tensor<Float>.ones([batchSize, inputDim], on: metal)
    let magmaW1 = Tensor<Float>.full([inputDim, hiddenDim], 0.01, on: metal)
    let magmaW2 = Tensor<Float>.full([hiddenDim, outputDim], 0.01, on: metal)
    magmaX.markForMaterialization()
    magmaW1.markForMaterialization()
    magmaW2.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxX = MLXArray.ones([batchSize, inputDim])
    let mlxW1 = MLXArray.ones([inputDim, hiddenDim]) * Float(0.01)
    let mlxW2 = MLXArray.ones([hiddenDim, outputDim]) * Float(0.01)
    eval(mlxX, mlxW1, mlxW2)

    // Benchmark Magma
    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let h = magmaX.matmul(magmaW1).gelu()
        let output = h.matmul(magmaW2)
        output.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    // Benchmark MLX
    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let h = MLXNN.gelu(matmul(mlxX, mlxW1))
        let output = matmul(h, mlxW2)
        eval(output)
    }

    let result = BenchmarkResult(
        name: "MLP-Large",
        category: "compound",
        shape: "128x1024->4096->1024",
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

// Attention - Small
do {
    let batch = 2
    let heads = 8
    let seqLen = 64
    let headDim = 64

    // Magma setup
    let magmaQ = Tensor<Float>.ones([batch, heads, seqLen, headDim], on: metal)
    let magmaK = Tensor<Float>.ones([batch, heads, seqLen, headDim], on: metal)
    let magmaV = Tensor<Float>.ones([batch, heads, seqLen, headDim], on: metal)
    magmaQ.markForMaterialization()
    magmaK.markForMaterialization()
    magmaV.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxQ = MLXArray.ones([batch, heads, seqLen, headDim])
    let mlxK = MLXArray.ones([batch, heads, seqLen, headDim])
    let mlxV = MLXArray.ones([batch, heads, seqLen, headDim])
    let scale = Float(1.0 / Foundation.sqrt(Float(headDim)))
    eval(mlxQ, mlxK, mlxV)

    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let kT = magmaK.transposeLastTwo()
        let scores = magmaQ.batchedMatmul(kT)
        let attnWeights = scores.softmax(dim: -1)
        let output = attnWeights.batchedMatmul(magmaV)
        output.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let kT = mlxK.transposed(0, 1, 3, 2)
        let scores = matmul(mlxQ, kT) * scale
        let attnWeights = softmax(scores, axis: -1)
        let output = matmul(attnWeights, mlxV)
        eval(output)
    }

    let result = BenchmarkResult(
        name: "Attention-Small",
        category: "compound",
        shape: "[\(batch),\(heads),\(seqLen),\(headDim)]",
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

// Attention - Large
do {
    let batch = 4
    let heads = 16
    let seqLen = 256
    let headDim = 64

    // Magma setup
    let magmaQ = Tensor<Float>.ones([batch, heads, seqLen, headDim], on: metal)
    let magmaK = Tensor<Float>.ones([batch, heads, seqLen, headDim], on: metal)
    let magmaV = Tensor<Float>.ones([batch, heads, seqLen, headDim], on: metal)
    magmaQ.markForMaterialization()
    magmaK.markForMaterialization()
    magmaV.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxQ = MLXArray.ones([batch, heads, seqLen, headDim])
    let mlxK = MLXArray.ones([batch, heads, seqLen, headDim])
    let mlxV = MLXArray.ones([batch, heads, seqLen, headDim])
    let scaleLarge = Float(1.0 / Foundation.sqrt(Float(headDim)))
    eval(mlxQ, mlxK, mlxV)

    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let kT = magmaK.transposeLastTwo()
        let scores = magmaQ.batchedMatmul(kT)
        let attnWeights = scores.softmax(dim: -1)
        let output = attnWeights.batchedMatmul(magmaV)
        output.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let kT = mlxK.transposed(0, 1, 3, 2)
        let scores = matmul(mlxQ, kT) * scaleLarge
        let attnWeights = softmax(scores, axis: -1)
        let output = matmul(attnWeights, mlxV)
        eval(output)
    }

    let result = BenchmarkResult(
        name: "Attention-Large",
        category: "compound",
        shape: "[\(batch),\(heads),\(seqLen),\(headDim)]",
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

print()

// MARK: - Gradient (Forward + Backward) Benchmarks

print("═══════════════════════════════════════════════════════════════════════════")
print("GRADIENT OPERATIONS (FORWARD + BACKWARD)")
print("═══════════════════════════════════════════════════════════════════════════")

// Differentiable weight struct for Magma gradient benchmarks
struct MLPWeights: Differentiable {
    var w1: Tensor<Float>
    var w2: Tensor<Float>
}

// MLP forward+backward - Small
do {
    let batchSize = 32
    let inputDim = 512
    let hiddenDim = 1024
    let outputDim = 256

    // Magma setup
    let magmaX = Tensor<Float>.ones([batchSize, inputDim], on: metal)
    var magmaWeights = MLPWeights(
        w1: Tensor<Float>.full([inputDim, hiddenDim], 0.01, on: metal),
        w2: Tensor<Float>.full([hiddenDim, outputDim], 0.01, on: metal)
    )
    magmaX.markForMaterialization()
    magmaWeights.w1.markForMaterialization()
    magmaWeights.w2.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxX = MLXArray.ones([batchSize, inputDim])
    let mlxW1 = MLXArray.ones([inputDim, hiddenDim]) * Float(0.01)
    let mlxW2 = MLXArray.ones([hiddenDim, outputDim]) * Float(0.01)
    eval(mlxX, mlxW1, mlxW2)

    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let (loss, grads) = valueWithGradient(at: magmaWeights) { w -> Tensor<Float> in
            let h = magmaX.matmul(w.w1).gelu()
            return h.matmul(w.w2).sum()
        }
        loss.markForMaterialization()
        grads.w1.markForMaterialization()
        grads.w2.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    let mlxForwardBackward = MLX.valueAndGrad { (params: [MLXArray]) -> [MLXArray] in
        let h = MLXNN.gelu(matmul(mlxX, params[0]))
        return [matmul(h, params[1]).sum()]
    }
    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let (loss, _) = mlxForwardBackward([mlxW1, mlxW2])
        eval(loss)
    }

    let result = BenchmarkResult(
        name: "MLP-Grad-Small",
        category: "gradient",
        shape: "32x512->1024->256",
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

// MLP forward+backward - Large
do {
    let batchSize = 128
    let inputDim = 1024
    let hiddenDim = 4096
    let outputDim = 1024

    // Magma setup
    let magmaX = Tensor<Float>.ones([batchSize, inputDim], on: metal)
    var magmaWeights = MLPWeights(
        w1: Tensor<Float>.full([inputDim, hiddenDim], 0.01, on: metal),
        w2: Tensor<Float>.full([hiddenDim, outputDim], 0.01, on: metal)
    )
    magmaX.markForMaterialization()
    magmaWeights.w1.markForMaterialization()
    magmaWeights.w2.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxX = MLXArray.ones([batchSize, inputDim])
    let mlxW1 = MLXArray.ones([inputDim, hiddenDim]) * Float(0.01)
    let mlxW2 = MLXArray.ones([hiddenDim, outputDim]) * Float(0.01)
    eval(mlxX, mlxW1, mlxW2)

    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let (loss, grads) = valueWithGradient(at: magmaWeights) { w -> Tensor<Float> in
            let h = magmaX.matmul(w.w1).gelu()
            return h.matmul(w.w2).sum()
        }
        loss.markForMaterialization()
        grads.w1.markForMaterialization()
        grads.w2.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    let mlxForwardBackward = MLX.valueAndGrad { (params: [MLXArray]) -> [MLXArray] in
        let h = MLXNN.gelu(matmul(mlxX, params[0]))
        return [matmul(h, params[1]).sum()]
    }
    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let (loss, _) = mlxForwardBackward([mlxW1, mlxW2])
        eval(loss)
    }

    let result = BenchmarkResult(
        name: "MLP-Grad-Large",
        category: "gradient",
        shape: "128x1024->4096->1024",
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

// Attention forward+backward
struct AttnWeights: Differentiable {
    var q: Tensor<Float>
    var k: Tensor<Float>
    var v: Tensor<Float>
}

do {
    let batch = 4
    let heads = 16
    let seqLen = 256
    let headDim = 64

    // Magma setup
    var magmaWeights = AttnWeights(
        q: Tensor<Float>.ones([batch, heads, seqLen, headDim], on: metal),
        k: Tensor<Float>.ones([batch, heads, seqLen, headDim], on: metal),
        v: Tensor<Float>.ones([batch, heads, seqLen, headDim], on: metal)
    )
    magmaWeights.q.markForMaterialization()
    magmaWeights.k.markForMaterialization()
    magmaWeights.v.markForMaterialization()
    LazyTensorBarrier(on: metal)

    // MLX setup
    let mlxQ = MLXArray.ones([batch, heads, seqLen, headDim])
    let mlxK = MLXArray.ones([batch, heads, seqLen, headDim])
    let mlxV = MLXArray.ones([batch, heads, seqLen, headDim])
    let scaleLargeGrad = Float(1.0 / Foundation.sqrt(Float(headDim)))
    eval(mlxQ, mlxK, mlxV)

    let magmaTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let (loss, grads) = valueWithGradient(at: magmaWeights) { w -> Tensor<Float> in
            let kT = w.k.transpose(-1, -2)
            let scores = w.q.batchedMatmul(kT).softmax(dim: -1)
            return scores.batchedMatmul(w.v).sum()
        }
        loss.markForMaterialization()
        grads.q.markForMaterialization()
        grads.k.markForMaterialization()
        grads.v.markForMaterialization()
        LazyTensorBarrier(on: metal)
    }

    let mlxAttnGrad = MLX.valueAndGrad { (params: [MLXArray]) -> [MLXArray] in
        let kT = params[1].transposed(0, 1, 3, 2)
        let scores = softmax(matmul(params[0], kT) * scaleLargeGrad, axis: -1)
        return [matmul(scores, params[2]).sum()]
    }
    let mlxTime = measureTime(iterations: config.measurementIterations, warmup: config.warmupIterations) {
        let (loss, _) = mlxAttnGrad([mlxQ, mlxK, mlxV])
        eval(loss)
    }

    let result = BenchmarkResult(
        name: "Attention-Grad-Large",
        category: "gradient",
        shape: "[\(batch),\(heads),\(seqLen),\(headDim)]",
        magmaMeanMs: magmaTime.mean,
        magmaMinMs: magmaTime.min,
        mlxMeanMs: mlxTime.mean,
        mlxMinMs: mlxTime.min
    )
    results.append(result)
    print(result.summary)
}

print()

// MARK: - Summary

print("═══════════════════════════════════════════════════════════════════════════")
print("SUMMARY")
print("═══════════════════════════════════════════════════════════════════════════")
print()

let grouped = Dictionary(grouping: results) { $0.category }
for (category, categoryResults) in grouped.sorted(by: { $0.key < $1.key }) {
    print("\(category.uppercased())")
    print(String(repeating: "-", count: 50))
    let avgMagma = categoryResults.reduce(0.0) { $0 + $1.magmaMeanMs } / Double(categoryResults.count)
    let avgMLX = categoryResults.reduce(0.0) { $0 + $1.mlxMeanMs } / Double(categoryResults.count)
    let avgSpeedup = avgMLX / avgMagma
    print("  Benchmarks: \(categoryResults.count)")
    print("  Avg Magma: \(String(format: "%.3f", avgMagma)) ms")
    print("  Avg MLX:   \(String(format: "%.3f", avgMLX)) ms")
    if avgSpeedup > 1.0 {
        print("  Average:   Magma \(String(format: "%.2f", avgSpeedup))x faster")
    } else {
        print("  Average:   MLX \(String(format: "%.2f", 1.0 / avgSpeedup))x faster")
    }
    print()
}

// Overall summary
let totalMagma = results.reduce(0.0) { $0 + $1.magmaMeanMs }
let totalMLX = results.reduce(0.0) { $0 + $1.mlxMeanMs }
let overallSpeedup = totalMLX / totalMagma

print("OVERALL")
print(String(repeating: "=", count: 50))
print("Total benchmarks: \(results.count)")
print("Total Magma time: \(String(format: "%.1f", totalMagma)) ms")
print("Total MLX time:   \(String(format: "%.1f", totalMLX)) ms")
if overallSpeedup > 1.0 {
    print("Overall: Magma is \(String(format: "%.2f", overallSpeedup))x faster than MLX")
} else {
    print("Overall: MLX is \(String(format: "%.2f", 1.0 / overallSpeedup))x faster than Magma")
}
print()
print("Done!")

#else
print("ERROR: MetalHLO not available")
exit(1)
#endif
