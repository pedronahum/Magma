// MLX vs Magma Metal Comparison Benchmark
// Compares performance AND accuracy between MLX and Magma's Metal backend
// Tests both raw matmul AND fused operations where Magma's optimization passes shine

import Foundation
import MLX
import MLXNN
import Magma
import LazyTensor
import XLARuntime

#if os(macOS) && canImport(MetalHLO)
import MetalHLO
#endif

// MARK: - Benchmark Configuration

let warmupIterations = 3
let benchmarkIterations = 10
let accuracyTolerance: Float = 1e-4  // Relative tolerance for accuracy checks

// MARK: - Accuracy Metrics

struct AccuracyResult {
    let maxAbsError: Float
    let meanAbsError: Float
    let maxRelError: Float
    let passed: Bool

    var statusIcon: String { passed ? "✓" : "✗" }
}

func compareArrays(mlxArray: MLXArray, magmaScalars: [Float], tolerance: Float = accuracyTolerance) -> AccuracyResult {
    let mlxScalars = mlxArray.asArray(Float.self)

    guard mlxScalars.count == magmaScalars.count else {
        return AccuracyResult(maxAbsError: Float.infinity, meanAbsError: Float.infinity, maxRelError: Float.infinity, passed: false)
    }

    var maxAbsErr: Float = 0
    var sumAbsErr: Float = 0
    var maxRelErr: Float = 0

    for (mlx, magma) in zip(mlxScalars, magmaScalars) {
        let absErr = abs(mlx - magma)
        maxAbsErr = max(maxAbsErr, absErr)
        sumAbsErr += absErr

        let denom = max(abs(mlx), 1e-8)
        let relErr = absErr / denom
        maxRelErr = max(maxRelErr, relErr)
    }

    let meanAbsErr = sumAbsErr / Float(mlxScalars.count)
    let passed = maxRelErr < tolerance || maxAbsErr < 1e-5

    return AccuracyResult(maxAbsError: maxAbsErr, meanAbsError: meanAbsErr, maxRelError: maxRelErr, passed: passed)
}

// MARK: - Data Conversion Helpers

func mlxToMagma(_ mlxArray: MLXArray, device: XLARuntime.Device) -> Tensor<Float> {
    let data = mlxArray.asArray(Float.self)
    let shape = mlxArray.shape.map { Int($0) }
    return Tensor<Float>(data, shape: shape, on: device)
}

// MARK: - Benchmark Result

struct BenchmarkResult {
    let operation: String
    let size: String
    let mlxTimeMs: Double
    let magmaTimeMs: Double
    let accuracy: AccuracyResult

    var speedup: Double { mlxTimeMs / magmaTimeMs }
    var speedupIcon: String { speedup >= 1.0 ? "✓" : "" }
}

var allResults: [BenchmarkResult] = []

// MARK: - Helper Functions

func printHeader(_ title: String) {
    print()
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║  \(title.padding(toLength: 76, withPad: " ", startingAt: 0))║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()
}

func printTableHeader() {
    print("Operation          │ Size           │ MLX (ms)  │ Magma (ms)│ Speedup │ Accuracy")
    print("───────────────────┼────────────────┼───────────┼───────────┼─────────┼──────────")
}

func printResult(_ result: BenchmarkResult) {
    let opStr = result.operation.padding(toLength: 18, withPad: " ", startingAt: 0)
    let sizeStr = result.size.padding(toLength: 14, withPad: " ", startingAt: 0)
    let mlxStr = String(format: "%9.2f", result.mlxTimeMs)
    let magmaStr = String(format: "%9.2f", result.magmaTimeMs)
    let speedupStr = result.speedup >= 1.0
        ? String(format: "%5.2fx ✓", result.speedup)
        : String(format: "%5.2fx  ", result.speedup)
    let accStr = result.accuracy.passed
        ? String(format: "✓ e=%.1e", result.accuracy.maxRelError)
        : String(format: "✗ e=%.1e", result.accuracy.maxRelError)
    print("\(opStr) │ \(sizeStr) │ \(mlxStr) │ \(magmaStr) │ \(speedupStr) │ \(accStr)")
}

// MARK: - Main Benchmark

printHeader("MLX vs MAGMA METAL COMPARISON BENCHMARK (WITH ACCURACY)")

print("Configuration:")
print("  Warmup iterations: \(warmupIterations)")
print("  Benchmark iterations: \(benchmarkIterations)")
print("  Accuracy tolerance: \(accuracyTolerance)")
print()

#if os(macOS) && canImport(MetalHLO)
print("System Info:")
if let deviceName = MetalBackend.deviceName {
    print("  Metal Device: \(deviceName)")
}
print("  MLX Version: 0.21.x")
print()

guard Backend.metal.isAvailable else {
    print("ERROR: Metal backend not available")
    exit(1)
}

let metalDevice = Device(backend: .metal, index: 0)

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 1: Raw Matrix Multiplication
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("1. RAW MATRIX MULTIPLICATION")
printTableHeader()

for size in [256, 512, 1024] {
    // Create MLX data (source of truth)
    let mlxA = MLXRandom.uniform(low: -1, high: 1, [size, size])
    let mlxB = MLXRandom.uniform(low: -1, high: 1, [size, size])
    MLX.eval(mlxA, mlxB)

    // Convert to Magma
    let magmaA = mlxToMagma(mlxA, device: metalDevice)
    let magmaB = mlxToMagma(mlxB, device: metalDevice)
    magmaA.markForMaterialization()
    magmaB.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    // Warmup MLX
    for _ in 0..<warmupIterations {
        let r = MLX.matmul(mlxA, mlxB)
        MLX.eval(r)
    }

    // Warmup Magma
    for _ in 0..<warmupIterations {
        let z = magmaA.matmul(magmaB)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    // Benchmark MLX
    var mlxTime: Double = 0
    var mlxResult: MLXArray!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        mlxResult = MLX.matmul(mlxA, mlxB)
        MLX.eval(mlxResult)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Benchmark Magma
    var magmaTime: Double = 0
    var magmaResult: Tensor<Float>!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        magmaResult = magmaA.matmul(magmaB)
        LazyTensorBarrier(on: metalDevice)
        let _ = magmaResult.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    // Compare accuracy
    let accuracy = compareArrays(mlxArray: mlxResult, magmaScalars: magmaResult.scalars())

    let result = BenchmarkResult(
        operation: "matmul",
        size: "[\(size),\(size)]",
        mlxTimeMs: mlxTime,
        magmaTimeMs: magmaTime,
        accuracy: accuracy
    )
    allResults.append(result)
    printResult(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 2: MatMul + Bias + ReLU (Fused Pattern)
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("2. MATMUL + BIAS + RELU (Fusion Pattern)")
print("This tests MatMulBiasActivationPattern - fuses matmul, bias add, and activation")
print()
printTableHeader()

for (batch, inSize, outSize) in [(64, 512, 512), (64, 1024, 1024), (128, 768, 3072)] {
    // Create MLX data
    let mlxX = MLXRandom.uniform(low: -1, high: 1, [batch, inSize])
    let mlxW = MLXRandom.uniform(low: -1, high: 1, [inSize, outSize])
    let mlxB = MLXRandom.uniform(low: -1, high: 1, [outSize])
    MLX.eval(mlxX, mlxW, mlxB)

    // Convert to Magma
    let magmaX = mlxToMagma(mlxX, device: metalDevice)
    let magmaW = mlxToMagma(mlxW, device: metalDevice)
    let magmaB = mlxToMagma(mlxB, device: metalDevice)
    magmaX.markForMaterialization()
    magmaW.markForMaterialization()
    magmaB.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    // Warmup
    for _ in 0..<warmupIterations {
        let h = MLX.matmul(mlxX, mlxW)
        let hBias = h + mlxB
        let r = MLX.maximum(hBias, MLXArray(0.0))
        MLX.eval(r)
    }

    for _ in 0..<warmupIterations {
        let h = magmaX.matmul(magmaW)
        let hBias = h + magmaB
        let z = hBias.relu()
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    // Benchmark MLX
    var mlxTime: Double = 0
    var mlxResult: MLXArray!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let h = MLX.matmul(mlxX, mlxW)
        let hBias = h + mlxB
        mlxResult = MLX.maximum(hBias, MLXArray(0.0))
        MLX.eval(mlxResult)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Benchmark Magma
    var magmaTime: Double = 0
    var magmaResult: Tensor<Float>!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let h = magmaX.matmul(magmaW)
        let hBias = h + magmaB
        magmaResult = hBias.relu()
        LazyTensorBarrier(on: metalDevice)
        let _ = magmaResult.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    let accuracy = compareArrays(mlxArray: mlxResult, magmaScalars: magmaResult.scalars())

    let result = BenchmarkResult(
        operation: "matmul+bias+relu",
        size: "[\(batch),\(inSize)]→\(outSize)",
        mlxTimeMs: mlxTime,
        magmaTimeMs: magmaTime,
        accuracy: accuracy
    )
    allResults.append(result)
    printResult(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 3: Softmax
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("3. SOFTMAX (Fusion Pattern)")
print("This tests SoftmaxPattern - fuses exp, sum, and divide operations")
print()
printTableHeader()

for (batch, seqLen) in [(32, 512), (32, 1024), (64, 2048)] {
    let mlxX = MLXRandom.uniform(low: -5, high: 5, [batch, seqLen])
    MLX.eval(mlxX)

    let magmaX = mlxToMagma(mlxX, device: metalDevice)
    magmaX.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    // Warmup
    for _ in 0..<warmupIterations {
        let r = MLX.softmax(mlxX, axis: -1)
        MLX.eval(r)
    }
    for _ in 0..<warmupIterations {
        let z = magmaX.softmax(dim: -1)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    // Benchmark MLX
    var mlxTime: Double = 0
    var mlxResult: MLXArray!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        mlxResult = MLX.softmax(mlxX, axis: -1)
        MLX.eval(mlxResult)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Benchmark Magma
    var magmaTime: Double = 0
    var magmaResult: Tensor<Float>!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        magmaResult = magmaX.softmax(dim: -1)
        LazyTensorBarrier(on: metalDevice)
        let _ = magmaResult.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    let accuracy = compareArrays(mlxArray: mlxResult, magmaScalars: magmaResult.scalars())

    let result = BenchmarkResult(
        operation: "softmax",
        size: "[\(batch),\(seqLen)]",
        mlxTimeMs: mlxTime,
        magmaTimeMs: magmaTime,
        accuracy: accuracy
    )
    allResults.append(result)
    printResult(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 4: GELU Activation
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("4. GELU ACTIVATION (Fusion Pattern)")
print("This tests GeluPattern - fuses the GELU approximation operations")
print()
printTableHeader()

for size in [512, 1024, 2048] {
    let batch = 64

    let mlxX = MLXRandom.uniform(low: -3, high: 3, [batch, size])
    MLX.eval(mlxX)

    let magmaX = mlxToMagma(mlxX, device: metalDevice)
    magmaX.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    // Warmup
    // Note: Use geluApproximate to match MetalHLO's tanh approximation
    for _ in 0..<warmupIterations {
        let r = MLXNN.geluApproximate(mlxX)
        MLX.eval(r)
    }
    for _ in 0..<warmupIterations {
        let z = magmaX.gelu()
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    // Benchmark MLX
    var mlxTime: Double = 0
    var mlxResult: MLXArray!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        mlxResult = MLXNN.geluApproximate(mlxX)
        MLX.eval(mlxResult)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Benchmark Magma
    var magmaTime: Double = 0
    var magmaResult: Tensor<Float>!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        magmaResult = magmaX.gelu()
        LazyTensorBarrier(on: metalDevice)
        let _ = magmaResult.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    let accuracy = compareArrays(mlxArray: mlxResult, magmaScalars: magmaResult.scalars())

    let result = BenchmarkResult(
        operation: "gelu",
        size: "[\(batch),\(size)]",
        mlxTimeMs: mlxTime,
        magmaTimeMs: magmaTime,
        accuracy: accuracy
    )
    allResults.append(result)
    printResult(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 5: Layer Normalization
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("5. LAYER NORMALIZATION (Fusion Pattern)")
print("This tests LayerNormPattern - fuses mean, variance, normalize, scale, shift")
print()
printTableHeader()

for (batch, features) in [(64, 256), (64, 768), (128, 1024)] {
    let mlxX = MLXRandom.uniform(low: -3, high: 3, [batch, features])
    let mlxGamma = MLXArray.ones([features])
    let mlxBeta = MLXArray.zeros([features])
    MLX.eval(mlxX, mlxGamma, mlxBeta)

    let magmaX = mlxToMagma(mlxX, device: metalDevice)
    magmaX.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    // Create LayerNorm with unit weight and zero bias
    let ln = nn.LayerNorm(features, device: metalDevice)

    // Warmup
    for _ in 0..<warmupIterations {
        let mean = mlxX.mean(axis: -1, keepDims: true)
        let variance = (mlxX - mean).square().mean(axis: -1, keepDims: true)
        let normalized = (mlxX - mean) / (variance + 1e-5).sqrt()
        let r = normalized * mlxGamma + mlxBeta
        MLX.eval(r)
    }
    for _ in 0..<warmupIterations {
        let z = ln(magmaX)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    // Benchmark MLX
    var mlxTime: Double = 0
    var mlxResult: MLXArray!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let mean = mlxX.mean(axis: -1, keepDims: true)
        let variance = (mlxX - mean).square().mean(axis: -1, keepDims: true)
        let normalized = (mlxX - mean) / (variance + 1e-5).sqrt()
        mlxResult = normalized * mlxGamma + mlxBeta
        MLX.eval(mlxResult)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Benchmark Magma
    var magmaTime: Double = 0
    var magmaResult: Tensor<Float>!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        magmaResult = ln(magmaX)
        LazyTensorBarrier(on: metalDevice)
        let _ = magmaResult.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    let accuracy = compareArrays(mlxArray: mlxResult, magmaScalars: magmaResult.scalars())

    let result = BenchmarkResult(
        operation: "layernorm",
        size: "[\(batch),\(features)]",
        mlxTimeMs: mlxTime,
        magmaTimeMs: magmaTime,
        accuracy: accuracy
    )
    allResults.append(result)
    printResult(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 6: Transformer FFN (MatMul + Bias + GELU)
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("6. TRANSFORMER FFN (MatMul + Bias + GELU)")
print("Real-world pattern: First layer of transformer feed-forward network")
print()
printTableHeader()

for (batch, seqLen, hidden, ffnDim) in [(8, 128, 768, 3072), (8, 256, 768, 3072), (4, 512, 1024, 4096)] {
    let inputSize = batch * seqLen

    let mlxX = MLXRandom.uniform(low: -1, high: 1, [inputSize, hidden])
    let mlxW = MLXRandom.uniform(low: -0.1, high: 0.1, [hidden, ffnDim])
    let mlxB = MLXRandom.uniform(low: -0.1, high: 0.1, [ffnDim])
    MLX.eval(mlxX, mlxW, mlxB)

    let magmaX = mlxToMagma(mlxX, device: metalDevice)
    let magmaW = mlxToMagma(mlxW, device: metalDevice)
    let magmaB = mlxToMagma(mlxB, device: metalDevice)
    magmaX.markForMaterialization()
    magmaW.markForMaterialization()
    magmaB.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    // Warmup
    // Note: Use geluApproximate to match MetalHLO's tanh approximation
    for _ in 0..<warmupIterations {
        let h = MLX.matmul(mlxX, mlxW) + mlxB
        let r = MLXNN.geluApproximate(h)
        MLX.eval(r)
    }
    for _ in 0..<warmupIterations {
        let h = magmaX.matmul(magmaW) + magmaB
        let z = h.gelu()
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    // Benchmark MLX
    var mlxTime: Double = 0
    var mlxResult: MLXArray!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let h = MLX.matmul(mlxX, mlxW) + mlxB
        mlxResult = MLXNN.geluApproximate(h)
        MLX.eval(mlxResult)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Benchmark Magma
    var magmaTime: Double = 0
    var magmaResult: Tensor<Float>!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let h = magmaX.matmul(magmaW) + magmaB
        magmaResult = h.gelu()
        LazyTensorBarrier(on: metalDevice)
        let _ = magmaResult.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    let accuracy = compareArrays(mlxArray: mlxResult, magmaScalars: magmaResult.scalars())

    let result = BenchmarkResult(
        operation: "ffn_layer",
        size: "B\(batch)×S\(seqLen)×\(hidden)→\(ffnDim)",
        mlxTimeMs: mlxTime,
        magmaTimeMs: magmaTime,
        accuracy: accuracy
    )
    allResults.append(result)
    printResult(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 7: Full Attention Pattern
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("7. SCALED DOT-PRODUCT ATTENTION (Fusion Pattern)")
print("This tests ScaledDotProductAttentionPattern - fuses Q·K^T/√d · softmax · V")
print()
printTableHeader()

for (batch, heads, seqLen, headDim) in [(4, 8, 128, 64), (4, 8, 256, 64), (2, 12, 512, 64)] {
    let scale = 1.0 / sqrt(Float(headDim))

    let mlxQ = MLXRandom.uniform(low: -1, high: 1, [batch, heads, seqLen, headDim])
    let mlxK = MLXRandom.uniform(low: -1, high: 1, [batch, heads, seqLen, headDim])
    let mlxV = MLXRandom.uniform(low: -1, high: 1, [batch, heads, seqLen, headDim])
    MLX.eval(mlxQ, mlxK, mlxV)

    // Create fresh Magma tensors from same data for each benchmark
    func createMagmaTensors() -> (Tensor<Float>, Tensor<Float>, Tensor<Float>) {
        let q = mlxToMagma(mlxQ, device: metalDevice)
        let k = mlxToMagma(mlxK, device: metalDevice)
        let v = mlxToMagma(mlxV, device: metalDevice)
        q.markForMaterialization()
        k.markForMaterialization()
        v.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)
        return (q, k, v)
    }

    let (magmaQ, magmaK, magmaV) = createMagmaTensors()

    // Warmup
    for _ in 0..<warmupIterations {
        let scores = MLX.matmul(mlxQ, mlxK.transposed(0, 1, 3, 2)) * scale
        let attnWeights = MLX.softmax(scores, axis: -1)
        let r = MLX.matmul(attnWeights, mlxV)
        MLX.eval(r)
    }
    for _ in 0..<warmupIterations {
        let (q, k, v) = createMagmaTensors()
        let kT = k.transpose(-1, -2)
        let scores = q.batchedMatmul(kT) * Tensor<Float>.full([], scale, on: metalDevice)
        let attnWeights = scores.softmax(dim: -1)
        let z = attnWeights.batchedMatmul(v)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    // Benchmark MLX
    var mlxTime: Double = 0
    var mlxResult: MLXArray!
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let scores = MLX.matmul(mlxQ, mlxK.transposed(0, 1, 3, 2)) * scale
        let attnWeights = MLX.softmax(scores, axis: -1)
        mlxResult = MLX.matmul(attnWeights, mlxV)
        MLX.eval(mlxResult)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Benchmark Magma - use fresh tensors to avoid graph reuse issues
    var magmaTime: Double = 0
    var magmaResult: Tensor<Float>!
    for _ in 0..<benchmarkIterations {
        let (q, k, v) = createMagmaTensors()
        let start = CFAbsoluteTimeGetCurrent()
        let kT = k.transpose(-1, -2)
        let scores = q.batchedMatmul(kT) * Tensor<Float>.full([], scale, on: metalDevice)
        let attnWeights = scores.softmax(dim: -1)
        magmaResult = attnWeights.batchedMatmul(v)
        LazyTensorBarrier(on: metalDevice)
        let _ = magmaResult.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    let accuracy = compareArrays(mlxArray: mlxResult, magmaScalars: magmaResult.scalars())

    let result = BenchmarkResult(
        operation: "attention",
        size: "B\(batch)H\(heads)S\(seqLen)D\(headDim)",
        mlxTimeMs: mlxTime,
        magmaTimeMs: magmaTime,
        accuracy: accuracy
    )
    allResults.append(result)
    printResult(result)
}

// ═══════════════════════════════════════════════════════════════════════════════
// FINAL SUMMARY TABLE
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("COMPREHENSIVE SUMMARY TABLE")

print("╔════════════════════════════════════════════════════════════════════════════════════════════════╗")
print("║  Operation          │ Size              │ MLX (ms)  │ Magma (ms)│ Speedup │ MaxRelErr │ Match ║")
print("╠════════════════════════════════════════════════════════════════════════════════════════════════╣")

for result in allResults {
    let op = result.operation.padding(toLength: 18, withPad: " ", startingAt: 0)
    let size = result.size.padding(toLength: 17, withPad: " ", startingAt: 0)
    let mlx = String(format: "%9.2f", result.mlxTimeMs)
    let magma = String(format: "%9.2f", result.magmaTimeMs)
    let speedup = String(format: "%6.2fx", result.speedup)
    let err = String(format: "%.2e", result.accuracy.maxRelError)
    let match = result.accuracy.passed ? "  ✓  " : "  ✗  "
    print("║  \(op) │ \(size) │ \(mlx) │ \(magma) │ \(speedup) │ \(err) │\(match)║")
}

print("╚════════════════════════════════════════════════════════════════════════════════════════════════╝")
print()

// Summary statistics
let passedCount = allResults.filter { $0.accuracy.passed }.count
let totalCount = allResults.count
let fasterCount = allResults.filter { $0.speedup >= 1.0 }.count

print("ACCURACY SUMMARY:")
print("  ✓ Passed: \(passedCount)/\(totalCount) operations")
if passedCount < totalCount {
    print("  ✗ Failed:")
    for result in allResults where !result.accuracy.passed {
        print("    - \(result.operation) [\(result.size)]: maxRelErr=\(String(format: "%.2e", result.accuracy.maxRelError))")
    }
}
print()

print("PERFORMANCE SUMMARY:")
print("  Magma faster: \(fasterCount)/\(totalCount) operations")
print()

let avgSpeedup = allResults.reduce(0.0) { $0 + $1.speedup } / Double(totalCount)
print("  Average speedup: \(String(format: "%.2fx", avgSpeedup))")
print()

print("Legend:")
print("  • Speedup > 1.0x means Magma is faster")
print("  • MaxRelErr = max|mlx - magma| / max(|mlx|, 1e-8)")
print("  • Match ✓ = MaxRelErr < \(accuracyTolerance) or MaxAbsErr < 1e-5")
print()

#else
print("ERROR: This benchmark requires macOS with MetalHLO support")
#endif
