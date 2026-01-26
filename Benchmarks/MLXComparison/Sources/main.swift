// MLX vs Magma Metal Comparison Benchmark
// Compares performance between MLX and Magma's Metal backend
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

// MARK: - Helper Functions

func printHeader(_ title: String) {
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  \(title.padding(toLength: 64, withPad: " ", startingAt: 0))║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
}

func printTableHeader() {
    print("Operation          │ Size           │ MLX (ms)  │ Magma (ms)│ Speedup")
    print("───────────────────┼────────────────┼───────────┼───────────┼─────────")
}

func printResult(_ op: String, _ size: String, _ mlxMs: Double, _ magmaMs: Double) {
    let opStr = op.padding(toLength: 18, withPad: " ", startingAt: 0)
    let sizeStr = size.padding(toLength: 14, withPad: " ", startingAt: 0)
    let mlxStr = String(format: "%9.2f", mlxMs)
    let magmaStr = String(format: "%9.2f", magmaMs)
    let speedup = mlxMs / magmaMs
    let speedupStr = speedup >= 1.0 ? String(format: "%6.2fx ✓", speedup) : String(format: "%6.2fx", speedup)
    print("\(opStr) │ \(sizeStr) │ \(mlxStr) │ \(magmaStr) │ \(speedupStr)")
}

// MARK: - Main Benchmark

printHeader("MLX vs MAGMA METAL COMPARISON BENCHMARK")

print("Configuration:")
print("  Warmup iterations: \(warmupIterations)")
print("  Benchmark iterations: \(benchmarkIterations)")
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

for size in [256, 512, 1024, 2048] {
    // MLX
    let mlxA = MLXRandom.uniform(low: -1, high: 1, [size, size])
    let mlxB = MLXRandom.uniform(low: -1, high: 1, [size, size])

    for _ in 0..<warmupIterations {
        let _ = MLX.matmul(mlxA, mlxB)
        MLX.eval(mlxA)
    }

    var mlxTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let result = MLX.matmul(mlxA, mlxB)
        MLX.eval(result)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Magma
    let magmaA = Tensor<Float>.randn([size, size], on: metalDevice)
    let magmaB = Tensor<Float>.randn([size, size], on: metalDevice)
    magmaA.markForMaterialization()
    magmaB.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    for _ in 0..<warmupIterations {
        let z = magmaA.matmul(magmaB)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    var magmaTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let z = magmaA.matmul(magmaB)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    printResult("matmul", "[\(size),\(size)]", mlxTime, magmaTime)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 2: MatMul + Bias + ReLU (Fused Pattern)
// Tests: MatMulBiasActivationPattern fusion
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("2. MATMUL + BIAS + RELU (Fusion Pattern)")
print("This tests MatMulBiasActivationPattern - fuses matmul, bias add, and activation")
print()
printTableHeader()

for (batch, inSize, outSize) in [(64, 512, 512), (64, 1024, 1024), (128, 768, 3072)] {
    // MLX: Manual matmul + bias + relu
    let mlxX = MLXRandom.uniform(low: -1, high: 1, [batch, inSize])
    let mlxW = MLXRandom.uniform(low: -1, high: 1, [inSize, outSize])
    let mlxB = MLXRandom.uniform(low: -1, high: 1, [outSize])

    for _ in 0..<warmupIterations {
        let h = MLX.matmul(mlxX, mlxW)
        let hBias = h + mlxB
        let _ = MLX.maximum(hBias, MLXArray(0.0))
        MLX.eval(mlxX)
    }

    var mlxTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let h = MLX.matmul(mlxX, mlxW)
        let hBias = h + mlxB
        let result = MLX.maximum(hBias, MLXArray(0.0))
        MLX.eval(result)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Magma: matmul + bias + relu (will be fused by optimization pass)
    let magmaX = Tensor<Float>.randn([batch, inSize], on: metalDevice)
    let magmaW = Tensor<Float>.randn([inSize, outSize], on: metalDevice)
    let magmaB = Tensor<Float>.randn([outSize], on: metalDevice)
    magmaX.markForMaterialization()
    magmaW.markForMaterialization()
    magmaB.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    for _ in 0..<warmupIterations {
        let h = magmaX.matmul(magmaW)
        let hBias = h + magmaB
        let z = hBias.relu()
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    var magmaTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let h = magmaX.matmul(magmaW)
        let hBias = h + magmaB
        let z = hBias.relu()
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    printResult("matmul+bias+relu", "[\(batch),\(inSize)]→\(outSize)", mlxTime, magmaTime)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 3: Softmax
// Tests: SoftmaxPattern fusion (exp, sum, div)
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("3. SOFTMAX (Fusion Pattern)")
print("This tests SoftmaxPattern - fuses exp, sum, and divide operations")
print()
printTableHeader()

for (batch, seqLen) in [(32, 512), (32, 1024), (64, 2048)] {
    // MLX
    let mlxX = MLXRandom.uniform(low: -5, high: 5, [batch, seqLen])

    for _ in 0..<warmupIterations {
        let _ = MLX.softmax(mlxX, axis: -1)
        MLX.eval(mlxX)
    }

    var mlxTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let result = MLX.softmax(mlxX, axis: -1)
        MLX.eval(result)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Magma
    let magmaX = Tensor<Float>.randn([batch, seqLen], on: metalDevice)
    magmaX.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    for _ in 0..<warmupIterations {
        let z = magmaX.softmax(dim: -1)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    var magmaTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let z = magmaX.softmax(dim: -1)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    printResult("softmax", "[\(batch),\(seqLen)]", mlxTime, magmaTime)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 4: GELU Activation
// Tests: GeluPattern fusion
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("4. GELU ACTIVATION (Fusion Pattern)")
print("This tests GeluPattern - fuses the GELU approximation operations")
print()
printTableHeader()

for size in [512, 1024, 2048] {
    let batch = 64

    // MLX
    let mlxX = MLXRandom.uniform(low: -3, high: 3, [batch, size])

    for _ in 0..<warmupIterations {
        let _ = MLXNN.gelu(mlxX)
        MLX.eval(mlxX)
    }

    var mlxTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let result = MLXNN.gelu(mlxX)
        MLX.eval(result)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Magma
    let magmaX = Tensor<Float>.randn([batch, size], on: metalDevice)
    magmaX.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    for _ in 0..<warmupIterations {
        let z = magmaX.gelu()
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    var magmaTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let z = magmaX.gelu()
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    printResult("gelu", "[\(batch),\(size)]", mlxTime, magmaTime)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 5: Layer Normalization
// Tests: LayerNormPattern fusion (mean, variance, normalize, scale, shift)
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("5. LAYER NORMALIZATION (Fusion Pattern)")
print("This tests LayerNormPattern - fuses mean, variance, normalize, scale, shift")
print()
printTableHeader()

for (batch, features) in [(64, 256), (64, 768), (128, 1024)] {
    // MLX
    let mlxX = MLXRandom.uniform(low: -3, high: 3, [batch, features])
    let mlxGamma = MLXArray.ones([features])
    let mlxBeta = MLXArray.zeros([features])

    for _ in 0..<warmupIterations {
        // Manual LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
        let mean = mlxX.mean(axis: -1, keepDims: true)
        let variance = (mlxX - mean).square().mean(axis: -1, keepDims: true)
        let normalized = (mlxX - mean) / (variance + 1e-5).sqrt()
        let _ = normalized * mlxGamma + mlxBeta
        MLX.eval(mlxX)
    }

    var mlxTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let mean = mlxX.mean(axis: -1, keepDims: true)
        let variance = (mlxX - mean).square().mean(axis: -1, keepDims: true)
        let normalized = (mlxX - mean) / (variance + 1e-5).sqrt()
        let result = normalized * mlxGamma + mlxBeta
        MLX.eval(result)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Magma: Using nn.LayerNorm (will trigger fusion)
    let ln = nn.LayerNorm(features, device: metalDevice)
    let magmaX = Tensor<Float>.randn([batch, features], on: metalDevice)
    magmaX.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    for _ in 0..<warmupIterations {
        let z = ln(magmaX)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    var magmaTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let z = ln(magmaX)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    printResult("layernorm", "[\(batch),\(features)]", mlxTime, magmaTime)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 6: Transformer Linear Layer (MatMul + Bias + GELU)
// Real-world pattern: FFN first layer
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("6. TRANSFORMER FFN (MatMul + Bias + GELU)")
print("Real-world pattern: First layer of transformer feed-forward network")
print()
printTableHeader()

for (batch, seqLen, hidden, ffnDim) in [(8, 128, 768, 3072), (8, 256, 768, 3072), (4, 512, 1024, 4096)] {
    let inputSize = batch * seqLen

    // MLX
    let mlxX = MLXRandom.uniform(low: -1, high: 1, [inputSize, hidden])
    let mlxW = MLXRandom.uniform(low: -0.1, high: 0.1, [hidden, ffnDim])
    let mlxB = MLXRandom.uniform(low: -0.1, high: 0.1, [ffnDim])

    for _ in 0..<warmupIterations {
        let h = MLX.matmul(mlxX, mlxW) + mlxB
        let _ = MLXNN.gelu(h)
        MLX.eval(mlxX)
    }

    var mlxTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let h = MLX.matmul(mlxX, mlxW) + mlxB
        let result = MLXNN.gelu(h)
        MLX.eval(result)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Magma
    let magmaX = Tensor<Float>.randn([inputSize, hidden], on: metalDevice)
    let magmaW = Tensor<Float>.randn([hidden, ffnDim], on: metalDevice)
    let magmaB = Tensor<Float>.randn([ffnDim], on: metalDevice)
    magmaX.markForMaterialization()
    magmaW.markForMaterialization()
    magmaB.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    for _ in 0..<warmupIterations {
        let h = magmaX.matmul(magmaW) + magmaB
        let z = h.gelu()
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    var magmaTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let h = magmaX.matmul(magmaW) + magmaB
        let z = h.gelu()
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    printResult("ffn_layer", "B\(batch)×S\(seqLen)×\(hidden)→\(ffnDim)", mlxTime, magmaTime)
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK 7: Full Attention Pattern
// Tests: ScaledDotProductAttentionPattern (Q @ K^T / sqrt(d) @ V)
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("7. SCALED DOT-PRODUCT ATTENTION (Fusion Pattern)")
print("This tests ScaledDotProductAttentionPattern - fuses Q·K^T/√d · softmax · V")
print()
printTableHeader()

for (batch, heads, seqLen, headDim) in [(4, 8, 128, 64), (4, 8, 256, 64), (2, 12, 512, 64)] {
    let scale = 1.0 / sqrt(Float(headDim))

    // MLX: Manual attention
    let mlxQ = MLXRandom.uniform(low: -1, high: 1, [batch, heads, seqLen, headDim])
    let mlxK = MLXRandom.uniform(low: -1, high: 1, [batch, heads, seqLen, headDim])
    let mlxV = MLXRandom.uniform(low: -1, high: 1, [batch, heads, seqLen, headDim])

    for _ in 0..<warmupIterations {
        // Q @ K^T
        let scores = MLX.matmul(mlxQ, mlxK.transposed(0, 1, 3, 2)) * scale
        // softmax
        let attnWeights = MLX.softmax(scores, axis: -1)
        // @ V
        let _ = MLX.matmul(attnWeights, mlxV)
        MLX.eval(mlxQ)
    }

    var mlxTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let scores = MLX.matmul(mlxQ, mlxK.transposed(0, 1, 3, 2)) * scale
        let attnWeights = MLX.softmax(scores, axis: -1)
        let result = MLX.matmul(attnWeights, mlxV)
        MLX.eval(result)
        mlxTime += CFAbsoluteTimeGetCurrent() - start
    }
    mlxTime = mlxTime / Double(benchmarkIterations) * 1000

    // Magma: Using nn.MultiheadAttention which will trigger fusion
    let magmaQ = Tensor<Float>.randn([batch, heads, seqLen, headDim], on: metalDevice)
    let magmaK = Tensor<Float>.randn([batch, heads, seqLen, headDim], on: metalDevice)
    let magmaV = Tensor<Float>.randn([batch, heads, seqLen, headDim], on: metalDevice)
    magmaQ.markForMaterialization()
    magmaK.markForMaterialization()
    magmaV.markForMaterialization()
    LazyTensorBarrier(on: metalDevice)

    for _ in 0..<warmupIterations {
        // Manual attention in Magma using batchedMatmul for 4D tensors
        let kT = magmaK.transpose(-1, -2)
        let scores = magmaQ.batchedMatmul(kT) * Tensor<Float>.full([], scale, on: metalDevice)
        let attnWeights = scores.softmax(dim: -1)
        let z = attnWeights.batchedMatmul(magmaV)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
    }

    var magmaTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let kT = magmaK.transpose(-1, -2)
        let scores = magmaQ.batchedMatmul(kT) * Tensor<Float>.full([], scale, on: metalDevice)
        let attnWeights = scores.softmax(dim: -1)
        let z = attnWeights.batchedMatmul(magmaV)
        LazyTensorBarrier(on: metalDevice)
        let _ = z.scalars().first
        magmaTime += CFAbsoluteTimeGetCurrent() - start
    }
    magmaTime = magmaTime / Double(benchmarkIterations) * 1000

    printResult("attention", "B\(batch)H\(heads)S\(seqLen)D\(headDim)", mlxTime, magmaTime)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════════════════════════════

printHeader("SUMMARY")
print("Notes:")
print("  • Speedup > 1.0x means Magma is faster (marked with ✓)")
print("  • Speedup < 1.0x means MLX is faster")
print("  • Fusion patterns eliminate intermediate memory round-trips")
print("  • MLX uses custom Metal compute shaders optimized for Apple Silicon")
print("  • Magma uses MPSGraph with automatic operation fusion")
print()
print("Key Insights:")
print("  • Raw matmul: MLX wins for large matrices (optimized kernels)")
print("  • Fused operations: Magma competitive due to fusion passes")
print("  • Real workloads: Fusion patterns can offset raw kernel speed")
print()

#else
print("ERROR: This benchmark requires macOS with MetalHLO support")
#endif
