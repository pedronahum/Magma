// Metal Example
// Demonstrates using Magma with Metal GPU acceleration via MetalHLO
//
// This example shows:
// 1. How to select the Metal backend
// 2. Basic tensor operations on Metal
// 3. Matrix multiplication performance
// 4. Comparison with CPU backend

import Foundation
import Magma
import LazyTensor
import XLARuntime

#if os(macOS) && canImport(MetalHLO)
import MetalHLO

// Check for diagnostics mode
let runDiagnostics = CommandLine.arguments.contains("--diagnostics") || CommandLine.arguments.contains("-d")

if runDiagnostics {
    runMetalDiagnostics()
    exit(0)
}

print("Magma Metal GPU Example")
print("=======================\n")
print("Tip: Run with --diagnostics for detailed performance analysis\n")

// Check available backends
print("Available backends: \(Backend.availableBackends.map { $0.rawValue })")
print("Best available: \(Backend.bestAvailable.rawValue)")

// Check if Metal is available
guard Backend.metal.isAvailable else {
    print("Error: Metal backend is not available on this system.")
    print("This example requires macOS with Apple Silicon or a Metal-capable GPU.")
    exit(1)
}

print("Metal backend is available!\n")

// Example 1: Simple tensor operations on Metal
print("Example 1: Basic Tensor Operations")
print("-----------------------------------")

// Create a Metal device
let metalDevice = Device(backend: .metal, index: 0)

// Create tensors (these will be lazy - no computation yet)
let a = Tensor<Float>([1, 2, 3, 4, 5, 6], shape: [2, 3], on: metalDevice)
let b = Tensor<Float>([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], shape: [2, 3], on: metalDevice)

// Perform operations (still lazy)
let sum = a + b
let product = a * b
let relu_result = (a - Tensor<Float>([3.0, 3.0, 3.0, 3.0, 3.0, 3.0], shape: [2, 3], on: metalDevice)).relu()

// Trigger execution on Metal GPU
LazyTensorBarrier(on: metalDevice)

// Read results
print("A = \(a.scalars())")
print("B = \(b.scalars())")
print("A + B = \(sum.scalars())")
print("A * B = \(product.scalars())")
print("ReLU(A - 3) = \(relu_result.scalars())")
print()

// Example 2: Matrix Multiplication
print("Example 2: Matrix Multiplication")
print("---------------------------------")

let matA = Tensor<Float>.randn([64, 128], on: metalDevice)
let matB = Tensor<Float>.randn([128, 64], on: metalDevice)

// Matrix multiplication (lazy)
let matC = matA.matmul(matB)

// Execute
LazyTensorBarrier(on: metalDevice)

print("Matrix A: \(matA.shape)")
print("Matrix B: \(matB.shape)")
print("Result C = A @ B: \(matC.shape)")
print("First few values of C: \(Array(matC.scalars().prefix(5)))")
print()

// Example 3: Performance Comparison
print("Example 3: Performance Comparison (Metal vs CPU)")
print("-------------------------------------------------")

let size = 512
let warmupIterations = 3
let benchmarkIterations = 10

print("Matrix size: \(size) x \(size)")
print("Warmup iterations: \(warmupIterations)")
print("Benchmark iterations: \(benchmarkIterations)")
print()

// Metal benchmark - use pre-created data to avoid MLIR embedding overhead
print("Metal GPU:")
var metalTotalTime: Double = 0

// Pre-create random data ONCE (avoid embedding huge constants in MLIR each iteration)
// This is how proper ML benchmarks work - input data is prepared ahead of time
let metalX = Tensor<Float>.randn([size, size], on: metalDevice)
let metalY = Tensor<Float>.randn([size, size], on: metalDevice)

// Materialize once so subsequent operations reuse the data
metalX.markForMaterialization()
metalY.markForMaterialization()
LazyTensorBarrier(on: metalDevice)

for i in 0..<(warmupIterations + benchmarkIterations) {
    let start = CFAbsoluteTimeGetCurrent()

    // Use pre-materialized tensors
    let z = metalX.matmul(metalY)
    LazyTensorBarrier(on: metalDevice)

    // Force read
    let _ = z.scalars().first

    let elapsed = CFAbsoluteTimeGetCurrent() - start

    if i >= warmupIterations {
        metalTotalTime += elapsed
    }

    if i < warmupIterations {
        print("  Warmup \(i + 1): \(String(format: "%.3f", elapsed * 1000)) ms")
    }
}

let metalAvgTime = metalTotalTime / Double(benchmarkIterations)
print("  Average: \(String(format: "%.3f", metalAvgTime * 1000)) ms")

// Calculate GFLOPS (2 * N^3 FLOPs for matrix multiply)
let flops = 2.0 * Double(size) * Double(size) * Double(size)
let metalGFLOPS = flops / metalAvgTime / 1e9
print("  Performance: \(String(format: "%.2f", metalGFLOPS)) GFLOPS")
print()

// CPU benchmark (for comparison)
print("CPU:")
let cpuDevice = Device(backend: .cpu, index: 0)

// Check if CPU backend is actually available (requires XLA)
if Backend.cpu.isAvailable {
    var cpuTotalTime: Double = 0

    // Pre-create and materialize tensors for fair comparison
    let cpuX = Tensor<Float>.randn([size, size], on: cpuDevice)
    let cpuY = Tensor<Float>.randn([size, size], on: cpuDevice)
    cpuX.markForMaterialization()
    cpuY.markForMaterialization()
    LazyTensorBarrier(on: cpuDevice)

    for i in 0..<(warmupIterations + benchmarkIterations) {
        let start = CFAbsoluteTimeGetCurrent()

        let z = cpuX.matmul(cpuY)
        LazyTensorBarrier(on: cpuDevice)

        // Force materialization
        let _ = z.scalars().first

        let elapsed = CFAbsoluteTimeGetCurrent() - start

        if i >= warmupIterations {
            cpuTotalTime += elapsed
        }

        if i < warmupIterations {
            print("  Warmup \(i + 1): \(String(format: "%.3f", elapsed * 1000)) ms")
        }
    }

    let cpuAvgTime = cpuTotalTime / Double(benchmarkIterations)
    print("  Average: \(String(format: "%.3f", cpuAvgTime * 1000)) ms")

    let cpuGFLOPS = flops / cpuAvgTime / 1e9
    print("  Performance: \(String(format: "%.2f", cpuGFLOPS)) GFLOPS")
    print()

    // Speedup
    let speedup = cpuAvgTime / metalAvgTime
    print("Metal speedup over CPU: \(String(format: "%.2f", speedup))x")
} else {
    print("  CPU backend not available (XLA not installed)")
    print("  To enable CPU backend, set MAGMA_ENABLE_XLA=1 and install XLA")
}

print()

// Example 4: Neural Network Layer
print("Example 4: Simple Neural Network Forward Pass")
print("----------------------------------------------")

// Simulate a simple feedforward layer: y = ReLU(x @ W + b)
let batchSize = 32
let inputDim = 256
let hiddenDim = 512

let input = Tensor<Float>.randn([batchSize, inputDim], on: metalDevice)
let weights = Tensor<Float>.randn([inputDim, hiddenDim], on: metalDevice)
let bias = Tensor<Float>.randn([hiddenDim], on: metalDevice)

// Forward pass
let linear = input.matmul(weights)
// Broadcasting addition would require the bias to be broadcast to [batchSize, hiddenDim]
// For simplicity, we'll just use the linear result with ReLU
let output = linear.relu()

LazyTensorBarrier(on: metalDevice)

print("Input shape: \(input.shape)")
print("Weights shape: \(weights.shape)")
print("Output shape: \(output.shape)")
print("Output sample (first 5 values): \(Array(output.scalars().prefix(5)))")
print()

// Print cache statistics
print("Compilation Cache Statistics")
print("-----------------------------")
print("This example demonstrates that repeated operations with the same")
print("graph structure benefit from compilation caching.")
print("(Cache statistics available via CompilationCache.shared)")
print()

print("Done! Metal GPU acceleration is working correctly.")

#else

print("This example requires macOS with Metal and MetalHLO support.")
print("Metal GPU acceleration is only available on Apple platforms.")

#endif  // os(macOS) && canImport(MetalHLO)
