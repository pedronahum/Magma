// MLX vs Magma Metal Comparison Benchmark
// Compares matrix multiplication performance between MLX and Magma's Metal backend

import Foundation
import MLX
import Magma
import LazyTensor
import XLARuntime

#if os(macOS) && canImport(MetalHLO)
import MetalHLO
#endif

print("╔══════════════════════════════════════════════════════════════════╗")
print("║           MLX vs MAGMA METAL COMPARISON BENCHMARK                ║")
print("╚══════════════════════════════════════════════════════════════════╝\n")

let sizes = [128, 256, 512, 1024, 2048, 4096]
let warmupIterations = 3
let benchmarkIterations = 10

print("Warmup iterations: \(warmupIterations)")
print("Benchmark iterations: \(benchmarkIterations)")
print()

print("Size       │ MLX (ms)   │ MLX GFLOPS  │ Magma (ms) │ Magma GFLOPS│ MLX/Magma")
print("───────────┼────────────┼─────────────┼────────────┼─────────────┼──────────")

for size in sizes {
    let flops = 2.0 * Double(size) * Double(size) * Double(size)

    // ═══════════════════════════════════════════════════════════════════════
    // MLX Benchmark
    // ═══════════════════════════════════════════════════════════════════════

    // Create random matrices in MLX
    let mlxA = MLXRandom.uniform(low: -1, high: 1, [size, size])
    let mlxB = MLXRandom.uniform(low: -1, high: 1, [size, size])

    // Warmup MLX
    for _ in 0..<warmupIterations {
        let _ = MLX.matmul(mlxA, mlxB)
        MLX.eval(mlxA)  // Force synchronization
    }

    // Benchmark MLX
    var mlxTime: Double = 0
    for _ in 0..<benchmarkIterations {
        let start = CFAbsoluteTimeGetCurrent()
        let result = MLX.matmul(mlxA, mlxB)
        MLX.eval(result)  // Force synchronization
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        mlxTime += elapsed
    }
    mlxTime /= Double(benchmarkIterations)
    let mlxGFLOPS = flops / mlxTime / 1e9

    // ═══════════════════════════════════════════════════════════════════════
    // Magma Metal Benchmark
    // ═══════════════════════════════════════════════════════════════════════

    var magmaTime: Double = 0
    var magmaGFLOPS: Double = 0

    #if os(macOS) && canImport(MetalHLO)
    if Backend.metal.isAvailable {
        let metalDevice = Device(backend: .metal, index: 0)

        // Create and materialize tensors
        let magmaA = Tensor<Float>.randn([size, size], on: metalDevice)
        let magmaB = Tensor<Float>.randn([size, size], on: metalDevice)
        magmaA.markForMaterialization()
        magmaB.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        // Warmup Magma
        for _ in 0..<warmupIterations {
            let z = magmaA.matmul(magmaB)
            LazyTensorBarrier(on: metalDevice)
            let _ = z.scalars().first
        }

        // Benchmark Magma
        for _ in 0..<benchmarkIterations {
            let start = CFAbsoluteTimeGetCurrent()
            let z = magmaA.matmul(magmaB)
            LazyTensorBarrier(on: metalDevice)
            let _ = z.scalars().first
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            magmaTime += elapsed
        }
        magmaTime /= Double(benchmarkIterations)
        magmaGFLOPS = flops / magmaTime / 1e9
    }
    #endif

    // Print results
    let sizeStr = String(format: "%4dx%-4d", size, size)
    let mlxTimeStr = String(format: "%8.2f", mlxTime * 1000)
    let mlxGFLOPSStr = String(format: "%9.1f", mlxGFLOPS)

    if magmaTime > 0 {
        let magmaTimeStr = String(format: "%8.2f", magmaTime * 1000)
        let magmaGFLOPSStr = String(format: "%9.1f", magmaGFLOPS)
        let ratio = mlxTime / magmaTime
        let ratioStr = String(format: "%7.2fx", ratio)
        print("\(sizeStr)  │ \(mlxTimeStr) │ \(mlxGFLOPSStr)   │ \(magmaTimeStr) │ \(magmaGFLOPSStr)  │ \(ratioStr)")
    } else {
        print("\(sizeStr)  │ \(mlxTimeStr) │ \(mlxGFLOPSStr)   │      N/A   │       N/A   │      N/A")
    }
}

print()
print("═══════════════════════════════════════════════════════════════════")
print()
print("Notes:")
print("  - MLX/Magma > 1.0 means Magma Metal is faster")
print("  - MLX/Magma < 1.0 means MLX is faster")
print("  - Both frameworks use Metal GPU acceleration")
print("  - MLX is highly optimized by Apple for Apple Silicon")
print()

// Additional info
print("System Info:")
#if os(macOS) && canImport(MetalHLO)
if let deviceName = MetalBackend.deviceName {
    print("  Metal Device: \(deviceName)")
}
#endif
print("  MLX Version: 0.21.x")
print()
