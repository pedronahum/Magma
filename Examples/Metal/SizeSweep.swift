// Metal vs CPU Size Sweep Benchmark
// Tests performance across different matrix sizes to find GPU breakeven point

import Foundation
import Magma
import LazyTensor
import XLARuntime

#if os(macOS) && canImport(MetalHLO)
import MetalHLO

public func runSizeSweep() {
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          METAL vs CPU SIZE SWEEP BENCHMARK                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    let metalDevice = Device(backend: .metal, index: 0)
    let cpuDevice = Device(backend: .cpu, index: 0)

    let sizes = [128, 256, 512, 1024, 2048, 4096]
    let warmupIterations = 2
    let benchmarkIterations = 5

    print("Size       │ Metal (ms) │ Metal GFLOPS │ CPU (ms)   │ CPU GFLOPS │ Speedup")
    print("───────────┼────────────┼──────────────┼────────────┼────────────┼────────")

    for size in sizes {
        let flops = 2.0 * Double(size) * Double(size) * Double(size)

        // Metal benchmark
        let metalX = Tensor<Float>.randn([size, size], on: metalDevice)
        let metalY = Tensor<Float>.randn([size, size], on: metalDevice)
        metalX.markForMaterialization()
        metalY.markForMaterialization()
        LazyTensorBarrier(on: metalDevice)

        var metalTime: Double = 0
        for i in 0..<(warmupIterations + benchmarkIterations) {
            let start = CFAbsoluteTimeGetCurrent()
            let z = metalX.matmul(metalY)
            LazyTensorBarrier(on: metalDevice)
            let _ = z.scalars().first
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            if i >= warmupIterations {
                metalTime += elapsed
            }
        }
        metalTime /= Double(benchmarkIterations)
        let metalGFLOPS = flops / metalTime / 1e9

        // CPU benchmark (if available)
        var cpuTime: Double = 0
        var cpuGFLOPS: Double = 0
        var speedup: Double = 0

        if Backend.cpu.isAvailable {
            let cpuX = Tensor<Float>.randn([size, size], on: cpuDevice)
            let cpuY = Tensor<Float>.randn([size, size], on: cpuDevice)
            cpuX.markForMaterialization()
            cpuY.markForMaterialization()
            LazyTensorBarrier(on: cpuDevice)

            for i in 0..<(warmupIterations + benchmarkIterations) {
                let start = CFAbsoluteTimeGetCurrent()
                let z = cpuX.matmul(cpuY)
                LazyTensorBarrier(on: cpuDevice)
                let _ = z.scalars().first
                let elapsed = CFAbsoluteTimeGetCurrent() - start
                if i >= warmupIterations {
                    cpuTime += elapsed
                }
            }
            cpuTime /= Double(benchmarkIterations)
            cpuGFLOPS = flops / cpuTime / 1e9
            speedup = cpuTime / metalTime
        }

        let sizeStr = String(format: "%4dx%-4d", size, size)
        let metalTimeStr = String(format: "%8.2f", metalTime * 1000)
        let metalGFLOPSStr = String(format: "%10.1f", metalGFLOPS)

        if Backend.cpu.isAvailable {
            let cpuTimeStr = String(format: "%8.2f", cpuTime * 1000)
            let cpuGFLOPSStr = String(format: "%10.1f", cpuGFLOPS)
            let speedupStr = speedup >= 1.0 ?
                String(format: " %5.2fx", speedup) :
                String(format: " %5.2fx", speedup)
            print("\(sizeStr)  │ \(metalTimeStr) │ \(metalGFLOPSStr)   │ \(cpuTimeStr) │ \(cpuGFLOPSStr) │\(speedupStr)")
        } else {
            print("\(sizeStr)  │ \(metalTimeStr) │ \(metalGFLOPSStr)   │     N/A    │        N/A │    N/A")
        }
    }

    print()
    print("Note: Speedup > 1.0 means Metal is faster than CPU")
    print("      First iteration for each size includes compilation overhead")
}

#endif
