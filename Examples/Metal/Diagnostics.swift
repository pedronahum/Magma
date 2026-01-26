// Metal Performance Diagnostics
// Run this to understand where time is being spent

import Foundation
import Magma
import LazyTensor
import XLARuntime

#if os(macOS) && canImport(MetalHLO)
import MetalHLO

/// Runs detailed performance diagnostics on the Metal backend
public func runMetalDiagnostics() {
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║              METAL PERFORMANCE DIAGNOSTICS                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝\n")

    let metalDevice = Device(backend: .metal, index: 0)

    // Test 1: Direct MetalHLO execution (bypassing LazyTensor)
    print("TEST 1: Direct MetalHLO Execution (bypassing LazyTensor)")
    print("─────────────────────────────────────────────────────────")
    directMetalHLOTest()
    print()

    // Test 2: Compilation caching
    print("TEST 2: Compilation Cache Behavior")
    print("───────────────────────────────────")
    compilationCacheTest()
    print()

    // Test 3: LazyTensor overhead
    print("TEST 3: LazyTensor Overhead Analysis")
    print("─────────────────────────────────────")
    lazyTensorOverheadTest(device: metalDevice)
    print()

    // Test 4: Data transfer overhead
    print("TEST 4: Data Transfer Overhead")
    print("──────────────────────────────")
    dataTransferTest()
    print()

    print("═══════════════════════════════════════════════════════════════════")
    print("DIAGNOSIS COMPLETE")
    print("═══════════════════════════════════════════════════════════════════")
}

/// Direct MetalHLO execution without LazyTensor
func directMetalHLOTest() {
    do {
        let client = try MetalHLOClient.create()
        let size = 512

        // Simple MLIR for matmul
        let mlir = """
        module @matmul_test {
          func.func @main(%arg0: tensor<\(size)x\(size)xf32>, %arg1: tensor<\(size)x\(size)xf32>) -> (tensor<\(size)x\(size)xf32>) {
            %0 = stablehlo.dot %arg0, %arg1 : (tensor<\(size)x\(size)xf32>, tensor<\(size)x\(size)xf32>) -> tensor<\(size)x\(size)xf32>
            return %0 : tensor<\(size)x\(size)xf32>
          }
        }
        """

        // Time compilation
        let compileStart = CFAbsoluteTimeGetCurrent()
        let executable = try client.compile(mlir)
        let compileTime = CFAbsoluteTimeGetCurrent() - compileStart
        print("  Compilation: \(String(format: "%.3f", compileTime * 1000)) ms")

        // Create input data
        let dataA = (0..<size*size).map { _ in Float.random(in: -1...1) }
        let dataB = (0..<size*size).map { _ in Float.random(in: -1...1) }

        // Time buffer creation (now optimized - no type conversion)
        let bufferStart = CFAbsoluteTimeGetCurrent()
        let bufferA = client.createBuffer(dataA, shape: [size, size])
        let bufferB = client.createBuffer(dataB, shape: [size, size])
        let bufferTime = CFAbsoluteTimeGetCurrent() - bufferStart
        print("  Buffer creation: \(String(format: "%.3f", bufferTime * 1000)) ms")

        // Time execution with detailed timing
        let execStart = CFAbsoluteTimeGetCurrent()
        let (outputs, timing) = try executable.executeWithTiming([bufferA, bufferB])
        let execTime = CFAbsoluteTimeGetCurrent() - execStart

        print("  Execution (wall clock): \(String(format: "%.3f", execTime * 1000)) ms")
        print("    - Encode time: \(String(format: "%.3f", timing.encodeTimeNs / 1_000_000)) ms")
        print("    - GPU time: \(String(format: "%.3f", timing.gpuTimeNs / 1_000_000)) ms")
        print("    - Total (internal): \(String(format: "%.3f", timing.totalMs)) ms")

        // Time readback
        let readbackStart = CFAbsoluteTimeGetCurrent()
        let _ = try outputs[0].toFloatArray()
        let readbackTime = CFAbsoluteTimeGetCurrent() - readbackStart
        print("  Readback: \(String(format: "%.3f", readbackTime * 1000)) ms")

        let total = compileTime + bufferTime + execTime + readbackTime
        print("  TOTAL: \(String(format: "%.3f", total * 1000)) ms")

    } catch {
        print("  ERROR: \(error)")
    }
}

/// Test compilation cache behavior
func compilationCacheTest() {
    do {
        let client = try MetalHLOClient.create()

        let mlir = """
        module @cache_test {
          func.func @main(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> (tensor<256x256xf32>) {
            %0 = stablehlo.add %arg0, %arg1 : tensor<256x256xf32>
            return %0 : tensor<256x256xf32>
          }
        }
        """

        // Clear cache
        MetalCompilationCache.shared.clear()

        var times: [Double] = []
        for i in 0..<5 {
            let start = CFAbsoluteTimeGetCurrent()
            let _ = try client.compile(mlir)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            times.append(elapsed)
            print("  Compile \(i + 1): \(String(format: "%.3f", elapsed)) ms")
        }

        print()
        print("  Note: MetalHLO has internal compilation cache in MetalExecutor")
        print("  Magma also has MetalCompilationCache at LazyTensor level")

    } catch {
        print("  ERROR: \(error)")
    }
}

/// Analyze LazyTensor overhead
func lazyTensorOverheadTest(device: Device) {
    let size = 256

    // Create tensors
    let a = Tensor<Float>.randn([size, size], on: device)
    let b = Tensor<Float>.randn([size, size], on: device)

    // Warm up
    let _ = a.matmul(b)
    LazyTensorBarrier(on: device)

    // Time multiple iterations
    var times: [Double] = []
    for i in 0..<5 {
        let c = a.matmul(b)

        let start = CFAbsoluteTimeGetCurrent()
        LazyTensorBarrier(on: device)
        let _ = c.scalars().first
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        print("  LazyTensor iteration \(i + 1): \(String(format: "%.3f", elapsed)) ms")
    }

    let avg = times.reduce(0, +) / Double(times.count)
    print("  Average: \(String(format: "%.3f", avg)) ms")
}

/// Test data transfer overhead
func dataTransferTest() {
    do {
        let client = try MetalHLOClient.create()
        let sizes = [256, 512, 1024]

        for size in sizes {
            let elementCount = size * size
            let byteCount = elementCount * 4  // float32

            let data = (0..<elementCount).map { _ in Float.random(in: -1...1) }

            // H2D (Host to Device) - now optimized
            let h2dStart = CFAbsoluteTimeGetCurrent()
            let buffer = client.createBuffer(data, shape: [size, size])
            let h2dTime = (CFAbsoluteTimeGetCurrent() - h2dStart) * 1000

            // D2H (Device to Host)
            let d2hStart = CFAbsoluteTimeGetCurrent()
            let _ = try buffer.toFloatArray()
            let d2hTime = (CFAbsoluteTimeGetCurrent() - d2hStart) * 1000

            let bandwidth_h2d = Double(byteCount) / (h2dTime / 1000) / (1024*1024*1024)
            let bandwidth_d2h = Double(byteCount) / (d2hTime / 1000) / (1024*1024*1024)

            print("  \(size)x\(size) (\(byteCount/1024) KB):")
            print("    H2D: \(String(format: "%.3f", h2dTime)) ms (\(String(format: "%.2f", bandwidth_h2d)) GB/s)")
            print("    D2H: \(String(format: "%.3f", d2hTime)) ms (\(String(format: "%.2f", bandwidth_d2h)) GB/s)")
        }

    } catch {
        print("  ERROR: \(error)")
    }
}

#endif
