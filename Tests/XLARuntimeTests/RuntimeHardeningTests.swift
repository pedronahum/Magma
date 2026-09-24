// Magma - Runtime hardening tests
// Regressions for the PJRT runtime layer found while preparing the alpha:
// - executables with more than 16 outputs overflowed a fixed output array;
// - the output-count cache kept entries for destroyed executables;
// - PJRT error messages were dropped (only "code N" reached the user);
// - toHost / createBuffer did not check host element sizes;
// - a client could be destroyed while its buffers/executables were alive;
// - unsupported output element types were silently reported as f32.

import Foundation
import Testing
import CXLARuntime
@testable import XLARuntime
@testable import LazyTensor
@testable import Magma

/// MLIR for `func(x: tensor<4xf32>) -> (x + 0, x + 1, ..., x + n-1)`.
private func manyOutputsModule(_ n: Int) -> String {
    var body = ""
    var results: [String] = []
    for i in 0..<n {
        body += "    %c\(i) = stablehlo.constant dense<\(Float(i))> : tensor<4xf32>\n"
        body += "    %r\(i) = stablehlo.add %arg0, %c\(i) : tensor<4xf32>\n"
        results.append("%r\(i)")
    }
    let types = Array(repeating: "tensor<4xf32>", count: n).joined(separator: ", ")
    return """
    module @many_outputs {
      func.func @main(%arg0: tensor<4xf32>) -> (\(types)) {
    \(body)    return \(results.joined(separator: ", ")) : \(types)
      }
    }
    """
}

/// Runs `exe` once through `PJRT_ExecuteWrapper` directly and reports how many
/// outputs it produced and how many handles the storage PJRT wrote them into
/// can hold. Values alone cannot catch an overflow of that storage: PJRT
/// writes the handles contiguously and they read back correctly even while
/// the write overruns neighbouring thread-local memory.
private func rawExecuteStorage(
    _ exe: PJRTExecutable, _ input: PJRTBuffer
) throws -> (outputs: Int, capacity: Int) {
    var inputHandles: [UnsafeMutableRawPointer?] = [input.handle]
    var outputsPtr: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
    var numOutputs = 0
    let code = inputHandles.withUnsafeMutableBufferPointer { inputs in
        PJRT_ExecuteWrapper(exe.handle, inputs.baseAddress, 1, &outputsPtr, &numOutputs)
    }
    guard code == SW_PJRT_Error_OK, let outputs = outputsPtr else {
        throw XLAError.executionFailed("PJRT_ExecuteWrapper failed with code \(code)")
    }
    // Ask for the capacity before anything else can execute on this thread.
    let capacity = PJRT_Testing_OutputStorageCapacity(outputs)
    for i in 0..<numOutputs { PJRT_DestroyBuffer(outputs[i]) }
    return (numOutputs, capacity)
}

@Suite("Runtime Hardening Tests", .serialized,
       .enabled(if: PluginAvailability.cpu, "CPU PJRT plugin not available"))
struct RuntimeHardeningTests {

    // MARK: - More than 16 outputs

    @Test("an executable with 24 outputs returns every output with correct values")
    func manyOutputs() throws {
        let client = try PJRTClient.create(backend: .cpu)
        let exe = try client.compile(manyOutputsModule(24))
        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)

        // Execute twice: the second run reuses the (grown) thread-local storage.
        for _ in 0..<2 {
            let outs = try exe.execute([x])
            #expect(outs.count == 24)
            for (i, out) in outs.enumerated() {
                let offset = Float(i)
                #expect(try out.toFloatArray() == [1 + offset, 2 + offset, 3 + offset, 4 + offset])
            }
        }
        #expect(exe.executionCount == 2)
    }

    @Test("execute output storage holds every output of a 24-output executable")
    func manyOutputsStorageCapacity() throws {
        let client = try PJRTClient.create(backend: .cpu)
        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)

        // Small first, then large, then small again: the storage must grow
        // for the large executable and still fit the small one afterwards.
        for n in [3, 24, 17, 5] {
            let exe = try client.compile(manyOutputsModule(n))
            let (outputs, capacity) = try rawExecuteStorage(exe, x)
            #expect(outputs == n)
            #expect(capacity >= n, "\(n) outputs written into storage for \(capacity)")
        }
    }

    // Checks values only; the storage size itself is covered by
    // manyOutputsStorageCapacity (a barrier runs through the same wrapper).
    @Test("a barrier materializing 24 tensors returns every value")
    func barrierWithManyOutputs() throws {
        TensorRegistry.shared.clearAll()
        let x = Tensor<Float>([1, 2, 3, 4], shape: [4])
        let results = (0..<24).map { i in x * Tensor<Float>.full([4], Float(i + 1)) }
        for r in results { r.markForMaterialization() }
        LazyTensorBarrier()

        for (i, r) in results.enumerated() {
            let k = Float(i + 1)
            #expect(r.scalars() == [k, 2 * k, 3 * k, 4 * k])
        }
    }

    @Test("destroying an executable evicts its cached output count")
    func outputCountCacheIsEvictedOnDestroy() throws {
        let client = try PJRTClient.create(backend: .cpu)
        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)

        var exe: PJRTExecutable? = try client.compile(manyOutputsModule(20))
        _ = try exe!.execute([x])   // populates the cache
        let handle = try #require(exe!.handle)
        #expect(PJRT_Testing_HasCachedOutputCount(handle))

        exe = nil   // PJRT_DestroyExecutable runs in deinit
        // A stale entry would be reused if the allocator hands this address
        // to a later executable with a different output count.
        #expect(!PJRT_Testing_HasCachedOutputCount(handle))
    }

    @Test("output counts stay correct while executables are created and destroyed")
    func outputCountsStayCorrectUnderChurn() throws {
        let client = try PJRTClient.create(backend: .cpu)
        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)

        // Alternate output counts so a stale cache entry at a recycled address
        // would report the wrong number of outputs.
        for round in 0..<40 {
            let n = round % 2 == 0 ? 1 : 20
            // The executable is a temporary, destroyed right after executing.
            let outs = try client.compile(manyOutputsModule(n)).execute([x])
            #expect(outs.count == n)
            #expect(try outs.last?.toFloatArray() == [1, 2, 3, 4].map { $0 + Float(n - 1) })
        }
    }

    // MARK: - Error messages

    @Test("a compile error carries XLA's diagnostic, not just a code")
    func compileErrorMessage() throws {
        let client = try PJRTClient.create(backend: .cpu)
        let bad = """
        module @bad {
          func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            %0 = stablehlo.not_a_real_op %arg0 : tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        do {
            _ = try client.compile(bad)
            Issue.record("compiling invalid MLIR should throw")
        } catch let error as XLAError {
            guard case .compilationFailed(let message) = error else {
                Issue.record("expected compilationFailed, got \(error)")
                return
            }
            #expect(message.contains("not_a_real_op"), "message was: \(message)")
        }
    }

    @Test("an execute error carries PJRT's message")
    func executeErrorMessage() throws {
        let client = try PJRTClient.create(backend: .cpu)
        let exe = try client.compile(manyOutputsModule(1))
        do {
            _ = try exe.execute([])   // the program expects one argument
            Issue.record("executing with the wrong argument count should throw")
        } catch let error as XLAError {
            guard case .executionFailed(let message) = error else {
                Issue.record("expected executionFailed, got \(error)")
                return
            }
            // More than the bare status: PJRT explains the argument mismatch.
            #expect(message.contains("):"), "message was: \(message)")
            #expect(message.lowercased().contains("argument"), "message was: \(message)")
        }
        #expect(exe.executionCount == 0)
    }

    @Test("a rejected plugin load records why")
    func pluginMismatchMessage() throws {
        _ = try PJRTClient.create(backend: .cpu)   // ensure the CPU plugin is loaded
        PJRT_ClearLastErrorMessage()
        #expect(PJRT_LoadPlugin("/nonexistent/other_plugin.so") != SW_PJRT_Error_OK)
        let message = PJRT_GetLastErrorMessage().map { String(cString: $0) }
        #expect(message?.contains("mismatch") == true, "message was: \(message ?? "nil")")
    }

    // MARK: - Host element size validation

    @Test("toHost rejects a Swift type whose width differs from the buffer's")
    func toHostChecksElementWidth() throws {
        let client = try PJRTClient.create(backend: .cpu)
        let buffer = try client.createBuffer([1, 2, 3] as [Float], shape: [3], elementType: .float32)
        #expect(throws: XLAError.self) { try buffer.toHost(Double.self) }
        #expect(throws: XLAError.self) { try buffer.toHost(UInt8.self) }
        #expect(try buffer.toHost(Float.self) == [1, 2, 3])
    }

    @Test("createBuffer rejects data whose byte count does not match the shape")
    func createBufferChecksByteCount() throws {
        let client = try PJRTClient.create(backend: .cpu)
        // Too few elements for the shape.
        #expect(throws: XLAError.self) {
            try client.createBuffer([1, 2, 3] as [Float], shape: [4], elementType: .float32)
        }
        // Right count, wrong width: 4 Doubles are 32 bytes, 4 f32 are 16.
        #expect(throws: XLAError.self) {
            try client.createBuffer([1, 2, 3, 4] as [Double], shape: [4], elementType: .float32)
        }
        let ok = try client.createBuffer([1, 2, 3, 4] as [Double], shape: [2, 2], elementType: .float64)
        #expect(try ok.toHost(Double.self) == [1, 2, 3, 4])
    }

    @Test("toFloatArray converts non-f32 outputs numerically")
    func toFloatArrayConverts() throws {
        let client = try PJRTClient.create(backend: .cpu)
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf64>, tensor<4xi1>, tensor<4xbf16>, tensor<4xf16>) {
            %0 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xi32>
            %1 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xf64>
            %z = stablehlo.constant dense<2.0> : tensor<4xf32>
            %2 = stablehlo.compare GT, %arg0, %z : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
            %3 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xbf16>
            %4 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xf16>
            return %0, %1, %2, %3, %4 : tensor<4xi32>, tensor<4xf64>, tensor<4xi1>, tensor<4xbf16>, tensor<4xf16>
          }
        }
        """
        let exe = try client.compile(mlir)
        let x = try client.createBuffer([1, -2, 3, 0.5] as [Float], shape: [4], elementType: .float32)
        let outs = try exe.execute([x])
        #expect(outs.map(\.elementType) == [.int32, .float64, .bool, .bfloat16, .float16])
        #expect(try outs[0].toFloatArray() == [1, -2, 3, 0])
        #expect(try outs[1].toFloatArray() == [1, -2, 3, 0.5])
        #expect(try outs[2].toFloatArray() == [0, 0, 1, 0])
        #expect(try outs[3].toFloatArray() == [1, -2, 3, 0.5])
        #expect(try outs[4].toFloatArray() == [1, -2, 3, 0.5])
        #expect(try outs[1].toHost(Double.self) == [1, -2, 3, 0.5])
    }

    // MARK: - Unsupported element types

    @Test("an output with an unsupported element type throws instead of posing as f32")
    func unsupportedOutputElementType() throws {
        let client = try PJRTClient.create(backend: .cpu)
        let mlir = """
        module @m {
          func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf8E4M3FN> {
            %0 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xf8E4M3FN>
            return %0 : tensor<4xf8E4M3FN>
          }
        }
        """
        let exe = try client.compile(mlir)
        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        do {
            _ = try exe.execute([x])
            Issue.record("an f8 output should be rejected")
        } catch let error as XLAError {
            #expect("\(error)".contains("unsupported PJRT element type"), "error was: \(error)")
        }
    }

    // MARK: - Client lifetime

    @Test("buffers and executables keep their client alive")
    func clientOutlivesItsObjects() throws {
        weak var weakClient: PJRTClient?
        var buffer: PJRTBuffer?
        var exe: PJRTExecutable?
        do {
            let client = try PJRTClient.create(backend: .cpu)
            weakClient = client
            buffer = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
            exe = try client.compile(manyOutputsModule(2))
        }
        // The only strong references to the client are now the buffer and executable.
        #expect(weakClient != nil)
        let outs = try #require(exe).execute([try #require(buffer)])
        #expect(try outs[1].toFloatArray() == [2, 3, 4, 5])

        buffer = nil
        exe = nil
        #expect(weakClient != nil)       // the outputs still hold it
        _ = outs
    }

    // MARK: - Concurrency

    @Test("executionCount is exact under concurrent execution")
    func concurrentExecutionCount() throws {
        let client = try PJRTClient.create(backend: .cpu)
        let exe = try client.compile(manyOutputsModule(20))
        let x = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4], elementType: .float32)
        let threads = 8
        let perThread = 25
        let failures = LockedCounter()

        DispatchQueue.concurrentPerform(iterations: threads) { _ in
            for _ in 0..<perThread {
                do {
                    let outs = try exe.execute([x])
                    if try outs.count != 20 || outs[19].toFloatArray() != [20, 21, 22, 23] {
                        failures.increment()
                    }
                    // Each thread has its own storage; it must fit all 20 outputs.
                    let (outputs, capacity) = try rawExecuteStorage(exe, x)
                    if outputs != 20 || capacity < 20 {
                        failures.increment()
                    }
                } catch {
                    failures.increment()
                }
            }
        }
        #expect(failures.value == 0)
        #expect(exe.executionCount == threads * perThread)
    }

    @Test("concurrent CPU client creation is safe")
    func concurrentClientCreation() {
        let failures = LockedCounter()
        DispatchQueue.concurrentPerform(iterations: 8) { _ in
            if (try? PJRTClient.create(backend: .cpu)) == nil { failures.increment() }
        }
        #expect(failures.value == 0)
    }
}

private final class LockedCounter: @unchecked Sendable {
    private let lock = Foundation.NSLock()
    private var count = 0
    func increment() { lock.lock(); count += 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return count }
}

// MARK: - Plugin path resolution (no plugin is loaded by these tests)

@Suite("Plugin Resolution Tests")
struct PluginResolutionTests {
    private static let ext: String = {
        #if os(macOS)
        return "dylib"
        #else
        return "so"
        #endif
    }()

    @Test("MAGMA_XLA_PATH accepts the lib-prefixed plugin name")
    func libPrefixedName() {
        let target = "/xla/libpjrt_c_api_cpu_plugin.\(Self.ext)"
        let r = Backend.resolvePlugin(for: .cpu, environment: ["MAGMA_XLA_PATH": "/xla"],
                                      fileExists: { $0 == target })
        #expect(r.path == target)
        #expect(r.searched.first == "/xla/pjrt_c_api_cpu_plugin.\(Self.ext)")
    }

    @Test("MAGMA_XLA_PATH accepts JAX's xla_cuda_plugin name for GPU")
    func jaxCudaName() {
        let target = "/xla/xla_cuda_plugin.\(Self.ext)"
        let r = Backend.resolvePlugin(for: .gpu, environment: ["MAGMA_XLA_PATH": "/xla"],
                                      fileExists: { $0 == target })
        #expect(r.path == target)
    }

    @Test("a missing plugin under MAGMA_XLA_PATH falls back to the system paths")
    func fallsBackToSystemPaths() {
        let r = Backend.resolvePlugin(for: .cpu, environment: ["MAGMA_XLA_PATH": "/empty"],
                                      fileExists: { $0.hasPrefix("/usr/local/lib/pjrt_c_api_cpu") })
        #expect(r.path == "/usr/local/lib/pjrt_c_api_cpu_plugin.\(Self.ext)")
        #expect(r.searched.contains("/empty/pjrt_c_api_cpu_plugin.\(Self.ext)"))
    }

    @Test("TPU still finds libtpu when MAGMA_XLA_PATH has no TPU plugin")
    func tpuFallsBackToLibtpu() {
        let env = ["MAGMA_XLA_PATH": "/xla", "TPU_LIBRARY_PATH": "/opt/tpu/libtpu.so"]
        let r = Backend.resolvePlugin(for: .tpu, environment: env,
                                      fileExists: { $0 == "/opt/tpu/libtpu.so" })
        #expect(r.path == "/opt/tpu/libtpu.so")

        let standard = Backend.resolvePlugin(for: .tpu, environment: ["MAGMA_XLA_PATH": "/xla"],
                                             fileExists: { $0 == "/usr/lib/libtpu.so" })
        #expect(standard.path == "/usr/lib/libtpu.so")
    }

    @Test("MAGMA_PJRT_PLUGIN_<BACKEND> pins the plugin and applies only to that backend")
    func perBackendOverride() {
        let env = ["MAGMA_PJRT_PLUGIN_GPU": "/custom/cuda.so", "MAGMA_XLA_PATH": "/xla"]
        let everything: (String) -> Bool = { _ in true }
        #expect(Backend.resolvePlugin(for: .gpu, environment: env, fileExists: everything).path
                == "/custom/cuda.so")
        #expect(Backend.resolvePlugin(for: .cpu, environment: env, fileExists: everything).path
                == "/xla/pjrt_c_api_cpu_plugin.\(Self.ext)")

        // A pinned file that does not exist is reported, not silently replaced.
        let missing = Backend.resolvePlugin(for: .gpu, environment: env,
                                            fileExists: { $0 != "/custom/cuda.so" })
        #expect(missing.path == nil)
        #expect(missing.searched == ["/custom/cuda.so"])
    }

    @Test("nothing found: every candidate is listed and none is returned")
    func nothingFound() {
        let r = Backend.resolvePlugin(for: .cpu, environment: ["MAGMA_XLA_PATH": "/xla"],
                                      fileExists: { _ in false })
        #expect(r.path == nil)
        #expect(r.searched.contains("/xla/pjrt_c_api_cpu_plugin.\(Self.ext)"))
        #expect(r.searched.contains("/opt/xla/lib/pjrt_c_api_cpu_plugin.\(Self.ext)"))
        #expect(Set(r.searched).count == r.searched.count)
    }

    @Test("client creation without a plugin lists the searched paths and MAGMA_XLA_PATH")
    func missingPluginError() {
        // TPU is never loaded in this process; pin it to a file that does not
        // exist so resolution fails before any plugin is touched.
        setenv("MAGMA_PJRT_PLUGIN_TPU", "/nonexistent/libtpu-for-test.so", 1)
        defer { unsetenv("MAGMA_PJRT_PLUGIN_TPU") }

        #expect(!Backend.tpu.isAvailable)
        do {
            _ = try PJRTClient.create(backend: .tpu)
            Issue.record("creating a client without a plugin should throw")
        } catch {
            let text = "\(error)"
            #expect(text.contains("/nonexistent/libtpu-for-test.so"), "error was: \(text)")
            #expect(text.contains("MAGMA_XLA_PATH"), "error was: \(text)")
            #expect(text.contains("MAGMA_PJRT_PLUGIN_TPU"), "error was: \(text)")
        }
    }
}
