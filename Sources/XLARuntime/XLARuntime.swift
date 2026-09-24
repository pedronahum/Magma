// Magma - XLARuntime
// Swift wrapper around PJRT for XLA execution
//
// This module provides:
// - PJRTClient: Device management and program compilation
// - PJRTDevice: Device abstraction (CPU, GPU, TPU)
// - PJRTBuffer: On-device data buffers
// - PJRTExecutable: Compiled XLA programs

import CXLARuntime
import Foundation
#if os(Linux)
import Glibc
#else
import Darwin
#endif

// MARK: - Backend Selection

/// Available XLA backends
public enum Backend: String, Sendable {
    case cpu
    case gpu
    case tpu
    case metal  // Metal GPU via MetalHLO (macOS only)

    /// The outcome of searching for a backend's PJRT plugin.
    struct PluginResolution: Equatable {
        /// The plugin file to load, or nil when none was found.
        let path: String?
        /// Every candidate checked, in order (for error messages).
        let searched: [String]
    }

    /// Name of the environment variable that pins this backend's plugin to an
    /// exact file, e.g. `MAGMA_PJRT_PLUGIN_GPU`.
    var pluginOverrideVariable: String {
        "MAGMA_PJRT_PLUGIN_\(rawValue.uppercased())"
    }

    /// Plugin file names tried in each search directory, most specific first.
    var pluginFileNames: [String] {
        let ext = Self.libExtension
        var names = ["pjrt_c_api_\(rawValue)_plugin.\(ext)",
                     "libpjrt_c_api_\(rawValue)_plugin.\(ext)"]
        switch self {
        case .gpu: names.append("xla_cuda_plugin.\(ext)")   // name used by JAX's CUDA wheels
        case .tpu: names.append("libtpu.\(ext)")
        case .cpu, .metal: break
        }
        return names
    }

    /// Find this backend's PJRT plugin.
    ///
    /// Search order:
    /// 1. `MAGMA_PJRT_PLUGIN_<BACKEND>` (`MAGMA_PJRT_PLUGIN_CPU`, `_GPU`, `_TPU`):
    ///    the full path of the plugin file for that backend. When set, it is the
    ///    only candidate; a missing file is reported rather than silently
    ///    falling back to another plugin.
    /// 2. `MAGMA_XLA_PATH`: a directory holding the plugin, under any of
    ///    `pluginFileNames` (`pjrt_c_api_<backend>_plugin.<ext>`, the same with a
    ///    `lib` prefix, and JAX's `xla_cuda_plugin.so` for GPU).
    /// 3. TPU only: `TPU_LIBRARY_PATH`, then the standard Cloud TPU `libtpu.so`
    ///    locations.
    /// 4. The system directories (`/opt/xla/lib`, `/usr/local/lib`, ...).
    ///
    /// `environment` and `fileExists` are injectable for testing.
    static func resolvePlugin(
        for backend: Backend,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        fileExists: (String) -> Bool = { access($0, F_OK) == 0 }
    ) -> PluginResolution {
        func value(_ name: String) -> String? {
            guard let raw = environment[name], !raw.isEmpty else { return nil }
            return raw
        }

        if let pinned = value(backend.pluginOverrideVariable) {
            return PluginResolution(path: fileExists(pinned) ? pinned : nil, searched: [pinned])
        }

        var candidates: [String] = []
        if let base = value("MAGMA_XLA_PATH") {
            candidates += backend.pluginFileNames.map { "\(base)/\($0)" }
        }
        if backend == .tpu {
            if let tpuLibrary = value("TPU_LIBRARY_PATH") { candidates.append(tpuLibrary) }
            candidates += ["/usr/lib/libtpu.so", "/usr/local/lib/libtpu.so", "/lib/libtpu.so"]
        }
        for directory in systemPluginDirectories {
            candidates += backend.pluginFileNames.map { "\(directory)/\($0)" }
        }

        var searched: [String] = []
        for candidate in candidates where !searched.contains(candidate) {
            searched.append(candidate)
            if fileExists(candidate) {
                return PluginResolution(path: candidate, searched: searched)
            }
        }
        return PluginResolution(path: nil, searched: searched)
    }

    private static var systemPluginDirectories: [String] {
        #if os(macOS)
        return ["/usr/local/lib", "/opt/xla/lib"]
        #else
        return ["/opt/xla/lib", "/usr/local/lib", "/opt/magma/lib"]
        #endif
    }

    /// The plugin file this backend would load, or nil when none is installed.
    ///
    /// See `PJRTClient.create(backend:cpuDeviceCount:)` for the search order;
    /// set `MAGMA_XLA_PATH` to the plugin's directory or
    /// `MAGMA_PJRT_PLUGIN_<BACKEND>` to its full path to override it.
    public var resolvedPluginPath: String? {
        Self.resolvePlugin(for: self).path
    }

    /// The plugin path to report: the resolved one, else the first candidate.
    func pluginPath() -> String {
        let resolution = Self.resolvePlugin(for: self)
        return resolution.path ?? resolution.searched.first ?? pluginFileNames[0]
    }

    private static var libExtension: String {
        #if os(macOS)
        return "dylib"
        #else
        return "so"
        #endif
    }

    /// Check if this backend's plugin is available on the system
    public var isAvailable: Bool {
        switch self {
        case .metal:
            #if os(macOS) && canImport(MetalHLO)
            return MetalBackend.isAvailable
            #else
            return false
            #endif
        default:
            return resolvedPluginPath != nil
        }
    }

    /// Get all available backends on this system
    public static var availableBackends: [Backend] {
        [.cpu, .gpu, .tpu, .metal].filter { $0.isAvailable }
    }

    /// Get the best available backend (TPU > Metal > GPU > CPU)
    public static var bestAvailable: Backend {
        if Backend.tpu.isAvailable { return .tpu }
        #if os(macOS)
        if Backend.metal.isAvailable { return .metal }
        #endif
        if Backend.gpu.isAvailable { return .gpu }
        return .cpu
    }

    /// Whether a PJRT client for this backend reserves a large fraction of device
    /// memory at creation. CUDA GPU and TPU clients do (e.g. ~75-80% of a unified
    /// pool on an NVIDIA GB10), so more than one concurrent client can exhaust
    /// memory and freeze the machine; CPU and Metal do not.
    public var reservesLargeMemory: Bool {
        switch self {
        case .gpu, .tpu: return true
        case .cpu, .metal: return false
        }
    }
}

// MARK: - TPU Environment Detection

/// Utilities for detecting and configuring TPU environments
public struct TPUEnvironment {

    /// Check if running on a Google Cloud TPU VM
    public static var isTPUVM: Bool {
        // Check for TPU-specific environment variables
        if getenv("TPU_NAME") != nil { return true }
        if getenv("TPU_CHIPS_PER_HOST_BOUNDS") != nil { return true }

        // Check for libtpu.so
        return Backend.tpu.isAvailable
    }

    /// Get the TPU topology string (e.g., "2x2x1" for v4-8)
    public static var topology: String? {
        if let chips = getenv("TPU_CHIPS_PER_HOST_BOUNDS") {
            return String(cString: chips)
        }
        return nil
    }

    /// Get the TPU name from environment
    public static var tpuName: String? {
        if let name = getenv("TPU_NAME") {
            return String(cString: name)
        }
        return nil
    }

    /// Get the TPU type (e.g., "v4-8", "v3-8")
    public static var tpuType: String? {
        // Try to infer from accelerator type
        if let accelType = getenv("ACCELERATOR_TYPE") {
            return String(cString: accelType)
        }
        // Fallback to TPU name parsing
        if let name = tpuName {
            // TPU names often contain the type
            return name
        }
        return nil
    }

    /// Number of TPU chips available on this host
    public static var numChips: Int {
        if let bounds = topology {
            // Parse "AxBxC" format
            let parts = bounds.split(separator: "x").compactMap { Int($0) }
            if parts.count >= 3 {
                return parts.reduce(1, *)
            }
        }
        // Default single-host assumption
        return Backend.tpu.isAvailable ? 4 : 0
    }

    /// Check if this is a multi-host TPU pod
    public static var isMultiHost: Bool {
        if let hosts = getenv("TPU_HOST_BOUNDS") {
            let hostStr = String(cString: hosts)
            let parts = hostStr.split(separator: "x").compactMap { Int($0) }
            return parts.reduce(1, *) > 1
        }
        return false
    }

    /// Print TPU environment information
    public static func printInfo() {
        print("TPU Environment:")
        print("  Is TPU VM: \(isTPUVM)")
        if isTPUVM {
            if let name = tpuName {
                print("  TPU Name: \(name)")
            }
            if let type = tpuType {
                print("  TPU Type: \(type)")
            }
            if let topo = topology {
                print("  Topology: \(topo)")
            }
            print("  Chips: \(numChips)")
            print("  Multi-host: \(isMultiHost)")
            print("  Plugin path: \(Backend.tpu.pluginPath())")
        } else {
            print("  Not running on a TPU VM")
            print("  Available backends: \(Backend.availableBackends.map { $0.rawValue })")
        }
    }
}

// MARK: - Device

/// Represents a device for computation
public struct Device: Hashable, Sendable, CustomStringConvertible {
    /// Device type
    public let backend: Backend

    /// Device index (for multi-device setups)
    public let index: Int

    /// Default device (CPU:0)
    public static let `default` = Device(backend: .cpu, index: 0)

    public var description: String {
        "\(backend.rawValue.uppercased()):\(index)"
    }

    public init(backend: Backend, index: Int = 0) {
        self.backend = backend
        self.index = index
    }
}

// MARK: - Errors

/// Errors from XLA runtime operations
public enum XLAError: Error, CustomStringConvertible {
    case clientCreationFailed(String)
    case compilationFailed(String)
    case executionFailed(String)
    case bufferCreationFailed(String)
    case bufferTransferFailed(String)
    case deviceNotFound(String)
    case notImplemented(String)
    case noDeviceAvailable

    public var description: String {
        switch self {
        case .clientCreationFailed(let msg): return "Client creation failed: \(msg)"
        case .compilationFailed(let msg): return "Compilation failed: \(msg)"
        case .executionFailed(let msg): return "Execution failed: \(msg)"
        case .bufferCreationFailed(let msg): return "Buffer creation failed: \(msg)"
        case .bufferTransferFailed(let msg): return "Buffer transfer failed: \(msg)"
        case .deviceNotFound(let msg): return "Device not found: \(msg)"
        case .notImplemented(let msg): return "Not implemented: \(msg)"
        case .noDeviceAvailable: return "No device available"
        }
    }
}

/// Describe a failed PJRT wrapper call: the status code name plus the plugin's
/// own message (e.g. the XLA compile diagnostic) when one was recorded.
///
/// Callers clear the thread's last message with `PJRT_ClearLastErrorMessage()`
/// right before the wrapper call, so a message here always belongs to it.
func pjrtFailureDescription(_ operation: String, _ code: SW_PJRT_Error_Code) -> String {
    let names = ["OK", "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED",
                 "NOT_FOUND", "ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED",
                 "FAILED_PRECONDITION", "ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED",
                 "INTERNAL", "UNAVAILABLE", "DATA_LOSS", "UNAUTHENTICATED"]
    let raw = Int(code.rawValue)
    let status = names.indices.contains(raw) ? names[raw] : "code \(raw)"
    if let message = PJRT_GetLastErrorMessage() {
        return "\(operation) failed (\(status)): \(String(cString: message))"
    }
    return "\(operation) failed (\(status))"
}

// MARK: - Element Types

/// PJRT element types matching StableHLO types
public enum ElementType: Sendable {
    case bool
    case int8, int16, int32, int64
    case uint8, uint16, uint32, uint64
    case float16, float32, float64
    case bfloat16
    case complex64, complex128

    /// Size in bytes
    public var sizeInBytes: Int {
        switch self {
        case .bool, .int8, .uint8: return 1
        case .int16, .uint16, .float16, .bfloat16: return 2
        case .int32, .uint32, .float32: return 4
        case .int64, .uint64, .float64, .complex64: return 8
        case .complex128: return 16
        }
    }

    /// Construct from the C API type (device -> host query path). Nil for
    /// element types Magma does not model (F8 variants, S4/U4, token, ...).
    init?(cType: SW_PJRT_Buffer_Type) {
        switch cType {
        case SW_PJRT_Buffer_Type_PRED: self = .bool
        case SW_PJRT_Buffer_Type_S8:   self = .int8
        case SW_PJRT_Buffer_Type_S16:  self = .int16
        case SW_PJRT_Buffer_Type_S32:  self = .int32
        case SW_PJRT_Buffer_Type_S64:  self = .int64
        case SW_PJRT_Buffer_Type_U8:   self = .uint8
        case SW_PJRT_Buffer_Type_U16:  self = .uint16
        case SW_PJRT_Buffer_Type_U32:  self = .uint32
        case SW_PJRT_Buffer_Type_U64:  self = .uint64
        case SW_PJRT_Buffer_Type_F16:  self = .float16
        case SW_PJRT_Buffer_Type_F32:  self = .float32
        case SW_PJRT_Buffer_Type_F64:  self = .float64
        case SW_PJRT_Buffer_Type_BF16: self = .bfloat16
        case SW_PJRT_Buffer_Type_C64:  self = .complex64
        case SW_PJRT_Buffer_Type_C128: self = .complex128
        default: return nil
        }
    }

    /// Convert to C API type
    var toCType: SW_PJRT_Buffer_Type {
        switch self {
        case .bool: return SW_PJRT_Buffer_Type_PRED
        case .int8: return SW_PJRT_Buffer_Type_S8
        case .int16: return SW_PJRT_Buffer_Type_S16
        case .int32: return SW_PJRT_Buffer_Type_S32
        case .int64: return SW_PJRT_Buffer_Type_S64
        case .uint8: return SW_PJRT_Buffer_Type_U8
        case .uint16: return SW_PJRT_Buffer_Type_U16
        case .uint32: return SW_PJRT_Buffer_Type_U32
        case .uint64: return SW_PJRT_Buffer_Type_U64
        case .float16: return SW_PJRT_Buffer_Type_F16
        case .float32: return SW_PJRT_Buffer_Type_F32
        case .float64: return SW_PJRT_Buffer_Type_F64
        case .bfloat16: return SW_PJRT_Buffer_Type_BF16
        case .complex64: return SW_PJRT_Buffer_Type_C64
        case .complex128: return SW_PJRT_Buffer_Type_C128
        }
    }
}

// MARK: - PJRTClient

/// Client for managing devices and compiling programs
public final class PJRTClient: @unchecked Sendable {

    /// The backend this client uses
    public let backend: Backend

    /// Opaque handle to PJRT_Client
    private var handle: UnsafeMutableRawPointer?

    /// Available devices
    public private(set) var devices: [PJRTDevice] = []

    /// Platform name
    public private(set) var platformName: String = ""

    // MARK: Concurrent-accelerator-client guard
    //
    // Accelerator PJRT plugins (CUDA GPU, TPU) reserve a large fraction of device
    // memory per client. On a unified-memory machine there is no separate VRAM to
    // fail into, so a second concurrent client exhausts memory and freezes the
    // whole box. We refuse to create more than one live accelerator client at a
    // time unless explicitly opted out. Slots are counted here and released in
    // deinit, so a client that goes out of scope frees its slot automatically.
    nonisolated(unsafe) private static var liveAcceleratorClients = 0
    private static let acceleratorLock = NSLock()

    /// Whether this instance holds an accelerator-slot reservation to release.
    private var holdsAcceleratorSlot = false

    /// Env opt-out for the concurrent-accelerator-client guard.
    private static var concurrentAcceleratorClientsAllowed: Bool {
        guard let raw = getenv("MAGMA_ALLOW_CONCURRENT_ACCEL_CLIENTS") else { return false }
        let value = String(cString: raw).lowercased()
        return value == "1" || value == "true" || value == "yes"
    }

    private init(backend: Backend) {
        self.backend = backend
    }

    deinit {
        if let handle = handle {
            PJRT_DestroyClient(handle)
        }
        releaseAcceleratorSlot()
    }

    /// Release this client's accelerator-slot reservation, if it holds one.
    private func releaseAcceleratorSlot() {
        guard holdsAcceleratorSlot else { return }
        Self.acceleratorLock.lock()
        Self.liveAcceleratorClients -= 1
        holdsAcceleratorSlot = false
        Self.acceleratorLock.unlock()
    }

    /// Create a client for the specified backend.
    ///
    /// - Parameter cpuDeviceCount: when set (CPU backend only), the CPU plugin is
    ///   asked to expose this many virtual devices via its `cpu_device_count`
    ///   create option. This lets multi-device code be developed and tested on
    ///   CPU without physical accelerators. Ignored/unset for other backends.
    public static func create(backend: Backend = .cpu, cpuDeviceCount: Int? = nil) throws -> PJRTClient {
        let client = PJRTClient(backend: backend)

        // Reserve an accelerator slot before touching the plugin, refusing a
        // second concurrent client that would over-commit device memory. The
        // reservation is released by `client`'s deinit — including when creation
        // throws below and the client is discarded — so counts stay balanced.
        if backend.reservesLargeMemory {
            acceleratorLock.lock()
            if liveAcceleratorClients > 0 && !concurrentAcceleratorClientsAllowed {
                let live = liveAcceleratorClients
                acceleratorLock.unlock()
                throw XLAError.clientCreationFailed(
                    "refusing to create a second concurrent \(backend) client: " +
                    "\(live) accelerator client(s) already live. Each reserves most of " +
                    "device memory, so concurrent clients can exhaust it and freeze the " +
                    "machine. Release the existing client first, or set " +
                    "MAGMA_ALLOW_CONCURRENT_ACCEL_CLIENTS=1 to override.")
            }
            liveAcceleratorClients += 1
            client.holdsAcceleratorSlot = true
            acceleratorLock.unlock()
        }

        // Locate and load the PJRT plugin
        let resolution = Backend.resolvePlugin(for: backend)
        guard let pluginPath = resolution.path else {
            throw XLAError.clientCreationFailed(
                "no PJRT plugin found for the \(backend) backend. Searched:\n" +
                resolution.searched.map { "  \($0)" }.joined(separator: "\n") +
                "\nSet MAGMA_XLA_PATH to the directory that contains the plugin, or " +
                "\(backend.pluginOverrideVariable) to the plugin file's full path.")
        }

        PJRT_ClearLastErrorMessage()
        let errorCode = PJRT_LoadPlugin(pluginPath)

        if errorCode != SW_PJRT_Error_OK {
            throw XLAError.clientCreationFailed(
                pjrtFailureDescription("Loading plugin '\(pluginPath)'", errorCode))
        }

        // Create the client, optionally requesting several virtual CPU devices.
        var clientHandle: UnsafeMutableRawPointer?
        let createError: SW_PJRT_Error_Code
        PJRT_ClearLastErrorMessage()
        if let cpuDeviceCount {
            createError = PJRT_CreateClientWithCpuDeviceCount(Int64(cpuDeviceCount), &clientHandle)
        } else {
            createError = PJRT_CreateClient(&clientHandle)
        }

        if createError != SW_PJRT_Error_OK {
            throw XLAError.clientCreationFailed(pjrtFailureDescription("PJRT_Client_Create", createError))
        }

        guard let handle = clientHandle else {
            throw XLAError.clientCreationFailed("PJRT_CreateClient returned NULL")
        }

        client.handle = handle

        // Get platform name
        var namePtr: UnsafePointer<CChar>?
        if PJRT_GetPlatformName(handle, &namePtr) == SW_PJRT_Error_OK, let name = namePtr {
            client.platformName = String(cString: name)
        }

        // Enumerate devices
        var devicesPtr: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
        var numDevices: Int = 0

        if PJRT_GetAddressableDevices(handle, &devicesPtr, &numDevices) == SW_PJRT_Error_OK {
            if let devices = devicesPtr {
                for i in 0..<numDevices {
                    if let deviceHandle = devices[i] {
                        var deviceId: Int32 = 0
                        var kindPtr: UnsafePointer<CChar>?

                        PJRT_GetDeviceId(deviceHandle, &deviceId)
                        PJRT_GetDeviceKind(deviceHandle, &kindPtr)

                        let kind = kindPtr.map { String(cString: $0) } ?? "unknown"
                        let device = PJRTDevice(
                            id: Int(deviceId),
                            kind: kind,
                            client: client,
                            handle: deviceHandle
                        )
                        client.devices.append(device)
                    }
                }
            }
        }

        return client
    }

    /// Get the default device
    public var defaultDevice: PJRTDevice? {
        devices.first
    }

    /// Number of addressable devices on this client. 1 for a normal CPU/GPU
    /// client; more when the client was created with several devices (e.g. a
    /// CPU client with `cpuDeviceCount`, or a multi-GPU/TPU host).
    public var deviceCount: Int { devices.count }

    /// The addressable device at a logical index (0-based), or nil if out of
    /// range. `client.devices` remains available for the full list.
    public func device(at index: Int) -> PJRTDevice? {
        devices.indices.contains(index) ? devices[index] : nil
    }

    /// Create a buffer from host data.
    ///
    /// `data` is copied as raw bytes, so its byte count must equal
    /// `shape.product * elementType.sizeInBytes` (for example `[Float]` with
    /// `.float32`, `[Int32]` with `.int32`); otherwise this throws
    /// `XLAError.bufferCreationFailed` rather than reading past `data`.
    public func createBuffer<T>(
        _ data: [T],
        shape: [Int],
        elementType: ElementType,
        device: PJRTDevice? = nil
    ) throws -> PJRTBuffer {
        let targetDevice = device ?? defaultDevice
        guard let targetDevice = targetDevice else {
            throw XLAError.noDeviceAvailable
        }

        guard let clientHandle = handle, let deviceHandle = targetDevice.handle else {
            throw XLAError.bufferCreationFailed("Invalid handles")
        }

        guard shape.allSatisfy({ $0 >= 0 }) else {
            throw XLAError.bufferCreationFailed("negative dimension in shape \(shape)")
        }
        let elementCount = shape.reduce(1, *)
        let expectedBytes = elementCount * elementType.sizeInBytes
        let providedBytes = data.count * MemoryLayout<T>.stride
        guard providedBytes == expectedBytes else {
            throw XLAError.bufferCreationFailed(
                "shape \(shape) of \(elementType) needs \(expectedBytes) bytes, but " +
                "\(data.count) \(T.self) value(s) provide \(providedBytes)")
        }

        let dims = shape.map { Int64($0) }
        var bufferHandle: UnsafeMutableRawPointer?

        PJRT_ClearLastErrorMessage()
        let errorCode = data.withUnsafeBytes { dataPtr in
            dims.withUnsafeBufferPointer { dimsPtr in
                PJRT_CreateBuffer(
                    clientHandle,
                    dataPtr.baseAddress,
                    elementType.toCType,
                    dimsPtr.baseAddress,
                    dims.count,
                    deviceHandle,
                    &bufferHandle
                )
            }
        }

        if errorCode != SW_PJRT_Error_OK {
            throw XLAError.bufferCreationFailed(pjrtFailureDescription("PJRT_CreateBuffer", errorCode))
        }

        guard let buffer = bufferHandle else {
            throw XLAError.bufferCreationFailed("PJRT_CreateBuffer returned NULL")
        }

        return PJRTBuffer(
            handle: buffer,
            shape: shape,
            elementType: elementType,
            device: targetDevice
        )
    }

    /// Compile StableHLO MLIR to an executable
    public func compile(_ mlir: String) throws -> PJRTExecutable {
        guard let clientHandle = handle else {
            throw XLAError.compilationFailed("Client not initialized")
        }

        var executableHandle: UnsafeMutableRawPointer?
        PJRT_ClearLastErrorMessage()
        let errorCode = PJRT_CompileWrapper(clientHandle, mlir, &executableHandle)

        if errorCode != SW_PJRT_Error_OK {
            throw XLAError.compilationFailed(pjrtFailureDescription("PJRT_Client_Compile", errorCode))
        }

        guard let executable = executableHandle else {
            throw XLAError.compilationFailed("PJRT_Compile returned NULL")
        }

        return PJRTExecutable(
            handle: executable,
            client: self,
            devices: devices
        )
    }

    /// Compile StableHLO MLIR with replication / SPMD / Shardy options.
    ///
    /// - Parameters:
    ///   - numReplicas: number of data-parallel replicas. Values > 1 make a
    ///     multi-device executable whose cross-replica collectives (e.g.
    ///     `all_reduce`) reduce across replicas.
    ///   - numPartitions: number of SPMD partitions. Values > 1 require a client
    ///     with at least `numReplicas * numPartitions` devices.
    ///   - useSPMDPartitioning: enable XLA's SPMD partitioner.
    ///   - useShardyPartitioner: run the Shardy propagation + partitioning
    ///     pipeline (consumes `sdy.mesh` / `sdy.sharding` annotations in the
    ///     module). Requires the plugin to have been built with Shardy.
    ///
    /// With `numReplicas: 1, numPartitions: 1, useSPMDPartitioning: false,
    /// useShardyPartitioner: false` this is equivalent to `compile(_:)`.
    public func compile(
        _ mlir: String,
        numReplicas: Int = 1,
        numPartitions: Int = 1,
        useSPMDPartitioning: Bool = true,
        useShardyPartitioner: Bool = false
    ) throws -> PJRTExecutable {
        guard let clientHandle = handle else {
            throw XLAError.compilationFailed("Client not initialized")
        }

        var executableHandle: UnsafeMutableRawPointer?
        PJRT_ClearLastErrorMessage()
        let errorCode = PJRT_CompileWrapperSPMD(
            clientHandle,
            mlir,
            Int64(numReplicas),
            Int64(numPartitions),
            useSPMDPartitioning ? 1 : 0,
            useShardyPartitioner ? 1 : 0,
            &executableHandle
        )

        if errorCode != SW_PJRT_Error_OK {
            throw XLAError.compilationFailed(pjrtFailureDescription(
                "PJRT_Client_Compile (replicas: \(numReplicas), partitions: \(numPartitions))",
                errorCode))
        }

        guard let executable = executableHandle else {
            throw XLAError.compilationFailed("PJRT_CompileWrapperSPMD returned NULL")
        }

        return PJRTExecutable(
            handle: executable,
            client: self,
            devices: devices
        )
    }
}

// MARK: - Buffer Distribution (scatter / replicate / gather)

extension PJRTClient {
    /// Place a copy of `hostData` on each of the first `count` devices.
    /// Used for replicated inputs (e.g. DDP parameters).
    public func replicate<T>(
        _ hostData: [T],
        shape: [Int],
        elementType: ElementType,
        count: Int
    ) throws -> [PJRTBuffer] {
        precondition(count > 0, "count must be positive")
        guard deviceCount >= count else {
            throw XLAError.noDeviceAvailable
        }
        return try (0..<count).map { d in
            try createBuffer(hostData, shape: shape, elementType: elementType, device: devices[d])
        }
    }

    /// Shard `hostData` (row-major, `shape[0]` = leading dim) into `count`
    /// contiguous slices along axis 0, one buffer per device. `shape[0]` must be
    /// divisible by `count`. Used for sharded inputs (e.g. an SPMD data batch).
    public func scatterAlongAxis0<T>(
        _ hostData: [T],
        shape: [Int],
        elementType: ElementType,
        count: Int
    ) throws -> [PJRTBuffer] {
        precondition(count > 0, "count must be positive")
        precondition(!shape.isEmpty, "scatter needs a leading dimension")
        guard deviceCount >= count else { throw XLAError.noDeviceAvailable }
        guard shape[0] % count == 0 else {
            throw XLAError.bufferCreationFailed(
                "axis-0 size \(shape[0]) is not divisible by device count \(count)")
        }
        let rowsPer = shape[0] / count
        let inner = shape.dropFirst().reduce(1, *)      // elements per leading index
        let sliceElems = rowsPer * inner
        let shardShape = [rowsPer] + Array(shape.dropFirst())
        return try (0..<count).map { d in
            let slice = Array(hostData[(d * sliceElems)..<((d + 1) * sliceElems)])
            return try createBuffer(slice, shape: shardShape, elementType: elementType, device: devices[d])
        }
    }

    /// Concatenate per-device row shards (float32) back into a single host array
    /// and its full shape (axis-0 gather). Shards must share their trailing dims.
    public func gatherAlongAxis0(_ shards: [PJRTBuffer]) throws -> (data: [Float], shape: [Int]) {
        guard let first = shards.first else { return ([], []) }
        var data: [Float] = []
        var totalRows = 0
        for shard in shards {
            data.append(contentsOf: try shard.toFloatArray())
            totalRows += shard.shape.first ?? 1
        }
        let shape = [totalRows] + Array(first.shape.dropFirst())
        return (data, shape)
    }
}

// MARK: - PJRTDevice

/// Represents a PJRT device
public class PJRTDevice: @unchecked Sendable {
    public let id: Int
    public let kind: String
    public weak var client: PJRTClient?
    internal var handle: UnsafeMutableRawPointer?

    init(id: Int, kind: String, client: PJRTClient, handle: UnsafeMutableRawPointer?) {
        self.id = id
        self.kind = kind
        self.client = client
        self.handle = handle
    }

    public var description: String {
        "\(kind):\(id)"
    }
}

// MARK: - PJRTBuffer

/// On-device data buffer
public final class PJRTBuffer: @unchecked Sendable {

    internal var handle: UnsafeMutableRawPointer?
    public let shape: [Int]
    public let elementType: ElementType
    public let device: PJRTDevice

    /// The client that owns this buffer's device memory. Held strongly so the
    /// client (and its PJRT_Client) outlives every buffer it allocated.
    private let owningClient: PJRTClient?

    public var elementCount: Int {
        shape.isEmpty ? 1 : shape.reduce(1, *)
    }

    public var sizeInBytes: Int {
        elementCount * elementType.sizeInBytes
    }

    init(handle: UnsafeMutableRawPointer, shape: [Int], elementType: ElementType, device: PJRTDevice) {
        self.handle = handle
        self.shape = shape
        self.elementType = elementType
        self.device = device
        self.owningClient = device.client
    }

    deinit {
        if let handle = handle {
            PJRT_DestroyBuffer(handle)
        }
    }

    /// Copy the buffer's raw contents to host memory as `T` (synchronous).
    ///
    /// No conversion happens: `T` must have the element type's width (e.g.
    /// `Float` or `Int32` for 4-byte types, `Double` for `.float64`), otherwise
    /// this throws `XLAError.bufferTransferFailed`. Use `toFloatArray()` to get
    /// values converted to `Float` from any real element type.
    public func toHost<T>(_ type: T.Type) throws -> [T] {
        guard let bufferHandle = handle else {
            throw XLAError.bufferTransferFailed("Buffer handle not available")
        }
        guard MemoryLayout<T>.stride == elementType.sizeInBytes else {
            throw XLAError.bufferTransferFailed(
                "cannot read a \(elementType) buffer (\(elementType.sizeInBytes)-byte elements) " +
                "as \(T.self) (\(MemoryLayout<T>.stride) bytes)")
        }

        // Allocate raw memory and copy data from device
        let rawBuffer = UnsafeMutableRawPointer.allocate(
            byteCount: max(sizeInBytes, 1),
            alignment: MemoryLayout<T>.alignment
        )
        defer { rawBuffer.deallocate() }

        PJRT_ClearLastErrorMessage()
        let errorCode = PJRT_BufferToHost(bufferHandle, rawBuffer, sizeInBytes)

        if errorCode != SW_PJRT_Error_OK {
            throw XLAError.bufferTransferFailed(pjrtFailureDescription("PJRT_Buffer_ToHostBuffer", errorCode))
        }

        // Convert to typed array
        let typedPointer = rawBuffer.bindMemory(to: T.self, capacity: elementCount)
        return Array(UnsafeBufferPointer(start: typedPointer, count: elementCount))
    }

    /// Copy the buffer to host and convert each element to `Float`.
    ///
    /// Float32 buffers are copied as-is; integer, bool (0/1), float16,
    /// bfloat16 and float64 buffers are converted numerically. Complex buffers
    /// throw `XLAError.bufferTransferFailed`.
    public func toFloatArray() throws -> [Float] {
        switch elementType {
        case .float32: return try toHost(Float.self)
        case .float64: return try toHost(Double.self).map { Float($0) }
        case .int8: return try toHost(Int8.self).map { Float($0) }
        case .int16: return try toHost(Int16.self).map { Float($0) }
        case .int32: return try toHost(Int32.self).map { Float($0) }
        case .int64: return try toHost(Int64.self).map { Float($0) }
        case .uint8: return try toHost(UInt8.self).map { Float($0) }
        case .uint16: return try toHost(UInt16.self).map { Float($0) }
        case .uint32: return try toHost(UInt32.self).map { Float($0) }
        case .uint64: return try toHost(UInt64.self).map { Float($0) }
        case .bool: return try toHost(UInt8.self).map { $0 != 0 ? 1 : 0 }
        case .bfloat16: return try toHost(UInt16.self).map { Float(bitPattern: UInt32($0) << 16) }
        case .float16: return try toHost(UInt16.self).map(Self.halfToFloat)
        case .complex64, .complex128:
            throw XLAError.bufferTransferFailed("cannot convert a \(elementType) buffer to Float")
        }
    }

    /// IEEE 754 binary16 bits to Float (portable; `Float16` is not available
    /// on every platform Magma builds for).
    private static func halfToFloat(_ bits: UInt16) -> Float {
        let sign: Float = (bits & 0x8000) != 0 ? -1 : 1
        let exponent = Int((bits >> 10) & 0x1F)
        let mantissa = Float(bits & 0x3FF)
        switch exponent {
        case 0: return sign * mantissa * 0x1p-24                     // zero / subnormal
        case 0x1F: return mantissa == 0 ? sign * .infinity : .nan
        default: return sign * (1 + mantissa / 1024) * Float(sign: .plus, exponent: exponent - 15, significand: 1)
        }
    }
}

// MARK: - PJRTExecutable

/// Compiled XLA program ready for execution
public final class PJRTExecutable: @unchecked Sendable {

    internal var handle: UnsafeMutableRawPointer?
    /// The client that compiled this executable. Held strongly so the client
    /// (and its PJRT_Client) outlives every executable it compiled.
    public let client: PJRTClient?
    public let devices: [PJRTDevice]

    private let executionCountLock = NSLock()
    private var _executionCount = 0

    /// Number of successful executions (single- or multi-device). Safe to read
    /// while other threads execute this executable.
    public var executionCount: Int {
        executionCountLock.lock()
        defer { executionCountLock.unlock() }
        return _executionCount
    }

    private func recordExecution() {
        executionCountLock.lock()
        _executionCount += 1
        executionCountLock.unlock()
    }

    init(handle: UnsafeMutableRawPointer, client: PJRTClient, devices: [PJRTDevice]) {
        self.handle = handle
        self.client = client
        self.devices = devices
    }

    deinit {
        if let handle = handle {
            PJRT_DestroyExecutable(handle)
        }
    }

    /// Execute the program with input buffers
    public func execute(_ inputs: [PJRTBuffer]) throws -> [PJRTBuffer] {
        guard let execHandle = handle else {
            throw XLAError.executionFailed("Executable handle not available")
        }

        guard let device = devices.first else {
            throw XLAError.noDeviceAvailable
        }

        // Collect input handles
        var inputHandles: [UnsafeMutableRawPointer?] = inputs.map { $0.handle }
        var outputsPtr: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
        var numOutputs: Int = 0

        PJRT_ClearLastErrorMessage()
        let errorCode = inputHandles.withUnsafeMutableBufferPointer { inputsPtr in
            PJRT_ExecuteWrapper(
                execHandle,
                inputsPtr.baseAddress,
                inputs.count,
                &outputsPtr,
                &numOutputs
            )
        }

        if errorCode != SW_PJRT_Error_OK {
            throw XLAError.executionFailed(pjrtFailureDescription("PJRT_LoadedExecutable_Execute", errorCode))
        }

        recordExecution()

        // The wrapper's output array is thread-local and reused by the next
        // execute, so copy the handles out before wrapping them.
        guard let outputHandles = outputsPtr else { return [] }
        let handles = (0..<numOutputs).map { outputHandles[$0] }
        return try wrapOutputs(handles, device: device)
    }

    /// Wrap raw output handles in PJRTBuffers, querying each one's shape and
    /// element type. If an output cannot be wrapped (e.g. an element type
    /// Magma does not support), the handles not yet wrapped are destroyed so
    /// nothing leaks, and the error is thrown.
    private func wrapOutputs(_ handles: [UnsafeMutableRawPointer?], device: PJRTDevice) throws -> [PJRTBuffer] {
        var outputs: [PJRTBuffer] = []
        outputs.reserveCapacity(handles.count)
        for (index, handle) in handles.enumerated() {
            guard let handle else { continue }
            do {
                outputs.append(try makeOutputBuffer(handle, device: device))
            } catch {
                PJRT_DestroyBuffer(handle)
                for rest in handles[(index + 1)...] {
                    if let rest { PJRT_DestroyBuffer(rest) }
                }
                throw error
            }
        }
        return outputs
    }

    /// Build a PJRTBuffer around a raw output handle, querying its shape and
    /// element type. Shared by single- and multi-device execute. Does not take
    /// ownership of `handle` when it throws.
    private func makeOutputBuffer(_ handle: UnsafeMutableRawPointer, device: PJRTDevice) throws -> PJRTBuffer {
        var dimsPtr: UnsafePointer<Int64>?
        var numDims = 0
        PJRT_ClearLastErrorMessage()
        let dimsCode = PJRT_GetBufferDimensions(handle, &dimsPtr, &numDims)
        guard dimsCode == SW_PJRT_Error_OK else {
            throw XLAError.executionFailed(pjrtFailureDescription("Querying output dimensions", dimsCode))
        }
        var shape: [Int] = []
        if let dims = dimsPtr {
            for j in 0..<numDims { shape.append(Int(dims[j])) }
        }

        var cType = SW_PJRT_Buffer_Type_F32
        PJRT_ClearLastErrorMessage()
        let typeCode = PJRT_GetBufferElementType(handle, &cType)
        guard typeCode == SW_PJRT_Error_OK, let elementType = ElementType(cType: cType) else {
            throw XLAError.executionFailed(pjrtFailureDescription(
                "Querying output element type",
                typeCode == SW_PJRT_Error_OK ? SW_PJRT_Error_UNIMPLEMENTED : typeCode))
        }
        return PJRTBuffer(handle: handle, shape: shape, elementType: elementType, device: device)
    }

    /// Execute across multiple devices.
    ///
    /// `inputsPerDevice[d]` are the arguments for device `d`; every device must
    /// supply the same number of arguments, and each buffer must be resident on
    /// the corresponding device (see `PJRTClient.createBuffer(_:…, device:)`).
    /// Returns per-device outputs. The executable must have been compiled for
    /// `inputsPerDevice.count` devices (e.g. `numReplicas`/`numPartitions = N`).
    public func executeMultiDevice(inputsPerDevice: [[PJRTBuffer]]) throws -> [[PJRTBuffer]] {
        guard let execHandle = handle else {
            throw XLAError.executionFailed("Executable handle not available")
        }
        guard !devices.isEmpty else { throw XLAError.noDeviceAvailable }

        let numDevices = inputsPerDevice.count
        guard numDevices > 0 else { return [] }
        let numArgs = inputsPerDevice[0].count
        for devInputs in inputsPerDevice where devInputs.count != numArgs {
            throw XLAError.executionFailed("Every device must supply the same number of arguments")
        }

        // Flatten input handles row-major: [dev0 args…, dev1 args…, …].
        var inputHandles: [UnsafeMutableRawPointer?] = []
        inputHandles.reserveCapacity(numDevices * numArgs)
        for devInputs in inputsPerDevice {
            for buf in devInputs { inputHandles.append(buf.handle) }
        }

        var outputsFlat: UnsafeMutablePointer<UnsafeMutableRawPointer?>?
        var numOutputs = 0
        PJRT_ClearLastErrorMessage()
        let errorCode = inputHandles.withUnsafeMutableBufferPointer { ptr in
            PJRT_ExecuteMultiDevice(
                execHandle, ptr.baseAddress, numDevices, numArgs, &outputsFlat, &numOutputs)
        }
        if errorCode != SW_PJRT_Error_OK {
            throw XLAError.executionFailed(pjrtFailureDescription(
                "Multi-device PJRT_LoadedExecutable_Execute (\(numDevices) devices)", errorCode))
        }
        recordExecution()
        defer { if let outputsFlat { PJRT_FreeOutputList(outputsFlat) } }

        guard let flat = outputsFlat else {
            return Array(repeating: [], count: numDevices)
        }
        var result: [[PJRTBuffer]] = []
        result.reserveCapacity(numDevices)
        for d in 0..<numDevices {
            let device = devices.indices.contains(d) ? devices[d] : devices[0]
            let handles = (0..<numOutputs).map { flat[d * numOutputs + $0] }
            do {
                result.append(try wrapOutputs(handles, device: device))
            } catch {
                // Devices after `d` were never wrapped; release their outputs.
                for later in (d + 1)..<numDevices {
                    for o in 0..<numOutputs {
                        if let h = flat[later * numOutputs + o] { PJRT_DestroyBuffer(h) }
                    }
                }
                throw error
            }
        }
        return result
    }
}

// MARK: - Execution Timing

/// Timing breakdown for profiled execution
public struct ExecutionTiming: Sendable {
    public let h2dCreateNs: UInt64
    public let executeNs: UInt64
    public let d2hInitiateNs: UInt64
    public let d2hAwaitNs: UInt64
    public let bufferDestroyNs: UInt64
    public let totalNs: UInt64
    public let numInputs: Int
    public let numOutputs: Int

    init(from timing: SW_PJRT_ExecutionTiming) {
        self.h2dCreateNs = timing.h2d_create_ns
        self.executeNs = timing.execute_ns
        self.d2hInitiateNs = timing.d2h_initiate_ns
        self.d2hAwaitNs = timing.d2h_await_ns
        self.bufferDestroyNs = timing.buffer_destroy_ns
        self.totalNs = timing.total_ns
        self.numInputs = timing.num_inputs
        self.numOutputs = timing.num_outputs
    }

    public var totalMs: Double {
        Double(totalNs) / 1_000_000.0
    }

    public var executeMs: Double {
        Double(executeNs) / 1_000_000.0
    }
}
