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

    /// Get the default plugin path for this backend
    func pluginPath() -> String {
        // Check environment variable first
        if let envPath = getenv("MAGMA_XLA_PATH") {
            let basePath = String(cString: envPath)
            // If the path already contains the library directory, use it directly
            return "\(basePath)/pjrt_c_api_\(rawValue)_plugin.\(Self.libExtension)"
        }

        // TPU has special paths on Cloud TPU VMs
        if self == .tpu {
            return Self.tpuPluginPath() ?? "/usr/lib/libtpu.so"
        }

        #if os(macOS)
        let systemPaths = ["/usr/local/lib", "/opt/xla/lib"]
        #else
        let systemPaths = ["/opt/xla/lib", "/usr/local/lib", "/opt/magma/lib"]
        #endif

        for path in systemPaths {
            let fullPath = "\(path)/pjrt_c_api_\(rawValue)_plugin.\(Self.libExtension)"
            if access(fullPath, F_OK) == 0 {
                return fullPath
            }
        }

        return systemPaths[0] + "/pjrt_c_api_\(rawValue)_plugin.\(Self.libExtension)"
    }

    /// Find the TPU plugin on Cloud TPU VMs
    private static func tpuPluginPath() -> String? {
        // Standard TPU library locations on Cloud TPU VMs
        let tpuPaths = [
            "/usr/lib/libtpu.so",                              // Standard location
            "/usr/local/lib/libtpu.so",                        // Alternative
            "/lib/libtpu.so",                                  // Fallback
        ]

        // Check TPU_LIBRARY_PATH environment variable first
        if let envPath = getenv("TPU_LIBRARY_PATH") {
            let path = String(cString: envPath)
            if access(path, F_OK) == 0 {
                return path
            }
        }

        // Search standard locations
        for path in tpuPaths {
            if access(path, F_OK) == 0 {
                return path
            }
        }

        return nil
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
            let path = pluginPath()
            return access(path, F_OK) == 0
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

    /// Construct from the C API type (device -> host query path).
    init(cType: SW_PJRT_Buffer_Type) {
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
        default: self = .float32
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

        // Load the PJRT plugin
        let pluginPath = backend.pluginPath()
        let errorCode = PJRT_LoadPlugin(pluginPath)

        if errorCode != SW_PJRT_Error_OK {
            if let errorMsg = PJRT_GetLastError() {
                throw XLAError.clientCreationFailed("Failed to load plugin '\(pluginPath)': \(String(cString: errorMsg))")
            }
            throw XLAError.clientCreationFailed("Failed to load plugin '\(pluginPath)': error code \(errorCode.rawValue)")
        }

        // Create the client, optionally requesting several virtual CPU devices.
        var clientHandle: UnsafeMutableRawPointer?
        let createError: SW_PJRT_Error_Code
        if let cpuDeviceCount {
            createError = PJRT_CreateClientWithCpuDeviceCount(Int64(cpuDeviceCount), &clientHandle)
        } else {
            createError = PJRT_CreateClient(&clientHandle)
        }

        if createError != SW_PJRT_Error_OK {
            throw XLAError.clientCreationFailed("PJRT_CreateClient failed with code \(createError.rawValue)")
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

    /// Create a buffer from host data
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

        let dims = shape.map { Int64($0) }
        var bufferHandle: UnsafeMutableRawPointer?

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
            throw XLAError.bufferCreationFailed("PJRT_CreateBuffer failed with code \(errorCode.rawValue)")
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
        let errorCode = PJRT_CompileWrapper(clientHandle, mlir, &executableHandle)

        if errorCode != SW_PJRT_Error_OK {
            throw XLAError.compilationFailed("Compilation failed with code \(errorCode.rawValue)")
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

    /// Compile StableHLO MLIR with SPMD / Shardy partitioning options.
    ///
    /// - Parameters:
    ///   - numPartitions: number of partitions to partition across. Values > 1
    ///     require a client with at least that many devices.
    ///   - useSPMDPartitioning: enable XLA's SPMD partitioner.
    ///   - useShardyPartitioner: run the Shardy propagation + partitioning
    ///     pipeline (consumes `sdy.mesh` / `sdy.sharding` annotations in the
    ///     module). Requires the plugin to have been built with Shardy.
    ///
    /// With `numPartitions: 1, useSPMDPartitioning: false, useShardyPartitioner:
    /// false` this is equivalent to `compile(_:)`.
    public func compile(
        _ mlir: String,
        numPartitions: Int,
        useSPMDPartitioning: Bool = true,
        useShardyPartitioner: Bool = false
    ) throws -> PJRTExecutable {
        guard let clientHandle = handle else {
            throw XLAError.compilationFailed("Client not initialized")
        }

        var executableHandle: UnsafeMutableRawPointer?
        let errorCode = PJRT_CompileWrapperSPMD(
            clientHandle,
            mlir,
            Int64(numPartitions),
            useSPMDPartitioning ? 1 : 0,
            useShardyPartitioner ? 1 : 0,
            &executableHandle
        )

        if errorCode != SW_PJRT_Error_OK {
            if let errorMsg = PJRT_GetLastError() {
                throw XLAError.compilationFailed("SPMD compilation failed: \(String(cString: errorMsg))")
            }
            throw XLAError.compilationFailed("SPMD compilation failed with code \(errorCode.rawValue)")
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
    }

    deinit {
        if let handle = handle {
            PJRT_DestroyBuffer(handle)
        }
    }

    /// Copy buffer contents to host (synchronous)
    public func toHost<T>(_ type: T.Type) throws -> [T] {
        guard let bufferHandle = handle else {
            throw XLAError.bufferTransferFailed("Buffer handle not available")
        }

        // Allocate raw memory and copy data from device
        let rawBuffer = UnsafeMutableRawPointer.allocate(
            byteCount: sizeInBytes,
            alignment: MemoryLayout<T>.alignment
        )
        defer { rawBuffer.deallocate() }

        let errorCode = PJRT_BufferToHost(bufferHandle, rawBuffer, sizeInBytes)

        if errorCode != SW_PJRT_Error_OK {
            throw XLAError.bufferTransferFailed("Transfer failed with code \(errorCode.rawValue)")
        }

        // Convert to typed array
        let typedPointer = rawBuffer.bindMemory(to: T.self, capacity: elementCount)
        return Array(UnsafeBufferPointer(start: typedPointer, count: elementCount))
    }

    /// Copy buffer to Float array
    public func toFloatArray() throws -> [Float] {
        try toHost(Float.self)
    }
}

// MARK: - PJRTExecutable

/// Compiled XLA program ready for execution
public final class PJRTExecutable: @unchecked Sendable {

    internal var handle: UnsafeMutableRawPointer?
    public weak var client: PJRTClient?
    public let devices: [PJRTDevice]

    public private(set) var executionCount: Int = 0

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
            throw XLAError.executionFailed("Execution failed with code \(errorCode.rawValue)")
        }

        executionCount += 1

        // Create output buffers
        var outputs: [PJRTBuffer] = []
        if let outputHandles = outputsPtr {
            for i in 0..<numOutputs {
                if let outputHandle = outputHandles[i] {
                    // Query dimensions
                    var dimsPtr: UnsafePointer<Int64>?
                    var numDims: Int = 0
                    PJRT_GetBufferDimensions(outputHandle, &dimsPtr, &numDims)

                    var shape: [Int] = []
                    if let dims = dimsPtr {
                        for j in 0..<numDims {
                            shape.append(Int(dims[j]))
                        }
                    }

                    // Query the real output element type; fall back to .float32
                    // only if the query fails (preserves prior behavior on error).
                    var elementType: ElementType = .float32
                    var cType = SW_PJRT_Buffer_Type_F32
                    if PJRT_GetBufferElementType(outputHandle, &cType) == SW_PJRT_Error_OK {
                        elementType = ElementType(cType: cType)
                    }

                    let buffer = PJRTBuffer(
                        handle: outputHandle,
                        shape: shape,
                        elementType: elementType,
                        device: device
                    )
                    outputs.append(buffer)
                }
            }
        }

        return outputs
    }

    /// Build a PJRTBuffer around a raw output handle, querying its shape and
    /// element type. Shared by single- and multi-device execute.
    private func makeOutputBuffer(_ handle: UnsafeMutableRawPointer, device: PJRTDevice) -> PJRTBuffer {
        var dimsPtr: UnsafePointer<Int64>?
        var numDims = 0
        PJRT_GetBufferDimensions(handle, &dimsPtr, &numDims)
        var shape: [Int] = []
        if let dims = dimsPtr {
            for j in 0..<numDims { shape.append(Int(dims[j])) }
        }
        var elementType: ElementType = .float32
        var cType = SW_PJRT_Buffer_Type_F32
        if PJRT_GetBufferElementType(handle, &cType) == SW_PJRT_Error_OK {
            elementType = ElementType(cType: cType)
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
        let errorCode = inputHandles.withUnsafeMutableBufferPointer { ptr in
            PJRT_ExecuteMultiDevice(
                execHandle, ptr.baseAddress, numDevices, numArgs, &outputsFlat, &numOutputs)
        }
        if errorCode != SW_PJRT_Error_OK {
            throw XLAError.executionFailed("Multi-device execution failed with code \(errorCode.rawValue)")
        }
        executionCount += 1
        defer { if let outputsFlat { PJRT_FreeOutputList(outputsFlat) } }

        var result: [[PJRTBuffer]] = []
        result.reserveCapacity(numDevices)
        for d in 0..<numDevices {
            let device = devices.indices.contains(d) ? devices[d] : devices[0]
            var devOutputs: [PJRTBuffer] = []
            if let flat = outputsFlat {
                for o in 0..<numOutputs {
                    if let h = flat[d * numOutputs + o] {
                        devOutputs.append(makeOutputBuffer(h, device: device))
                    }
                }
            }
            result.append(devOutputs)
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
