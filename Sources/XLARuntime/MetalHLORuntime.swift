// Magma - MetalHLO Runtime
// Metal GPU backend via MetalHLO for macOS
//
// This module provides:
// - MetalBackend: Static utilities for Metal availability
// - MetalHLOClient: Device management and program compilation
// - MetalHLOBuffer: On-device data buffers
// - MetalHLOExecutable: Compiled programs

#if os(macOS) && canImport(MetalHLO)

import MetalHLO
import StableHLO
import Metal

// To avoid ambiguity between MetalHLO.ElementType and XLARuntime.ElementType,
// we create wrapper types and use explicit type references.

// MARK: - Metal Backend Detection

/// Utilities for detecting Metal backend availability
public struct MetalBackend {

    /// Check if Metal is available on this system
    public static var isAvailable: Bool {
        // Check for Metal device
        guard MTLCreateSystemDefaultDevice() != nil else {
            return false
        }
        return true
    }

    /// Get the default Metal device name
    public static var deviceName: String? {
        return MTLCreateSystemDefaultDevice()?.name
    }
}

// MARK: - MetalHLO Client

/// Client for managing Metal devices and compiling programs via MetalHLO
public final class MetalHLOClient: @unchecked Sendable {

    /// The underlying MetalHLO client
    private let client: MetalHLO.Client

    /// Device name
    public var deviceName: String {
        client.deviceName
    }

    private init(client: MetalHLO.Client) {
        self.client = client
    }

    /// Create a Metal client
    public static func create() throws -> MetalHLOClient {
        let client = try MetalHLO.Client.create()
        return MetalHLOClient(client: client)
    }

    /// Create a buffer from host data
    public func createBuffer<T: Numeric>(
        _ data: [T],
        shape: [Int],
        elementType: XLARuntime.ElementType
    ) throws -> MetalHLOBuffer {
        // Convert Magma ElementType to MetalHLO ElementType
        let metalElementType = convertToMetalHLOElementType(elementType)
        let buffer = try client.createBuffer(data, shape: shape, elementType: metalElementType)
        return MetalHLOBuffer(buffer: buffer, shape: shape, elementType: elementType)
    }

    /// Create a buffer from Float data (optimized fast path)
    public func createBuffer(
        _ data: [Float],
        shape: [Int]
    ) -> MetalHLOBuffer {
        // Use the optimized non-throwing version that bypasses type conversion
        let buffer = client.createBuffer(data, shape: shape)
        return MetalHLOBuffer(buffer: buffer, shape: shape, elementType: .float32)
    }

    /// Create a buffer from Int32 data (optimized fast path)
    public func createBuffer(
        _ data: [Int32],
        shape: [Int]
    ) -> MetalHLOBuffer {
        let buffer = client.createBuffer(data, shape: shape)
        return MetalHLOBuffer(buffer: buffer, shape: shape, elementType: .int32)
    }

    /// Create a buffer from Int64 data (optimized fast path)
    public func createBuffer(
        _ data: [Int64],
        shape: [Int]
    ) -> MetalHLOBuffer {
        let buffer = client.createBuffer(data, shape: shape)
        return MetalHLOBuffer(buffer: buffer, shape: shape, elementType: .int64)
    }

    /// Compile StableHLO MLIR to an executable
    public func compile(_ mlir: String) throws -> MetalHLOExecutable {
        let executable = try client.compile(mlir)
        return MetalHLOExecutable(executable: executable, client: self)
    }
}

// MARK: - MetalHLO Buffer

/// On-device data buffer for Metal
public final class MetalHLOBuffer: @unchecked Sendable {

    /// The underlying MetalHLO buffer
    internal let buffer: MetalHLO.Buffer

    /// Shape of the tensor
    public let shape: [Int]

    /// Element type
    public let elementType: XLARuntime.ElementType

    /// Number of elements
    public var elementCount: Int {
        shape.isEmpty ? 1 : shape.reduce(1, *)
    }

    /// Size in bytes
    public var sizeInBytes: Int {
        elementCount * elementType.sizeInBytes
    }

    internal init(buffer: MetalHLO.Buffer, shape: [Int], elementType: XLARuntime.ElementType) {
        self.buffer = buffer
        self.shape = shape
        self.elementType = elementType
    }

    /// Copy buffer contents to host (synchronous)
    public func toHost<T>(_ type: T.Type) throws -> [T] {
        // Use the appropriate MetalHLO transfer method based on type
        if type == Float.self {
            return try buffer.toFloatArray() as! [T]
        } else if type == Int32.self {
            return try buffer.toInt32Array() as! [T]
        } else if type == Int64.self {
            return try buffer.toInt64Array() as! [T]
        } else if type == Bool.self {
            return try buffer.toBoolArray() as! [T]
        } else {
            // Fallback to Float and convert
            let floats = try buffer.toFloatArray()
            return floats as! [T]
        }
    }

    /// Copy buffer to Float array
    public func toFloatArray() throws -> [Float] {
        try buffer.toFloatArray()
    }
}

// MARK: - MetalHLO Executable

/// Compiled program ready for execution on Metal
public final class MetalHLOExecutable: @unchecked Sendable {

    /// The underlying MetalHLO executable
    private let executable: MetalHLO.Executable

    /// Reference to the client (for buffer creation)
    private weak var client: MetalHLOClient?

    /// Number of executions
    public private(set) var executionCount: Int = 0

    /// Number of inputs expected
    public var inputCount: Int {
        executable.inputCount
    }

    /// Number of outputs produced
    public var outputCount: Int {
        executable.outputCount
    }

    internal init(executable: MetalHLO.Executable, client: MetalHLOClient) {
        self.executable = executable
        self.client = client
    }

    /// Execute the program with input buffers
    public func execute(_ inputs: [MetalHLOBuffer]) throws -> [MetalHLOBuffer] {
        // Extract underlying MetalHLO buffers
        let metalInputs = inputs.map { $0.buffer }

        // Execute via MetalHLO
        let metalOutputs = try executable.execute(metalInputs)

        executionCount += 1

        // Wrap outputs in MetalHLOBuffer
        return metalOutputs.enumerated().map { (index, metalBuffer) in
            // Get output type from executable
            let outputType = executable.outputTypes[index]
            return MetalHLOBuffer(
                buffer: metalBuffer,
                shape: outputType.shape,
                elementType: convertFromMetalHLOElementType(outputType.elementType)
            )
        }
    }

    /// Execute with timing information
    public func executeWithTiming(_ inputs: [MetalHLOBuffer]) throws -> ([MetalHLOBuffer], MetalExecutionTiming) {
        let metalInputs = inputs.map { $0.buffer }

        let (metalOutputs, timing) = try executable.executeWithTiming(metalInputs)

        executionCount += 1

        let outputs = metalOutputs.enumerated().map { (index, metalBuffer) in
            let outputType = executable.outputTypes[index]
            return MetalHLOBuffer(
                buffer: metalBuffer,
                shape: outputType.shape,
                elementType: convertFromMetalHLOElementType(outputType.elementType)
            )
        }

        let metalTiming = MetalExecutionTiming(
            encodeTimeNs: UInt64(timing.encodeTime * 1_000_000_000),
            gpuTimeNs: UInt64(timing.gpuTime * 1_000_000_000),
            totalTimeNs: UInt64(timing.totalTime * 1_000_000_000)
        )

        return (outputs, metalTiming)
    }
}

// MARK: - Execution Timing

/// Timing breakdown for Metal execution
public struct MetalExecutionTiming: Sendable {
    public let encodeTimeNs: UInt64
    public let gpuTimeNs: UInt64
    public let totalTimeNs: UInt64

    public var totalMs: Double {
        Double(totalTimeNs) / 1_000_000.0
    }

    public var gpuMs: Double {
        Double(gpuTimeNs) / 1_000_000.0
    }
}

// MARK: - Type Conversions

/// Convert XLARuntime.ElementType to MetalHLO.ElementType
public func convertToMetalHLOElementType(_ type: XLARuntime.ElementType) -> MetalHLO.ElementType {
    switch type {
    case .bool: return .int1
    case .int8: return .int8
    case .int16: return .int16
    case .int32: return .int32
    case .int64: return .int64
    case .uint8: return .uint8
    case .uint16: return .uint16
    case .uint32: return .uint32
    case .uint64: return .uint64
    case .float16: return .float16
    case .float32: return .float32
    case .float64: return .float64
    case .bfloat16: return .bfloat16
    case .complex64, .complex128:
        // MetalHLO doesn't support complex types, fall back to float32
        return .float32
    }
}

/// Convert MetalHLO.ElementType to XLARuntime.ElementType
public func convertFromMetalHLOElementType(_ type: MetalHLO.ElementType) -> XLARuntime.ElementType {
    switch type {
    case .int1: return .bool
    case .int8: return .int8
    case .int16: return .int16
    case .int32: return .int32
    case .int64: return .int64
    case .uint8: return .uint8
    case .uint16: return .uint16
    case .uint32: return .uint32
    case .uint64: return .uint64
    case .float16: return .float16
    case .float32: return .float32
    case .float64: return .float64
    case .bfloat16: return .bfloat16
    }
}

/// Convert DType to MetalHLO.ElementType
public func convertDTypeToMetalHLOElementType(_ dtype: DType) -> MetalHLO.ElementType {
    switch dtype {
    case .bool: return .int1
    case .int8: return .int8
    case .int16: return .int16
    case .int32: return .int32
    case .int64: return .int64
    case .uint8: return .uint8
    case .uint16: return .uint16
    case .uint32: return .uint32
    case .uint64: return .uint64
    case .float16: return .float16
    case .float32: return .float32
    case .float64: return .float64
    case .bfloat16: return .bfloat16
    }
}

/// Convert MetalHLO.ElementType to DType
public func convertMetalHLOElementTypeToDType(_ type: MetalHLO.ElementType) -> DType {
    switch type {
    case .int1: return .bool
    case .int8: return .int8
    case .int16: return .int16
    case .int32: return .int32
    case .int64: return .int64
    case .uint8: return .uint8
    case .uint16: return .uint16
    case .uint32: return .uint32
    case .uint64: return .uint64
    case .float16: return .float16
    case .float32: return .float32
    case .float64: return .float64
    case .bfloat16: return .bfloat16
    }
}

#endif // os(macOS) && canImport(MetalHLO)
