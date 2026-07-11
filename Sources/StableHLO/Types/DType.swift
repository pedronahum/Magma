// Magma - StableHLO
// Data type definitions

/// Tensor element data types
public enum DType: String, Hashable, Sendable {
    case bool = "i1"
    case int8 = "i8"
    case int16 = "i16"
    case int32 = "i32"
    case int64 = "i64"
    case uint8 = "ui8"
    case uint16 = "ui16"
    case uint32 = "ui32"
    case uint64 = "ui64"
    case float16 = "f16"
    case float32 = "f32"
    case float64 = "f64"
    case bfloat16 = "bf16"
    
    /// MLIR type string representation
    public var mlirName: String { rawValue }

    /// Largest finite-magnitude literal for this dtype, used as a ±infinity
    /// sentinel for max/min reductions (StableHLO text can't carry raw inf here).
    public var maxFiniteLiteral: String {
        switch self {
        case .float16:  return "6.5504e+4"                       // f16 max
        case .bfloat16: return "3.3895314e+38"                   // bf16 max
        case .float32:  return "3.4028235e+38"                   // f32 max
        case .float64:  return "1.7976931348623157e+308"         // f64 max
        case .int8:     return "127"
        case .int16:    return "32767"
        case .int32:    return "2147483647"
        case .int64:    return "9223372036854775807"
        case .uint8:    return "255"
        case .uint16:   return "65535"
        case .uint32:   return "4294967295"
        case .uint64:   return "18446744073709551615"
        case .bool:     return "1"
        }
    }

    /// Zero literal for this dtype (for pad fill values etc.).
    public var zeroLiteral: String {
        if self == .bool { return "false" }
        return isFloatingPoint ? "0.0" : "0"
    }

    /// Size in bytes
    public var byteSize: Int {
        switch self {
        case .bool, .int8, .uint8: return 1
        case .int16, .uint16, .float16, .bfloat16: return 2
        case .int32, .uint32, .float32: return 4
        case .int64, .uint64, .float64: return 8
        }
    }
    
    /// Whether this is a floating point type
    public var isFloatingPoint: Bool {
        switch self {
        case .float16, .float32, .float64, .bfloat16: return true
        default: return false
        }
    }
    
    /// Whether this is a signed integer type
    public var isSignedInteger: Bool {
        switch self {
        case .int8, .int16, .int32, .int64: return true
        default: return false
        }
    }
    
    /// Whether this is an unsigned integer type
    public var isUnsignedInteger: Bool {
        switch self {
        case .uint8, .uint16, .uint32, .uint64: return true
        default: return false
        }
    }
}

// MARK: - Swift Type Mapping

/// Protocol for types that can be tensor elements
public protocol TensorScalar {
    static var dtype: DType { get }
}

extension Float: TensorScalar {
    public static var dtype: DType { .float32 }
}

extension Double: TensorScalar {
    public static var dtype: DType { .float64 }
}

extension Int32: TensorScalar {
    public static var dtype: DType { .int32 }
}

extension Int64: TensorScalar {
    public static var dtype: DType { .int64 }
}

extension Int: TensorScalar {
    public static var dtype: DType { .int64 }
}

extension Bool: TensorScalar {
    public static var dtype: DType { .bool }
}

// MARK: - Floating Point Tensor Scalar

/// Protocol for floating point types that can be used in neural network computations.
///
/// This protocol extends `TensorScalar` with the mathematical operations needed
/// for deep learning: arithmetic, comparison, and elementary functions.
/// Types conforming to this protocol can be used in differentiable tensor computations.
///
/// Conforming types:
/// - `Float` (float32) - Most common for training
/// - `Double` (float64) - Higher precision when needed
///
/// Example:
/// ```swift
/// func myLayer<Scalar: FloatingPointTensorScalar>(input: Tensor<Scalar>) -> Tensor<Scalar> {
///     return input.relu() * Scalar(2.0)
/// }
/// ```
public protocol FloatingPointTensorScalar: TensorScalar, BinaryFloatingPoint, ExpressibleByFloatLiteral
where Self.RawSignificand: FixedWidthInteger {
    /// Create an instance from a Float value
    init(_ value: Float)

    /// Create an instance from a Double value
    init(_ value: Double)
}

extension Float: FloatingPointTensorScalar {}
extension Double: FloatingPointTensorScalar {}

// MARK: - Additional Integer Conformances

extension UInt8: TensorScalar {
    public static var dtype: DType { .uint8 }
}

extension Int8: TensorScalar {
    public static var dtype: DType { .int8 }
}

extension UInt16: TensorScalar {
    public static var dtype: DType { .uint16 }
}

extension Int16: TensorScalar {
    public static var dtype: DType { .int16 }
}

extension UInt32: TensorScalar {
    public static var dtype: DType { .uint32 }
}

extension UInt64: TensorScalar {
    public static var dtype: DType { .uint64 }
}
