// Magma - StableHLO
// SSA Value representation

/// Represents a value in SSA form
public struct Value: Hashable, Sendable, CustomStringConvertible {
    /// Unique identifier within the function
    public let id: Int

    /// Type of this value
    public let type: TensorType

    /// Whether this is a block argument (for control flow regions)
    public let isBlockArg: Bool

    /// Prefix for block argument names (e.g., "cond" -> %cond_0)
    public let blockArgPrefix: String?

    /// SSA name (e.g., "%0" or "%cond_0" for block args)
    public var name: String {
        if isBlockArg, let prefix = blockArgPrefix {
            // id=-1 means use prefix as-is (raw name)
            if id < 0 {
                return "%\(prefix)"
            }
            return "%\(prefix)_\(id)"
        }
        return "%\(id)"
    }

    public var description: String { "\(name): \(type)" }

    /// Create a value
    public init(id: Int, type: TensorType, isBlockArg: Bool = false, blockArgPrefix: String? = nil) {
        self.id = id
        self.type = type
        self.isBlockArg = isBlockArg
        self.blockArgPrefix = blockArgPrefix
    }
}

/// Comparison direction for stablehlo.compare
public enum CompareDirection: String, Sendable {
    case eq = "EQ"
    case ne = "NE"
    case lt = "LT"
    case le = "LE"
    case gt = "GT"
    case ge = "GE"
}

/// Represents a function argument
public struct Argument: Hashable, Sendable, CustomStringConvertible {
    /// Argument index
    public let index: Int
    
    /// Type of this argument
    public let type: TensorType
    
    /// Argument name (e.g., "%arg0")
    public var name: String { "%arg\(index)" }
    
    public var description: String { "\(name): \(type)" }
    
    /// Create an argument
    public init(index: Int, type: TensorType) {
        self.index = index
        self.type = type
    }
    
    /// Convert to Value for use in operations
    public var asValue: Value {
        // Use negative IDs for arguments to distinguish from regular values
        Value(id: -(index + 1), type: type)
    }
}

extension Value {
    /// Whether this value is an argument (negative ID)
    public var isArgument: Bool { id < 0 }
    
    /// Argument index if this is an argument
    public var argumentIndex: Int? {
        isArgument ? -(id + 1) : nil
    }
    
    /// Display name accounting for arguments and block args
    public var displayName: String {
        // Check block arg first (block args with id=-1 should use raw name, not %arg)
        if isBlockArg, let prefix = blockArgPrefix {
            // id<0 means use prefix as-is (raw name)
            if id < 0 {
                return "%\(prefix)"
            }
            return "%\(prefix)_\(id)"
        }
        // Then check function arguments (negative id without blockArg flag)
        if let argIdx = argumentIndex {
            return "%arg\(argIdx)"
        }
        return "%\(id)"
    }
}
