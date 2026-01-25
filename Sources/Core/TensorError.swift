// Magma - TensorError
// Comprehensive error handling for tensor operations

import StableHLO

// MARK: - Tensor Error Types

/// Errors that can occur during tensor operations
public enum TensorError: Error, CustomStringConvertible {

    // MARK: Shape Errors

    /// Shape mismatch in binary operation
    case shapeMismatch(
        operation: String,
        lhsShape: [Int],
        rhsShape: [Int],
        message: String? = nil
    )

    /// Shape incompatible with operation
    case incompatibleShape(
        operation: String,
        shape: [Int],
        requirement: String
    )

    /// Invalid reshape
    case invalidReshape(
        from: [Int],
        to: [Int]
    )

    /// Invalid broadcast
    case invalidBroadcast(
        from: [Int],
        to: [Int]
    )

    /// Axis out of bounds
    case axisOutOfBounds(
        axis: Int,
        rank: Int
    )

    /// Index out of bounds
    case indexOutOfBounds(
        index: Int,
        dimension: Int,
        size: Int
    )

    // MARK: Device Errors

    /// Device mismatch between tensors
    case deviceMismatch(
        lhsDevice: String,
        rhsDevice: String,
        operation: String
    )

    /// Operation not supported on device
    case unsupportedDevice(
        operation: String,
        device: String
    )

    // MARK: Type Errors

    /// Type mismatch between tensors
    case typeMismatch(
        lhsType: DType,
        rhsType: DType,
        operation: String
    )

    /// Invalid type for operation
    case invalidType(
        operation: String,
        actualType: DType,
        expectedTypes: [DType]
    )

    // MARK: Value Errors

    /// Empty tensor where non-empty required
    case emptyTensor(operation: String)

    /// Scalar required but tensor has multiple elements
    case notScalar(shape: [Int])

    /// Invalid value in operation
    case invalidValue(
        operation: String,
        value: String,
        reason: String
    )

    // MARK: - Description

    public var description: String {
        switch self {
        case let .shapeMismatch(operation, lhsShape, rhsShape, message):
            var result = """
            Shape mismatch in '\(operation)':
              Left operand shape:  \(formatShape(lhsShape))
              Right operand shape: \(formatShape(rhsShape))
            """
            if let msg = message {
                result += "\n  \(msg)"
            }
            result += "\n" + shapeMismatchHelp(operation: operation, lhsShape: lhsShape, rhsShape: rhsShape)
            return result

        case let .incompatibleShape(operation, shape, requirement):
            return """
            Incompatible shape for '\(operation)':
              Shape: \(formatShape(shape))
              Requirement: \(requirement)
            """

        case let .invalidReshape(from, to):
            let fromCount = from.reduce(1, *)
            let toCount = to.reduce(1, *)
            return """
            Invalid reshape:
              From shape \(formatShape(from)) (\(fromCount) elements)
              To shape \(formatShape(to)) (\(toCount) elements)
              Total elements must be equal
            """

        case let .invalidBroadcast(from, to):
            return """
            Invalid broadcast:
              Cannot broadcast \(formatShape(from)) to \(formatShape(to))
              Broadcasting rules:
              - Shapes are compared from right to left
              - Dimensions must be equal or one must be 1
            """

        case let .axisOutOfBounds(axis, rank):
            let validRange = rank > 0 ? "[-\(rank), \(rank - 1)]" : "none (tensor is scalar)"
            return """
            Axis out of bounds:
              Axis: \(axis)
              Tensor rank: \(rank)
              Valid axis range: \(validRange)
            """

        case let .indexOutOfBounds(index, dimension, size):
            return """
            Index out of bounds:
              Index \(index) is out of bounds for dimension \(dimension) with size \(size)
              Valid range: [0, \(size - 1)] or [-\(size), -1]
            """

        case let .deviceMismatch(lhsDevice, rhsDevice, operation):
            return """
            Device mismatch in '\(operation)':
              Left operand device:  \(lhsDevice)
              Right operand device: \(rhsDevice)
              All operands must be on the same device.
              Use .to(device:) to move tensors to a common device.
            """

        case let .unsupportedDevice(operation, device):
            return """
            Operation '\(operation)' not supported on device '\(device)'
            """

        case let .typeMismatch(lhsType, rhsType, operation):
            return """
            Type mismatch in '\(operation)':
              Left operand type:  \(lhsType)
              Right operand type: \(rhsType)
              Use .to(dtype:) to convert tensors to a common type.
            """

        case let .invalidType(operation, actualType, expectedTypes):
            let expected = expectedTypes.map(\.rawValue).joined(separator: ", ")
            return """
            Invalid type for '\(operation)':
              Actual type: \(actualType)
              Expected one of: \(expected)
            """

        case let .emptyTensor(operation):
            return "Empty tensor provided to '\(operation)'. Operation requires non-empty tensor."

        case let .notScalar(shape):
            return """
            Expected scalar tensor but got shape \(formatShape(shape))
            Use .item() only on scalar tensors (shape [] or [1])
            """

        case let .invalidValue(operation, value, reason):
            return """
            Invalid value in '\(operation)':
              Value: \(value)
              Reason: \(reason)
            """
        }
    }

    // MARK: - Helpers

    private func formatShape(_ shape: [Int]) -> String {
        if shape.isEmpty {
            return "[] (scalar)"
        }
        return "[\(shape.map(String.init).joined(separator: ", "))]"
    }

    private func shapeMismatchHelp(operation: String, lhsShape: [Int], rhsShape: [Int]) -> String {
        switch operation.lowercased() {
        case "matmul", "dot", "mm":
            return matmulShapeHelp(lhsShape: lhsShape, rhsShape: rhsShape)
        case "add", "subtract", "multiply", "divide", "+", "-", "*", "/":
            return broadcastingHelp(lhsShape: lhsShape, rhsShape: rhsShape)
        default:
            return ""
        }
    }

    private func matmulShapeHelp(lhsShape: [Int], rhsShape: [Int]) -> String {
        guard lhsShape.count >= 2 && rhsShape.count >= 2 else {
            return "Hint: Matrix multiplication requires at least 2D tensors"
        }
        let lhsK = lhsShape[lhsShape.count - 1]
        let rhsK = rhsShape[rhsShape.count - 2]
        if lhsK != rhsK {
            return """
            Hint: For matmul, the inner dimensions must match:
              A[..., M, K] @ B[..., K, N] -> C[..., M, N]
              Here: A[..., \(lhsK)] does not match B[\(rhsK), ...]
            """
        }
        return ""
    }

    private func broadcastingHelp(lhsShape: [Int], rhsShape: [Int]) -> String {
        // Find where broadcasting fails
        let maxRank = max(lhsShape.count, rhsShape.count)
        let lhsPadded = Array(repeating: 1, count: maxRank - lhsShape.count) + lhsShape
        let rhsPadded = Array(repeating: 1, count: maxRank - rhsShape.count) + rhsShape

        for i in 0..<maxRank {
            let l = lhsPadded[i]
            let r = rhsPadded[i]
            if l != r && l != 1 && r != 1 {
                return """
                Hint: Broadcasting failed at dimension \(i):
                  Left size: \(l), Right size: \(r)
                  For broadcasting, dimensions must be equal or one must be 1
                """
            }
        }
        return ""
    }
}

// MARK: - Shape Debugging Utilities

/// Debug utilities for understanding tensor operations
public enum TensorDebug {

    /// Print detailed shape information
    public static func describeShape(_ shape: [Int], name: String = "tensor") {
        let elementCount = shape.isEmpty ? 1 : shape.reduce(1, *)
        print("\(name):")
        print("  Shape: \(formatShape(shape))")
        print("  Rank: \(shape.count)")
        print("  Elements: \(elementCount)")
        if !shape.isEmpty {
            print("  Dimensions: \(shape.enumerated().map { "dim\($0.offset)=\($0.element)" }.joined(separator: ", "))")
        }
    }

    /// Check if two shapes are broadcastable
    public static func areBroadcastable(_ shape1: [Int], _ shape2: [Int]) -> Bool {
        let maxRank = max(shape1.count, shape2.count)
        let padded1 = Array(repeating: 1, count: maxRank - shape1.count) + shape1
        let padded2 = Array(repeating: 1, count: maxRank - shape2.count) + shape2

        for i in 0..<maxRank {
            if padded1[i] != padded2[i] && padded1[i] != 1 && padded2[i] != 1 {
                return false
            }
        }
        return true
    }

    /// Compute the broadcast result shape
    public static func broadcastResultShape(_ shape1: [Int], _ shape2: [Int]) -> [Int]? {
        guard areBroadcastable(shape1, shape2) else { return nil }

        let maxRank = max(shape1.count, shape2.count)
        let padded1 = Array(repeating: 1, count: maxRank - shape1.count) + shape1
        let padded2 = Array(repeating: 1, count: maxRank - shape2.count) + shape2

        return (0..<maxRank).map { max(padded1[$0], padded2[$0]) }
    }

    /// Check if a reshape is valid
    public static func isValidReshape(from: [Int], to: [Int]) -> Bool {
        let fromElements = from.isEmpty ? 1 : from.reduce(1, *)
        let toElements = to.isEmpty ? 1 : to.reduce(1, *)
        return fromElements == toElements
    }

    /// Describe matmul shape requirements
    public static func matmulRequirements(_ lhs: [Int], _ rhs: [Int]) {
        print("Matrix multiplication shape analysis:")
        print("  Left:  \(formatShape(lhs))")
        print("  Right: \(formatShape(rhs))")

        guard lhs.count >= 2 && rhs.count >= 2 else {
            print("  ERROR: Both tensors must have rank >= 2")
            return
        }

        let lhsM = lhs[lhs.count - 2]
        let lhsK = lhs[lhs.count - 1]
        let rhsK = rhs[rhs.count - 2]
        let rhsN = rhs[rhs.count - 1]

        print("  Left inner dimensions:  [\(lhsM), \(lhsK)]")
        print("  Right inner dimensions: [\(rhsK), \(rhsN)]")

        if lhsK == rhsK {
            let resultInner = [lhsM, rhsN]
            print("  ✓ Compatible: K dimension matches (K=\(lhsK))")
            print("  Result inner dimensions: \(resultInner)")
        } else {
            print("  ✗ Incompatible: K dimensions don't match")
            print("    Left K:  \(lhsK)")
            print("    Right K: \(rhsK)")
        }
    }

    private static func formatShape(_ shape: [Int]) -> String {
        if shape.isEmpty {
            return "[] (scalar)"
        }
        return "[\(shape.map(String.init).joined(separator: ", "))]"
    }
}

// MARK: - Assertion Helpers with Better Messages

/// Better assertion helpers for tensor operations
public enum TensorAssert {

    /// Assert shapes are broadcastable
    public static func broadcastable(
        _ lhs: [Int],
        _ rhs: [Int],
        operation: String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        guard TensorDebug.areBroadcastable(lhs, rhs) else {
            fatalError(TensorError.shapeMismatch(operation: operation, lhsShape: lhs, rhsShape: rhs).description,
                       file: file, line: line)
        }
    }

    /// Assert shapes are equal
    public static func shapesEqual(
        _ lhs: [Int],
        _ rhs: [Int],
        operation: String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        guard lhs == rhs else {
            fatalError(TensorError.shapeMismatch(
                operation: operation,
                lhsShape: lhs,
                rhsShape: rhs,
                message: "Shapes must be exactly equal"
            ).description, file: file, line: line)
        }
    }

    /// Assert axis is valid for rank
    public static func validAxis(
        _ axis: Int,
        rank: Int,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        let normalizedAxis = axis < 0 ? axis + rank : axis
        guard normalizedAxis >= 0 && normalizedAxis < rank else {
            fatalError(TensorError.axisOutOfBounds(axis: axis, rank: rank).description,
                       file: file, line: line)
        }
    }

    /// Assert valid reshape
    public static func validReshape(
        from: [Int],
        to: [Int],
        file: StaticString = #file,
        line: UInt = #line
    ) {
        guard TensorDebug.isValidReshape(from: from, to: to) else {
            fatalError(TensorError.invalidReshape(from: from, to: to).description,
                       file: file, line: line)
        }
    }

    /// Assert valid matmul shapes
    public static func validMatmul(
        lhs: [Int],
        rhs: [Int],
        file: StaticString = #file,
        line: UInt = #line
    ) {
        guard lhs.count >= 2 && rhs.count >= 2 else {
            fatalError(TensorError.incompatibleShape(
                operation: "matmul",
                shape: lhs.count < 2 ? lhs : rhs,
                requirement: "Requires at least 2D tensors"
            ).description, file: file, line: line)
        }

        let lhsK = lhs[lhs.count - 1]
        let rhsK = rhs[rhs.count - 2]

        guard lhsK == rhsK else {
            fatalError(TensorError.shapeMismatch(
                operation: "matmul",
                lhsShape: lhs,
                rhsShape: rhs,
                message: "Inner dimensions must match: A[..., K] @ B[K, ...], but got K=\(lhsK) and K=\(rhsK)"
            ).description, file: file, line: line)
        }
    }

    /// Assert tensor is non-empty
    public static func nonEmpty(
        shape: [Int],
        operation: String,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        let elementCount = shape.isEmpty ? 1 : shape.reduce(1, *)
        guard elementCount > 0 else {
            fatalError(TensorError.emptyTensor(operation: operation).description,
                       file: file, line: line)
        }
    }

    /// Assert tensor is scalar
    public static func isScalar(
        shape: [Int],
        file: StaticString = #file,
        line: UInt = #line
    ) {
        let elementCount = shape.isEmpty ? 1 : shape.reduce(1, *)
        guard elementCount == 1 else {
            fatalError(TensorError.notScalar(shape: shape).description,
                       file: file, line: line)
        }
    }
}
