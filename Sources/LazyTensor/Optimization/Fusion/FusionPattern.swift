// Magma - Fusion Pattern Infrastructure
// Defines the protocol and types for pattern-based operation fusion
//
// Fusion patterns detect sequences of operations that can be replaced
// with a single fused operation, reducing kernel launches and memory traffic.

import Foundation

// MARK: - Fusion Match

/// Represents a successful pattern match in the graph
public struct FusionMatch {
    /// The operations that were matched (in dependency order)
    public let operations: [LazyTensorHandle]

    /// Indices of matched operations in the graph's node list
    public let indices: [Int]

    /// Metadata extracted during matching (e.g., scale factor, activation type)
    public let metadata: [String: Any]

    /// The final output of the matched pattern
    public var output: LazyTensorHandle? {
        operations.last
    }

    public init(operations: [LazyTensorHandle], indices: [Int], metadata: [String: Any] = [:]) {
        self.operations = operations
        self.indices = indices
        self.metadata = metadata
    }
}

// MARK: - Fusion Pattern Protocol

/// Protocol for implementing fusion patterns
///
/// A fusion pattern detects a specific sequence of operations in the graph
/// and replaces them with a single fused operation.
///
/// Pattern matching is done in two phases:
/// 1. `match(in:)` - Searches for the pattern and returns a match if found
/// 2. `apply(_:to:)` - Replaces the matched operations with the fused version
public protocol FusionPattern {
    /// Unique name for this pattern
    var name: String { get }

    /// Priority for pattern matching (higher = tried first)
    /// Default is 0. Use higher values for patterns that subsume others.
    var priority: Int { get }

    /// Search for this pattern in the graph
    ///
    /// - Parameter graph: The graph to search
    /// - Returns: A FusionMatch if the pattern was found, nil otherwise
    func match(in graph: IRGraph) -> FusionMatch?

    /// Apply the fusion to the graph
    ///
    /// - Parameters:
    ///   - match: The match returned by `match(in:)`
    ///   - graph: The graph to transform
    /// - Returns: The transformed graph with fused operations
    func apply(_ match: FusionMatch, to graph: IRGraph) -> IRGraph
}

// MARK: - Default Implementation

extension FusionPattern {
    public var priority: Int { 0 }
}

// MARK: - Pattern Matching Helpers

/// Helper functions for pattern matching
public enum PatternMatchingHelpers {

    /// Get the operation that produces a tensor
    public static func getProducingOp(
        _ handle: LazyTensorHandle,
        in graph: IRGraph
    ) -> (opKind: OpKind, inputs: [LazyTensorHandle], attributes: [String: Any])? {
        guard let irNode = handle.irNode else { return nil }
        if case .operation(let op, let inputs, let attrs) = irNode {
            return (op, inputs, attrs)
        }
        return nil
    }

    /// Check if a tensor is a constant with a specific value
    public static func isConstantWithValue(_ handle: LazyTensorHandle, _ value: Float, tolerance: Float = 1e-6) -> Bool {
        guard let irNode = handle.irNode else { return false }
        if case .constant(let values, _) = irNode {
            return values.count == 1 && abs(values[0] - value) < tolerance
        }
        return false
    }

    /// Check if a tensor is a scalar constant
    public static func isScalarConstant(_ handle: LazyTensorHandle) -> Bool {
        guard let irNode = handle.irNode else { return false }
        if case .constant(_, let shape) = irNode {
            return shape.isEmpty || shape == [1]
        }
        return false
    }

    /// Extract scalar constant value
    public static func getScalarValue(_ handle: LazyTensorHandle) -> Float? {
        guard let irNode = handle.irNode else { return nil }
        if case .constant(let values, _) = irNode, values.count == 1 {
            return values[0]
        }
        return nil
    }

    /// Check if a handle has exactly one consumer in the graph
    public static func hasSingleConsumer(_ handle: LazyTensorHandle, in graph: IRGraph) -> Bool {
        var consumerCount = 0
        for node in graph.nodes {
            guard let irNode = node.irNode else { continue }
            if case .operation(_, let inputs, _) = irNode {
                if inputs.contains(where: { $0.id == handle.id }) {
                    consumerCount += 1
                    if consumerCount > 1 { return false }
                }
            }
        }
        // Also check if it's a graph output
        if graph.outputs.contains(where: { $0.id == handle.id }) {
            consumerCount += 1
        }
        return consumerCount == 1
    }

    /// Find all operations of a specific kind
    public static func findOperations(ofKind kind: OpKind, in graph: IRGraph) -> [LazyTensorHandle] {
        graph.nodes.filter { node in
            if let irNode = node.irNode,
               case .operation(let op, _, _) = irNode,
               op == kind {
                return true
            }
            return false
        }
    }

    /// Build operation lookup map
    public static func buildOperationMap(_ graph: IRGraph) -> [UInt64: (LazyTensorHandle, OpKind, [LazyTensorHandle], [String: Any])] {
        var map: [UInt64: (LazyTensorHandle, OpKind, [LazyTensorHandle], [String: Any])] = [:]
        for node in graph.nodes {
            if let irNode = node.irNode,
               case .operation(let op, let inputs, let attrs) = irNode {
                map[node.id] = (node, op, inputs, attrs)
            }
        }
        return map
    }
}

// MARK: - Graph Transformation Helpers

extension IRGraph {

    /// Replace matched operations with fused operations
    ///
    /// - Parameters:
    ///   - matched: Operations to remove
    ///   - fusedOps: Fused operations to insert
    /// - Returns: New graph with replacements
    public func replacing(_ matched: [LazyTensorHandle], with fusedOps: [LazyTensorHandle]) -> IRGraph {
        let matchedIds = Set(matched.map { $0.id })

        let newGraph = IRGraph()

        // Copy nodes, replacing matched with fused
        for node in nodes {
            if matchedIds.contains(node.id) {
                continue // Skip matched nodes
            }
            newGraph.nodes.append(node)
        }

        // Add fused operations
        newGraph.nodes.append(contentsOf: fusedOps)

        // Re-sort topologically
        newGraph.buildTopologicalOrder()

        // Update outputs - replace any matched outputs with fused outputs
        let outputMap: [UInt64: LazyTensorHandle] = {
            var map: [UInt64: LazyTensorHandle] = [:]
            // The last matched operation's output should map to the fused output
            if let lastMatched = matched.last, let fusedOutput = fusedOps.last {
                map[lastMatched.id] = fusedOutput
            }
            return map
        }()

        newGraph.outputs = outputs.map { output in
            outputMap[output.id] ?? output
        }

        return newGraph
    }

    /// Update all references from one tensor to another
    public func updateReferences(from oldId: UInt64, to newHandle: LazyTensorHandle) {
        for node in nodes {
            guard let irNode = node.irNode else { continue }
            if case .operation(let op, let inputs, let attrs) = irNode {
                let updatedInputs = inputs.map { input -> LazyTensorHandle in
                    input.id == oldId ? newHandle : input
                }
                if updatedInputs != inputs {
                    node.irNode = .operation(op: op, inputs: updatedInputs, attributes: attrs)
                }
            }
        }
    }
}

// MARK: - Activation Types

/// Common activation functions for fused operations
public enum ActivationType: String, Sendable {
    case none
    case relu
    case gelu
    case geluApproximate
    case sigmoid
    case tanh
    case silu
    case leakyRelu
    case elu
}

/// Helper to identify activation operations
extension OpKind {
    /// Whether this operation is an activation function
    public var isActivation: Bool {
        switch self {
        case .relu, .sigmoid, .gelu, .leakyRelu, .elu, .silu, .tanh:
            return true
        default:
            return false
        }
    }

    /// Convert to ActivationType
    public var asActivationType: ActivationType? {
        switch self {
        case .relu: return .relu
        case .gelu: return .gelu
        case .sigmoid: return .sigmoid
        case .tanh: return .tanh
        case .silu: return .silu
        case .leakyRelu: return .leakyRelu
        case .elu: return .elu
        default: return nil
        }
    }
}

// MARK: - Input Array Comparison

private func != (lhs: [LazyTensorHandle], rhs: [LazyTensorHandle]) -> Bool {
    guard lhs.count == rhs.count else { return true }
    for (l, r) in zip(lhs, rhs) {
        if l.id != r.id { return true }
    }
    return false
}
