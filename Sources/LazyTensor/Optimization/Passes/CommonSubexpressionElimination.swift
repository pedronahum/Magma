// Magma - Common Subexpression Elimination Pass
// Reuses identical computations by detecting operations with the same
// inputs and attributes.
//
// This optimization reduces redundant computation when the same operation
// appears multiple times in a graph.

import Foundation

/// Key for identifying identical expressions
///
/// Two operations are considered identical if they have:
/// 1. Same operation type
/// 2. Same input tensor IDs (in order)
/// 3. Same attributes (for operations that have them)
struct ExpressionKey: Hashable {
    let opKind: OpKind
    let inputIds: [UInt64]
    let attributesHash: Int

    init(opKind: OpKind, inputs: [LazyTensorHandle], attributes: [String: Any]) {
        self.opKind = opKind
        self.inputIds = inputs.map { $0.id }

        // Compute a hash for attributes
        // This is a simplified version - in production, you'd want more robust hashing
        var hasher = Hasher()
        for (key, value) in attributes.sorted(by: { $0.key < $1.key }) {
            hasher.combine(key)
            if let intVal = value as? Int {
                hasher.combine(intVal)
            } else if let floatVal = value as? Float {
                hasher.combine(floatVal)
            } else if let intArray = value as? [Int] {
                for i in intArray { hasher.combine(i) }
            } else if let stringVal = value as? String {
                hasher.combine(stringVal)
            }
            // Other types are ignored for hashing purposes
        }
        self.attributesHash = hasher.finalize()
    }
}

/// Common Subexpression Elimination Pass
///
/// Identifies operations that compute the same value and eliminates duplicates.
/// When two operations have identical inputs and attributes, only one is kept
/// and all references to the duplicate are replaced with the original.
///
/// Example:
/// ```
/// a = input()
/// b = a * 2
/// c = a * 2        // DUPLICATE of b
/// d = b + c        // Uses both b and c
/// output(d)
/// ```
/// After CSE:
/// ```
/// a = input()
/// b = a * 2
/// d = b + b        // c replaced with b
/// output(d)
/// ```
public final class CommonSubexpressionEliminationPass: OptimizationPass {

    public let name = "cse"
    public let dependencies: [String] = ["dce"]
    public let enabledByDefault = true

    public init() {}

    public func run(on graph: IRGraph) -> IRGraph {
        // Build topological order if not already done
        if graph.nodes.isEmpty {
            graph.buildTopologicalOrder()
        }

        // Map from expression key to the tensor that computes it
        var expressionToTensor: [ExpressionKey: LazyTensorHandle] = [:]

        // Map from replaced tensor ID to its replacement
        var replacements: [UInt64: LazyTensorHandle] = [:]

        // Process nodes in topological order
        for node in graph.nodes {
            guard let irNode = node.irNode else { continue }

            switch irNode {
            case .operation(let opKind, let inputs, let attributes):
                // Apply existing replacements to inputs
                let resolvedInputs = inputs.map { input -> LazyTensorHandle in
                    replacements[input.id] ?? input
                }

                // Create expression key with resolved inputs
                let key = ExpressionKey(opKind: opKind, inputs: resolvedInputs, attributes: attributes)

                if let existing = expressionToTensor[key] {
                    // Found a duplicate - replace this node with the existing one
                    replacements[node.id] = existing
                } else {
                    // First occurrence - update inputs if any were replaced
                    if resolvedInputs != inputs {
                        // Need to update the node's inputs
                        node.irNode = .operation(op: opKind, inputs: resolvedInputs, attributes: attributes)
                    }
                    expressionToTensor[key] = node
                }

            case .constant(let values, let shape):
                // Constants with same values can be deduplicated
                let key = ExpressionKey(
                    opKind: .add, // Placeholder - constants don't have a real OpKind
                    inputs: [],
                    attributes: ["const_values": values.prefix(8).map { $0 }, "const_shape": shape]
                )

                if let existing = expressionToTensor[key] {
                    replacements[node.id] = existing
                } else {
                    expressionToTensor[key] = node
                }

            case .data, .whileLoopTraced:
                // Data nodes and while loops are not CSE candidates
                break
            }
        }

        // If no replacements were made, return original graph
        if replacements.isEmpty {
            return graph
        }

        // Build new graph with replacements applied
        let newGraph = IRGraph()

        // Filter out replaced nodes and update references
        for node in graph.nodes {
            // Skip nodes that were replaced
            if replacements[node.id] != nil {
                continue
            }

            // Update inputs for operations
            if let irNode = node.irNode {
                switch irNode {
                case .operation(let opKind, let inputs, let attributes):
                    let resolvedInputs = inputs.map { input -> LazyTensorHandle in
                        replacements[input.id] ?? input
                    }
                    if resolvedInputs != inputs {
                        node.irNode = .operation(op: opKind, inputs: resolvedInputs, attributes: attributes)
                    }
                default:
                    break
                }
            }

            newGraph.nodes.append(node)
        }

        // Update outputs with replacements
        newGraph.outputs = graph.outputs.map { output -> LazyTensorHandle in
            replacements[output.id] ?? output
        }

        return newGraph
    }
}

// MARK: - Array Equality for LazyTensorHandle

/// Allow comparison of input arrays
extension Array where Element == LazyTensorHandle {
    static func != (lhs: [LazyTensorHandle], rhs: [LazyTensorHandle]) -> Bool {
        guard lhs.count == rhs.count else { return true }
        for (l, r) in zip(lhs, rhs) {
            if l.id != r.id { return true }
        }
        return false
    }
}
