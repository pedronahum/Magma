// Magma - Common Subexpression Elimination Pass
// Reuses identical computations by detecting operations with the same
// inputs and attributes.
//
// This optimization reduces redundant computation when the same operation
// appears multiple times in a graph.

import Foundation
import StableHLO

/// Key for identifying identical expressions
///
/// Two operations are considered identical if they have:
/// 1. Same operation type
/// 2. Same input tensor IDs (in order)
/// 3. Same attributes (compared exactly, whatever their type)
/// 4. Same result shape and dtype
///
/// Two constants are identical only if their shape, dtype and every value
/// (bit for bit) match. Keys compare their contents, not just a hash, so a
/// hash collision can never merge different expressions.
struct ExpressionKey: Hashable {
    let opKind: OpKind?
    let inputIds: [UInt64]
    let attributes: String
    let shape: [Int]?
    let dtype: DType?
    let constantBits: [UInt32]?

    init(opKind: OpKind, inputs: [LazyTensorHandle], attributes: [String: Any],
         output: LazyTensorHandle? = nil) {
        self.opKind = opKind
        self.inputIds = inputs.map { $0.id }
        // Every attribute type counts (e.g. a `convert`'s DType target, which a
        // numeric-only hash used to ignore).
        self.attributes = IRGraph.attributesDescription(attributes)
        self.shape = output?.shape
        self.dtype = output?.dtype
        self.constantBits = nil
    }

    init(constant values: [Float], shape: [Int], dtype: DType) {
        self.opKind = nil
        self.inputIds = []
        self.attributes = ""
        self.shape = shape
        self.dtype = dtype
        self.constantBits = values.map(\.bitPattern)
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
                // RNG operations must never be CSE'd — each call produces
                // independent random values even with identical inputs
                if opKind == .rngUniform || opKind == .rngNormal {
                    let resolvedInputs = inputs.map { input -> LazyTensorHandle in
                        replacements[input.id] ?? input
                    }
                    if resolvedInputs != inputs {
                        node.irNode = .operation(op: opKind, inputs: resolvedInputs, attributes: attributes)
                    }
                    break
                }

                // Apply existing replacements to inputs
                let resolvedInputs = inputs.map { input -> LazyTensorHandle in
                    replacements[input.id] ?? input
                }

                // Create expression key with resolved inputs
                let key = ExpressionKey(
                    opKind: opKind, inputs: resolvedInputs, attributes: attributes, output: node)

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
                // Constants can be deduplicated only when every value matches:
                // comparing a sample merged large constants that differed
                // elsewhere.
                let key = ExpressionKey(constant: values, shape: shape, dtype: node.dtype)

                if let existing = expressionToTensor[key] {
                    replacements[node.id] = existing
                } else {
                    expressionToTensor[key] = node
                }

            case .data, .whileLoopTraced:
                // Data nodes and while loops are not CSE candidates
                break
            #if os(macOS) && canImport(MetalHLO)
            case .metalData:
                break
            #endif
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
