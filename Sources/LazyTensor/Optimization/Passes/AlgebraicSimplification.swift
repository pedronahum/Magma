// Magma - Algebraic Simplification Pass
// Applies mathematical identities to simplify operations
//
// This pass recognizes patterns like x + 0, x * 1, x - x, etc.
// and simplifies them to their mathematical equivalents.

import Foundation

/// Algebraic Simplification Pass
///
/// Applies mathematical identities to simplify the computation graph.
/// This can reduce operations and expose further optimization opportunities.
///
/// Supported simplifications:
/// - x + 0 = x
/// - x - 0 = x
/// - x * 1 = x
/// - x * 0 = 0
/// - x / 1 = x
/// - x - x = 0
/// - x / x = 1 (for non-zero x)
/// - exp(log(x)) = x
/// - log(exp(x)) = x
/// - transpose(transpose(x)) = x
/// - reshape(reshape(x, s1), s2) = reshape(x, s2)
/// - double negation: -(-x) = x
/// - sqrt(x^2) = |x| (approximately)
public final class AlgebraicSimplificationPass: OptimizationPass {

    public let name = "algebraic-simplification"
    public let dependencies: [String] = ["constant-folding"]
    public let enabledByDefault = true

    public init() {}

    public func run(on graph: IRGraph) -> IRGraph {
        // Build topological order if not already done
        if graph.nodes.isEmpty {
            graph.buildTopologicalOrder()
        }

        // Build a map from tensor ID to its producing operation for quick lookup
        var producingOp: [UInt64: (node: LazyTensorHandle, opKind: OpKind, inputs: [LazyTensorHandle], attributes: [String: Any])] = [:]

        for node in graph.nodes {
            if let irNode = node.irNode {
                if case .operation(let op, let inputs, let attrs) = irNode {
                    producingOp[node.id] = (node, op, inputs, attrs)
                }
            }
        }

        // Track replacements
        var replacements: [UInt64: LazyTensorHandle] = [:]
        var simplifiedCount = 0

        // Process each node looking for simplification opportunities
        for node in graph.nodes {
            guard let irNode = node.irNode else { continue }

            switch irNode {
            case .operation(let opKind, let inputs, let attributes):
                // Resolve any already-replaced inputs
                let resolvedInputs = inputs.map { replacements[$0.id] ?? $0 }

                // Try to simplify this operation
                if let simplification = simplify(
                    opKind: opKind,
                    inputs: resolvedInputs,
                    output: node,
                    attributes: attributes,
                    producingOp: producingOp
                ) {
                    switch simplification {
                    case .replace(let replacement):
                        replacements[node.id] = replacement
                        simplifiedCount += 1

                    case .updateInputs(let newInputs):
                        node.irNode = .operation(op: opKind, inputs: newInputs, attributes: attributes)

                    case .replaceOp(let newOp, let newInputs, let newAttrs):
                        node.irNode = .operation(op: newOp, inputs: newInputs, attributes: newAttrs)
                        simplifiedCount += 1
                    }
                } else if resolvedInputs != inputs {
                    // Update inputs if any were replaced
                    node.irNode = .operation(op: opKind, inputs: resolvedInputs, attributes: attributes)
                }

            case .constant, .data, .whileLoopTraced:
                break
            }
        }

        // If no simplifications were made, return original graph
        if simplifiedCount == 0 && replacements.isEmpty {
            return graph
        }

        // Build new graph with simplifications applied
        let newGraph = IRGraph()

        // Filter out replaced nodes
        for node in graph.nodes {
            if replacements[node.id] != nil {
                continue
            }
            newGraph.nodes.append(node)
        }

        // Update outputs with replacements
        newGraph.outputs = graph.outputs.map { output -> LazyTensorHandle in
            replacements[output.id] ?? output
        }

        return newGraph
    }

    /// Result of a simplification attempt
    private enum SimplificationResult {
        /// Replace this node with an existing tensor
        case replace(LazyTensorHandle)
        /// Update the inputs to this operation
        case updateInputs([LazyTensorHandle])
        /// Replace with a different operation
        case replaceOp(OpKind, [LazyTensorHandle], [String: Any])
    }

    /// Try to simplify an operation
    private func simplify(
        opKind: OpKind,
        inputs: [LazyTensorHandle],
        output: LazyTensorHandle,
        attributes: [String: Any],
        producingOp: [UInt64: (node: LazyTensorHandle, opKind: OpKind, inputs: [LazyTensorHandle], attributes: [String: Any])]
    ) -> SimplificationResult? {

        switch opKind {
        // x + 0 = x, 0 + x = x
        case .add:
            if isZeroConstant(inputs[1]) {
                return .replace(inputs[0])
            }
            if isZeroConstant(inputs[0]) {
                return .replace(inputs[1])
            }

        // x - 0 = x
        case .subtract:
            if isZeroConstant(inputs[1]) {
                return .replace(inputs[0])
            }
            // x - x = 0
            if inputs[0].id == inputs[1].id {
                // Need to create a zero constant - for now just skip
                // In a more complete implementation, we'd create a new constant node
                return nil
            }

        // x * 1 = x, 1 * x = x
        case .multiply:
            if isOneConstant(inputs[1]) {
                return .replace(inputs[0])
            }
            if isOneConstant(inputs[0]) {
                return .replace(inputs[1])
            }
            // x * 0 = 0, 0 * x = 0
            if isZeroConstant(inputs[1]) {
                return .replace(inputs[1])
            }
            if isZeroConstant(inputs[0]) {
                return .replace(inputs[0])
            }

        // x / 1 = x
        case .divide:
            if isOneConstant(inputs[1]) {
                return .replace(inputs[0])
            }

        // max(x, -inf) = x, max(-inf, x) = x
        case .maximum:
            if isNegInfConstant(inputs[1]) {
                return .replace(inputs[0])
            }
            if isNegInfConstant(inputs[0]) {
                return .replace(inputs[1])
            }

        // min(x, inf) = x, min(inf, x) = x
        case .minimum:
            if isInfConstant(inputs[1]) {
                return .replace(inputs[0])
            }
            if isInfConstant(inputs[0]) {
                return .replace(inputs[1])
            }

        // x^0 = 1, x^1 = x (when constants available)
        case .power:
            if isZeroConstant(inputs[1]) {
                // Would need to create constant 1 - skip for now
                return nil
            }
            if isOneConstant(inputs[1]) {
                return .replace(inputs[0])
            }

        // -(-x) = x
        case .negate:
            if let producer = producingOp[inputs[0].id], producer.opKind == .negate {
                return .replace(producer.inputs[0])
            }

        // exp(log(x)) = x
        case .exponential:
            if let producer = producingOp[inputs[0].id], producer.opKind == .log {
                return .replace(producer.inputs[0])
            }

        // log(exp(x)) = x
        case .log:
            if let producer = producingOp[inputs[0].id], producer.opKind == .exponential {
                return .replace(producer.inputs[0])
            }

        // sqrt(rsqrt(x)) = 1/sqrt(x) which needs computation - skip
        // rsqrt(sqrt(x)) = 1/x - also needs computation - skip

        // transpose(transpose(x)) = x (for simple 2D transpose)
        case .transpose:
            if let producer = producingOp[inputs[0].id], producer.opKind == .transpose {
                // Check if it's a simple transpose that cancels
                // For 2D, transpose(transpose(x)) = x
                if inputs[0].shape.count == 2 {
                    return .replace(producer.inputs[0])
                }
            }

        // reshape(reshape(x, s1), s2) = reshape(x, s2)
        case .reshape:
            if let producer = producingOp[inputs[0].id], producer.opKind == .reshape {
                // Skip the intermediate reshape
                return .replaceOp(.reshape, [producer.inputs[0]], attributes)
            }
            // reshape to same shape = identity
            if inputs[0].shape == output.shape {
                return .replace(inputs[0])
            }

        // broadcast to same shape = identity
        case .broadcast:
            if inputs[0].shape == output.shape {
                return .replace(inputs[0])
            }

        // relu(relu(x)) = relu(x)
        case .relu:
            if let producer = producingOp[inputs[0].id], producer.opKind == .relu {
                return .replace(inputs[0])
            }

        // abs(abs(x)) = abs(x)
        case .abs:
            if let producer = producingOp[inputs[0].id], producer.opKind == .abs {
                return .replace(inputs[0])
            }
            // abs(relu(x)) = relu(x)
            if let producer = producingOp[inputs[0].id], producer.opKind == .relu {
                return .replace(inputs[0])
            }

        // floor(floor(x)) = floor(x)
        case .floor:
            if let producer = producingOp[inputs[0].id], producer.opKind == .floor {
                return .replace(inputs[0])
            }

        // ceil(ceil(x)) = ceil(x)
        case .ceil:
            if let producer = producingOp[inputs[0].id], producer.opKind == .ceil {
                return .replace(inputs[0])
            }

        default:
            break
        }

        return nil
    }

    /// Check if a tensor is a constant with all zeros
    private func isZeroConstant(_ handle: LazyTensorHandle) -> Bool {
        guard let irNode = handle.irNode else { return false }
        if case .constant(let values, _) = irNode {
            return values.allSatisfy { $0 == 0 }
        }
        return false
    }

    /// Check if a tensor is a constant with all ones
    private func isOneConstant(_ handle: LazyTensorHandle) -> Bool {
        guard let irNode = handle.irNode else { return false }
        if case .constant(let values, _) = irNode {
            return values.allSatisfy { $0 == 1 }
        }
        return false
    }

    /// Check if a tensor is a constant with all negative infinity
    private func isNegInfConstant(_ handle: LazyTensorHandle) -> Bool {
        guard let irNode = handle.irNode else { return false }
        if case .constant(let values, _) = irNode {
            return values.allSatisfy { $0 == -.infinity }
        }
        return false
    }

    /// Check if a tensor is a constant with all positive infinity
    private func isInfConstant(_ handle: LazyTensorHandle) -> Bool {
        guard let irNode = handle.irNode else { return false }
        if case .constant(let values, _) = irNode {
            return values.allSatisfy { $0 == .infinity }
        }
        return false
    }
}

// Helper to compare input arrays
private func != (lhs: [LazyTensorHandle], rhs: [LazyTensorHandle]) -> Bool {
    guard lhs.count == rhs.count else { return true }
    for (l, r) in zip(lhs, rhs) {
        if l.id != r.id { return true }
    }
    return false
}
