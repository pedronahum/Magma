// Magma - Dead Code Elimination Pass
// Removes operations whose outputs are never used
//
// This is a fundamental optimization that eliminates wasted computation.
// It works by marking all output tensors as "live" and propagating liveness
// backwards through the graph.

import Foundation

/// Dead Code Elimination Pass
///
/// Removes operations whose outputs are not used by any live computation.
/// An operation is "live" if:
/// 1. It produces a graph output, or
/// 2. Its output is used by another live operation
///
/// Example:
/// ```
/// a = input()
/// b = a * 2        // used by c
/// c = b + 1        // output
/// d = a * 3        // DEAD - not used
/// output(c)
/// ```
/// After DCE, operation 'd' is removed.
public final class DeadCodeEliminationPass: OptimizationPass {

    public let name = "dce"
    public let dependencies: [String] = []
    public let enabledByDefault = true

    public init() {}

    public func run(on graph: IRGraph) -> IRGraph {
        // Build topological order if not already done
        if graph.nodes.isEmpty {
            graph.buildTopologicalOrder()
        }

        // 1. Mark all output tensor IDs as "live"
        var live = Set<UInt64>(graph.outputs.map { $0.id })

        // 2. Backward pass: mark inputs of live ops as live
        // We iterate in reverse topological order (from outputs to inputs)
        var changed = true
        while changed {
            changed = false

            for node in graph.nodes.reversed() {
                // If this node is live, mark all its inputs as live
                if live.contains(node.id) {
                    if let irNode = node.irNode {
                        let inputs = getInputs(from: irNode)
                        for input in inputs {
                            if !live.contains(input.id) {
                                live.insert(input.id)
                                changed = true
                            }
                        }
                    }
                }
            }
        }

        // 3. Filter to keep only live nodes
        let liveNodes = graph.nodes.filter { live.contains($0.id) }

        // 4. Create new graph with live nodes only
        let newGraph = IRGraph()
        newGraph.nodes = liveNodes
        newGraph.outputs = graph.outputs

        return newGraph
    }

    /// Extract input handles from an IR node
    private func getInputs(from node: IRNode) -> [LazyTensorHandle] {
        switch node {
        case .constant, .data:
            return []

        case .operation(_, let inputs, _):
            return inputs

        case .whileLoopTraced(_, let initialValues, _, _, let bodyNodes):
            // For while loops, both initial values and body nodes contribute to liveness
            // Body nodes are internal, so we only return initial values as external inputs
            return initialValues
        }
    }
}
