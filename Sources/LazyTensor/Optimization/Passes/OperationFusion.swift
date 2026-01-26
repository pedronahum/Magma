// Magma - Operation Fusion Pass
// NOTE: Fusion patterns have been migrated to MetalHLO for backend-specific optimization.
//
// MetalHLO performs pattern detection at the HLO level, allowing Magma to emit pure
// StableHLO that is compatible with all backends (Metal, CPU via PJRT, TPU via PJRT, etc.)
//
// This pass is kept for backward compatibility but performs no transformations.

import Foundation

/// Operation Fusion Pass
///
/// This pass previously detected patterns of operations that could be fused and replaced
/// them with single fused operations. This functionality has been moved to MetalHLO.
///
/// The pass is retained for backward compatibility with existing code that registers
/// custom patterns or expects the pass to exist in the PassManager.
public final class OperationFusionPass: OptimizationPass {

    public let name = "operation-fusion"
    public let dependencies: [String] = ["algebraic-simplification"]
    public let enabledByDefault = false  // Disabled since fusion is now done in MetalHLO

    /// Registered fusion patterns (empty by default)
    private var patterns: [FusionPattern] = []

    /// Maximum number of fusion iterations (prevents infinite loops)
    public var maxIterations: Int = 100

    public init() {
        // No patterns registered by default.
        // Fusion is now handled by MetalHLO at the HLO level.
    }

    /// Register a custom fusion pattern
    ///
    /// Custom patterns can still be registered if needed for specific use cases.
    public func registerPattern(_ pattern: FusionPattern) {
        patterns.append(pattern)
        patterns.sort { $0.priority > $1.priority }
    }

    public func run(on graph: IRGraph) -> IRGraph {
        // If no patterns are registered, return the graph unchanged
        guard !patterns.isEmpty else {
            return graph
        }

        // Build topological order if not already done
        if graph.nodes.isEmpty {
            graph.buildTopologicalOrder()
        }

        var currentGraph = graph
        var changed = true
        var iteration = 0

        // Fixed-point iteration: keep applying patterns until no more changes
        while changed && iteration < maxIterations {
            changed = false
            iteration += 1

            for pattern in patterns {
                if let match = pattern.match(in: currentGraph) {
                    // Apply the fusion
                    currentGraph = pattern.apply(match, to: currentGraph)
                    changed = true

                    // Rebuild topological order after transformation
                    currentGraph.buildTopologicalOrder()

                    // Break to restart pattern matching from the beginning
                    break
                }
            }
        }

        return currentGraph
    }
}
