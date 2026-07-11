// Magma - PassManager
// Optimization pass infrastructure for transforming computation graphs
//
// Inspired by MLIR's pass infrastructure and the optimization specification
// for achieving MLX-competitive performance.

import Foundation

// MARK: - Optimization Pass Protocol

/// Protocol for optimization passes that transform computation graphs
///
/// Passes operate on IRGraph and return a (potentially) transformed graph.
/// Each pass should be idempotent - running it multiple times should have
/// the same effect as running it once.
public protocol OptimizationPass {
    /// Unique name for this pass
    var name: String { get }

    /// Names of passes that must run before this one
    var dependencies: [String] { get }

    /// Whether this pass is enabled by default
    var enabledByDefault: Bool { get }

    /// Run the pass on a graph, returning the transformed graph
    func run(on graph: IRGraph) -> IRGraph
}

// MARK: - Default Implementation

extension OptimizationPass {
    public var dependencies: [String] { [] }
    public var enabledByDefault: Bool { true }
}

// MARK: - Pass Metrics

/// Records timing and statistics for optimization passes
public final class PassMetrics: @unchecked Sendable {
    /// Shared metrics instance
    public static let shared = PassMetrics()

    private let lock = NSLock()

    /// Timing data per pass
    private var timings: [String: [TimeInterval]] = [:]

    /// Transformation counts per pass
    private var transformCounts: [String: Int] = [:]

    private init() {}

    /// Record a pass execution
    public func record(pass: String, time: TimeInterval, transformations: Int = 0) {
        lock.lock()
        defer { lock.unlock() }

        if timings[pass] == nil {
            timings[pass] = []
        }
        timings[pass]!.append(time)

        if transformations > 0 {
            transformCounts[pass, default: 0] += transformations
        }
    }

    /// Get total time spent in a pass
    public func totalTime(for pass: String) -> TimeInterval {
        lock.lock()
        defer { lock.unlock() }
        return timings[pass]?.reduce(0, +) ?? 0
    }

    /// Get average time per invocation
    public func averageTime(for pass: String) -> TimeInterval {
        lock.lock()
        defer { lock.unlock() }
        guard let times = timings[pass], !times.isEmpty else { return 0 }
        return times.reduce(0, +) / Double(times.count)
    }

    /// Get invocation count
    public func invocationCount(for pass: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return timings[pass]?.count ?? 0
    }

    /// Get transformation count
    public func transformationCount(for pass: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return transformCounts[pass] ?? 0
    }

    /// Reset all metrics
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        timings.removeAll()
        transformCounts.removeAll()
    }

    /// Print a summary of all pass metrics
    public func printSummary() {
        lock.lock()
        let sortedPasses = timings.keys.sorted { totalTime(for: $0) > totalTime(for: $1) }
        lock.unlock()

        print("=== Optimization Pass Metrics ===")
        print(String(format: "%-30s %10s %10s %10s %10s", "Pass", "Total(ms)", "Avg(ms)", "Count", "Transforms"))

        for pass in sortedPasses {
            let total = totalTime(for: pass) * 1000
            let avg = averageTime(for: pass) * 1000
            let count = invocationCount(for: pass)
            let transforms = transformationCount(for: pass)
            print(String(format: "%-30s %10.2f %10.2f %10d %10d", pass, total, avg, count, transforms))
        }
    }
}

// MARK: - Pass Manager

/// Manages and executes optimization passes on computation graphs
///
/// The PassManager handles:
/// - Pass registration and dependency ordering
/// - Enabling/disabling passes
/// - Running passes in the correct order
/// - Collecting metrics
public final class PassManager: @unchecked Sendable {

    /// Shared instance with default passes registered
    public static let shared = PassManager()

    private var passes: [OptimizationPass] = []
    private var passMap: [String: OptimizationPass] = [:]
    private var enabled: Set<String> = []
    private let lock = NSLock()

    /// Whether to collect detailed metrics
    public var collectMetrics: Bool = true

    /// Whether to print debug info during optimization
    public var verbose: Bool = false

    /// Create a new pass manager
    public init() {
        registerDefaultPasses()
    }

    /// Register default optimization passes
    private func registerDefaultPasses() {
        // Phase 1: Basic optimizations
        register(DeadCodeEliminationPass())
        register(CommonSubexpressionEliminationPass())
        register(ConstantFoldingPass())
        register(AlgebraicSimplificationPass())

        // Phase 2: Fusion
        register(OperationFusionPass())

        // Phase 3: Layout optimization (to be added)
        // register(LayoutOptimizationPass())
    }

    /// Register a pass
    public func register(_ pass: OptimizationPass) {
        lock.lock()
        defer { lock.unlock() }

        passes.append(pass)
        passMap[pass.name] = pass

        if pass.enabledByDefault {
            enabled.insert(pass.name)
        }
    }

    /// Enable a pass by name
    public func enable(_ passName: String) {
        lock.lock()
        defer { lock.unlock() }
        enabled.insert(passName)
    }

    /// Disable a pass by name
    public func disable(_ passName: String) {
        lock.lock()
        defer { lock.unlock() }
        enabled.remove(passName)
    }

    /// Check if a pass is enabled
    public func isEnabled(_ passName: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return enabled.contains(passName)
    }

    /// Get sorted passes respecting dependencies
    private func sortedPasses() -> [OptimizationPass] {
        // Simple topological sort based on dependencies
        var result: [OptimizationPass] = []
        var added = Set<String>()

        func add(_ pass: OptimizationPass) {
            if added.contains(pass.name) { return }

            // Add dependencies first
            for depName in pass.dependencies {
                if let dep = passMap[depName] {
                    add(dep)
                }
            }

            added.insert(pass.name)
            result.append(pass)
        }

        for pass in passes {
            add(pass)
        }

        return result
    }

    /// Run all enabled passes on a graph
    ///
    /// - Parameter graph: The graph to optimize
    /// - Returns: The optimized graph
    public func run(on graph: IRGraph) -> IRGraph {
        lock.lock()
        let sortedPassList = sortedPasses()
        let enabledSet = enabled
        let shouldCollectMetrics = collectMetrics
        let isVerbose = verbose
        lock.unlock()

        var result = graph

        for pass in sortedPassList {
            guard enabledSet.contains(pass.name) else { continue }

            // Date().timeIntervalSinceReferenceDate matches CFAbsoluteTimeGetCurrent
            // (seconds since the 2001 reference date) and is available on Linux.
            let startTime = Date().timeIntervalSinceReferenceDate
            let nodesBefore = result.nodes.count

            result = pass.run(on: result)

            let elapsed = Date().timeIntervalSinceReferenceDate - startTime
            let nodesAfter = result.nodes.count
            let transformations = max(0, nodesBefore - nodesAfter)

            if shouldCollectMetrics {
                PassMetrics.shared.record(pass: pass.name, time: elapsed, transformations: transformations)
            }

            if isVerbose {
                print("[\(pass.name)] \(nodesBefore) -> \(nodesAfter) nodes, \(String(format: "%.2fms", elapsed * 1000))")
            }
        }

        return result
    }

    /// Run passes until no more transformations occur (fixed-point iteration)
    ///
    /// - Parameters:
    ///   - graph: The graph to optimize
    ///   - maxIterations: Maximum number of iterations
    /// - Returns: The optimized graph
    public func runToFixedPoint(on graph: IRGraph, maxIterations: Int = 10) -> IRGraph {
        var result = graph
        var previousHash = result.computeHash()

        for iteration in 0..<maxIterations {
            result = run(on: result)

            let currentHash = result.computeHash()
            if currentHash == previousHash {
                if verbose {
                    print("Fixed point reached after \(iteration + 1) iterations")
                }
                break
            }
            previousHash = currentHash
        }

        return result
    }

    /// Get list of all registered passes
    public var registeredPasses: [String] {
        lock.lock()
        defer { lock.unlock() }
        return passes.map { $0.name }
    }

    /// Get list of enabled passes
    public var enabledPasses: [String] {
        lock.lock()
        defer { lock.unlock() }
        return passes.filter { enabled.contains($0.name) }.map { $0.name }
    }
}

// MARK: - Optimization Level

/// Predefined optimization levels
public enum OptimizationLevel {
    /// No optimizations
    case O0

    /// Basic optimizations (DCE, CSE, constant folding)
    case O1

    /// Standard optimizations (O1 + fusion patterns)
    case O2

    /// Aggressive optimizations (O2 + layout optimization)
    case O3

    /// Configure pass manager for this optimization level
    public func configure(_ manager: PassManager) {
        // First disable all
        for pass in manager.registeredPasses {
            manager.disable(pass)
        }

        switch self {
        case .O0:
            // No passes enabled
            break

        case .O1:
            // Basic passes
            manager.enable("dce")
            manager.enable("cse")
            manager.enable("constant-folding")
            manager.enable("algebraic-simplification")

        case .O2:
            // O1 + fusion
            OptimizationLevel.O1.configure(manager)
            manager.enable("operation-fusion")

        case .O3:
            // O2 + layout
            OptimizationLevel.O2.configure(manager)
            manager.enable("layout-optimization")
        }
    }
}
