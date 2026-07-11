// Magma - LazyTensor
// x10-style lazy execution with graph tracing and compilation caching
//
// This module provides:
// - LazyTensorHandle: Reference to a node in the computation graph
// - IRNode: Operation representation in the graph
// - IRGraph: Full computation graph with topological ordering
// - LazyTensorBarrier: Triggers compilation and execution

import Foundation
import Synchronization
import StableHLO
import XLARuntime

// MARK: - Lazy Tensor Handle

/// Handle to a lazy tensor value in the computation graph
///
/// LazyTensorHandle represents a tensor that may or may not be materialized.
/// Operations on lazy tensors build up a computation graph that is
/// compiled and executed when `LazyTensorBarrier()` is called.
public final class LazyTensorHandle: @unchecked Sendable {

    /// Unique identifier for this handle
    public let id: UInt64

    /// Shape of the tensor
    public let shape: [Int]

    /// Data type of tensor elements
    public let dtype: DType

    /// Device where this tensor resides
    public let device: Device

    /// Incremental structural hash — computed automatically when irNode is set.
    /// Two structurally equivalent DAGs produce the same hash regardless of tensor IDs.
    /// Used for fast-path cache lookup at barrier time.
    public private(set) var structuralHash: UInt64 = 0

    /// The IR node that produces this tensor (nil if materialized)
    public var irNode: IRNode? {
        didSet {
            #if os(macOS) && canImport(MetalHLO)
            // The incremental structural hash is only consumed by the Metal
            // fast-path cache (computeFastPathHash). On CPU/GPU/TPU it is pure
            // overhead — recomputed on every op assignment and never read — so
            // skip it there.
            if let node = irNode {
                structuralHash = IRNode.computeIncrementalHash(node: node, shape: shape, dtype: dtype)
            }
            #endif
            // Invalidate persistent subgraph cache entry for this handle.
            // Required when irNode changes (e.g., after materialization to .metalData),
            // so stale subtrees are not replayed in subsequent barriers.
            invalidateSubgraphCache(for: self)
        }
    }

    /// Materialized buffer (nil if not yet computed)
    public var materializedBuffer: PJRTBuffer?

    /// Whether this tensor has been materialized
    public var isMaterialized: Bool {
        if materializedBuffer != nil { return true }
        #if os(macOS) && canImport(MetalHLO)
        if case .metalData = irNode { return true }
        #endif
        return false
    }

    /// Whether this tensor is "live" - explicitly requested by the user for materialization.
    /// Live tensors are outputs that must be computed. Non-live tensors are intermediates
    /// that may be garbage collected if no longer needed.
    /// Inspired by TensorFlow Swift's lazy tensor design.
    public var isLive: Bool = false

    /// Create a new lazy tensor handle
    public init(id: UInt64, shape: [Int], dtype: DType, device: Device) {
        self.id = id
        self.shape = shape
        self.dtype = dtype
        self.device = device
    }
}

// MARK: - IR Node

/// Represents an operation in the computation graph
public indirect enum IRNode: @unchecked Sendable {
    /// Data from a materialized PJRT buffer (stays on device)
    case data(PJRTBuffer)

    /// Data from a materialized Metal buffer (stays on GPU, no D2H copy)
    #if os(macOS) && canImport(MetalHLO)
    case metalData(MetalHLOBuffer)
    #endif

    /// Constant value
    case constant(values: [Float], shape: [Int])

    /// Operation with inputs
    case operation(op: OpKind, inputs: [LazyTensorHandle], attributes: [String: Any])

    /// While loop with traced body
    /// - iterations: Number of loop iterations
    /// - initialValues: Input handles for loop-carried state
    /// - bodyInputs: Placeholder handles used during body tracing (map to block args)
    /// - bodyOutputs: Output handles from body tracing (become loop outputs)
    /// - bodyNodes: All nodes created during body tracing (the body's computation graph)
    case whileLoopTraced(
        iterations: Int,
        initialValues: [LazyTensorHandle],
        bodyInputs: [LazyTensorHandle],
        bodyOutputs: [LazyTensorHandle],
        bodyNodes: [LazyTensorHandle]
    )


    // MARK: - Incremental Hashing

    /// Compute a structural hash for this node, incorporating input hashes recursively.
    /// This enables O(1) graph hash computation at barrier time.
    static func computeIncrementalHash(node: IRNode, shape: [Int], dtype: DType) -> UInt64 {
        var hasher = Hasher()
        hasher.combine(shape)
        hasher.combine(dtype)

        switch node {
        case .data:
            hasher.combine("data")

        #if os(macOS) && canImport(MetalHLO)
        case .metalData:
            hasher.combine("data")
        #endif

        case .constant(_, let constShape):
            // All constants are promoted (threshold = Int.max), so exclude values
            hasher.combine("promoted_const")
            hasher.combine(constShape)

        case .operation(let op, let inputs, let attributes):
            hasher.combine(op.rawValue)
            // Incorporate input hashes in order — this captures DAG structure
            for input in inputs {
                hasher.combine(input.structuralHash)
            }
            IRNode.hashAttributes(attributes, into: &hasher)

        case .whileLoopTraced(let iterations, let initialValues, _, let bodyOutputs, _):
            hasher.combine("while")
            hasher.combine(iterations)
            for input in initialValues {
                hasher.combine(input.structuralHash)
            }
            for output in bodyOutputs {
                hasher.combine(output.structuralHash)
            }
        }

        return UInt64(hasher.finalize().magnitude)
    }

    /// Hash operation attributes in a stable, sorted order.
    static func hashAttributes(_ attributes: [String: Any], into hasher: inout Hasher) {
        for (key, value) in attributes.sorted(by: { $0.key < $1.key }) {
            hasher.combine(key)
            switch value {
            case let v as Int: hasher.combine(v)
            case let v as Float: hasher.combine(v)
            case let v as Double: hasher.combine(v)
            case let v as Bool: hasher.combine(v)
            case let v as String: hasher.combine(v)
            case let v as [Int]: v.forEach { hasher.combine($0) }
            case let v as [Float]: v.forEach { hasher.combine($0) }
            case let v as [[Int]]: v.forEach { inner in inner.forEach { hasher.combine($0) } }
            default:
                hasher.combine(String(describing: value))
            }
        }
    }
}

// MARK: - Operation Kinds

/// All supported operation types
public enum OpKind: String, Sendable {
    // Elementwise binary
    case add, subtract, multiply, divide
    case maximum, minimum, power

    // Elementwise unary
    case negate, abs, exponential, log
    case sqrt, rsqrt, sine, cosine, tanh
    case floor, ceil

    // Type conversion
    case convert

    // Matrix operations
    case matmul, batchedMatmul, transpose, reshape, broadcast

    // Reductions
    case reduceSum, reduceMax, reduceMin, reduceMean

    // Activations
    case relu, sigmoid, softmax, gelu, leakyRelu, elu, silu

    // Clamp/Clip
    case clamp

    // Convolutions
    case conv1d, conv2d, convTranspose2d, maxPool2d, avgPool2d

    // Normalization
    case batchNorm, layerNorm

    // Comparison
    case equal, less, greater, select

    // Indexing/Slicing
    case slice, pad, gather, scatter, concatenate

    // Control Flow
    case whileLoop, cond

    // Random Number Generation
    case rngUniform, rngNormal

}

// MARK: - IR Graph

/// Complete computation graph ready for compilation
public final class IRGraph: @unchecked Sendable {

    /// Nodes in topological order
    public var nodes: [LazyTensorHandle] = []

    /// Output handles
    public var outputs: [LazyTensorHandle] = []

    /// Create an empty graph
    public init() {}

    /// Add an output to the graph
    public func addOutput(_ handle: LazyTensorHandle) {
        outputs.append(handle)
    }

    /// Emit StableHLO MLIR for this graph
    public func emitStableHLO(name: String) -> String {
        let emitter = StableHLOEmitter(graph: self)
        return emitter.emit(name: name)
    }
}

// MARK: - Execution Context
//
// Inspired by TensorFlow Swift's LazyTensorContext, this provides thread-local
// state for lazy tensor execution. This enables safe concurrent execution of
// independent tensor computations without cross-thread interference.

/// Thread-local execution context for lazy tensor operations
///
/// Each thread maintains its own context with:
/// - Local pending tensors (not shared across threads)
/// - Configuration flags for the current execution
/// - Statistics for this thread's executions
///
/// This design is inspired by TensorFlow Swift's LazyTensorContext pattern.
public final class ExecutionContext: @unchecked Sendable {

    /// Thread-local storage key
    private static let threadLocalKey = "Magma.ExecutionContext"

    /// Get the current thread's execution context
    public static var current: ExecutionContext {
        #if os(Linux)
        // On Linux, use a simple global for now (proper TLS requires more setup)
        return _globalContext
        #else
        if let context = Thread.current.threadDictionary[threadLocalKey] as? ExecutionContext {
            return context
        }
        let context = ExecutionContext()
        Thread.current.threadDictionary[threadLocalKey] = context
        return context
        #endif
    }

    /// Whether shape tracking is enabled (for debugging)
    public var isShapeTrackingEnabled: Bool = false

    /// Whether to promote constants to inputs (for cache optimization)
    public var shouldPromoteConstants: Bool = true

    /// Local tensor ID counter (thread-local to avoid contention)
    private var localNextId: UInt64 = 0

    /// Local pending tensors (per-thread, per-device)
    private var localPending: [String: [LazyTensorHandle]] = [:]
    private var localPendingSet: [String: Set<UInt64>] = [:]

    /// Statistics for this context
    public var executionCount: Int = 0
    public var totalNodesExecuted: Int = 0

    /// Private initializer - use ExecutionContext.current instead
    fileprivate init() {}

    /// Generate a locally-unique tensor ID
    /// Combined with thread ID for global uniqueness
    public func nextLocalId() -> UInt64 {
        let id = localNextId
        localNextId += 1
        return id
    }

    /// Mark a tensor for local materialization
    public func markForMaterialization(_ handle: LazyTensorHandle) {
        let key = handle.device.description
        if localPendingSet[key] == nil {
            localPendingSet[key] = Set()
            localPending[key] = []
        }
        if !localPendingSet[key]!.contains(handle.id) {
            localPendingSet[key]!.insert(handle.id)
            localPending[key]!.append(handle)
        }
    }

    /// Take pending tensors for a device (thread-local)
    public func takeLocalPending(for device: Device) -> [LazyTensorHandle] {
        let key = device.description
        let result = localPending[key] ?? []
        localPending[key] = []
        localPendingSet[key] = Set()
        return result
    }

    /// Clear all local state
    public func reset() {
        localPending.removeAll()
        localPendingSet.removeAll()
        executionCount = 0
        totalNodesExecuted = 0
    }
}

#if os(Linux)
/// Global context fallback for Linux (until proper TLS is implemented)
nonisolated(unsafe) private var _globalContext = ExecutionContext()
#endif

// MARK: - Tensor Registry

/// Global registry for tracking live lazy tensors
public final class TensorRegistry: @unchecked Sendable {

    /// Shared instance
    public static let shared = TensorRegistry()

    /// Next available tensor ID. Atomic so the per-op hot path doesn't take the
    /// registry lock (which also guards the pending dictionaries).
    private let nextIdCounter = Atomic<UInt64>(0)

    /// Pending (unmaterialized) tensors by device - these are tensors that need to be computed
    /// Only tensors explicitly marked for materialization are added here
    private var pending: [String: [LazyTensorHandle]] = [:]

    /// Tensors that have been marked for materialization (to avoid duplicates)
    private var pendingSet: [String: Set<UInt64>] = [:]

    /// Lock for thread safety
    private let lock = NSLock()

    private init() {}

    /// Generate a new unique tensor ID (lock-free).
    public func nextTensorId() -> UInt64 {
        nextIdCounter.wrappingAdd(1, ordering: .relaxed).oldValue
    }

    /// Register a pending tensor (for internal tracking - doesn't mark for output)
    public func registerPending(_ handle: LazyTensorHandle) {
        // Record to tracer if tracing is active (for while loop body capture)
        WhileLoopTracer.shared.recordNode(handle)

        // This is otherwise a no-op for intermediate tensors
        // Only tensors marked via markForMaterialization will become outputs
    }

    /// Mark a tensor for materialization (this tensor will be computed as an output)
    /// Call this when the user explicitly requests tensor values (e.g., scalars())
    public func markForMaterialization(_ handle: LazyTensorHandle) {
        lock.lock()
        defer { lock.unlock() }

        // Skip if already materialized
        if handle.isMaterialized {
            return
        }

        let key = handle.device.description
        if pendingSet[key] == nil {
            pendingSet[key] = Set()
            pending[key] = []
        }

        // Only add if not already pending
        if !pendingSet[key]!.contains(handle.id) {
            pendingSet[key]!.insert(handle.id)
            pending[key]!.append(handle)
        }
    }

    /// Get and clear pending tensors for a device
    public func takePending(for device: Device) -> [LazyTensorHandle] {
        lock.lock()
        defer { lock.unlock() }
        let key = device.description
        let result = pending[key] ?? []
        pending[key] = []
        pendingSet[key] = Set()
        return result
    }

    /// Current pending (unmaterialized) tensor count
    public var pendingCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return pending.values.reduce(0) { $0 + $1.count }
    }

    /// Clear all pending tensors
    public func clearAll() {
        lock.lock()
        defer { lock.unlock() }
        pending.removeAll()
        pendingSet.removeAll()
    }
}

// MARK: - Constant Promotion
//
// Inspired by TensorFlow Swift's LazyTensorTraceCache, constant promotion converts
// small scalar/vector constants into function inputs. This enables cache reuse
// across computations that differ only in constant values (e.g., learning rates,
// epsilon values, different batch constants).

/// Represents a constant that has been promoted to a function input
public struct PromotedConstant: Sendable {
    /// Original node ID in the graph
    public let originalNodeId: UInt64

    /// Shape of the constant
    public let shape: [Int]

    /// Data type
    public let dtype: DType

    /// The actual values (extracted from the graph)
    public let values: [Float]

    /// Index of this constant in the promoted inputs list
    public let inputIndex: Int
}

/// Result of constant promotion analysis
public struct ConstantPromotionResult: Sendable {
    /// Hash computed without constant values (structure-only)
    public let structuralHash: String

    /// Constants that were promoted to inputs
    public let promotedConstants: [PromotedConstant]

    /// Whether promotion was applied
    public var wasPromoted: Bool { !promotedConstants.isEmpty }
}

// MARK: - Compilation Cache

/// Cache for compiled executables with constant promotion support
///
/// The cache uses structural hashing (graph structure without constant values)
/// to enable cache reuse across computations that differ only in scalar constants.
/// This is particularly beneficial for training loops where hyperparameters
/// like learning rate or epsilon vary between iterations.
public final class CompilationCache: @unchecked Sendable {

    /// Shared cache instance
    public static let shared = CompilationCache()

    /// Cache entries keyed by structural hash
    private var cache: [String: PJRTExecutable] = [:]

    /// MLIR text cache keyed by structural hash.
    /// Stores the emitted MLIR text alongside compiled executables so that if the
    /// executable cache is cleared, re-compilation can skip re-emission entirely.
    private var mlirCache: [String: String] = [:]

    /// Lock for thread safety
    private let lock = NSLock()

    /// Maximum number of elements for a constant to be promotable
    /// All constants should be promoted to enable passing data as input buffers
    /// instead of embedding large arrays in MLIR text (which causes parsing issues)
    public static let promotionThreshold: Int = Int.max

    private init() {}

    /// Look up a cached executable
    public func get(hash: String) -> PJRTExecutable? {
        lock.lock()
        defer { lock.unlock() }
        return cache[hash]
    }

    /// Store an executable in the cache
    public func put(hash: String, executable: PJRTExecutable) {
        lock.lock()
        defer { lock.unlock() }
        cache[hash] = executable
    }

    /// Look up cached MLIR text
    public func getMlir(hash: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return mlirCache[hash]
    }

    /// Store MLIR text in the cache
    public func putMlir(hash: String, mlir: String) {
        lock.lock()
        defer { lock.unlock() }
        mlirCache[hash] = mlir
    }

    /// Clear all cached entries
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
        mlirCache.removeAll()
        hitCount = 0
        missCount = 0
        promotionHitCount = 0
        mlirHitCount = 0
    }

    /// Cache statistics
    public var hitCount: Int = 0
    public var missCount: Int = 0
    public var promotionHitCount: Int = 0  // Hits due to constant promotion
    public var mlirHitCount: Int = 0       // Hits on MLIR text cache (skipped re-emission)

    /// Cache hit rate
    public var hitRate: Double {
        let total = hitCount + missCount
        return total > 0 ? Double(hitCount) / Double(total) : 0
    }

    /// Percentage of hits that were due to constant promotion
    public var promotionBenefit: Double {
        guard hitCount > 0 else { return 0 }
        return Double(promotionHitCount) / Double(hitCount)
    }
}

// MARK: - Cached Graph Metadata

/// Metadata stored alongside a cached executable to enable fast-path
/// buffer assembly without re-analyzing the graph.
public struct CachedGraphMetadata: Sendable {
    /// Number of data (on-device buffer) inputs expected
    public let dataInputCount: Int

    /// Promoted constant slot descriptors, in input order
    public let promotedConstantSlots: [PromotedConstantSlot]

    /// Total number of outputs
    public let outputCount: Int
}

/// Describes a promoted constant input slot (shape/dtype only, no values).
public struct PromotedConstantSlot: Sendable {
    public let shape: [Int]
    public let dtype: DType
}

// MARK: - Metal Compilation Cache

#if os(macOS) && canImport(MetalHLO)

/// Cache entry combining executable and metadata for fast-path replay.
public struct MetalCacheEntry: Sendable {
    public let executable: MetalHLOExecutable
    public let metadata: CachedGraphMetadata
    /// Values for each promoted constant slot, in input order.
    /// Stored so the fast path can re-create constant buffers without re-traversing
    /// the (pre-optimization) graph — which would find different constants than the
    /// post-optimization graph the executable was compiled from.
    public let promotedConstantValues: [[Float]]
}

/// Cache for compiled Metal executables
public final class MetalCompilationCache: @unchecked Sendable {

    /// Shared cache instance
    public static let shared = MetalCompilationCache()

    /// Slow-path cache keyed by post-optimization structural hash
    private var cache: [String: MetalHLOExecutable] = [:]

    /// Fast-path cache keyed by incremental (pre-optimization) hash
    private var fastCache: [UInt64: MetalCacheEntry] = [:]

    /// MLIR text cache keyed by post-optimization structural hash.
    ///
    /// Stores the emitted MLIR alongside compiled executables so that if the
    /// executable cache is cleared (or future eviction is added), re-compilation
    /// can skip re-emission — the most expensive CPU-side step before compilation.
    /// Inspired by TensorFlow Swift's TFFunctionBuilder node-level caching strategy.
    private var mlirCache: [String: String] = [:]

    /// Lock for thread safety
    private let lock = NSLock()

    private init() {}

    /// Look up a cached executable (slow path, post-optimization hash)
    public func get(hash: String) -> MetalHLOExecutable? {
        lock.lock()
        defer { lock.unlock() }
        return cache[hash]
    }

    /// Store an executable in the cache (slow path)
    public func put(hash: String, executable: MetalHLOExecutable) {
        lock.lock()
        defer { lock.unlock() }
        cache[hash] = executable
    }

    /// Look up a cached entry by incremental hash (fast path)
    public func getFast(hash: UInt64) -> MetalCacheEntry? {
        lock.lock()
        defer { lock.unlock() }
        return fastCache[hash]
    }

    /// Store a cache entry by incremental hash (fast path)
    public func putFast(hash: UInt64, entry: MetalCacheEntry) {
        lock.lock()
        defer { lock.unlock() }
        fastCache[hash] = entry
    }

    /// Look up cached MLIR text (slow path, post-optimization hash)
    public func getMlir(hash: String) -> String? {
        lock.lock()
        defer { lock.unlock() }
        return mlirCache[hash]
    }

    /// Store MLIR text in the cache alongside the executable
    public func putMlir(hash: String, mlir: String) {
        lock.lock()
        defer { lock.unlock() }
        mlirCache[hash] = mlir
    }

    /// Clear all cached entries
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
        fastCache.removeAll()
        mlirCache.removeAll()
        hitCount = 0
        missCount = 0
        fastHitCount = 0
        mlirHitCount = 0
    }

    /// Cache statistics
    public var hitCount: Int = 0
    public var missCount: Int = 0
    public var fastHitCount: Int = 0
    public var mlirHitCount: Int = 0

    /// Cache hit rate
    public var hitRate: Double {
        let total = hitCount + missCount + fastHitCount
        return total > 0 ? Double(hitCount + fastHitCount) / Double(total) : 0
    }
}

#endif

// MARK: - Lazy Tensor Barrier

/// Global XLA client (lazily initialized)
/// Access is protected by _clientLock
nonisolated(unsafe) private var _globalClient: PJRTClient?
private let _clientLock = NSLock()

/// Resolve which backend to actually execute on.
///
/// Priority:
/// 1. `MAGMA_DEFAULT_BACKEND` env override (`cpu`/`gpu`/`tpu`/`metal`), when that
///    plugin is actually available.
/// 2. The requested backend, when its plugin is available.
/// 3. The best available backend (TPU > GPU > CPU) as a fallback, so execution
///    still works on machines where the requested plugin — most often the CPU
///    plugin — is not installed but another accelerator is present.
public func resolveExecutionBackend(requested: Backend) -> Backend {
    // Explicit override wins, but only if its plugin is present.
    if let raw = ProcessInfo.processInfo.environment["MAGMA_DEFAULT_BACKEND"],
       let override = Backend(rawValue: raw.lowercased()),
       override.isAvailable {
        return override
    }

    // Use the requested backend when its plugin is present.
    if requested.isAvailable {
        return requested
    }

    // Otherwise fall back to whatever is available so execution can proceed.
    // (bestAvailable returns .cpu when nothing else is found, preserving the
    // previous behavior of surfacing a plugin-load error to the caller.)
    return Backend.bestAvailable
}

/// Get or create the global XLA client
public func getGlobalClient(backend: Backend = .cpu) throws -> PJRTClient {
    _clientLock.lock()
    defer { _clientLock.unlock() }

    if let client = _globalClient {
        return client
    }

    let resolved = resolveExecutionBackend(requested: backend)
    let client = try PJRTClient.create(backend: resolved)
    _globalClient = client
    return client
}

// MARK: - Metal Client

#if os(macOS) && canImport(MetalHLO)

/// Global MetalHLO client (lazily initialized)
nonisolated(unsafe) private var _globalMetalClient: MetalHLOClient?
private let _metalClientLock = NSLock()

/// Get or create the global MetalHLO client
public func getMetalHLOClient() throws -> MetalHLOClient {
    _metalClientLock.lock()
    defer { _metalClientLock.unlock() }

    if let client = _globalMetalClient {
        return client
    }

    let client = try MetalHLOClient.create()
    _globalMetalClient = client
    return client
}

#endif

/// Trigger compilation and execution of all pending operations
///
/// When called, all lazy tensors on the specified device are:
/// 1. Collected into a graph
/// 2. Validated for shape/type errors
/// 3. Analyzed for constant promotion (cache optimization)
/// 4. Converted to StableHLO MLIR
/// 5. Compiled (or fetched from cache)
/// 6. Executed
/// 7. Results stored back in tensor handles
///
/// Example:
/// ```swift
/// let y = x.matmul(w).relu()  // Lazy - no computation yet
/// LazyTensorBarrier()          // Compile and execute everything
/// print(y.scalars())           // Now we can read the results
/// ```
public func LazyTensorBarrier(on device: Device = .default) {
    // Route to Metal-specific barrier if using Metal backend
    #if os(macOS) && canImport(MetalHLO)
    if device.backend == .metal {
        MetalLazyTensorBarrier(on: device)
        return
    }
    #endif

    // 1. Collect all pending tensors for this device
    let pending = TensorRegistry.shared.takePending(for: device)
    if pending.isEmpty {
        return // Nothing to execute
    }

    // Find all outputs (tensors that need to be materialized)
    let outputs = pending.filter { !$0.isMaterialized }
    if outputs.isEmpty {
        return
    }

    // Mark outputs as live (for future memory optimization)
    for output in outputs {
        output.isLive = true
    }

    // 2. Build IR graph
    let graph = IRGraph()
    for output in outputs {
        graph.addOutput(output)
    }
    graph.buildTopologicalOrder()

    // 3. Validate graph (catch errors early with better messages)
    do {
        try graph.validate()
    } catch {
        print("Magma: Graph validation failed: \(error)")
        return
    }

    // 3.5. Run optimization passes
    let optimizedGraph: IRGraph
    if ProcessInfo.processInfo.environment["MAGMA_NO_OPT"] != "1" {
        let passManager = PassManager.shared
        optimizedGraph = passManager.run(on: graph)

        if ProcessInfo.processInfo.environment["MAGMA_DEBUG"] == "1" {
            let originalCount = graph.nodes.count
            let optimizedCount = optimizedGraph.nodes.count
            if originalCount != optimizedCount {
                print("Magma: Optimization reduced \(originalCount) -> \(optimizedCount) nodes")
            }
        }
    } else {
        optimizedGraph = graph
    }

    // 4. Analyze for constant promotion (use optimized graph)
    let promotionResult = optimizedGraph.analyzeForConstantPromotion()
    let structuralHash = promotionResult.structuralHash

    // 5. Check compilation cache using structural hash
    let cache = CompilationCache.shared
    var executable: PJRTExecutable
    if let cached = cache.get(hash: structuralHash) {
        cache.hitCount += 1
        if promotionResult.wasPromoted {
            cache.promotionHitCount += 1
        }
        executable = cached
    } else {
        cache.missCount += 1

        // 6. Emit StableHLO MLIR — use cached MLIR text if available to skip re-emission
        let mlir: String
        if let cachedMlir = cache.getMlir(hash: structuralHash) {
            cache.mlirHitCount += 1
            mlir = cachedMlir
        } else {
            let emitter = StableHLOEmitter(graph: optimizedGraph)
            mlir = emitter.emit(
                name: "lazy_graph_\(structuralHash.prefix(8))",
                promotedConstants: promotionResult.promotedConstants
            )
            cache.putMlir(hash: structuralHash, mlir: mlir)
        }

        // 7. Compile
        do {
            let client = try getGlobalClient(backend: device.backend)
            executable = try client.compile(mlir)
            cache.put(hash: structuralHash, executable: executable)
        } catch {
            // Always write failing MLIR to debug file for first unique failure
            let debugPath = "/tmp/magma_debug_\(structuralHash.prefix(8)).mlir"
            // Check if we haven't already written this hash
            if !FileManager.default.fileExists(atPath: debugPath) {
                print("Magma: MLIR that failed to compile (hash=\(structuralHash.prefix(8))):")
                print(mlir.prefix(3000))
                print("... (truncated)")
                if let data = mlir.data(using: .utf8) {
                    let url = URL(fileURLWithPath: debugPath)
                    try? data.write(to: url)
                    print("Magma: Full MLIR written to \(debugPath)")
                }
            }
            print("Magma: Compilation failed: \(error)")
            return
        }
    }

    // 8. Collect input buffers (use optimized graph)
    // First: data nodes (pre-materialized large tensors)
    var inputBuffers: [PJRTBuffer] = []
    for node in optimizedGraph.nodes {
        if case .data(let buffer) = node.irNode {
            inputBuffers.append(buffer)
        }
    }

    // Second: promoted constants (need to create buffers for their values)
    if promotionResult.wasPromoted {
        do {
            let client = try getGlobalClient(backend: device.backend)
            for promoted in promotionResult.promotedConstants.sorted(by: { $0.inputIndex < $1.inputIndex }) {
                let buffer = try client.createBuffer(
                    promoted.values,
                    shape: promoted.shape,
                    elementType: .float32,
                    device: nil
                )
                inputBuffers.append(buffer)
            }
        } catch {
            print("Magma: Failed to create buffers for promoted constants: \(error)")
            return
        }
    }

    // 9. Execute
    do {
        let outputBuffers = try executable.execute(inputBuffers)

        // 10. Update tensor handles with results
        for (i, output) in outputs.enumerated() {
            if i < outputBuffers.count {
                let buffer = outputBuffers[i]
                output.materializedBuffer = buffer
                // Set irNode to .data so the tensor can be used as input in future computations
                // This replaces the computation graph with a direct reference to the materialized buffer
                output.irNode = .data(buffer)
            }
        }
    } catch {
        print("Magma: Execution failed: \(error)")
    }
}

// MARK: - Metal Lazy Tensor Barrier

#if os(macOS) && canImport(MetalHLO)
import MetalHLO

/// Compute a fast-path cache key from output tensors' incremental hashes.
/// This is O(n) in the number of outputs, not O(graph_size).
private func computeFastPathHash(outputs: [LazyTensorHandle]) -> UInt64 {
    var hasher = Hasher()
    hasher.combine(outputs.count)
    for output in outputs {
        hasher.combine(output.structuralHash)
        hasher.combine(output.shape)
        hasher.combine(output.dtype)
    }

    // Constant VALUES are intentionally excluded from `structuralHash` so the
    // compiled executable can be reused across differing constant values (the
    // slow-path executable cache is keyed by that hash). But on a fast-path
    // hit we reuse the constant values captured at first compile
    // (`entry.promotedConstantValues`). So the fast hash MUST distinguish
    // constant values — otherwise a graph with identical structure but changed
    // constants (e.g. a new input batch or scalar built as a constant) would
    // hit the cache and silently execute a prior iteration's constant data.
    //
    // Fold constant values in via a deterministic DFS. This walk is O(graph)
    // but far cheaper than the emit+compile it still skips; when constants
    // differ, the miss falls through to the slow path, which reuses the
    // executable (constant-excluded structural hash) with the current values.
    var visited = Set<UInt64>()
    func foldConstants(_ handle: LazyTensorHandle) {
        guard !visited.contains(handle.id) else { return }
        visited.insert(handle.id)
        guard let node = handle.irNode else { return }
        switch node {
        case .constant(let values, let constShape):
            hasher.combine("const")
            hasher.combine(constShape)
            hasher.combine(values)
        case .operation(_, let inputs, _):
            for input in inputs { foldConstants(input) }
        case .whileLoopTraced(_, let initialValues, _, _, _):
            for input in initialValues { foldConstants(input) }
        case .data, .metalData:
            break
        }
    }
    for output in outputs { foldConstants(output) }

    return UInt64(hasher.finalize().magnitude)
}

/// Lightweight input collection from output tensors for the fast path.
/// Walks the DAG from outputs collecting leaf nodes (data/metalData and constants)
/// in topological order, without building a full IRGraph or running optimization.
///
/// Returns (dataBuffers, constants) in the order the executable expects.
/// Lightweight DAG walk that collects only data inputs (`.metalData` / `.data` nodes).
/// Constants are NOT collected here — the fast path uses cached constant values from
/// the `MetalCacheEntry` instead of re-traversing the pre-optimization graph.
private func collectDataInputsFastPath(
    outputs: [LazyTensorHandle]
) -> [LazyTensorHandle] {
    var dataInputs: [LazyTensorHandle] = []
    var visited = Set<UInt64>()

    func visit(_ handle: LazyTensorHandle) {
        guard !visited.contains(handle.id) else { return }
        visited.insert(handle.id)

        guard let node = handle.irNode else { return }
        switch node {
        case .metalData, .data:
            dataInputs.append(handle)

        case .constant:
            break  // skip — constants come from cache entry

        case .operation(_, let inputs, _):
            for input in inputs {
                visit(input)
            }

        case .whileLoopTraced(_, let initialValues, _, _, _):
            for input in initialValues {
                visit(input)
            }
        }
    }

    for output in outputs {
        visit(output)
    }

    return dataInputs
}


/// Metal-specific lazy tensor barrier
///
/// Uses a two-tier cache: the fast path checks the incremental (pre-optimization) hash
/// and skips graph building, optimization, and MLIR emission entirely on cache hit.
/// The slow path falls through to the full pipeline on cache miss.

private func MetalLazyTensorBarrier(on device: Device) {
    // 1. Collect all pending tensors for this device
    let pending = TensorRegistry.shared.takePending(for: device)
    if pending.isEmpty {
        return
    }

    let outputs = pending.filter { !$0.isMaterialized }
    if outputs.isEmpty {
        return
    }

    for output in outputs {
        output.isLive = true
    }

    let debugEnabled = ProcessInfo.processInfo.environment["MAGMA_DEBUG"] == "1"
    let cache = MetalCompilationCache.shared

    // 2. Fast-path: check incremental hash before building any graph
    let fastHash = computeFastPathHash(outputs: outputs)

    if let entry = cache.getFast(hash: fastHash) {
        // Collect data inputs only (no constant traversal — constants come from cache entry)
        let dataInputHandles = collectDataInputsFastPath(outputs: outputs)

        // Sanity check: data input count must match
        if dataInputHandles.count == entry.metadata.dataInputCount {
            cache.fastHitCount += 1

            if debugEnabled {
                print("Magma [Metal]: Fast-path cache hit (hash=\(fastHash))")
            }

            // Assemble input buffers: data inputs first, then cached constant values
            var inputBuffers: [MetalHLOBuffer] = []

            // Data inputs
            var fastPathFailed = false
            for handle in dataInputHandles {
                if case .metalData(let metalBuffer) = handle.irNode {
                    inputBuffers.append(metalBuffer)
                } else if case .data(let pjrtBuffer) = handle.irNode {
                    do {
                        let client = try getMetalHLOClient()
                        let hostData = try pjrtBuffer.toFloatArray()
                        let metalBuffer = client.createBuffer(hostData, shape: handle.shape)
                        inputBuffers.append(metalBuffer)
                    } catch {
                        print("Magma [Metal]: Fast-path PJRT conversion failed: \(error)")
                        fastPathFailed = true
                        break
                    }
                }
            }

            if !fastPathFailed {
                // Promoted constant inputs — use cached values (avoids re-traversing pre-opt graph)
                do {
                    let client = try getMetalHLOClient()
                    for (i, values) in entry.promotedConstantValues.enumerated() {
                        let slot = entry.metadata.promotedConstantSlots[i]
                        let buffer = client.createBuffer(values, shape: slot.shape)
                        inputBuffers.append(buffer)
                    }
                } catch {
                    print("Magma [Metal]: Fast-path constant buffer creation failed: \(error)")
                    fastPathFailed = true
                }
            }

            if !fastPathFailed {
                // Execute and update handles
                do {
                    let outputBuffers = try entry.executable.execute(inputBuffers)
                    for (i, output) in outputs.enumerated() {
                        if i < outputBuffers.count {
                            output.materializedBuffer = nil
                            output.irNode = .metalData(outputBuffers[i])
                        }
                    }
                    return  // Fast path succeeded
                } catch {
                    print("Magma [Metal]: Fast-path execution failed: \(error)")
                    return
                }
            }
        }
        // Fall through to slow path if data count mismatches or execution failed
    }

    // ── Slow path: full graph build → optimize → emit → compile ──

    // 3. Build IR graph
    let graph = IRGraph()
    for output in outputs {
        graph.addOutput(output)
    }
    graph.buildTopologicalOrder()

    // 4. Validate graph
    do {
        try graph.validate()
    } catch {
        print("Magma [Metal]: Graph validation failed: \(error)")
        return
    }

    // 5. Run optimization passes
    let optimizedGraph: IRGraph
    if ProcessInfo.processInfo.environment["MAGMA_NO_OPT"] != "1" {
        let passManager = PassManager.shared
        optimizedGraph = passManager.run(on: graph)

        if debugEnabled {
            let originalCount = graph.nodes.count
            let optimizedCount = optimizedGraph.nodes.count
            if originalCount != optimizedCount {
                print("Magma [Metal]: Optimization reduced \(originalCount) -> \(optimizedCount) nodes")
            }
        }
    } else {
        optimizedGraph = graph
    }

    // 6. Analyze for constant promotion
    let promotionResult = optimizedGraph.analyzeForConstantPromotion()
    let structuralHash = promotionResult.structuralHash

    // 7. Check slow-path cache
    var executable: MetalHLOExecutable
    if let cached = cache.get(hash: structuralHash) {
        cache.hitCount += 1
        executable = cached
    } else {
        cache.missCount += 1

        // 8. Emit StableHLO MLIR — use cached MLIR text if available to skip re-emission.
        // The MLIR cache stores emitted text keyed by structural hash so that if the
        // executable cache is cleared, re-compilation skips the StableHLOEmitter entirely.
        let mlir: String
        if let cachedMlir = cache.getMlir(hash: structuralHash) {
            cache.mlirHitCount += 1
            mlir = cachedMlir
            if debugEnabled {
                print("Magma [Metal]: MLIR cache hit (hash=\(structuralHash.prefix(8))), skipping emission")
            }
        } else {
            let emitter = StableHLOEmitter(graph: optimizedGraph)
            mlir = emitter.emit(
                name: "metal_graph_\(structuralHash.prefix(8))",
                promotedConstants: promotionResult.promotedConstants
            )
            cache.putMlir(hash: structuralHash, mlir: mlir)
        }

        if debugEnabled {
            print("Magma [Metal]: Generated MLIR:")
            print(mlir)
        }

        // 9. Compile
        do {
            let client = try getMetalHLOClient()
            executable = try client.compile(mlir)

            if debugEnabled {
                print("Magma [Metal]: Compiled executable - inputs: \(executable.inputCount), outputs: \(executable.outputCount)")
            }

            cache.put(hash: structuralHash, executable: executable)
        } catch {
            let debugPath = "/tmp/magma_metal_debug_\(structuralHash.prefix(8)).mlir"
            if !FileManager.default.fileExists(atPath: debugPath) {
                print("Magma [Metal]: MLIR that failed to compile (hash=\(structuralHash.prefix(8))):")
                print(mlir.prefix(3000))
                if let data = mlir.data(using: .utf8) {
                    try? data.write(to: URL(fileURLWithPath: debugPath))
                }
            }
            print("Magma [Metal]: Compilation failed: \(error)")
            return
        }
    }

    // Store in fast cache for future iterations
    let dataInputCount = optimizedGraph.nodes.filter { node in
        if case .data = node.irNode { return true }
        if case .metalData = node.irNode { return true }
        return false
    }.count
    let sortedPromoted = promotionResult.promotedConstants.sorted(by: { $0.inputIndex < $1.inputIndex })
    let metadata = CachedGraphMetadata(
        dataInputCount: dataInputCount,
        promotedConstantSlots: sortedPromoted.map { PromotedConstantSlot(shape: $0.shape, dtype: $0.dtype) },
        outputCount: outputs.count
    )
    let promotedConstantValues = sortedPromoted.map { $0.values }
    cache.putFast(hash: fastHash, entry: MetalCacheEntry(
        executable: executable,
        metadata: metadata,
        promotedConstantValues: promotedConstantValues
    ))

    // 10. Collect input buffers
    var inputBuffers: [MetalHLOBuffer] = []

    for node in optimizedGraph.nodes {
        if case .metalData(let metalBuffer) = node.irNode {
            inputBuffers.append(metalBuffer)
        } else if case .data(let pjrtBuffer) = node.irNode {
            do {
                let client = try getMetalHLOClient()
                let hostData = try pjrtBuffer.toFloatArray()
                let metalBuffer = client.createBuffer(hostData, shape: node.shape)
                inputBuffers.append(metalBuffer)
            } catch {
                print("Magma [Metal]: Failed to convert PJRT buffer to Metal: \(error)")
                return
            }
        }
    }

    if promotionResult.wasPromoted {
        do {
            let client = try getMetalHLOClient()
            for promoted in promotionResult.promotedConstants.sorted(by: { $0.inputIndex < $1.inputIndex }) {
                let buffer = client.createBuffer(promoted.values, shape: promoted.shape)
                inputBuffers.append(buffer)
            }
        } catch {
            print("Magma [Metal]: Failed to create constant buffers: \(error)")
            return
        }
    }

    // 11. Execute
    do {
        let outputBuffers = try executable.execute(inputBuffers)
        for (i, output) in outputs.enumerated() {
            if i < outputBuffers.count {
                output.materializedBuffer = nil
                output.irNode = .metalData(outputBuffers[i])

                if debugEnabled {
                    let hostData = try outputBuffers[i].toFloatArray()
                    print("Magma [Metal]: Output \(i) - shape: \(output.shape), first few: \(Array(hostData.prefix(5)))")
                }
            }
        }
    } catch {
        print("Magma [Metal]: Execution failed: \(error)")
    }
}

#endif // os(macOS) && canImport(MetalHLO)

/// Execute a graph synchronously and return output buffers
///
/// This is a lower-level API that executes a pre-built graph.
/// For most use cases, prefer LazyTensorBarrier().
public func executeGraph(_ graph: IRGraph, on device: Device = .default) throws -> [PJRTBuffer] {
    graph.buildTopologicalOrder()

    // Run optimization passes
    let optimizedGraph: IRGraph
    if ProcessInfo.processInfo.environment["MAGMA_NO_OPT"] != "1" {
        let passManager = PassManager.shared
        optimizedGraph = passManager.run(on: graph)
    } else {
        optimizedGraph = graph
    }

    let emitter = StableHLOEmitter(graph: optimizedGraph)
    let graphHash = optimizedGraph.computeHash()
    let mlir = emitter.emit(name: "graph_\(graphHash.prefix(8))")

    // Check cache
    let cache = CompilationCache.shared
    var executable: PJRTExecutable

    if let cached = cache.get(hash: graphHash) {
        cache.hitCount += 1
        executable = cached
    } else {
        cache.missCount += 1
        let client = try getGlobalClient(backend: device.backend)
        executable = try client.compile(mlir)
        cache.put(hash: graphHash, executable: executable)
    }

    // Collect input buffers
    var inputBuffers: [PJRTBuffer] = []
    for node in graph.nodes {
        if case .data(let buffer) = node.irNode {
            inputBuffers.append(buffer)
        }
    }

    return try executable.execute(inputBuffers)
}

// MARK: - While Loop Tracing

/// Context for tracing a function body for while loop emission
///
/// During tracing, we create placeholder inputs that represent loop-carried variables.
/// All operations performed using these placeholders are recorded, and the final
/// outputs are captured. This traced graph can then be emitted as a stablehlo.while.
public final class WhileLoopTracer: @unchecked Sendable {

    /// Whether tracing is currently active
    public private(set) var isTracing: Bool = false

    /// Placeholder inputs created for this trace (represent block arguments)
    public private(set) var bodyInputs: [LazyTensorHandle] = []

    /// Nodes created during tracing (the body computation graph)
    public private(set) var tracedNodes: [LazyTensorHandle] = []

    /// Set of traced node IDs for O(1) lookup
    private var tracedNodeIds: Set<UInt64> = []

    /// Starting tensor ID for this trace (to identify traced nodes)
    private var startingId: UInt64 = 0

    /// Thread-local tracer key
    private static let threadLocalKey = "WhileLoopTracer_ThreadLocal"

    /// Thread-local tracer instance (each thread gets its own tracer)
    public static var shared: WhileLoopTracer {
        if let existing = Thread.current.threadDictionary[threadLocalKey] as? WhileLoopTracer {
            return existing
        }
        let tracer = WhileLoopTracer()
        Thread.current.threadDictionary[threadLocalKey] = tracer
        return tracer
    }

    fileprivate init() {}

    /// Begin tracing a while loop body
    ///
    /// Creates placeholder inputs for each initial value. Operations performed
    /// on these placeholders will be recorded as the body graph.
    ///
    /// - Parameter initialShapes: Shapes and dtypes of loop-carried variables
    /// - Returns: Placeholder handles to use as body inputs
    public func beginTrace(initialShapes: [(shape: [Int], dtype: DType)]) -> [LazyTensorHandle] {
        precondition(!isTracing, "Cannot nest while loop traces")
        isTracing = true
        bodyInputs = []
        tracedNodes = []
        tracedNodeIds = []
        startingId = TensorRegistry.shared.nextTensorId() + 1000000  // Use high IDs for placeholders

        // Create placeholder inputs (these represent block arguments in MLIR)
        for (i, spec) in initialShapes.enumerated() {
            let id = startingId + UInt64(i)
            let handle = LazyTensorHandle(id: id, shape: spec.shape, dtype: spec.dtype, device: .default)
            // Mark as a "placeholder" - no irNode means it's an external input
            handle.irNode = nil
            bodyInputs.append(handle)
        }

        return bodyInputs
    }

    /// Record a node created during tracing
    ///
    /// Call this from operation creation to track nodes belonging to the trace.
    public func recordNode(_ handle: LazyTensorHandle) {
        guard isTracing else { return }
        if !tracedNodeIds.contains(handle.id) {
            tracedNodeIds.insert(handle.id)
            tracedNodes.append(handle)
        }
    }

    /// Check if a handle is a placeholder input for current trace
    public func isPlaceholder(_ handle: LazyTensorHandle) -> Bool {
        guard isTracing else { return false }
        return bodyInputs.contains { $0.id == handle.id }
    }

    /// End tracing and return the captured graph
    ///
    /// - Parameter bodyOutputs: The output handles from the body function
    /// - Returns: Tuple of (bodyInputs, bodyOutputs, tracedNodes)
    public func endTrace(bodyOutputs: [LazyTensorHandle]) -> (
        inputs: [LazyTensorHandle],
        outputs: [LazyTensorHandle],
        nodes: [LazyTensorHandle]
    ) {
        precondition(isTracing, "Not currently tracing")
        isTracing = false

        let result = (inputs: bodyInputs, outputs: bodyOutputs, nodes: tracedNodes)

        // Clear state
        bodyInputs = []
        tracedNodes = []
        tracedNodeIds = []

        return result
    }

    /// Abort tracing (for error recovery)
    public func abortTrace() {
        isTracing = false
        bodyInputs = []
        tracedNodes = []
        tracedNodeIds = []
    }
}

// MARK: - Metrics

/// Print compilation and execution metrics
public func PrintMetrics() {
    let cache = CompilationCache.shared
    let hitRatePercent = Int(cache.hitRate * 100)
    let promotionBenefitPercent = Int(cache.promotionBenefit * 100)
    print("=== Magma Metrics ===")
    print("Cache hits: \(cache.hitCount)")
    print("Cache misses: \(cache.missCount)")
    print("Hit rate: \(hitRatePercent)%")
    print("Promotion hits: \(cache.promotionHitCount) (\(promotionBenefitPercent)% of hits from constant promotion)")
    print("Pending tensors: \(TensorRegistry.shared.pendingCount)")
}

// MARK: - NSLock (for cross-platform)

#if os(Linux)
import Glibc

final class NSLock: @unchecked Sendable {
    private var mutex = pthread_mutex_t()

    init() {
        pthread_mutex_init(&mutex, nil)
    }

    deinit {
        pthread_mutex_destroy(&mutex)
    }

    func lock() {
        pthread_mutex_lock(&mutex)
    }

    func unlock() {
        pthread_mutex_unlock(&mutex)
    }
}
#else
import Foundation
#endif
