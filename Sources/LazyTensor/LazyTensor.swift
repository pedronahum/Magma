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
            // Invalidate the persistent subgraph cache entry for this handle when
            // its node *changes* (e.g. after materialization to .metalData), so
            // stale subtrees are not replayed in subsequent barriers. The common
            // case — a freshly built op assigning irNode for the first time
            // (oldValue == nil) — cannot have a cache entry yet (the handle was
            // just allocated), so skip the lookup entirely. This keeps the per-op
            // hot path off the cache lock and out of the dictionary.
            if oldValue != nil {
                invalidateSubgraphCache(for: self)
            }
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

    /// Optional Shardy sharding for this tensor. When set (and the graph declares
    /// a `mesh`), the emitter attaches it: as a `{sdy.sharding=…}` attribute on a
    /// data-input argument, or as a `sdy.sharding_constraint` on an intermediate
    /// value. Referenced axes must exist in the graph's mesh.
    public var sharding: TensorSharding?

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

    // Collectives (cross-replica). Attributes: "replicaGroups" ([[Int]]) and,
    // for allReduce, "reduction" (String, e.g. "add"/"maximum").
    case allReduce, allReduceMean

}

// MARK: - IR Graph

/// Complete computation graph ready for compilation
public final class IRGraph: @unchecked Sendable {

    /// Nodes in topological order
    public var nodes: [LazyTensorHandle] = []

    /// Output handles
    public var outputs: [LazyTensorHandle] = []

    /// Optional Shardy device mesh for this graph. When set, it is emitted as an
    /// `sdy.mesh` in the module header and is the mesh that node shardings refer
    /// to. Required for the emitter to attach any node `sharding`.
    public var mesh: DeviceMesh?

    /// Create an empty graph
    public init() {}

    /// Add an output to the graph
    public func addOutput(_ handle: LazyTensorHandle) {
        outputs.append(handle)
    }

    /// Validate every node sharding against the graph's mesh (name, rank, axes).
    /// Throws the first `ShardingError` found; no-op when there is no mesh or no
    /// shardings.
    public func validateShardings() throws {
        guard let mesh else {
            // If any node is sharded, a mesh is required.
            for node in nodes where node.sharding != nil {
                throw ShardingError.meshNameMismatch(expected: "<graph.mesh not set>",
                                                     got: node.sharding!.meshName)
            }
            return
        }
        for node in nodes {
            if let sharding = node.sharding {
                try sharding.validate(against: mesh, rank: node.shape.count)
            }
        }
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
        // Record to the tracer only while a while-loop body is being traced.
        // Gate on a global atomic count first (a relaxed load) so the common
        // no-tracing path never touches thread-local storage — `shared` otherwise
        // does a Thread.threadDictionary lookup + cast on every single op.
        if _globalActiveTraceCount.load(ordering: .relaxed) > 0 {
            WhileLoopTracer.shared.recordNode(handle)
        }

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

    /// Fast-path (trace) cache keyed by a hash of the *raw* graph structure plus
    /// constant values. A hit lets a repeated barrier skip graph building,
    /// optimization, analysis, emission, and compilation entirely — the single
    /// biggest per-barrier cost (optimization alone is ~40-50%). Mirrors the
    /// Metal backend's fast path.
    private var fastCache: [UInt64: PJRTTraceCacheEntry] = [:]
    private let fastCacheMaxEntries = 4096

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

    /// Look up a fast-path (trace) entry.
    public func getFast(hash: UInt64) -> PJRTTraceCacheEntry? {
        lock.lock()
        defer { lock.unlock() }
        return fastCache[hash]
    }

    /// Store a fast-path (trace) entry (size-capped to bound memory).
    public func putFast(hash: UInt64, entry: PJRTTraceCacheEntry) {
        lock.lock()
        defer { lock.unlock() }
        if fastCache[hash] == nil && fastCache.count >= fastCacheMaxEntries { return }
        fastCache[hash] = entry
    }

    /// Clear all cached entries
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        cache.removeAll()
        mlirCache.removeAll()
        fastCache.removeAll()
        hitCount = 0
        missCount = 0
        promotionHitCount = 0
        mlirHitCount = 0
        fastHitCount = 0
    }

    /// Cache statistics
    public var hitCount: Int = 0
    public var missCount: Int = 0
    public var promotionHitCount: Int = 0  // Hits due to constant promotion
    public var mlirHitCount: Int = 0       // Hits on MLIR text cache (skipped re-emission)
    public var fastHitCount: Int = 0       // Fast-path (trace) hits (skipped the whole pipeline)

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

/// Fast-path (trace) cache entry for the PJRT backend: everything needed to
/// re-execute a repeated graph structure without rebuilding/optimizing it.
public struct PJRTTraceCacheEntry: @unchecked Sendable {
    public let executable: PJRTExecutable
    /// Number of `.data` inputs the executable expects (a sanity check on a hit).
    public let dataInputCount: Int
    /// Device buffers for the promoted constants, in input order. Created once on
    /// the compiling miss and reused on every hit: the fast key folds in constant
    /// values, so a hit guarantees they still match — no host→device re-upload.
    /// PJRT does not donate/consume input buffers (data inputs are likewise reused
    /// across barriers), so sharing these is safe.
    public let promotedConstantBuffers: [PJRTBuffer]
    /// The optimized-graph structural hash, kept for the self-verify mode.
    public let structuralHash: String
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

/// Global XLA clients, cached per *resolved* backend (lazily initialized).
///
/// Keyed by the resolved backend (see `resolveExecutionBackend`) rather than a
/// single slot, so two requests that resolve to the same physical backend share
/// one client, while a request for a genuinely different backend gets its own
/// client instead of silently receiving the first-created one.
///
/// Note: PJRT plugins register process-global state and the C shim loads a
/// single plugin per process (see `PJRT_LoadPlugin`), so in practice only one
/// accelerator backend can be live per process today. Requesting a second,
/// different backend therefore surfaces the plugin-mismatch error loudly rather
/// than silently running on the wrong device — the per-backend cache is what
/// makes that failure explicit and lets true multi-plugin support drop in later.
///
/// Access is protected by _clientLock
nonisolated(unsafe) private var _globalClients: [Backend: PJRTClient] = [:]
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

/// Get or create the global XLA client for a backend.
///
/// The requested backend is first resolved (env override / availability /
/// fallback) and the client is cached under that resolved backend, so repeated
/// calls for the same physical device reuse one client.
public func getGlobalClient(backend: Backend = .cpu) throws -> PJRTClient {
    // Resolve outside the lock: it only reads process environment and plugin
    // availability, and keeps the critical section to the cache lookup/insert.
    let resolved = resolveExecutionBackend(requested: backend)

    _clientLock.lock()
    defer { _clientLock.unlock() }

    if let client = _globalClients[resolved] {
        return client
    }

    let client = try PJRTClient.create(backend: resolved)
    _globalClients[resolved] = client
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
/// Compute the PJRT fast-path (trace) key for a set of output handles.
///
/// A single post-order DFS hashes the *raw* graph's full structure — op kinds,
/// shapes, dtypes, relative input indices (so DAG sharing is captured), and
/// attributes — AND constant VALUES. Two barriers with the same key therefore
/// have an identical structure and identical constants, so a cached executable
/// and cached promoted-constant values are safe to reuse. (`handle.structuralHash`
/// is only maintained on macOS, so this recomputes structure from scratch — still
/// far cheaper than the optimize+emit+compile it skips.)
///
/// Returns `nil` when the graph contains a traced while-loop: its body is not
/// captured by this key, so such graphs must not be fast-path cached. Otherwise
/// returns the key together with the `.data` input handles in the order the
/// executable expects them (same DFS), so the caller needs only this one walk.
func computePJRTTraceKey(
    outputs: [LazyTensorHandle]
) -> (key: UInt64, dataInputs: [LazyTensorHandle])? {
    var hasher = Hasher()
    hasher.combine(outputs.count)
    var indexOf: [UInt64: Int] = [:]
    var next = 0
    var visited = Set<UInt64>()
    var hasWhileLoop = false
    var dataInputs: [LazyTensorHandle] = []

    func visit(_ handle: LazyTensorHandle) {
        guard visited.insert(handle.id).inserted else { return }

        if let node = handle.irNode {
            switch node {
            case .operation(_, let inputs, _):
                for input in inputs { visit(input) }
            case .whileLoopTraced(_, let initialValues, _, _, _):
                hasWhileLoop = true
                for input in initialValues { visit(input) }
            case .constant, .data:
                break
            #if os(macOS) && canImport(MetalHLO)
            case .metalData:
                break
            #endif
            }
        }

        // Assign this node's structural index in post-order (topological) so that
        // inputs are referenced by an already-assigned relative index.
        indexOf[handle.id] = next
        next += 1

        guard let node = handle.irNode else {
            hasher.combine(9); hasher.combine(handle.shape); hasher.combine(handle.dtype)
            return
        }
        switch node {
        case .constant(let values, let shape):
            hasher.combine(0); hasher.combine(shape); hasher.combine(handle.dtype); hasher.combine(values)
        case .data:
            hasher.combine(1); hasher.combine(handle.shape); hasher.combine(handle.dtype)
            dataInputs.append(handle)
        case .operation(let op, let inputs, let attributes):
            hasher.combine(2); hasher.combine(op.rawValue)
            hasher.combine(handle.shape); hasher.combine(handle.dtype)
            hasher.combine(inputs.count)
            for input in inputs { hasher.combine(indexOf[input.id] ?? -1) }
            for (k, v) in attributes.sorted(by: { $0.key < $1.key }) {
                hasher.combine(k); hasher.combine(String(describing: v))
            }
        case .whileLoopTraced(let iterations, let initialValues, _, _, _):
            hasher.combine(3); hasher.combine(iterations)
            hasher.combine(handle.shape); hasher.combine(handle.dtype)
            for input in initialValues { hasher.combine(indexOf[input.id] ?? -1) }
        #if os(macOS) && canImport(MetalHLO)
        case .metalData:
            hasher.combine(1); hasher.combine(handle.shape); hasher.combine(handle.dtype)
        #endif
        }
    }

    for output in outputs { visit(output) }
    for output in outputs { hasher.combine(indexOf[output.id] ?? -1) }

    if hasWhileLoop { return nil }
    return (key: UInt64(hasher.finalize().magnitude), dataInputs: dataInputs)
}

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

    let cache = CompilationCache.shared
    let traceCacheDisabled = ProcessInfo.processInfo.environment["MAGMA_NO_TRACE_CACHE"] == "1"
    let verifyTraceCache = ProcessInfo.processInfo.environment["MAGMA_VERIFY_TRACE_CACHE"] == "1"

    // Fast path (trace cache): skip graph build + optimization + emission +
    // compilation for a repeated graph structure. `fastKey` folds in constant
    // values (nil ⇒ contains a while-loop ⇒ ineligible).
    let traceKeyResult = traceCacheDisabled ? nil : computePJRTTraceKey(outputs: outputs)
    if let (fastKey, dataInputs) = traceKeyResult, let entry = cache.getFast(hash: fastKey) {
        if verifyTraceCache {
            // Rebuild + optimize + hash the slow way and assert the fast key maps
            // to the same structural graph (catches a fast-key collision). Read-only
            // w.r.t. `outputs`. Only used on validation runs — it defeats the perf win.
            let vg = IRGraph()
            for output in outputs { vg.addOutput(output) }
            vg.buildTopologicalOrder()
            let vOpt = ProcessInfo.processInfo.environment["MAGMA_NO_OPT"] == "1"
                ? vg : PassManager.shared.run(on: vg)
            let vHash = vOpt.analyzeForConstantPromotion().structuralHash
            if vHash != entry.structuralHash {
                print("Magma: TRACE-CACHE VERIFY FAILED (fastKey=\(fastKey)): "
                    + "structuralHash \(vHash) != cached \(entry.structuralHash)")
            }
        }
        if dataInputs.count == entry.dataInputCount {
            var ok = true
            var inputBuffers: [PJRTBuffer] = []
            inputBuffers.reserveCapacity(dataInputs.count + entry.promotedConstantBuffers.count)
            for handle in dataInputs {
                if case .data(let buffer) = handle.irNode {
                    inputBuffers.append(buffer)
                } else {
                    ok = false; break   // unexpected leaf kind — bail to slow path
                }
            }
            if ok {
                // Promoted constants: reuse the buffers created at compile time —
                // no client call, no host→device upload.
                inputBuffers.append(contentsOf: entry.promotedConstantBuffers)
                do {
                    let outputBuffers = try entry.executable.execute(inputBuffers)
                    for (i, output) in outputs.enumerated() where i < outputBuffers.count {
                        output.materializedBuffer = outputBuffers[i]
                        output.irNode = .data(outputBuffers[i])
                    }
                    cache.fastHitCount += 1
                    return  // Fast path succeeded
                } catch {
                    // Any failure falls through to the slow path below, which
                    // rebuilds everything from scratch.
                    print("Magma: trace-cache fast path failed, using slow path: \(error)")
                }
            }
        }
        // Count mismatch or bail: fall through to slow path (does not re-cache).
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
    var optimizedDataOrder: [UInt64] = []
    for node in optimizedGraph.nodes {
        if case .data(let buffer) = node.irNode {
            inputBuffers.append(buffer)
            optimizedDataOrder.append(node.id)
        }
    }

    // The fast path collects data inputs by walking the RAW graph from outputs
    // (the `dataInputs` returned by computePJRTTraceKey). Only cache a trace entry
    // when that raw order matches the optimized graph's data order the executable
    // was compiled against — otherwise a future fast hit would feed buffers in the
    // wrong slots. (Reordering/DCE of data leaves ⇒ mismatch ⇒ no caching.)
    let traceOrderMatches: Bool
    if let rawDataInputs = traceKeyResult?.dataInputs {
        traceOrderMatches = rawDataInputs.map { $0.id } == optimizedDataOrder
    } else {
        traceOrderMatches = false
    }

    // Second: promoted constants (need to create buffers for their values).
    // Capture them so the trace entry can reuse them on future fast hits.
    var promotedConstantBuffers: [PJRTBuffer] = []
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
                promotedConstantBuffers.append(buffer)
            }
        } catch {
            print("Magma: Failed to create buffers for promoted constants: \(error)")
            return
        }
    }

    // 9. Execute
    do {
        let outputBuffers = try executable.execute(inputBuffers)

        // 9.5 Store a fast-path (trace) entry so a repeated barrier can skip the
        // whole pipeline. Only when the key is eligible (no while-loop) and the
        // raw/optimized data-input orders match (fast-path input assembly is sound).
        if let fastKey = traceKeyResult?.key, traceOrderMatches {
            cache.putFast(hash: fastKey, entry: PJRTTraceCacheEntry(
                executable: executable,
                dataInputCount: optimizedDataOrder.count,
                promotedConstantBuffers: promotedConstantBuffers,
                structuralHash: structuralHash
            ))
        }

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

/// How a graph input (`.data` node) is provided to each replica when running
/// data-parallel.
public enum ReplicaInputDistribution: Sendable {
    /// Same value on every replica (e.g. a model parameter). The input's baked
    /// host value is copied to each device.
    case replicated
    /// A distinct value per replica (e.g. this replica's data batch). Provide one
    /// float array per replica, each matching the input's shape.
    case perReplica([[Float]])
}

/// The high-level multi-device barrier: compile a traced graph for `numReplicas`
/// data-parallel replicas and execute it across that many devices of `client`,
/// returning each replica's outputs.
///
/// Each `.data` input node is distributed per `distribution` (keyed by the input
/// buffer's identity); inputs not listed default to `.replicated`. The graph is
/// traced once with the per-replica shapes, so cross-replica collectives in the
/// graph (e.g. `all_reduce`) reduce across the replicas. This is the primitive
/// DDP builds on: replicate parameters, provide each replica its data shard, and
/// sync gradients with an all-reduce in the graph.
///
/// Currently supports float32 inputs. Does not use the single-device compilation
/// cache, so it never disturbs the ordinary barrier.
public func executeGraphReplicated(
    _ graph: IRGraph,
    numReplicas: Int,
    distribution: [ObjectIdentifier: ReplicaInputDistribution] = [:],
    client: PJRTClient
) throws -> [[PJRTBuffer]] {
    precondition(numReplicas > 0, "numReplicas must be positive")
    guard client.deviceCount >= numReplicas else {
        throw XLAError.executionFailed(
            "client has \(client.deviceCount) device(s), need \(numReplicas) for \(numReplicas) replicas")
    }

    graph.buildTopologicalOrder()
    let optimizedGraph: IRGraph
    if ProcessInfo.processInfo.environment["MAGMA_NO_OPT"] != "1" {
        optimizedGraph = PassManager.shared.run(on: graph)
    } else {
        optimizedGraph = graph
    }

    let emitter = StableHLOEmitter(graph: optimizedGraph)
    let mlir = emitter.emit(name: "replicated_graph_\(optimizedGraph.computeHash().prefix(8))")

    let executable = try client.compile(mlir, numReplicas: numReplicas, useSPMDPartitioning: false)

    // Collect .data inputs in the same order the emitter used for func args.
    var dataBuffers: [PJRTBuffer] = []
    for node in optimizedGraph.nodes {
        if case .data(let buffer) = node.irNode { dataBuffers.append(buffer) }
    }

    // Build per-replica input buffers, each resident on its replica's device.
    var perReplica: [[PJRTBuffer]] = Array(repeating: [], count: numReplicas)
    for buf in dataBuffers {
        switch distribution[ObjectIdentifier(buf)] ?? .replicated {
        case .replicated:
            let host = try buf.toFloatArray()
            for d in 0..<numReplicas {
                perReplica[d].append(try client.createBuffer(
                    host, shape: buf.shape, elementType: .float32, device: client.devices[d]))
            }
        case .perReplica(let datas):
            guard datas.count == numReplicas else {
                throw XLAError.executionFailed(
                    "perReplica data count \(datas.count) != numReplicas \(numReplicas)")
            }
            for d in 0..<numReplicas {
                perReplica[d].append(try client.createBuffer(
                    datas[d], shape: buf.shape, elementType: .float32, device: client.devices[d]))
            }
        }
    }

    return try executable.executeMultiDevice(inputsPerDevice: perReplica)
}

// MARK: - While Loop Tracing

/// Context for tracing a function body for while loop emission
///
/// During tracing, we create placeholder inputs that represent loop-carried variables.
/// All operations performed using these placeholders are recorded, and the final
/// outputs are captured. This traced graph can then be emitted as a stablehlo.while.
/// Global count of threads currently tracing a while-loop body. `registerPending`
/// checks this (a relaxed atomic load) on the per-op hot path to avoid the
/// thread-local `WhileLoopTracer.shared` lookup when nobody is tracing — which is
/// the overwhelmingly common case. Kept in sync by beginTrace/endTrace/abortTrace.
nonisolated(unsafe) let _globalActiveTraceCount = Atomic<Int>(0)

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
        _globalActiveTraceCount.wrappingAdd(1, ordering: .relaxed)
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
        _globalActiveTraceCount.wrappingSubtract(1, ordering: .relaxed)

        let result = (inputs: bodyInputs, outputs: bodyOutputs, nodes: tracedNodes)

        // Clear state
        bodyInputs = []
        tracedNodes = []
        tracedNodeIds = []

        return result
    }

    /// Abort tracing (for error recovery)
    public func abortTrace() {
        // Guard the decrement so a stray abort (when not tracing) can't underflow
        // the global count.
        if isTracing {
            isTracing = false
            _globalActiveTraceCount.wrappingSubtract(1, ordering: .relaxed)
        }
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
