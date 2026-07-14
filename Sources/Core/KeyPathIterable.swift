// Magma - KeyPathIterable + reflection key-path helpers (experimental / PoC)
//
// The S4TF `KeyPathIterable`, backed by the standard-library reflection SPI
// `_forEachFieldWithKeyPath` instead of compiler synthesis (never upstreamed).
// This is the foundation for value-semantic differentiable models + a generic
// optimizer that walks a `Model.TangentVector` (the "Design A" path).
//
// The free `reflect*KeyPaths` functions work on ANY nominal type with no
// conformance — used to walk a synthesized `TangentVector` (which can't be
// extended to add a protocol). `KeyPathIterable` is the model's opt-in marker,
// with the same helpers as static methods.
//
// STATUS: experimental. Uses an underscored SPI (`@_spi(Reflection)`) — works on
// the current toolchain (Swift 6.3.3, aarch64-Linux) but is not stable API. Key
// paths are memoized per type. Recursion into nested KeyPathIterable properties
// (layer-of-layers) is future work; this covers flat models (Tensor fields
// direct), enough to prove the mechanism end to end.

@_spi(Reflection) import Swift
import Foundation

// Per-type key-path cache (reflection runs once per type).
nonisolated(unsafe) private var _keyPathCache: [ObjectIdentifier: [AnyKeyPath]] = [:]
private let _keyPathCacheLock = NSLock()

/// All stored-property key paths of `T`, in declaration order (memoized). Works
/// on any nominal type via reflection — no protocol conformance required.
public func reflectStoredKeyPaths<T>(of type: T.Type) -> [PartialKeyPath<T>] {
    let oid = ObjectIdentifier(T.self)

    _keyPathCacheLock.lock()
    let cached = _keyPathCache[oid]
    _keyPathCacheLock.unlock()
    if let cached {
        return cached.compactMap { $0 as? PartialKeyPath<T> }
    }

    var kps: [PartialKeyPath<T>] = []
    _ = _forEachFieldWithKeyPath(of: T.self, options: []) { _, kp in
        kps.append(kp)
        return true
    }

    _keyPathCacheLock.lock()
    _keyPathCache[oid] = kps.map { $0 as AnyKeyPath }
    _keyPathCacheLock.unlock()
    return kps
}

/// Writable key paths to the stored properties of type `V` (declaration order).
public func reflectWritableKeyPaths<T, V>(of type: T.Type, to _: V.Type) -> [WritableKeyPath<T, V>] {
    reflectStoredKeyPaths(of: type).compactMap { $0 as? WritableKeyPath<T, V> }
}

/// Read-only key paths to the stored properties of type `V` (declaration order).
public func reflectKeyPaths<T, V>(of type: T.Type, to _: V.Type) -> [KeyPath<T, V>] {
    reflectStoredKeyPaths(of: type).compactMap { $0 as? KeyPath<T, V> }
}

/// A model whose stored `Tensor` properties can be enumerated as key paths.
/// Conformance is empty — the default implementations use reflection.
public protocol KeyPathIterable {}

extension KeyPathIterable {
    /// Writable key paths to the stored properties of type `V`.
    public static func writableKeyPaths<V>(to type: V.Type) -> [WritableKeyPath<Self, V>] {
        reflectWritableKeyPaths(of: Self.self, to: type)
    }

    /// Read-only key paths to the stored properties of type `V`.
    public static func keyPaths<V>(to type: V.Type) -> [KeyPath<Self, V>] {
        reflectKeyPaths(of: Self.self, to: type)
    }
}
