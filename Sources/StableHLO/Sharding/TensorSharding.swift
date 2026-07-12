// Magma - Tensor sharding (Shardy `#sdy.sharding`)
//
// Pure-Swift model of how a tensor is partitioned across a `DeviceMesh`, emitting
// the `#sdy.sharding<...>` attribute text the Shardy partitioner consumes.
// Ported/adapted from SwiftIR's SwiftIRShardingLite and tied to `DeviceMesh` via
// `validate(against:rank:)` so the two form one coherent unit rather than
// independent islands. No native dependency — string generation only.

import Foundation

/// Errors from validating a `TensorSharding` against a `DeviceMesh`.
public enum ShardingError: Error, Equatable, Sendable {
    case meshNameMismatch(expected: String, got: String)
    case rankMismatch(expected: Int, got: Int)
    case unknownAxis(String, mesh: String)
    case axisUsedMoreThanOnce(String)
}

/// How a single tensor dimension is sharded across mesh axes.
///
/// A *closed* dimension is fully specified (`{"x"}`); an *open* dimension
/// (`{"x", ?}` / `{?}`) permits the propagation pass to add further sharding.
public struct DimensionSharding: Equatable, Hashable, Sendable {
    public let axes: [String]
    /// If true the dimension is fully specified; if false propagation may extend it.
    public let isClosed: Bool
    /// Optional propagation priority. Reserved: not yet emitted in MLIR text until
    /// the exact `sdy` priority syntax is validated end-to-end (see P2 SPMD test).
    public let priority: Int?

    public init(axes: [String] = [], isClosed: Bool = true, priority: Int? = nil) {
        self.axes = axes
        self.isClosed = isClosed
        self.priority = priority
    }

    /// Fully replicated (closed): `{}`.
    public static var replicated: DimensionSharding {
        DimensionSharding()
    }

    /// Sharded on a single axis, closed: `{"axis"}`.
    public static func sharded(on axis: String) -> DimensionSharding {
        DimensionSharding(axes: [axis])
    }

    /// Open on the given axes: `{"axis", ?}` (or `{?}` when empty) — propagation
    /// may add more.
    public static func open(on axes: [String] = []) -> DimensionSharding {
        DimensionSharding(axes: axes, isClosed: false)
    }

    /// The dimension as it appears in the sharding list, honoring open/closed.
    public var mlirText: String {
        var tokens = axes.map { "\"\($0)\"" }
        if !isClosed { tokens.append("?") }   // open: propagation may add axes
        return "{\(tokens.joined(separator: ", "))}"
    }
}

/// A complete sharding for a tensor: one `DimensionSharding` per dimension, plus
/// any axes the tensor is replicated over.
public struct TensorSharding: Equatable, Hashable, Sendable, CustomStringConvertible {
    public let meshName: String
    public let dimShardings: [DimensionSharding]
    public let replicatedAxes: [String]

    public init(
        meshName: String,
        dimShardings: [DimensionSharding],
        replicatedAxes: [String] = []
    ) {
        self.meshName = meshName
        self.dimShardings = dimShardings
        self.replicatedAxes = replicatedAxes
    }

    /// Convenience: one entry per dimension, `nil` meaning replicated.
    public init(meshName: String, axisNames: [String?]) {
        self.meshName = meshName
        self.dimShardings = axisNames.map { axis in
            axis.map { DimensionSharding.sharded(on: $0) } ?? .replicated
        }
        self.replicatedAxes = []
    }

    /// A fully-replicated sharding for a rank-`rank` tensor.
    public static func replicated(meshName: String, rank: Int) -> TensorSharding {
        TensorSharding(
            meshName: meshName,
            dimShardings: Array(repeating: .replicated, count: rank)
        )
    }

    public var rank: Int { dimShardings.count }

    /// The full `#sdy.sharding<@mesh, [...]>` attribute text.
    public var mlirAttributeText: String {
        let dimStr = dimShardings.map(\.mlirText).joined(separator: ", ")
        var result = "#sdy.sharding<@\(meshName), [\(dimStr)]"
        if !replicatedAxes.isEmpty {
            let rep = replicatedAxes.map { "\"\($0)\"" }.joined(separator: ", ")
            result += ", replicated={\(rep)}"
        }
        result += ">"
        return result
    }

    public var description: String { mlirAttributeText }

    /// Validate this sharding against a mesh and tensor rank. Ensures the mesh
    /// name matches, the rank matches, every referenced axis exists in the mesh,
    /// and no axis is used more than once (an `sdy` requirement).
    public func validate(against mesh: DeviceMesh, rank: Int) throws {
        guard meshName == mesh.name else {
            throw ShardingError.meshNameMismatch(expected: mesh.name, got: meshName)
        }
        guard dimShardings.count == rank else {
            throw ShardingError.rankMismatch(expected: rank, got: dimShardings.count)
        }
        var used = Set<String>()
        func use(_ axis: String) throws {
            guard mesh.contains(axis: axis) else {
                throw ShardingError.unknownAxis(axis, mesh: mesh.name)
            }
            guard used.insert(axis).inserted else {
                throw ShardingError.axisUsedMoreThanOnce(axis)
            }
        }
        for dim in dimShardings {
            for axis in dim.axes { try use(axis) }
        }
        for axis in replicatedAxes { try use(axis) }
    }
}
