// Magma - Device mesh (Shardy `sdy.mesh`)
//
// Pure-Swift model of a logical device topology, emitting the `sdy.mesh` MLIR
// text that the Shardy partitioner consumes. Ported/adapted from SwiftIR's
// SwiftIRShardingLite; kept in the StableHLO module so it sits next to its
// consumer, `MLIRBuilder` (which emits the mesh into a module header). No native
// dependency — this is string generation only.

import Foundation

/// A single named axis of a device mesh (e.g. `MeshAxis(name: "data", size: 4)`).
public struct MeshAxis: Equatable, Hashable, Sendable {
    public let name: String
    public let size: Int

    public init(name: String, size: Int) {
        precondition(size > 0, "Mesh axis size must be positive")
        self.name = name
        self.size = size
    }

    /// The axis as it appears inside an `sdy.mesh` list, e.g. `"data"=4`.
    public var mlirText: String {
        "\"\(name)\"=\(size)"
    }
}

/// A logical device topology referenced by tensor shardings.
///
/// Example:
/// ```swift
/// let mesh = DeviceMesh.grid(name: "mesh", rows: 2, cols: 4)  // 8 devices
/// mesh.mlirText  // sdy.mesh @mesh = <["x"=2, "y"=4]>
/// ```
public struct DeviceMesh: Equatable, Hashable, Sendable {
    public let name: String
    public let axes: [MeshAxis]
    /// Optional explicit logical→physical device ordering.
    public let deviceIds: [Int]?

    public init(name: String, axes: [MeshAxis], deviceIds: [Int]? = nil) {
        self.name = name
        self.axes = axes
        self.deviceIds = deviceIds
    }

    // MARK: Factories

    /// 1-D mesh with a single named axis.
    public static func linear(name: String, axisName: String, size: Int) -> DeviceMesh {
        DeviceMesh(name: name, axes: [MeshAxis(name: axisName, size: size)])
    }

    /// 2-D mesh (rows × cols).
    public static func grid(
        name: String,
        rows: Int, cols: Int,
        rowAxis: String = "x", colAxis: String = "y"
    ) -> DeviceMesh {
        DeviceMesh(name: name, axes: [
            MeshAxis(name: rowAxis, size: rows),
            MeshAxis(name: colAxis, size: cols),
        ])
    }

    /// 3-D mesh (x × y × z).
    public static func cube(
        name: String,
        x: Int, y: Int, z: Int,
        xAxis: String = "x", yAxis: String = "y", zAxis: String = "z"
    ) -> DeviceMesh {
        DeviceMesh(name: name, axes: [
            MeshAxis(name: xAxis, size: x),
            MeshAxis(name: yAxis, size: y),
            MeshAxis(name: zAxis, size: z),
        ])
    }

    // MARK: Queries

    /// Total number of devices (product of axis sizes).
    public var deviceCount: Int {
        axes.reduce(1) { $0 * $1.size }
    }

    /// Look up an axis by name.
    public func axis(named name: String) -> MeshAxis? {
        axes.first { $0.name == name }
    }

    /// Whether an axis with this name exists in the mesh.
    public func contains(axis name: String) -> Bool {
        axis(named: name) != nil
    }

    // MARK: MLIR

    /// The full `sdy.mesh @name = <[...]>` op text.
    public var mlirText: String {
        let axesStr = axes.map(\.mlirText).joined(separator: ", ")
        var result = "sdy.mesh @\(name) = <[\(axesStr)]"
        if let ids = deviceIds {
            let idsStr = ids.map(String.init).joined(separator: ", ")
            result += ", device_ids=[\(idsStr)]"
        }
        result += ">"
        return result
    }
}
