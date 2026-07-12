// Magma - Sharding tests
// Verify the pure-Swift Shardy types (DeviceMesh / TensorSharding) and their
// integration into MLIRBuilder's module header. No XLA needed.

import Testing
@testable import StableHLO

@Suite("Device Mesh Tests")
struct DeviceMeshTests {

    @Test("linear mesh mlirText")
    func linear() {
        let mesh = DeviceMesh.linear(name: "mesh", axisName: "data", size: 4)
        #expect(mesh.deviceCount == 4)
        #expect(mesh.mlirText == "sdy.mesh @mesh = <[\"data\"=4]>")
    }

    @Test("grid mesh mlirText and deviceCount")
    func grid() {
        let mesh = DeviceMesh.grid(name: "mesh", rows: 2, cols: 4)
        #expect(mesh.deviceCount == 8)
        #expect(mesh.mlirText == "sdy.mesh @mesh = <[\"x\"=2, \"y\"=4]>")
    }

    @Test("cube mesh and axis lookup")
    func cube() {
        let mesh = DeviceMesh.cube(name: "m", x: 2, y: 2, z: 2)
        #expect(mesh.deviceCount == 8)
        #expect(mesh.axis(named: "y")?.size == 2)
        #expect(mesh.contains(axis: "z"))
        #expect(!mesh.contains(axis: "w"))
    }

    @Test("explicit device_ids render")
    func deviceIds() {
        let mesh = DeviceMesh(name: "m", axes: [MeshAxis(name: "x", size: 2)], deviceIds: [1, 0])
        #expect(mesh.mlirText == "sdy.mesh @m = <[\"x\"=2], device_ids=[1, 0]>")
    }
}

@Suite("Tensor Sharding Tests")
struct TensorShardingTests {

    @Test("dimension sharding: closed / open / replicated")
    func dimensionText() {
        #expect(DimensionSharding.replicated.mlirText == "{}")
        #expect(DimensionSharding.sharded(on: "x").mlirText == "{\"x\"}")
        #expect(DimensionSharding.open(on: ["x"]).mlirText == "{\"x\", ?}")
        #expect(DimensionSharding.open().mlirText == "{?}")
    }

    @Test("tensor sharding attribute text")
    func attributeText() {
        // batch on "x", features replicated
        let s = TensorSharding(meshName: "mesh", axisNames: ["x", nil])
        #expect(s.rank == 2)
        #expect(s.mlirAttributeText == "#sdy.sharding<@mesh, [{\"x\"}, {}]>")
    }

    @Test("replicated factory and replicatedAxes")
    func replicated() {
        #expect(TensorSharding.replicated(meshName: "m", rank: 2).mlirAttributeText
            == "#sdy.sharding<@m, [{}, {}]>")
        let s = TensorSharding(
            meshName: "m",
            dimShardings: [.sharded(on: "x"), .replicated],
            replicatedAxes: ["y"]
        )
        #expect(s.mlirAttributeText == "#sdy.sharding<@m, [{\"x\"}, {}], replicated={\"y\"}>")
    }

    // MARK: validation ties TensorSharding <-> DeviceMesh together

    @Test("validate accepts a well-formed sharding")
    func validateOK() throws {
        let mesh = DeviceMesh.grid(name: "mesh", rows: 2, cols: 4)
        let s = TensorSharding(meshName: "mesh", axisNames: ["x", "y"])
        try s.validate(against: mesh, rank: 2)   // must not throw
    }

    @Test("validate rejects rank mismatch")
    func validateRank() {
        let mesh = DeviceMesh.linear(name: "mesh", axisName: "x", size: 4)
        let s = TensorSharding(meshName: "mesh", axisNames: ["x"])
        #expect(throws: ShardingError.rankMismatch(expected: 2, got: 1)) {
            try s.validate(against: mesh, rank: 2)
        }
    }

    @Test("validate rejects unknown axis")
    func validateUnknownAxis() {
        let mesh = DeviceMesh.linear(name: "mesh", axisName: "x", size: 4)
        let s = TensorSharding(meshName: "mesh", axisNames: ["nope", nil])
        #expect(throws: ShardingError.unknownAxis("nope", mesh: "mesh")) {
            try s.validate(against: mesh, rank: 2)
        }
    }

    @Test("validate rejects an axis used twice")
    func validateDuplicateAxis() {
        let mesh = DeviceMesh.linear(name: "mesh", axisName: "x", size: 4)
        let s = TensorSharding(meshName: "mesh", axisNames: ["x", "x"])
        #expect(throws: ShardingError.axisUsedMoreThanOnce("x")) {
            try s.validate(against: mesh, rank: 2)
        }
    }

    @Test("validate rejects mesh-name mismatch")
    func validateMeshName() {
        let mesh = DeviceMesh.linear(name: "mesh", axisName: "x", size: 4)
        let s = TensorSharding(meshName: "other", axisNames: ["x", nil])
        #expect(throws: ShardingError.meshNameMismatch(expected: "mesh", got: "other")) {
            try s.validate(against: mesh, rank: 2)
        }
    }
}

@Suite("MLIRBuilder Sharding Integration")
struct MLIRBuilderShardingTests {

    @Test("declared mesh is emitted in the module header")
    func meshInHeader() {
        let builder = MLIRBuilder()
        builder.declareMesh(.grid(name: "mesh", rows: 2, cols: 4))
        let x = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let y = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let z = builder.add(x, y)

        let mlir = builder.build(name: "m", outputs: [z])

        // The mesh op appears inside the module, before func.func.
        let meshLine = "sdy.mesh @mesh = <[\"x\"=2, \"y\"=4]>"
        #expect(mlir.contains(meshLine))
        let meshIdx = mlir.range(of: meshLine)
        let funcIdx = mlir.range(of: "func.func @main")
        #expect(meshIdx != nil && funcIdx != nil)
        if let m = meshIdx, let f = funcIdx { #expect(m.lowerBound < f.lowerBound) }
        // Body still present.
        #expect(mlir.contains("stablehlo.add"))
    }

    @Test("no mesh declared leaves the module unchanged")
    func noMeshUnchanged() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let y = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let z = builder.add(x, y)
        let mlir = builder.build(name: "m", outputs: [z])

        #expect(!mlir.contains("sdy.mesh"))
        // Header is the plain single-device form.
        #expect(mlir.contains("module @m {\n  func.func @main"))
    }

    @Test("argument sharding renders as an arg attribute (full #sdy.sharding form)")
    func argumentSharding() {
        let builder = MLIRBuilder()
        builder.declareMesh(.grid(name: "mesh", rows: 2, cols: 4))
        // Shard %arg0's rows on "x"; leave %arg1 unsharded.
        let x = builder.argument(
            TensorType(shape: [2, 3], dtype: .float32),
            sharding: TensorSharding(meshName: "mesh", axisNames: ["x", nil])
        )
        let y = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let z = builder.add(x, y)
        let mlir = builder.build(name: "m", outputs: [z])

        #expect(mlir.contains("%arg0: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}, {}]>}"))
        // The unsharded arg keeps the plain form.
        #expect(mlir.contains("%arg1: tensor<2x3xf32>,") || mlir.contains("%arg1: tensor<2x3xf32>)"))
    }

    @Test("sharding_constraint emits an in-graph op with the bare sharding form")
    func shardingConstraintOp() {
        let builder = MLIRBuilder()
        builder.declareMesh(.linear(name: "mesh", axisName: "x", size: 4))
        let x = builder.argument(TensorType(shape: [4, 8], dtype: .float32))
        let c = builder.shardingConstraint(x, TensorSharding(meshName: "mesh", axisNames: ["x", nil]))
        let mlir = builder.build(name: "m", outputs: [c])

        // Bare form inside the op (no #sdy.sharding wrapper).
        #expect(mlir.contains("= sdy.sharding_constraint %arg0 <@mesh, [{\"x\"}, {}]> : tensor<4x8xf32>"))
        #expect(!mlir.contains("sdy.sharding_constraint %arg0 <#sdy.sharding"))
    }

    @Test("no sharding leaves the argument signature unchanged")
    func noArgShardingUnchanged() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let y = builder.add(x, x)
        let mlir = builder.build(name: "m", outputs: [y])
        #expect(!mlir.contains("sdy.sharding"))
        #expect(mlir.contains("func.func @main(%arg0: tensor<2x3xf32>)"))
    }
}

@Suite("Collective Emission Tests")
struct CollectiveEmissionTests {

    @Test("all_reduce emits a reduction region and replica_groups")
    func allReduceEmission() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [4], dtype: .float32))
        let r = builder.allReduce(x, reduction: "add", replicaGroups: [[0, 1]])
        let mlir = builder.build(name: "ar", outputs: [r])

        #expect(mlir.contains("\"stablehlo.all_reduce\"(%arg0)"))
        #expect(mlir.contains("replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>"))
        #expect(mlir.contains("stablehlo.add %ar_a, %ar_b : tensor<f32>"))
        #expect(mlir.contains("stablehlo.return %ar_r : tensor<f32>"))
        #expect(mlir.contains("(tensor<4xf32>) -> tensor<4xf32>"))
    }

    @Test("all_reduce reduction op and groups are parametric")
    func allReduceParametric() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [2], dtype: .float32))
        let r = builder.allReduce(x, reduction: "maximum", replicaGroups: [[0, 1, 2, 3]])
        let mlir = builder.build(name: "ar", outputs: [r])
        #expect(mlir.contains("replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>"))
        #expect(mlir.contains("stablehlo.maximum %ar_a, %ar_b"))
    }

    @Test("allReduceMean is all_reduce(add) scaled by 1/N")
    func allReduceMeanEmission() {
        let builder = MLIRBuilder()
        let g = builder.argument(TensorType(shape: [4], dtype: .float32))
        let m = builder.allReduceMean(g, replicaGroups: [[0, 1, 2, 3]])
        let mlir = builder.build(name: "mean", outputs: [m])
        // Sum reduction across the group, then a 1/4 = 0.25 scale.
        #expect(mlir.contains("stablehlo.add %ar_a, %ar_b"))
        #expect(mlir.contains("dense<0.25> : tensor<4xf32>"))
        #expect(mlir.contains("stablehlo.multiply"))
    }

    @Test("allReduceMean with a single replica is a no-op scale")
    func allReduceMeanSingle() {
        let builder = MLIRBuilder()
        let g = builder.argument(TensorType(shape: [4], dtype: .float32))
        let m = builder.allReduceMean(g, replicaGroups: [[0]])
        let mlir = builder.build(name: "mean", outputs: [m])
        // 1/1: no scaling multiply is emitted.
        #expect(mlir.contains("stablehlo.all_reduce"))
        #expect(!mlir.contains("stablehlo.multiply"))
    }
}
