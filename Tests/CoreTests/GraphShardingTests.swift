// Magma - Graph sharding tests
// LazyTensorHandle.sharding + IRGraph.mesh are threaded through the
// StableHLOEmitter: the mesh becomes an sdy.mesh, a sharded data input gets a
// {sdy.sharding=…} argument attribute, and a sharded intermediate gets a
// sdy.sharding_constraint. validateShardings checks node shardings against the
// mesh. Emission tests are pure Swift; the argument test needs the CPU plugin.

import Testing
@testable import LazyTensor
@testable import StableHLO
@testable import XLARuntime

@Suite("Graph Sharding Tests")
struct GraphShardingTests {
    private func handle(_ shape: [Int]) -> LazyTensorHandle {
        LazyTensorHandle(id: TensorRegistry.shared.nextTensorId(),
                         shape: shape, dtype: .float32, device: .default)
    }

    @Test("mesh + intermediate sharding_constraint are emitted")
    func emitsMeshAndConstraint() {
        let graph = IRGraph()
        graph.mesh = .linear(name: "mesh", axisName: "x", size: 2)
        let c = handle([4])
        c.irNode = .constant(values: [1, 2, 3, 4], shape: [4])
        c.sharding = TensorSharding(meshName: "mesh", axisNames: ["x"])
        graph.addOutput(c)

        let mlir = graph.emitStableHLO(name: "m")
        #expect(mlir.contains("sdy.mesh @mesh = <[\"x\"=2]>"))
        #expect(mlir.contains("sdy.sharding_constraint"))
        #expect(mlir.contains("<@mesh, [{\"x\"}]>"))
    }

    @Test("no mesh => no sdy annotations (unchanged output)")
    func noMeshNoSdy() {
        let graph = IRGraph()
        let c = handle([4])
        c.irNode = .constant(values: [1, 2, 3, 4], shape: [4])
        c.sharding = TensorSharding(meshName: "mesh", axisNames: ["x"])   // ignored w/o mesh
        graph.addOutput(c)

        let mlir = graph.emitStableHLO(name: "m")
        #expect(!mlir.contains("sdy"))
    }

    @Test("validateShardings accepts a well-formed sharding")
    func validateOK() throws {
        let graph = IRGraph()
        graph.mesh = .linear(name: "mesh", axisName: "x", size: 2)
        let c = handle([4])
        c.irNode = .constant(values: [1, 2, 3, 4], shape: [4])
        c.sharding = TensorSharding(meshName: "mesh", axisNames: ["x"])
        graph.addOutput(c)
        graph.buildTopologicalOrder()
        try graph.validateShardings()
    }

    @Test("validateShardings rejects an unknown axis")
    func validateUnknownAxis() {
        let graph = IRGraph()
        graph.mesh = .linear(name: "mesh", axisName: "x", size: 2)
        let c = handle([4])
        c.irNode = .constant(values: [1, 2, 3, 4], shape: [4])
        c.sharding = TensorSharding(meshName: "mesh", axisNames: ["nope"])
        graph.addOutput(c)
        graph.buildTopologicalOrder()
        #expect(throws: ShardingError.self) { try graph.validateShardings() }
    }

    @Test("validateShardings requires a mesh when a node is sharded")
    func validateRequiresMesh() {
        let graph = IRGraph()   // no mesh
        let c = handle([4])
        c.irNode = .constant(values: [1, 2, 3, 4], shape: [4])
        c.sharding = TensorSharding(meshName: "mesh", axisNames: ["x"])
        graph.addOutput(c)
        graph.buildTopologicalOrder()
        #expect(throws: ShardingError.self) { try graph.validateShardings() }
    }

    @Test("sharded data input becomes a {sdy.sharding} argument attribute")
    func argSharding() throws {
        try #require((try? PJRTClient.create(backend: .cpu)) != nil, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu)
        let buf = try client.createBuffer([1, 2, 3, 4] as [Float], shape: [4],
                                          elementType: .float32, device: client.devices[0])

        let graph = IRGraph()
        graph.mesh = .linear(name: "mesh", axisName: "x", size: 2)
        let x = handle([4]); x.irNode = .data(buf)
        x.sharding = TensorSharding(meshName: "mesh", axisNames: ["x"])
        let y = handle([4]); y.irNode = .operation(op: .add, inputs: [x, x], attributes: [:])
        graph.addOutput(y)

        let mlir = graph.emitStableHLO(name: "m")
        #expect(mlir.contains("{sdy.sharding = #sdy.sharding<@mesh, [{\"x\"}]>}"))
    }
}
