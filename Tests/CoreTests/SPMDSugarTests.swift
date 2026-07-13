// Magma - SPMD Tensor-sugar tests
// SPMD written in ordinary Tensor code: annotate tensors with `.sharded(...)`,
// build with `makeGraph(mesh:)`, and run with `executeGraphSharded`. This is the
// SPMD ergonomic parallel to `crossReplicaMean` for DDP. CPU-emulated, no OOM.

import Testing
@testable import Magma
@testable import LazyTensor
@testable import StableHLO
@testable import XLARuntime

@Suite("SPMD Sugar Tests", .serialized)
struct SPMDSugarTests {
    static let cpuAvailable: Bool = { (try? PJRTClient.create(backend: .cpu)) != nil }()

    @Test("row-sharded matmul via Tensor.sharded sugar matches reference")
    func spmdViaSugar() throws {
        try #require(Self.cpuAvailable, "CPU PJRT plugin not available")
        let client = try PJRTClient.create(backend: .cpu, cpuDeviceCount: 2)
        let dev0 = client.devices[0]

        // Global inputs: x [8,4] = 1..32, w [4,4] = identity (so x @ w == x).
        let xFull = (1...32).map(Float.init)
        var identity = [Float](repeating: 0, count: 16)
        for i in 0..<4 { identity[i * 4 + i] = 1 }
        let xBuf = try client.createBuffer(xFull, shape: [8, 4], elementType: .float32, device: dev0)
        let wBuf = try client.createBuffer(identity, shape: [4, 4], elementType: .float32, device: dev0)

        // The whole SPMD program, in Tensor code: shard x's rows over mesh axis
        // "x", replicate w, and mark the output row-sharded too.
        let x = Tensor<Float>.input(from: xBuf).sharded(on: "mesh", ["x", nil])
        let w = Tensor<Float>.input(from: wBuf)                       // replicated
        let y = x.matmul(w).sharded(on: "mesh", ["x", nil])

        let mesh = DeviceMesh.linear(name: "mesh", axisName: "x", size: 2)
        let outs = try executeGraphSharded(y.makeGraph(mesh: mesh), numDevices: 2, client: client)
        #expect(outs.count == 2)

        // Gather the per-device row shards; x @ I == x.
        let (gathered, shape) = try client.gatherAlongAxis0(outs.map { $0[0] })
        #expect(shape == [8, 4])
        #expect(gathered == xFull)
    }

    @Test("makeGraph(mesh:) sets the graph mesh; sharded() annotates the node")
    func annotationPlumbing() {
        let xBuf_shape = [4]
        // Pure-emission check: a constant tensor sharded on a mesh emits sdy.
        let c = Tensor<Float>([1, 2, 3, 4], shape: xBuf_shape)
            .sharded(on: "mesh", ["x"])
        let mlir = c.makeGraph(mesh: .linear(name: "mesh", axisName: "x", size: 2))
            .emitStableHLO(name: "m")
        #expect(mlir.contains("sdy.mesh @mesh = <[\"x\"=2]>"))
        #expect(mlir.contains("<@mesh, [{\"x\"}]>"))

        // No mesh => no sdy (annotation inert without a mesh).
        let bare = Tensor<Float>([1, 2, 3, 4], shape: xBuf_shape).sharded(on: "mesh", ["x"])
        #expect(!bare.makeGraph().emitStableHLO(name: "m").contains("sdy"))
    }
}
