// Magma - StableHLOEmitter Tests

import Testing
@testable import LazyTensor
@testable import StableHLO

@Suite("StableHLO Emitter Tests")
struct StableHLOEmitterTests {

    @Test("Simple add")
    func simpleAdd() {
        // Create two constant handles
        let id1 = TensorRegistry.shared.nextTensorId()
        let handle1 = LazyTensorHandle(id: id1, shape: [2, 3], dtype: .float32, device: .default)
        handle1.irNode = .constant(values: [1, 2, 3, 4, 5, 6], shape: [2, 3])

        let id2 = TensorRegistry.shared.nextTensorId()
        let handle2 = LazyTensorHandle(id: id2, shape: [2, 3], dtype: .float32, device: .default)
        handle2.irNode = .constant(values: [1, 1, 1, 1, 1, 1], shape: [2, 3])

        // Create add operation
        let id3 = TensorRegistry.shared.nextTensorId()
        let handle3 = LazyTensorHandle(id: id3, shape: [2, 3], dtype: .float32, device: .default)
        handle3.irNode = .operation(op: .add, inputs: [handle1, handle2], attributes: [:])

        // Build graph
        let graph = IRGraph()
        graph.addOutput(handle3)
        graph.buildTopologicalOrder()

        // Emit MLIR
        let emitter = StableHLOEmitter(graph: graph)
        let mlir = emitter.emit(name: "test_add")

        // Verify MLIR structure
        #expect(mlir.contains("module @test_add"))
        #expect(mlir.contains("stablehlo.add"))
        #expect(mlir.contains("stablehlo.constant"))
    }

    @Test("Matmul relu")
    func matmulRelu() {
        // Create input constants
        let id1 = TensorRegistry.shared.nextTensorId()
        let handle1 = LazyTensorHandle(id: id1, shape: [2, 3], dtype: .float32, device: .default)
        handle1.irNode = .constant(values: Array(repeating: 1.0, count: 6), shape: [2, 3])

        let id2 = TensorRegistry.shared.nextTensorId()
        let handle2 = LazyTensorHandle(id: id2, shape: [3, 4], dtype: .float32, device: .default)
        handle2.irNode = .constant(values: Array(repeating: 1.0, count: 12), shape: [3, 4])

        // Create matmul operation
        let id3 = TensorRegistry.shared.nextTensorId()
        let handle3 = LazyTensorHandle(id: id3, shape: [2, 4], dtype: .float32, device: .default)
        handle3.irNode = .operation(op: .matmul, inputs: [handle1, handle2], attributes: [:])

        // Create relu operation
        let id4 = TensorRegistry.shared.nextTensorId()
        let handle4 = LazyTensorHandle(id: id4, shape: [2, 4], dtype: .float32, device: .default)
        handle4.irNode = .operation(op: .relu, inputs: [handle3], attributes: [:])

        // Build graph
        let graph = IRGraph()
        graph.addOutput(handle4)
        graph.buildTopologicalOrder()

        // Emit MLIR
        let mlir = graph.emitStableHLO(name: "test_matmul_relu")

        // Verify MLIR structure
        #expect(mlir.contains("stablehlo.dot"))
        #expect(mlir.contains("stablehlo.maximum")) // ReLU is implemented as max(x, 0)
    }

    @Test("Reduce sum")
    func reduceSum() {
        // Create input constant
        let id1 = TensorRegistry.shared.nextTensorId()
        let handle1 = LazyTensorHandle(id: id1, shape: [2, 3, 4], dtype: .float32, device: .default)
        handle1.irNode = .constant(values: Array(repeating: 1.0, count: 24), shape: [2, 3, 4])

        // Create reduce sum operation
        let id2 = TensorRegistry.shared.nextTensorId()
        let handle2 = LazyTensorHandle(id: id2, shape: [], dtype: .float32, device: .default)
        handle2.irNode = .operation(op: .reduceSum, inputs: [handle1], attributes: ["axes": [0, 1, 2]])

        // Build graph
        let graph = IRGraph()
        graph.addOutput(handle2)

        // Emit MLIR
        let mlir = graph.emitStableHLO(name: "test_reduce")

        // Verify MLIR structure
        #expect(mlir.contains("stablehlo.reduce"))
        #expect(mlir.contains("stablehlo.add"))
    }

    @Test("Graph hash")
    func graphHash() {
        // Create a simple graph
        let id1 = TensorRegistry.shared.nextTensorId()
        let handle1 = LazyTensorHandle(id: id1, shape: [2, 3], dtype: .float32, device: .default)
        handle1.irNode = .constant(values: [1, 2, 3, 4, 5, 6], shape: [2, 3])

        let graph1 = IRGraph()
        graph1.addOutput(handle1)
        graph1.buildTopologicalOrder()

        // Create another identical graph
        let id2 = TensorRegistry.shared.nextTensorId()
        let handle2 = LazyTensorHandle(id: id2, shape: [2, 3], dtype: .float32, device: .default)
        handle2.irNode = .constant(values: [1, 2, 3, 4, 5, 6], shape: [2, 3])

        let graph2 = IRGraph()
        graph2.addOutput(handle2)
        graph2.buildTopologicalOrder()

        // Hashes should be stable
        let hash1 = graph1.computeHash()
        let hash2 = graph2.computeHash()

        #expect(!hash1.isEmpty)
        #expect(!hash2.isEmpty)
        // Note: Due to different tensor IDs, hashes might differ
    }

    @Test("Topological sort")
    func topologicalSort() {
        // Create a diamond-shaped graph:
        //     A
        //    / \
        //   B   C
        //    \ /
        //     D

        let idA = TensorRegistry.shared.nextTensorId()
        let handleA = LazyTensorHandle(id: idA, shape: [2, 3], dtype: .float32, device: .default)
        handleA.irNode = .constant(values: [1, 2, 3, 4, 5, 6], shape: [2, 3])

        let idB = TensorRegistry.shared.nextTensorId()
        let handleB = LazyTensorHandle(id: idB, shape: [2, 3], dtype: .float32, device: .default)
        handleB.irNode = .operation(op: .relu, inputs: [handleA], attributes: [:])

        let idC = TensorRegistry.shared.nextTensorId()
        let handleC = LazyTensorHandle(id: idC, shape: [2, 3], dtype: .float32, device: .default)
        handleC.irNode = .operation(op: .negate, inputs: [handleA], attributes: [:])

        let idD = TensorRegistry.shared.nextTensorId()
        let handleD = LazyTensorHandle(id: idD, shape: [2, 3], dtype: .float32, device: .default)
        handleD.irNode = .operation(op: .add, inputs: [handleB, handleC], attributes: [:])

        let graph = IRGraph()
        graph.addOutput(handleD)
        graph.buildTopologicalOrder()

        // Verify topological order
        #expect(graph.nodes.count == 4)

        // A must come before B, C, and D
        let idxA = graph.nodes.firstIndex(where: { $0.id == idA })!
        let idxB = graph.nodes.firstIndex(where: { $0.id == idB })!
        let idxC = graph.nodes.firstIndex(where: { $0.id == idC })!
        let idxD = graph.nodes.firstIndex(where: { $0.id == idD })!

        #expect(idxA < idxB)
        #expect(idxA < idxC)
        #expect(idxB < idxD)
        #expect(idxC < idxD)
    }
}
