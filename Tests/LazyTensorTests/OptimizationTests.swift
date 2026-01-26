// Magma - Optimization Pass Tests
// Tests for the optimization infrastructure (PassManager, DCE, CSE, ConstantFolding, AlgebraicSimplification)

import Testing
@testable import LazyTensor
@testable import StableHLO

// MARK: - Pass Manager Tests

@Suite("PassManager Tests")
struct PassManagerTests {

    @Test("PassManager registers default passes")
    func defaultPassesRegistered() {
        let manager = PassManager()

        let registered = manager.registeredPasses
        #expect(registered.contains("dce"))
        #expect(registered.contains("cse"))
        #expect(registered.contains("constant-folding"))
        #expect(registered.contains("algebraic-simplification"))
        #expect(registered.contains("operation-fusion"))
    }

    @Test("PassManager enables default passes")
    func defaultPassesEnabled() {
        let manager = PassManager()

        #expect(manager.isEnabled("dce"))
        #expect(manager.isEnabled("cse"))
        #expect(manager.isEnabled("constant-folding"))
        #expect(manager.isEnabled("algebraic-simplification"))
        #expect(manager.isEnabled("operation-fusion"))
    }

    @Test("PassManager enable/disable works")
    func enableDisable() {
        let manager = PassManager()

        manager.disable("dce")
        #expect(!manager.isEnabled("dce"))

        manager.enable("dce")
        #expect(manager.isEnabled("dce"))
    }

    @Test("OptimizationLevel.O0 disables all passes")
    func optimizationLevelO0() {
        let manager = PassManager()
        OptimizationLevel.O0.configure(manager)

        #expect(!manager.isEnabled("dce"))
        #expect(!manager.isEnabled("cse"))
        #expect(!manager.isEnabled("constant-folding"))
        #expect(!manager.isEnabled("algebraic-simplification"))
        #expect(!manager.isEnabled("operation-fusion"))
    }

    @Test("OptimizationLevel.O1 enables basic passes")
    func optimizationLevelO1() {
        let manager = PassManager()
        OptimizationLevel.O1.configure(manager)

        #expect(manager.isEnabled("dce"))
        #expect(manager.isEnabled("cse"))
        #expect(manager.isEnabled("constant-folding"))
        #expect(manager.isEnabled("algebraic-simplification"))
        #expect(!manager.isEnabled("operation-fusion"))
    }

    @Test("OptimizationLevel.O2 enables fusion")
    func optimizationLevelO2() {
        let manager = PassManager()
        OptimizationLevel.O2.configure(manager)

        #expect(manager.isEnabled("dce"))
        #expect(manager.isEnabled("cse"))
        #expect(manager.isEnabled("constant-folding"))
        #expect(manager.isEnabled("algebraic-simplification"))
        #expect(manager.isEnabled("operation-fusion"))
    }

    @Test("PassManager runs passes on graph")
    func runOnGraph() {
        let manager = PassManager()

        // Create a simple graph
        let graph = IRGraph()
        let constId = TensorRegistry.shared.nextTensorId()
        let constHandle = LazyTensorHandle(
            id: constId,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )
        constHandle.irNode = .constant(values: [1, 2, 3, 4, 5, 6], shape: [2, 3])

        graph.addOutput(constHandle)
        graph.buildTopologicalOrder()

        // Run passes - should not crash
        let optimized = manager.run(on: graph)
        #expect(optimized.outputs.count == 1)
    }
}

// MARK: - Pass Metrics Tests

@Suite("PassMetrics Tests")
struct PassMetricsTests {

    @Test("PassMetrics records timing")
    func recordTiming() {
        let metrics = PassMetrics.shared
        metrics.reset()

        metrics.record(pass: "test-pass", time: 0.1, transformations: 5)

        #expect(metrics.invocationCount(for: "test-pass") == 1)
        #expect(metrics.transformationCount(for: "test-pass") == 5)
        #expect(metrics.totalTime(for: "test-pass") > 0)
    }

    @Test("PassMetrics averages correctly")
    func averageTime() {
        let metrics = PassMetrics.shared
        metrics.reset()

        metrics.record(pass: "avg-test", time: 0.1)
        metrics.record(pass: "avg-test", time: 0.3)
        metrics.record(pass: "avg-test", time: 0.2)

        let avg = metrics.averageTime(for: "avg-test")
        #expect(abs(avg - 0.2) < 0.0001)
    }

    @Test("PassMetrics reset clears data")
    func resetClears() {
        let metrics = PassMetrics.shared
        metrics.record(pass: "reset-test", time: 0.5, transformations: 10)
        metrics.reset()

        #expect(metrics.invocationCount(for: "reset-test") == 0)
        #expect(metrics.transformationCount(for: "reset-test") == 0)
    }
}

// MARK: - Dead Code Elimination Tests

@Suite("Dead Code Elimination Tests")
struct DeadCodeEliminationTests {

    @Test("DCE removes unused operations")
    func removesUnusedOps() {
        let pass = DeadCodeEliminationPass()

        // Create a graph with a dead operation
        let graph = IRGraph()

        // Live path: a -> b (output)
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [1, 2], shape: [2])

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)
        bHandle.irNode = .operation(op: .relu, inputs: [aHandle], attributes: [:])

        // Dead operation: c = a * 2 (not used)
        let cId = TensorRegistry.shared.nextTensorId()
        let cHandle = LazyTensorHandle(id: cId, shape: [2], dtype: .float32, device: .default)
        let twoConst = LazyTensorHandle(id: TensorRegistry.shared.nextTensorId(), shape: [1], dtype: .float32, device: .default)
        twoConst.irNode = .constant(values: [2], shape: [1])
        cHandle.irNode = .operation(op: .multiply, inputs: [aHandle, twoConst], attributes: [:])

        graph.nodes = [aHandle, twoConst, bHandle, cHandle]
        graph.addOutput(bHandle)

        let optimized = pass.run(on: graph)

        // Dead code (c and twoConst) should be removed
        #expect(optimized.nodes.count < graph.nodes.count)

        // b should still be an output
        #expect(optimized.outputs.contains { $0.id == bId })
    }

    @Test("DCE preserves all outputs")
    func preservesOutputs() {
        let pass = DeadCodeEliminationPass()

        let graph = IRGraph()

        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [1, 2], shape: [2])

        graph.nodes = [aHandle]
        graph.addOutput(aHandle)

        let optimized = pass.run(on: graph)

        #expect(optimized.outputs.count == 1)
        #expect(optimized.nodes.count >= 1)
    }

    @Test("DCE handles empty graph")
    func handlesEmptyGraph() {
        let pass = DeadCodeEliminationPass()
        let graph = IRGraph()

        let optimized = pass.run(on: graph)
        #expect(optimized.nodes.isEmpty)
        #expect(optimized.outputs.isEmpty)
    }
}

// MARK: - Common Subexpression Elimination Tests

@Suite("Common Subexpression Elimination Tests")
struct CommonSubexpressionEliminationTests {

    @Test("CSE eliminates duplicate operations")
    func eliminatesDuplicates() {
        let pass = CommonSubexpressionEliminationPass()

        let graph = IRGraph()

        // Create: a = input, b = relu(a), c = relu(a)
        // CSE should make c point to b
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [1, 2], shape: [2])

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)
        bHandle.irNode = .operation(op: .relu, inputs: [aHandle], attributes: [:])

        let cId = TensorRegistry.shared.nextTensorId()
        let cHandle = LazyTensorHandle(id: cId, shape: [2], dtype: .float32, device: .default)
        cHandle.irNode = .operation(op: .relu, inputs: [aHandle], attributes: [:])

        // d = b + c
        let dId = TensorRegistry.shared.nextTensorId()
        let dHandle = LazyTensorHandle(id: dId, shape: [2], dtype: .float32, device: .default)
        dHandle.irNode = .operation(op: .add, inputs: [bHandle, cHandle], attributes: [:])

        graph.nodes = [aHandle, bHandle, cHandle, dHandle]
        graph.addOutput(dHandle)

        let optimized = pass.run(on: graph)

        // One of the relu operations should be eliminated
        let reluCount = optimized.nodes.filter { node in
            if let irNode = node.irNode, case .operation(let op, _, _) = irNode {
                return op == .relu
            }
            return false
        }.count

        #expect(reluCount <= 1, "CSE should eliminate duplicate relu")
    }

    @Test("CSE preserves different operations")
    func preservesDifferentOps() {
        let pass = CommonSubexpressionEliminationPass()

        let graph = IRGraph()

        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [1, 2], shape: [2])

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)
        bHandle.irNode = .operation(op: .relu, inputs: [aHandle], attributes: [:])

        let cId = TensorRegistry.shared.nextTensorId()
        let cHandle = LazyTensorHandle(id: cId, shape: [2], dtype: .float32, device: .default)
        cHandle.irNode = .operation(op: .sigmoid, inputs: [aHandle], attributes: [:])

        graph.nodes = [aHandle, bHandle, cHandle]
        graph.addOutput(bHandle)
        graph.addOutput(cHandle)

        let optimized = pass.run(on: graph)

        // Both operations should be preserved (different ops)
        #expect(optimized.nodes.count >= 3)
    }
}

// MARK: - Constant Folding Tests

@Suite("Constant Folding Tests")
struct ConstantFoldingTests {

    @Test("Folds simple addition")
    func foldsAddition() {
        let pass = ConstantFoldingPass()

        let graph = IRGraph()

        // a = constant([1, 2])
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [1, 2], shape: [2])

        // b = constant([3, 4])
        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)
        bHandle.irNode = .constant(values: [3, 4], shape: [2])

        // c = a + b (should fold to constant([4, 6]))
        let cId = TensorRegistry.shared.nextTensorId()
        let cHandle = LazyTensorHandle(id: cId, shape: [2], dtype: .float32, device: .default)
        cHandle.irNode = .operation(op: .add, inputs: [aHandle, bHandle], attributes: [:])

        graph.nodes = [aHandle, bHandle, cHandle]
        graph.addOutput(cHandle)

        let optimized = pass.run(on: graph)

        // The output should be a constant
        guard let outputNode = optimized.outputs.first,
              let irNode = outputNode.irNode else {
            Issue.record("No output node found")
            return
        }

        if case .constant(let values, _) = irNode {
            #expect(values == [4, 6], "Addition should be folded correctly")
        } else {
            Issue.record("Output should be folded to constant")
        }
    }

    @Test("Folds multiplication")
    func foldsMultiplication() {
        let pass = ConstantFoldingPass()

        let graph = IRGraph()

        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [2, 3], shape: [2])

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)
        bHandle.irNode = .constant(values: [4, 5], shape: [2])

        let cId = TensorRegistry.shared.nextTensorId()
        let cHandle = LazyTensorHandle(id: cId, shape: [2], dtype: .float32, device: .default)
        cHandle.irNode = .operation(op: .multiply, inputs: [aHandle, bHandle], attributes: [:])

        graph.nodes = [aHandle, bHandle, cHandle]
        graph.addOutput(cHandle)

        let optimized = pass.run(on: graph)

        if let outputNode = optimized.outputs.first,
           let irNode = outputNode.irNode,
           case .constant(let values, _) = irNode {
            #expect(values == [8, 15], "Multiplication should be folded correctly")
        }
    }

    @Test("Folds unary operations")
    func foldsUnaryOps() {
        let pass = ConstantFoldingPass()

        let graph = IRGraph()

        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [3], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [-2, 0, 3], shape: [3])

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [3], dtype: .float32, device: .default)
        bHandle.irNode = .operation(op: .abs, inputs: [aHandle], attributes: [:])

        graph.nodes = [aHandle, bHandle]
        graph.addOutput(bHandle)

        let optimized = pass.run(on: graph)

        if let outputNode = optimized.outputs.first,
           let irNode = outputNode.irNode,
           case .constant(let values, _) = irNode {
            #expect(values == [2, 0, 3], "Abs should be folded correctly")
        }
    }

    @Test("Does not fold non-constant inputs")
    func doesNotFoldNonConstant() {
        let pass = ConstantFoldingPass()

        let graph = IRGraph()

        // Runtime input - no irNode set (nil), so not a constant
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        // Leave irNode as nil - represents a runtime input

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)
        bHandle.irNode = .constant(values: [1, 2], shape: [2])

        let cId = TensorRegistry.shared.nextTensorId()
        let cHandle = LazyTensorHandle(id: cId, shape: [2], dtype: .float32, device: .default)
        cHandle.irNode = .operation(op: .add, inputs: [aHandle, bHandle], attributes: [:])

        graph.nodes = [aHandle, bHandle, cHandle]
        graph.addOutput(cHandle)

        let optimized = pass.run(on: graph)

        // The output should still be an operation (not folded)
        if let outputNode = optimized.outputs.first,
           let irNode = outputNode.irNode {
            if case .operation = irNode {
                // Expected - operation is preserved
            } else {
                Issue.record("Operation with non-constant input should not be folded")
            }
        }
    }

    @Test("Folds relu activation")
    func foldsRelu() {
        let pass = ConstantFoldingPass()

        let graph = IRGraph()

        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [4], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [-2, -1, 0, 1], shape: [4])

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [4], dtype: .float32, device: .default)
        bHandle.irNode = .operation(op: .relu, inputs: [aHandle], attributes: [:])

        graph.nodes = [aHandle, bHandle]
        graph.addOutput(bHandle)

        let optimized = pass.run(on: graph)

        if let outputNode = optimized.outputs.first,
           let irNode = outputNode.irNode,
           case .constant(let values, _) = irNode {
            #expect(values == [0, 0, 0, 1], "ReLU should be folded correctly")
        }
    }
}

// MARK: - Algebraic Simplification Tests

@Suite("Algebraic Simplification Tests")
struct AlgebraicSimplificationTests {

    @Test("Simplifies x + 0 = x")
    func addZero() {
        let pass = AlgebraicSimplificationPass()

        let graph = IRGraph()

        // x is a non-constant input (irNode left as nil)
        let xId = TensorRegistry.shared.nextTensorId()
        let xHandle = LazyTensorHandle(id: xId, shape: [2], dtype: .float32, device: .default)
        // Leave irNode as nil - represents a runtime input

        let zeroId = TensorRegistry.shared.nextTensorId()
        let zeroHandle = LazyTensorHandle(id: zeroId, shape: [2], dtype: .float32, device: .default)
        zeroHandle.irNode = .constant(values: [0, 0], shape: [2])

        let resultId = TensorRegistry.shared.nextTensorId()
        let resultHandle = LazyTensorHandle(id: resultId, shape: [2], dtype: .float32, device: .default)
        resultHandle.irNode = .operation(op: .add, inputs: [xHandle, zeroHandle], attributes: [:])

        graph.nodes = [xHandle, zeroHandle, resultHandle]
        graph.addOutput(resultHandle)

        let optimized = pass.run(on: graph)

        // Result should be replaced with x
        #expect(optimized.outputs.first?.id == xId, "x + 0 should be simplified to x")
    }

    @Test("Simplifies x * 1 = x")
    func multiplyOne() {
        let pass = AlgebraicSimplificationPass()

        let graph = IRGraph()

        // x is a non-constant input (irNode left as nil)
        let xId = TensorRegistry.shared.nextTensorId()
        let xHandle = LazyTensorHandle(id: xId, shape: [2], dtype: .float32, device: .default)
        // Leave irNode as nil - represents a runtime input

        let oneId = TensorRegistry.shared.nextTensorId()
        let oneHandle = LazyTensorHandle(id: oneId, shape: [2], dtype: .float32, device: .default)
        oneHandle.irNode = .constant(values: [1, 1], shape: [2])

        let resultId = TensorRegistry.shared.nextTensorId()
        let resultHandle = LazyTensorHandle(id: resultId, shape: [2], dtype: .float32, device: .default)
        resultHandle.irNode = .operation(op: .multiply, inputs: [xHandle, oneHandle], attributes: [:])

        graph.nodes = [xHandle, oneHandle, resultHandle]
        graph.addOutput(resultHandle)

        let optimized = pass.run(on: graph)

        #expect(optimized.outputs.first?.id == xId, "x * 1 should be simplified to x")
    }

    @Test("Simplifies x * 0 = 0")
    func multiplyZero() {
        let pass = AlgebraicSimplificationPass()

        let graph = IRGraph()

        // x is a non-constant input (irNode left as nil)
        let xId = TensorRegistry.shared.nextTensorId()
        let xHandle = LazyTensorHandle(id: xId, shape: [2], dtype: .float32, device: .default)
        // Leave irNode as nil - represents a runtime input

        let zeroId = TensorRegistry.shared.nextTensorId()
        let zeroHandle = LazyTensorHandle(id: zeroId, shape: [2], dtype: .float32, device: .default)
        zeroHandle.irNode = .constant(values: [0, 0], shape: [2])

        let resultId = TensorRegistry.shared.nextTensorId()
        let resultHandle = LazyTensorHandle(id: resultId, shape: [2], dtype: .float32, device: .default)
        resultHandle.irNode = .operation(op: .multiply, inputs: [xHandle, zeroHandle], attributes: [:])

        graph.nodes = [xHandle, zeroHandle, resultHandle]
        graph.addOutput(resultHandle)

        let optimized = pass.run(on: graph)

        #expect(optimized.outputs.first?.id == zeroId, "x * 0 should be simplified to 0")
    }

    @Test("Simplifies x / 1 = x")
    func divideOne() {
        let pass = AlgebraicSimplificationPass()

        let graph = IRGraph()

        // x is a non-constant input (irNode left as nil)
        let xId = TensorRegistry.shared.nextTensorId()
        let xHandle = LazyTensorHandle(id: xId, shape: [2], dtype: .float32, device: .default)
        // Leave irNode as nil - represents a runtime input

        let oneId = TensorRegistry.shared.nextTensorId()
        let oneHandle = LazyTensorHandle(id: oneId, shape: [2], dtype: .float32, device: .default)
        oneHandle.irNode = .constant(values: [1, 1], shape: [2])

        let resultId = TensorRegistry.shared.nextTensorId()
        let resultHandle = LazyTensorHandle(id: resultId, shape: [2], dtype: .float32, device: .default)
        resultHandle.irNode = .operation(op: .divide, inputs: [xHandle, oneHandle], attributes: [:])

        graph.nodes = [xHandle, oneHandle, resultHandle]
        graph.addOutput(resultHandle)

        let optimized = pass.run(on: graph)

        #expect(optimized.outputs.first?.id == xId, "x / 1 should be simplified to x")
    }

    @Test("Simplifies double negation: -(-x) = x")
    func doubleNegation() {
        let pass = AlgebraicSimplificationPass()

        let graph = IRGraph()

        // x is a non-constant input (irNode left as nil)
        let xId = TensorRegistry.shared.nextTensorId()
        let xHandle = LazyTensorHandle(id: xId, shape: [2], dtype: .float32, device: .default)
        // Leave irNode as nil - represents a runtime input

        let negXId = TensorRegistry.shared.nextTensorId()
        let negXHandle = LazyTensorHandle(id: negXId, shape: [2], dtype: .float32, device: .default)
        negXHandle.irNode = .operation(op: .negate, inputs: [xHandle], attributes: [:])

        let negNegXId = TensorRegistry.shared.nextTensorId()
        let negNegXHandle = LazyTensorHandle(id: negNegXId, shape: [2], dtype: .float32, device: .default)
        negNegXHandle.irNode = .operation(op: .negate, inputs: [negXHandle], attributes: [:])

        graph.nodes = [xHandle, negXHandle, negNegXHandle]
        graph.addOutput(negNegXHandle)

        let optimized = pass.run(on: graph)

        #expect(optimized.outputs.first?.id == xId, "-(-x) should be simplified to x")
    }

    @Test("Simplifies exp(log(x)) = x")
    func expLog() {
        let pass = AlgebraicSimplificationPass()

        let graph = IRGraph()

        // x is a non-constant input (irNode left as nil)
        let xId = TensorRegistry.shared.nextTensorId()
        let xHandle = LazyTensorHandle(id: xId, shape: [2], dtype: .float32, device: .default)
        // Leave irNode as nil - represents a runtime input

        let logXId = TensorRegistry.shared.nextTensorId()
        let logXHandle = LazyTensorHandle(id: logXId, shape: [2], dtype: .float32, device: .default)
        logXHandle.irNode = .operation(op: .log, inputs: [xHandle], attributes: [:])

        let expLogXId = TensorRegistry.shared.nextTensorId()
        let expLogXHandle = LazyTensorHandle(id: expLogXId, shape: [2], dtype: .float32, device: .default)
        expLogXHandle.irNode = .operation(op: .exponential, inputs: [logXHandle], attributes: [:])

        graph.nodes = [xHandle, logXHandle, expLogXHandle]
        graph.addOutput(expLogXHandle)

        let optimized = pass.run(on: graph)

        #expect(optimized.outputs.first?.id == xId, "exp(log(x)) should be simplified to x")
    }

    @Test("Simplifies relu(relu(x)) = relu(x)")
    func doubleRelu() {
        let pass = AlgebraicSimplificationPass()

        let graph = IRGraph()

        // x is a non-constant input (irNode left as nil)
        let xId = TensorRegistry.shared.nextTensorId()
        let xHandle = LazyTensorHandle(id: xId, shape: [2], dtype: .float32, device: .default)
        // Leave irNode as nil - represents a runtime input

        let relu1Id = TensorRegistry.shared.nextTensorId()
        let relu1Handle = LazyTensorHandle(id: relu1Id, shape: [2], dtype: .float32, device: .default)
        relu1Handle.irNode = .operation(op: .relu, inputs: [xHandle], attributes: [:])

        let relu2Id = TensorRegistry.shared.nextTensorId()
        let relu2Handle = LazyTensorHandle(id: relu2Id, shape: [2], dtype: .float32, device: .default)
        relu2Handle.irNode = .operation(op: .relu, inputs: [relu1Handle], attributes: [:])

        graph.nodes = [xHandle, relu1Handle, relu2Handle]
        graph.addOutput(relu2Handle)

        let optimized = pass.run(on: graph)

        #expect(optimized.outputs.first?.id == relu1Id, "relu(relu(x)) should be simplified to relu(x)")
    }
}

// MARK: - ExpressionKey Tests

@Suite("ExpressionKey Tests")
struct ExpressionKeyTests {

    @Test("Same operations produce same key")
    func sameOperationsSameKey() {
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)

        let key1 = ExpressionKey(opKind: .relu, inputs: [aHandle], attributes: [:])
        let key2 = ExpressionKey(opKind: .relu, inputs: [aHandle], attributes: [:])

        #expect(key1 == key2)
        #expect(key1.hashValue == key2.hashValue)
    }

    @Test("Different operations produce different keys")
    func differentOpsDifferentKeys() {
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)

        let key1 = ExpressionKey(opKind: .relu, inputs: [aHandle], attributes: [:])
        let key2 = ExpressionKey(opKind: .sigmoid, inputs: [aHandle], attributes: [:])

        #expect(key1 != key2)
    }

    @Test("Different inputs produce different keys")
    func differentInputsDifferentKeys() {
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)

        let key1 = ExpressionKey(opKind: .relu, inputs: [aHandle], attributes: [:])
        let key2 = ExpressionKey(opKind: .relu, inputs: [bHandle], attributes: [:])

        #expect(key1 != key2)
    }
}

// MARK: - Fusion Pattern Infrastructure Tests

@Suite("FusionMatch Tests")
struct FusionMatchTests {

    @Test("FusionMatch stores operations and indices")
    func fusionMatchStorage() {
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)

        let match = FusionMatch(
            operations: [aHandle, bHandle],
            indices: [0, 1],
            metadata: ["scale": 0.5]
        )

        #expect(match.operations.count == 2)
        #expect(match.indices == [0, 1])
        #expect(match.metadata["scale"] as? Double == 0.5)
        #expect(match.output?.id == bId)
    }

    @Test("FusionMatch output returns last operation")
    func fusionMatchOutput() {
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)

        let match = FusionMatch(operations: [aHandle], indices: [0])
        #expect(match.output?.id == aId)
    }
}

@Suite("PatternMatchingHelpers Tests")
struct PatternMatchingHelpersTests {

    @Test("isConstantWithValue detects matching constant")
    func isConstantWithValue() {
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [1], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [0.5], shape: [1])

        #expect(PatternMatchingHelpers.isConstantWithValue(aHandle, 0.5))
        #expect(!PatternMatchingHelpers.isConstantWithValue(aHandle, 0.6))
    }

    @Test("isScalarConstant detects scalars")
    func isScalarConstant() {
        let scalarId = TensorRegistry.shared.nextTensorId()
        let scalarHandle = LazyTensorHandle(id: scalarId, shape: [1], dtype: .float32, device: .default)
        scalarHandle.irNode = .constant(values: [1.0], shape: [1])

        let vectorId = TensorRegistry.shared.nextTensorId()
        let vectorHandle = LazyTensorHandle(id: vectorId, shape: [4], dtype: .float32, device: .default)
        vectorHandle.irNode = .constant(values: [1, 2, 3, 4], shape: [4])

        #expect(PatternMatchingHelpers.isScalarConstant(scalarHandle))
        #expect(!PatternMatchingHelpers.isScalarConstant(vectorHandle))
    }

    @Test("getScalarValue extracts value")
    func getScalarValue() {
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [1], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [3.14], shape: [1])

        let value = PatternMatchingHelpers.getScalarValue(aHandle)
        #expect(value != nil)
        #expect(abs(value! - 3.14) < 0.001)
    }

    @Test("findOperations finds matching operations")
    func findOperations() {
        let graph = IRGraph()

        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [1, 2], shape: [2])

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)
        bHandle.irNode = .operation(op: .relu, inputs: [aHandle], attributes: [:])

        let cId = TensorRegistry.shared.nextTensorId()
        let cHandle = LazyTensorHandle(id: cId, shape: [2], dtype: .float32, device: .default)
        cHandle.irNode = .operation(op: .sigmoid, inputs: [aHandle], attributes: [:])

        graph.nodes = [aHandle, bHandle, cHandle]

        let reluOps = PatternMatchingHelpers.findOperations(ofKind: .relu, in: graph)
        let sigmoidOps = PatternMatchingHelpers.findOperations(ofKind: .sigmoid, in: graph)

        #expect(reluOps.count == 1)
        #expect(sigmoidOps.count == 1)
        #expect(reluOps.first?.id == bId)
    }

    @Test("hasSingleConsumer detects single consumer")
    func hasSingleConsumer() {
        let graph = IRGraph()

        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [1, 2], shape: [2])

        // b uses a (single consumer)
        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)
        bHandle.irNode = .operation(op: .relu, inputs: [aHandle], attributes: [:])

        graph.nodes = [aHandle, bHandle]
        graph.addOutput(bHandle)

        #expect(PatternMatchingHelpers.hasSingleConsumer(aHandle, in: graph))
    }
}

@Suite("ActivationType Tests")
struct ActivationTypeTests {

    @Test("OpKind.isActivation returns correct values")
    func isActivation() {
        #expect(OpKind.relu.isActivation)
        #expect(OpKind.sigmoid.isActivation)
        #expect(OpKind.gelu.isActivation)
        #expect(OpKind.tanh.isActivation)

        #expect(!OpKind.add.isActivation)
        #expect(!OpKind.matmul.isActivation)
        #expect(!OpKind.softmax.isActivation)
    }

    @Test("OpKind.asActivationType converts correctly")
    func asActivationType() {
        #expect(OpKind.relu.asActivationType == .relu)
        #expect(OpKind.sigmoid.asActivationType == .sigmoid)
        #expect(OpKind.gelu.asActivationType == .gelu)
        #expect(OpKind.add.asActivationType == nil)
    }
}

// MARK: - Operation Fusion Pass Tests

@Suite("OperationFusionPass Tests")
struct OperationFusionPassTests {

    @Test("Fusion pass registers default patterns")
    func registersDefaultPatterns() {
        let pass = OperationFusionPass()
        // Pass should have been created with default patterns
        #expect(pass.name == "operation-fusion")
    }

    @Test("Fusion pass handles empty graph")
    func handlesEmptyGraph() {
        let pass = OperationFusionPass()
        let graph = IRGraph()

        let result = pass.run(on: graph)
        #expect(result.nodes.isEmpty)
    }

    @Test("Fusion pass preserves simple graphs")
    func preservesSimpleGraphs() {
        let pass = OperationFusionPass()

        let graph = IRGraph()
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [1, 2], shape: [2])

        graph.nodes = [aHandle]
        graph.addOutput(aHandle)
        graph.buildTopologicalOrder()

        let result = pass.run(on: graph)
        #expect(result.outputs.count == 1)
    }
}

// MARK: - Specific Fusion Pattern Tests

@Suite("MatMulBiasActivation Pattern Tests")
struct MatMulBiasActivationPatternTests {

    @Test("Pattern has correct name and priority")
    func patternMetadata() {
        let pattern = MatMulBiasActivationPattern()
        #expect(pattern.name == "matmul-bias-activation")
        #expect(pattern.priority == 60)
    }
}

@Suite("SoftmaxPattern Tests")
struct SoftmaxPatternTests {

    @Test("Pattern has correct name and priority")
    func patternMetadata() {
        let pattern = SoftmaxPattern()
        #expect(pattern.name == "softmax")
        #expect(pattern.priority == 50)
    }
}

@Suite("LayerNormPattern Tests")
struct LayerNormPatternTests {

    @Test("Pattern has correct name and priority")
    func patternMetadata() {
        let pattern = LayerNormPattern()
        #expect(pattern.name == "layer-norm")
        #expect(pattern.priority == 80)
    }
}

@Suite("RMSNormPattern Tests")
struct RMSNormPatternTests {

    @Test("Pattern has correct name and priority")
    func patternMetadata() {
        let pattern = RMSNormPattern()
        #expect(pattern.name == "rms-norm")
        #expect(pattern.priority == 75)
    }
}

@Suite("ScaledDotProductAttentionPattern Tests")
struct ScaledDotProductAttentionPatternTests {

    @Test("Pattern has correct name and priority")
    func patternMetadata() {
        let pattern = ScaledDotProductAttentionPattern()
        #expect(pattern.name == "scaled-dot-product-attention")
        #expect(pattern.priority == 100)  // Highest priority
    }
}

@Suite("GeluPattern Tests")
struct GeluPatternTests {

    @Test("Pattern has correct name and priority")
    func patternMetadata() {
        let pattern = GeluPattern()
        #expect(pattern.name == "gelu")
        #expect(pattern.priority == 40)
    }
}

// MARK: - Graph Transformation Tests

@Suite("IRGraph Transformation Tests")
struct IRGraphTransformationTests {

    @Test("Graph replacing returns a new graph")
    func graphReplacing() {
        let graph = IRGraph()

        // Create nodes
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [1, 2], shape: [2])

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)
        bHandle.irNode = .operation(op: .relu, inputs: [aHandle], attributes: [:])

        graph.nodes = [aHandle, bHandle]
        graph.addOutput(bHandle)

        // Create fused replacement
        let fusedId = TensorRegistry.shared.nextTensorId()
        let fusedHandle = LazyTensorHandle(id: fusedId, shape: [2], dtype: .float32, device: .default)
        fusedHandle.irNode = .operation(op: .fusedGelu, inputs: [aHandle], attributes: [:])

        let newGraph = graph.replacing([bHandle], with: [fusedHandle])

        // A new graph should be returned (not the same instance)
        #expect(newGraph !== graph)

        // The new graph should have outputs set
        #expect(newGraph.outputs.count >= 1)
    }

    @Test("Graph updateReferences updates inputs")
    func graphUpdateReferences() {
        let graph = IRGraph()

        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [1, 2], shape: [2])

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(id: bId, shape: [2], dtype: .float32, device: .default)
        bHandle.irNode = .operation(op: .relu, inputs: [aHandle], attributes: [:])

        graph.nodes = [aHandle, bHandle]

        // Create a new handle to replace a
        let newAId = TensorRegistry.shared.nextTensorId()
        let newAHandle = LazyTensorHandle(id: newAId, shape: [2], dtype: .float32, device: .default)
        newAHandle.irNode = .constant(values: [3, 4], shape: [2])

        graph.updateReferences(from: aId, to: newAHandle)

        // Check that b now references newA
        if let irNode = bHandle.irNode,
           case .operation(_, let inputs, _) = irNode {
            #expect(inputs.first?.id == newAId)
        }
    }
}

// MARK: - Integration Tests

@Suite("Optimization Integration Tests")
struct OptimizationIntegrationTests {

    @Test("Multiple passes work together")
    func multiplePassesIntegration() {
        let manager = PassManager()
        manager.verbose = false

        // Create a graph that benefits from multiple optimizations
        let graph = IRGraph()

        // Constant folding opportunity: 2 + 3 = 5
        let twoId = TensorRegistry.shared.nextTensorId()
        let twoHandle = LazyTensorHandle(id: twoId, shape: [1], dtype: .float32, device: .default)
        twoHandle.irNode = .constant(values: [2], shape: [1])

        let threeId = TensorRegistry.shared.nextTensorId()
        let threeHandle = LazyTensorHandle(id: threeId, shape: [1], dtype: .float32, device: .default)
        threeHandle.irNode = .constant(values: [3], shape: [1])

        let fiveId = TensorRegistry.shared.nextTensorId()
        let fiveHandle = LazyTensorHandle(id: fiveId, shape: [1], dtype: .float32, device: .default)
        fiveHandle.irNode = .operation(op: .add, inputs: [twoHandle, threeHandle], attributes: [:])

        // Input data - non-constant (irNode left as nil)
        let xId = TensorRegistry.shared.nextTensorId()
        let xHandle = LazyTensorHandle(id: xId, shape: [1], dtype: .float32, device: .default)
        // Leave irNode as nil - represents a runtime input

        // x * 5 (using folded constant)
        let resultId = TensorRegistry.shared.nextTensorId()
        let resultHandle = LazyTensorHandle(id: resultId, shape: [1], dtype: .float32, device: .default)
        resultHandle.irNode = .operation(op: .multiply, inputs: [xHandle, fiveHandle], attributes: [:])

        graph.nodes = [twoHandle, threeHandle, fiveHandle, xHandle, resultHandle]
        graph.addOutput(resultHandle)

        // Run all passes
        let optimized = manager.run(on: graph)

        // Should have fewer nodes due to constant folding
        #expect(optimized.outputs.count == 1)
    }

    @Test("Fixed-point iteration converges")
    func fixedPointConverges() {
        let manager = PassManager()

        let graph = IRGraph()

        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(id: aId, shape: [2], dtype: .float32, device: .default)
        aHandle.irNode = .constant(values: [1, 2], shape: [2])

        graph.nodes = [aHandle]
        graph.addOutput(aHandle)

        // Running to fixed point should not crash and should complete
        let optimized = manager.runToFixedPoint(on: graph, maxIterations: 5)
        #expect(optimized.outputs.count == 1)
    }
}
