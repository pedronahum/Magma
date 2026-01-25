// Magma - LazyTensor Tests
// These tests verify lazy evaluation without needing XLA installed (using mocks)

import Testing
@testable import LazyTensor
@testable import StableHLO

@Suite("LazyTensorHandle Tests")
struct LazyTensorHandleTests {

    @Test("Handle creation")
    func handleCreation() {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [32, 784],
            dtype: .float32,
            device: .default
        )

        #expect(handle.shape == [32, 784])
        #expect(handle.dtype == .float32)
        #expect(!handle.isMaterialized)
    }

    @Test("Tensor registry generates unique IDs")
    func tensorRegistry() {
        let id1 = TensorRegistry.shared.nextTensorId()
        let id2 = TensorRegistry.shared.nextTensorId()

        #expect(id1 != id2)
        #expect(id2 == id1 + 1)
    }
}

@Suite("IRNode Tests")
struct IRNodeTests {

    @Test("Constant node")
    func constantNode() {
        let node = IRNode.constant(values: [1, 2, 3, 4], shape: [2, 2])

        if case .constant(let values, let shape) = node {
            #expect(values == [1, 2, 3, 4])
            #expect(shape == [2, 2])
        } else {
            Issue.record("Expected constant node")
        }
    }

    @Test("Operation node")
    func operationNode() {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )

        let node = IRNode.operation(
            op: .add,
            inputs: [handle, handle],
            attributes: [:]
        )

        if case .operation(let op, let inputs, _) = node {
            #expect(op == .add)
            #expect(inputs.count == 2)
        } else {
            Issue.record("Expected operation node")
        }
    }
}

@Suite("OpKind Tests")
struct OpKindTests {

    @Test("OpKind raw values")
    func opKindRawValues() {
        #expect(OpKind.add.rawValue == "add")
        #expect(OpKind.matmul.rawValue == "matmul")
        #expect(OpKind.relu.rawValue == "relu")
    }
}

@Suite("CompilationCache Tests")
struct CompilationCacheTests {

    @Test("Cache hit rate is valid")
    func cacheHitRate() {
        let cache = CompilationCache.shared

        #expect(cache.hitRate >= 0.0)
        #expect(cache.hitRate <= 1.0)
    }
}

@Suite("IRGraph Tests")
struct IRGraphTests {

    @Test("Empty graph")
    func emptyGraph() {
        let graph = IRGraph()
        #expect(graph.nodes.isEmpty)
        #expect(graph.outputs.isEmpty)
    }

    @Test("Add output")
    func addOutput() {
        let graph = IRGraph()
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )

        graph.addOutput(handle)
        #expect(graph.outputs.count == 1)
    }
}

// MARK: - Liveness Tracking Tests (Inspired by TensorFlow Swift)

@Suite("Liveness Tracking Tests")
struct LivenessTrackingTests {

    @Test("Handle starts as not live")
    func handleStartsAsNotLive() {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )

        #expect(!handle.isLive, "New handles should not be live by default")
    }

    @Test("Handle can be marked live")
    func handleCanBeMarkedLive() {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )

        handle.isLive = true
        #expect(handle.isLive, "Handle should be live after marking")
    }
}

// MARK: - Constant Promotion Tests (Inspired by TensorFlow Swift)

@Suite("Constant Promotion Tests")
struct ConstantPromotionTests {

    @Test("PromotedConstant creation")
    func promotedConstantCreation() {
        let promoted = PromotedConstant(
            originalNodeId: 42,
            shape: [1],
            dtype: .float32,
            values: [3.14],
            inputIndex: 0
        )

        #expect(promoted.originalNodeId == 42)
        #expect(promoted.shape == [1])
        #expect(promoted.values == [3.14])
        #expect(promoted.inputIndex == 0)
    }

    @Test("ConstantPromotionResult with promoted constants")
    func constantPromotionResult() {
        let result = ConstantPromotionResult(
            structuralHash: "abc123",
            promotedConstants: [
                PromotedConstant(originalNodeId: 1, shape: [1], dtype: .float32, values: [1.0], inputIndex: 0),
                PromotedConstant(originalNodeId: 2, shape: [1], dtype: .float32, values: [2.0], inputIndex: 1)
            ]
        )

        #expect(result.structuralHash == "abc123")
        #expect(result.wasPromoted, "Should indicate promotion was applied")
        #expect(result.promotedConstants.count == 2)
    }

    @Test("Empty promotion result")
    func emptyPromotionResult() {
        let result = ConstantPromotionResult(
            structuralHash: "xyz789",
            promotedConstants: []
        )

        #expect(!result.wasPromoted, "Should indicate no promotion applied")
    }

    @Test("Promotion threshold")
    func promotionThreshold() {
        // Verify the threshold is reasonable (16 elements)
        #expect(CompilationCache.promotionThreshold == 16)
    }

    @Test("Small constant is promoted")
    func smallConstantIsPromoted() {
        // Create a graph with a small constant that should be promoted
        let graph = IRGraph()

        let constId = TensorRegistry.shared.nextTensorId()
        let constHandle = LazyTensorHandle(
            id: constId,
            shape: [1], // Small scalar - should be promoted
            dtype: .float32,
            device: .default
        )
        constHandle.irNode = .constant(values: [0.01], shape: [1])

        let inputId = TensorRegistry.shared.nextTensorId()
        let inputHandle = LazyTensorHandle(
            id: inputId,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )
        inputHandle.irNode = .constant(values: [1, 2, 3, 4, 5, 6], shape: [2, 3])

        let outputId = TensorRegistry.shared.nextTensorId()
        let outputHandle = LazyTensorHandle(
            id: outputId,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )
        outputHandle.irNode = .operation(op: .multiply, inputs: [inputHandle, constHandle], attributes: [:])

        graph.addOutput(outputHandle)
        graph.buildTopologicalOrder()

        let result = graph.analyzeForConstantPromotion()

        // Small constants should be promoted
        #expect(result.wasPromoted, "Small constants should be promoted")

        // The scalar constant [1] should be promoted, larger [2,3] should not
        let scalarPromoted = result.promotedConstants.first { $0.shape == [1] }
        #expect(scalarPromoted != nil, "Scalar constant should be promoted")
    }

    @Test("Large constant not promoted")
    func largeConstantNotPromoted() {
        // Create a graph with a large constant that should NOT be promoted
        let graph = IRGraph()

        // Large constant (> 16 elements)
        let largeConstId = TensorRegistry.shared.nextTensorId()
        let largeConstHandle = LazyTensorHandle(
            id: largeConstId,
            shape: [10, 10], // 100 elements - too large for promotion
            dtype: .float32,
            device: .default
        )
        largeConstHandle.irNode = .constant(values: Array(repeating: Float(1.0), count: 100), shape: [10, 10])

        graph.addOutput(largeConstHandle)
        graph.buildTopologicalOrder()

        let result = graph.analyzeForConstantPromotion()

        // Large constants should NOT be promoted
        let largePromoted = result.promotedConstants.first { $0.shape == [10, 10] }
        #expect(largePromoted == nil, "Large constants should not be promoted")
    }

    @Test("Structural hash differs only by structure")
    func structuralHashDiffersOnlyByStructure() {
        // Two graphs with same structure but different constant values
        // should have the SAME structural hash (enabling cache reuse)

        func createGraphWithScalar(_ value: Float) -> IRGraph {
            let graph = IRGraph()

            let constId = TensorRegistry.shared.nextTensorId()
            let constHandle = LazyTensorHandle(
                id: constId,
                shape: [1],
                dtype: .float32,
                device: .default
            )
            constHandle.irNode = .constant(values: [value], shape: [1])

            graph.addOutput(constHandle)
            graph.buildTopologicalOrder()
            return graph
        }

        let graph1 = createGraphWithScalar(1.0)
        let graph2 = createGraphWithScalar(999.0)

        let result1 = graph1.analyzeForConstantPromotion()
        let result2 = graph2.analyzeForConstantPromotion()

        // Structural hashes should be the same (different values, same structure)
        #expect(
            result1.structuralHash == result2.structuralHash,
            "Graphs with same structure but different scalar values should have same structural hash"
        )
    }
}

// MARK: - Graph Validation Tests

@Suite("Graph Validation Tests")
struct GraphValidationTests {

    @Test("Valid graph passes")
    func validGraphPasses() throws {
        let graph = IRGraph()

        // Create a valid graph: constant -> relu -> output
        let constId = TensorRegistry.shared.nextTensorId()
        let constHandle = LazyTensorHandle(
            id: constId,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )
        constHandle.irNode = .constant(values: [1, 2, 3, 4, 5, 6], shape: [2, 3])

        let reluId = TensorRegistry.shared.nextTensorId()
        let reluHandle = LazyTensorHandle(
            id: reluId,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )
        reluHandle.irNode = .operation(op: .relu, inputs: [constHandle], attributes: [:])

        graph.addOutput(reluHandle)
        graph.buildTopologicalOrder()

        // Should not throw
        try graph.validate()
    }

    @Test("Matmul shape mismatch fails")
    func matmulShapeMismatchFails() {
        let graph = IRGraph()

        // Create invalid matmul: [2,3] @ [5,4] - inner dims don't match
        let aId = TensorRegistry.shared.nextTensorId()
        let aHandle = LazyTensorHandle(
            id: aId,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )
        aHandle.irNode = .constant(values: Array(repeating: Float(1), count: 6), shape: [2, 3])

        let bId = TensorRegistry.shared.nextTensorId()
        let bHandle = LazyTensorHandle(
            id: bId,
            shape: [5, 4], // Inner dim 5 != 3
            dtype: .float32,
            device: .default
        )
        bHandle.irNode = .constant(values: Array(repeating: Float(1), count: 20), shape: [5, 4])

        let outputId = TensorRegistry.shared.nextTensorId()
        let outputHandle = LazyTensorHandle(
            id: outputId,
            shape: [2, 4],
            dtype: .float32,
            device: .default
        )
        outputHandle.irNode = .operation(op: .matmul, inputs: [aHandle, bHandle], attributes: [:])

        graph.addOutput(outputHandle)
        graph.buildTopologicalOrder()

        // Should throw shape mismatch error
        #expect(throws: IRGraph.ValidationError.self) {
            try graph.validate()
        }
    }

    @Test("Reshape element count mismatch fails")
    func reshapeElementCountMismatchFails() {
        let graph = IRGraph()

        // Create invalid reshape: [2,3] (6 elements) -> [2,2] (4 elements)
        let inputId = TensorRegistry.shared.nextTensorId()
        let inputHandle = LazyTensorHandle(
            id: inputId,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )
        inputHandle.irNode = .constant(values: [1, 2, 3, 4, 5, 6], shape: [2, 3])

        let outputId = TensorRegistry.shared.nextTensorId()
        let outputHandle = LazyTensorHandle(
            id: outputId,
            shape: [2, 2], // 4 elements != 6 elements
            dtype: .float32,
            device: .default
        )
        outputHandle.irNode = .operation(op: .reshape, inputs: [inputHandle], attributes: ["shape": [2, 2]])

        graph.addOutput(outputHandle)
        graph.buildTopologicalOrder()

        // Should throw shape mismatch error
        #expect(throws: IRGraph.ValidationError.self) {
            try graph.validate()
        }
    }

    @Test("Constant value count mismatch fails")
    func constantValueCountMismatchFails() {
        let graph = IRGraph()

        // Create invalid constant: shape [2,3] but only 4 values
        let constId = TensorRegistry.shared.nextTensorId()
        let constHandle = LazyTensorHandle(
            id: constId,
            shape: [2, 3], // 6 elements expected
            dtype: .float32,
            device: .default
        )
        constHandle.irNode = .constant(values: [1, 2, 3, 4], shape: [2, 3]) // Only 4 values!

        graph.addOutput(constHandle)
        graph.buildTopologicalOrder()

        // Should throw shape mismatch error
        #expect(throws: (any Error).self) {
            try graph.validate()
        }
    }

    @Test("Binary op missing input fails")
    func binaryOpMissingInputFails() {
        let graph = IRGraph()

        // Create invalid add with only one input
        let inputId = TensorRegistry.shared.nextTensorId()
        let inputHandle = LazyTensorHandle(
            id: inputId,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )
        inputHandle.irNode = .constant(values: [1, 2, 3, 4, 5, 6], shape: [2, 3])

        let outputId = TensorRegistry.shared.nextTensorId()
        let outputHandle = LazyTensorHandle(
            id: outputId,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )
        // Add with only one input - invalid!
        outputHandle.irNode = .operation(op: .add, inputs: [inputHandle], attributes: [:])

        graph.addOutput(outputHandle)
        graph.buildTopologicalOrder()

        // Should throw missing input error
        #expect(throws: IRGraph.ValidationError.self) {
            try graph.validate()
        }
    }
}

// MARK: - Execution Context Tests

@Suite("Execution Context Tests")
struct ExecutionContextTests {

    @Test("Context is accessible")
    func contextIsAccessible() {
        let context = ExecutionContext.current
        #expect(context != nil)
    }

    @Test("Default settings")
    func defaultSettings() {
        let context = ExecutionContext.current
        #expect(context.shouldPromoteConstants, "Constant promotion should be enabled by default")
        #expect(!context.isShapeTrackingEnabled, "Shape tracking should be disabled by default")
    }

    @Test("Local ID generation")
    func localIdGeneration() {
        let context = ExecutionContext.current
        let id1 = context.nextLocalId()
        let id2 = context.nextLocalId()
        #expect(id2 == id1 + 1, "Local IDs should increment")
    }

    @Test("Statistics tracking")
    func statisticsTracking() {
        let context = ExecutionContext.current
        let initialCount = context.executionCount

        context.executionCount += 1
        #expect(context.executionCount == initialCount + 1)
    }
}

// MARK: - Cache Statistics Tests

@Suite("Cache Statistics Tests")
struct CacheStatisticsTests {

    @Test("Promotion benefit calculation")
    func promotionBenefitCalculation() {
        let cache = CompilationCache.shared

        // Note: We can't easily test the actual cache behavior without XLA,
        // but we can test the statistics calculations

        #expect(cache.hitRate >= 0.0)
        #expect(cache.hitRate <= 1.0)

        #expect(cache.promotionBenefit >= 0.0)
        #expect(cache.promotionBenefit <= 1.0)
    }
}
