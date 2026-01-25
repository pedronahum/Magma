// Magma - LazyTensor Tests
// These tests verify lazy evaluation without needing XLA installed (using mocks)

import XCTest
@testable import LazyTensor
@testable import StableHLO

final class LazyTensorHandleTests: XCTestCase {

    func testHandleCreation() {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [32, 784],
            dtype: .float32,
            device: .default
        )

        XCTAssertEqual(handle.shape, [32, 784])
        XCTAssertEqual(handle.dtype, .float32)
        XCTAssertFalse(handle.isMaterialized)
    }

    func testTensorRegistry() {
        let id1 = TensorRegistry.shared.nextTensorId()
        let id2 = TensorRegistry.shared.nextTensorId()

        XCTAssertNotEqual(id1, id2)
        XCTAssertEqual(id2, id1 + 1)
    }
}

final class IRNodeTests: XCTestCase {

    func testConstantNode() {
        let node = IRNode.constant(values: [1, 2, 3, 4], shape: [2, 2])

        if case .constant(let values, let shape) = node {
            XCTAssertEqual(values, [1, 2, 3, 4])
            XCTAssertEqual(shape, [2, 2])
        } else {
            XCTFail("Expected constant node")
        }
    }

    func testOperationNode() {
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
            XCTAssertEqual(op, .add)
            XCTAssertEqual(inputs.count, 2)
        } else {
            XCTFail("Expected operation node")
        }
    }
}

final class OpKindTests: XCTestCase {

    func testOpKindRawValues() {
        XCTAssertEqual(OpKind.add.rawValue, "add")
        XCTAssertEqual(OpKind.matmul.rawValue, "matmul")
        XCTAssertEqual(OpKind.relu.rawValue, "relu")
    }
}

final class CompilationCacheTests: XCTestCase {

    func testCacheHitRate() {
        let cache = CompilationCache.shared

        // Reset for test (note: this affects shared state)
        // In a real implementation, we'd want a way to reset or create new instances

        XCTAssertGreaterThanOrEqual(cache.hitRate, 0.0)
        XCTAssertLessThanOrEqual(cache.hitRate, 1.0)
    }
}

final class IRGraphTests: XCTestCase {

    func testEmptyGraph() {
        let graph = IRGraph()
        XCTAssertTrue(graph.nodes.isEmpty)
        XCTAssertTrue(graph.outputs.isEmpty)
    }

    func testAddOutput() {
        let graph = IRGraph()
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )

        graph.addOutput(handle)
        XCTAssertEqual(graph.outputs.count, 1)
    }
}

// MARK: - Liveness Tracking Tests (Inspired by TensorFlow Swift)

final class LivenessTrackingTests: XCTestCase {

    func testHandleStartsAsNotLive() {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )

        XCTAssertFalse(handle.isLive, "New handles should not be live by default")
    }

    func testHandleCanBeMarkedLive() {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: [2, 3],
            dtype: .float32,
            device: .default
        )

        handle.isLive = true
        XCTAssertTrue(handle.isLive, "Handle should be live after marking")
    }
}

// MARK: - Constant Promotion Tests (Inspired by TensorFlow Swift)

final class ConstantPromotionTests: XCTestCase {

    func testPromotedConstantCreation() {
        let promoted = PromotedConstant(
            originalNodeId: 42,
            shape: [1],
            dtype: .float32,
            values: [3.14],
            inputIndex: 0
        )

        XCTAssertEqual(promoted.originalNodeId, 42)
        XCTAssertEqual(promoted.shape, [1])
        XCTAssertEqual(promoted.values, [3.14])
        XCTAssertEqual(promoted.inputIndex, 0)
    }

    func testConstantPromotionResult() {
        let result = ConstantPromotionResult(
            structuralHash: "abc123",
            promotedConstants: [
                PromotedConstant(originalNodeId: 1, shape: [1], dtype: .float32, values: [1.0], inputIndex: 0),
                PromotedConstant(originalNodeId: 2, shape: [1], dtype: .float32, values: [2.0], inputIndex: 1)
            ]
        )

        XCTAssertEqual(result.structuralHash, "abc123")
        XCTAssertTrue(result.wasPromoted, "Should indicate promotion was applied")
        XCTAssertEqual(result.promotedConstants.count, 2)
    }

    func testEmptyPromotionResult() {
        let result = ConstantPromotionResult(
            structuralHash: "xyz789",
            promotedConstants: []
        )

        XCTAssertFalse(result.wasPromoted, "Should indicate no promotion applied")
    }

    func testPromotionThreshold() {
        // Verify the threshold is reasonable (16 elements)
        XCTAssertEqual(CompilationCache.promotionThreshold, 16)
    }

    func testSmallConstantIsPromoted() {
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
        XCTAssertTrue(result.wasPromoted, "Small constants should be promoted")

        // The scalar constant [1] should be promoted, larger [2,3] should not
        let scalarPromoted = result.promotedConstants.first { $0.shape == [1] }
        XCTAssertNotNil(scalarPromoted, "Scalar constant should be promoted")
    }

    func testLargeConstantNotPromoted() {
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
        XCTAssertNil(largePromoted, "Large constants should not be promoted")
    }

    func testStructuralHashDiffersOnlyByStructure() {
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
        XCTAssertEqual(
            result1.structuralHash,
            result2.structuralHash,
            "Graphs with same structure but different scalar values should have same structural hash"
        )
    }
}

// MARK: - Graph Validation Tests

final class GraphValidationTests: XCTestCase {

    func testValidGraphPasses() throws {
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
        XCTAssertNoThrow(try graph.validate())
    }

    func testMatmulShapeMismatchFails() {
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
        XCTAssertThrowsError(try graph.validate()) { error in
            if case IRGraph.ValidationError.shapeMismatch = error {
                // Expected
            } else {
                XCTFail("Expected shapeMismatch error, got: \(error)")
            }
        }
    }

    func testReshapeElementCountMismatchFails() {
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
        XCTAssertThrowsError(try graph.validate()) { error in
            if case IRGraph.ValidationError.shapeMismatch = error {
                // Expected
            } else {
                XCTFail("Expected shapeMismatch error, got: \(error)")
            }
        }
    }

    func testConstantValueCountMismatchFails() {
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
        XCTAssertThrowsError(try graph.validate())
    }

    func testBinaryOpMissingInputFails() {
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
        XCTAssertThrowsError(try graph.validate()) { error in
            if case IRGraph.ValidationError.missingInput = error {
                // Expected
            } else {
                XCTFail("Expected missingInput error, got: \(error)")
            }
        }
    }
}

// MARK: - Execution Context Tests

final class ExecutionContextTests: XCTestCase {

    func testContextIsAccessible() {
        let context = ExecutionContext.current
        XCTAssertNotNil(context)
    }

    func testDefaultSettings() {
        let context = ExecutionContext.current
        XCTAssertTrue(context.shouldPromoteConstants, "Constant promotion should be enabled by default")
        XCTAssertFalse(context.isShapeTrackingEnabled, "Shape tracking should be disabled by default")
    }

    func testLocalIdGeneration() {
        let context = ExecutionContext.current
        let id1 = context.nextLocalId()
        let id2 = context.nextLocalId()
        XCTAssertEqual(id2, id1 + 1, "Local IDs should increment")
    }

    func testStatisticsTracking() {
        let context = ExecutionContext.current
        let initialCount = context.executionCount

        context.executionCount += 1
        XCTAssertEqual(context.executionCount, initialCount + 1)
    }
}

// MARK: - Cache Statistics Tests

final class CacheStatisticsTests: XCTestCase {

    func testPromotionBenefitCalculation() {
        let cache = CompilationCache.shared

        // Store original values
        let originalHits = cache.hitCount
        let originalMisses = cache.missCount
        let originalPromotionHits = cache.promotionHitCount

        // Note: We can't easily test the actual cache behavior without XLA,
        // but we can test the statistics calculations

        XCTAssertGreaterThanOrEqual(cache.hitRate, 0.0)
        XCTAssertLessThanOrEqual(cache.hitRate, 1.0)

        XCTAssertGreaterThanOrEqual(cache.promotionBenefit, 0.0)
        XCTAssertLessThanOrEqual(cache.promotionBenefit, 1.0)
    }
}
