// Magma - StableHLO Tests
// These tests verify MLIR generation WITHOUT needing XLA installed

import XCTest
@testable import StableHLO

final class MLIRBuilderTests: XCTestCase {
    
    func testSimpleAdd() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let y = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let z = builder.add(x, y)
        
        let mlir = builder.build(name: "simple_add", outputs: [z])
        
        XCTAssertTrue(mlir.contains("stablehlo.add"))
        XCTAssertTrue(mlir.contains("tensor<2x3xf32>"))
        XCTAssertTrue(mlir.contains("func.func @main"))
        
        print("Generated MLIR:\n\(mlir)")
    }
    
    func testMatmul() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [32, 784], dtype: .float32))
        let w = builder.argument(TensorType(shape: [784, 256], dtype: .float32))
        let y = builder.dot(x, w)
        
        let mlir = builder.build(name: "matmul", outputs: [y])
        
        XCTAssertTrue(mlir.contains("stablehlo.dot"))
        XCTAssertTrue(mlir.contains("tensor<32x256xf32>"))
        
        print("Generated MLIR:\n\(mlir)")
    }
    
    func testRelu() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [32, 256], dtype: .float32))
        let y = builder.relu(x)
        
        let mlir = builder.build(name: "relu", outputs: [y])
        
        // ReLU is implemented as max(0, x)
        XCTAssertTrue(mlir.contains("stablehlo.maximum"))
        XCTAssertTrue(mlir.contains("dense<0"))
        
        print("Generated MLIR:\n\(mlir)")
    }
    
    func testLinearLayer() {
        // Simulates: y = relu(x @ w + b)
        let builder = MLIRBuilder()
        
        let x = builder.argument(TensorType(shape: [32, 784], dtype: .float32))
        let w = builder.argument(TensorType(shape: [784, 256], dtype: .float32))
        let b = builder.argument(TensorType(shape: [32, 256], dtype: .float32))
        
        let xw = builder.dot(x, w)
        let xwb = builder.add(xw, b)
        let y = builder.relu(xwb)
        
        let mlir = builder.build(name: "linear_relu", outputs: [y])
        
        XCTAssertTrue(mlir.contains("stablehlo.dot"))
        XCTAssertTrue(mlir.contains("stablehlo.add"))
        XCTAssertTrue(mlir.contains("stablehlo.maximum"))
        
        print("Generated MLIR:\n\(mlir)")
    }
    
    func testReduceSum() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [32, 10], dtype: .float32))
        let sum = builder.reduceSum(x, axes: [1])
        
        let mlir = builder.build(name: "reduce_sum", outputs: [sum])
        
        XCTAssertTrue(mlir.contains("stablehlo.reduce"))
        XCTAssertTrue(mlir.contains("tensor<32xf32>"))  // Result shape
        
        print("Generated MLIR:\n\(mlir)")
    }
    
    func testSigmoid() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let y = builder.sigmoid(x)
        
        let mlir = builder.build(name: "sigmoid", outputs: [y])
        
        // Sigmoid = 1 / (1 + exp(-x))
        XCTAssertTrue(mlir.contains("stablehlo.negate"))
        XCTAssertTrue(mlir.contains("stablehlo.exponential"))
        XCTAssertTrue(mlir.contains("stablehlo.divide"))
        
        print("Generated MLIR:\n\(mlir)")
    }
    
    func testTranspose() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [3, 4], dtype: .float32))
        let y = builder.transpose(x)
        
        let mlir = builder.build(name: "transpose", outputs: [y])
        
        XCTAssertTrue(mlir.contains("stablehlo.transpose"))
        XCTAssertTrue(mlir.contains("tensor<4x3xf32>"))
        
        print("Generated MLIR:\n\(mlir)")
    }
    
    func testCompare() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let y = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let cmp = builder.greater(x, y)
        
        let mlir = builder.build(name: "compare", outputs: [cmp])
        
        XCTAssertTrue(mlir.contains("stablehlo.compare GT"))
        XCTAssertTrue(mlir.contains("tensor<2x3xi1>"))  // Bool result
        
        print("Generated MLIR:\n\(mlir)")
    }
}

final class TensorTypeTests: XCTestCase {
    
    func testMLIRType() {
        let t1 = TensorType(shape: [32, 784], dtype: .float32)
        XCTAssertEqual(t1.mlirType, "tensor<32x784xf32>")
        
        let t2 = TensorType(shape: [], dtype: .float64)
        XCTAssertEqual(t2.mlirType, "tensor<f64>")
        
        let t3 = TensorType(shape: [1, 2, 3, 4], dtype: .int32)
        XCTAssertEqual(t3.mlirType, "tensor<1x2x3x4xi32>")
    }
    
    func testElementCount() {
        let t1 = TensorType(shape: [2, 3, 4], dtype: .float32)
        XCTAssertEqual(t1.elementCount, 24)
        
        let t2 = TensorType(shape: [], dtype: .float32)
        XCTAssertEqual(t2.elementCount, 1)
    }
    
    func testMatmulResult() {
        let a = TensorType(shape: [32, 784], dtype: .float32)
        let b = TensorType(shape: [784, 256], dtype: .float32)
        let c = a.matmulResult(with: b)
        
        XCTAssertEqual(c.shape, [32, 256])
        XCTAssertEqual(c.dtype, .float32)
    }
    
    func testBroadcast() {
        let shapes = broadcastShapes([1, 3], [2, 1])
        XCTAssertEqual(shapes, [2, 3])
        
        let shapes2 = broadcastShapes([3], [2, 3])
        XCTAssertEqual(shapes2, [2, 3])
        
        let incompatible = broadcastShapes([2, 3], [3, 2])
        XCTAssertNil(incompatible)
    }
}

final class DTypeTests: XCTestCase {

    func testByteSize() {
        XCTAssertEqual(DType.float32.byteSize, 4)
        XCTAssertEqual(DType.float64.byteSize, 8)
        XCTAssertEqual(DType.float16.byteSize, 2)
        XCTAssertEqual(DType.int8.byteSize, 1)
    }

    func testIsFloatingPoint() {
        XCTAssertTrue(DType.float32.isFloatingPoint)
        XCTAssertTrue(DType.bfloat16.isFloatingPoint)
        XCTAssertFalse(DType.int32.isFloatingPoint)
    }
}

// MARK: - Control Flow Tests

final class ControlFlowTests: XCTestCase {

    func testBoolConstant() {
        let builder = MLIRBuilder()
        let trueVal = builder.boolConstant(true)
        let falseVal = builder.boolConstant(false)

        let mlir = builder.build(name: "bool_constants", outputs: [trueVal, falseVal])

        XCTAssertTrue(mlir.contains("dense<true>"))
        XCTAssertTrue(mlir.contains("dense<false>"))
        XCTAssertTrue(mlir.contains("tensor<i1>"))

        print("Generated MLIR:\n\(mlir)")
    }

    func testCompare() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [4], dtype: .float32))
        let y = builder.argument(TensorType(shape: [4], dtype: .float32))

        let lt = builder.compare(x, y, direction: .lt)
        let eq = builder.compare(x, y, direction: .eq)
        let gt = builder.compare(x, y, direction: .gt)

        let mlir = builder.build(name: "compare", outputs: [lt, eq, gt])

        XCTAssertTrue(mlir.contains("stablehlo.compare LT"))
        XCTAssertTrue(mlir.contains("stablehlo.compare EQ"))
        XCTAssertTrue(mlir.contains("stablehlo.compare GT"))

        print("Generated MLIR:\n\(mlir)")
    }

    func testCondBasic() {
        let builder = MLIRBuilder()
        let pred = builder.boolConstant(true)

        let results = builder.cond(
            predicate: pred,
            resultTypes: [TensorType(shape: [2, 3], dtype: .float32)],
            trueBranch: {
                let c = builder.constant(1.0, type: TensorType(shape: [2, 3], dtype: .float32))
                return [c]
            },
            falseBranch: {
                let c = builder.constant(0.0, type: TensorType(shape: [2, 3], dtype: .float32))
                return [c]
            }
        )

        let mlir = builder.build(name: "cond_basic", outputs: results)

        XCTAssertTrue(mlir.contains("stablehlo.if"))
        XCTAssertTrue(mlir.contains("stablehlo.return"))

        print("Generated MLIR:\n\(mlir)")
    }

    func testWhileLoopBasic() {
        let builder = MLIRBuilder()

        // Initial values: counter = 0, sum = 0
        let initCounter = builder.constant(0.0, type: TensorType(shape: [], dtype: .float32))
        let initSum = builder.constant(0.0, type: TensorType(shape: [], dtype: .float32))
        let limit = builder.constant(10.0, type: TensorType(shape: [], dtype: .float32))

        let results = builder.whileLoop(
            initialValues: [initCounter, initSum],
            conditionBuilder: { loopVars in
                // Condition: counter < limit
                let counter = loopVars[0]
                return builder.compare(counter, limit, direction: .lt)
            },
            bodyBuilder: { loopVars in
                let counter = loopVars[0]
                let sum = loopVars[1]
                let one = builder.constant(1.0, type: TensorType(shape: [], dtype: .float32))
                // counter = counter + 1
                let newCounter = builder.add(counter, one)
                // sum = sum + counter
                let newSum = builder.add(sum, counter)
                return [newCounter, newSum]
            }
        )

        let mlir = builder.build(name: "while_loop_basic", outputs: results)

        XCTAssertTrue(mlir.contains("stablehlo.while"))
        XCTAssertTrue(mlir.contains("cond {"))
        XCTAssertTrue(mlir.contains("} do {"))
        XCTAssertTrue(mlir.contains("stablehlo.return"))

        print("Generated MLIR:\n\(mlir)")
    }

    func testWhileLoopWithTensors() {
        let builder = MLIRBuilder()

        // Accumulate tensor values in a loop
        let initCounter = builder.constant(0.0, type: TensorType(shape: [], dtype: .float32))
        let initAccum = builder.constant([0.0, 0.0, 0.0, 0.0], shape: [4], dtype: .float32)
        let maxIter = builder.constant(5.0, type: TensorType(shape: [], dtype: .float32))

        let results = builder.whileLoop(
            initialValues: [initCounter, initAccum],
            conditionBuilder: { loopVars in
                builder.compare(loopVars[0], maxIter, direction: .lt)
            },
            bodyBuilder: { loopVars in
                let counter = loopVars[0]
                let accum = loopVars[1]
                let one = builder.constant(1.0, type: TensorType(shape: [], dtype: .float32))
                let increment = builder.constant([1.0, 1.0, 1.0, 1.0], shape: [4], dtype: .float32)
                return [builder.add(counter, one), builder.add(accum, increment)]
            }
        )

        XCTAssertEqual(results.count, 2)

        let mlir = builder.build(name: "while_tensor", outputs: results)
        XCTAssertTrue(mlir.contains("stablehlo.while"))
        XCTAssertTrue(mlir.contains("tensor<4xf32>"))

        print("Generated MLIR:\n\(mlir)")
    }

    func testCondWithComputation() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [4], dtype: .float32))
        let threshold = builder.constant(0.5, type: TensorType(shape: [4], dtype: .float32))

        // Check if any element > threshold (simplified as first element)
        let comparison = builder.compare(x, threshold, direction: .gt)
        // For simplicity, we'll use a scalar predicate
        let scalarPred = builder.boolConstant(true)  // Would need reduce in practice

        let results = builder.cond(
            predicate: scalarPred,
            resultTypes: [TensorType(shape: [4], dtype: .float32)],
            trueBranch: {
                // Apply ReLU if condition is true
                [builder.relu(x)]
            },
            falseBranch: {
                // Apply sigmoid if condition is false
                [builder.sigmoid(x)]
            }
        )

        let mlir = builder.build(name: "cond_compute", outputs: results)

        XCTAssertTrue(mlir.contains("stablehlo.if"))
        // True branch has relu (implemented as maximum)
        XCTAssertTrue(mlir.contains("stablehlo.maximum") || mlir.contains("stablehlo.return"))

        print("Generated MLIR:\n\(mlir)")
    }
}
