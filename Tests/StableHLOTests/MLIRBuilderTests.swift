// Magma - StableHLO Tests
// These tests verify MLIR generation WITHOUT needing XLA installed

import Testing
@testable import StableHLO

@Suite("MLIR Builder Tests")
struct MLIRBuilderTests {

    @Test("Simple add operation")
    func simpleAdd() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let y = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let z = builder.add(x, y)

        let mlir = builder.build(name: "simple_add", outputs: [z])

        #expect(mlir.contains("stablehlo.add"))
        #expect(mlir.contains("tensor<2x3xf32>"))
        #expect(mlir.contains("func.func @main"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("Matrix multiplication")
    func matmul() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [32, 784], dtype: .float32))
        let w = builder.argument(TensorType(shape: [784, 256], dtype: .float32))
        let y = builder.dot(x, w)

        let mlir = builder.build(name: "matmul", outputs: [y])

        #expect(mlir.contains("stablehlo.dot"))
        #expect(mlir.contains("tensor<32x256xf32>"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("ReLU activation")
    func relu() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [32, 256], dtype: .float32))
        let y = builder.relu(x)

        let mlir = builder.build(name: "relu", outputs: [y])

        // ReLU is implemented as max(0, x)
        #expect(mlir.contains("stablehlo.maximum"))
        #expect(mlir.contains("dense<0"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("Linear layer: y = relu(x @ w + b)")
    func linearLayer() {
        let builder = MLIRBuilder()

        let x = builder.argument(TensorType(shape: [32, 784], dtype: .float32))
        let w = builder.argument(TensorType(shape: [784, 256], dtype: .float32))
        let b = builder.argument(TensorType(shape: [32, 256], dtype: .float32))

        let xw = builder.dot(x, w)
        let xwb = builder.add(xw, b)
        let y = builder.relu(xwb)

        let mlir = builder.build(name: "linear_relu", outputs: [y])

        #expect(mlir.contains("stablehlo.dot"))
        #expect(mlir.contains("stablehlo.add"))
        #expect(mlir.contains("stablehlo.maximum"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("Reduce sum")
    func reduceSum() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [32, 10], dtype: .float32))
        let sum = builder.reduceSum(x, axes: [1])

        let mlir = builder.build(name: "reduce_sum", outputs: [sum])

        #expect(mlir.contains("stablehlo.reduce"))
        #expect(mlir.contains("tensor<32xf32>"))  // Result shape

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("Sigmoid activation")
    func sigmoid() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let y = builder.sigmoid(x)

        let mlir = builder.build(name: "sigmoid", outputs: [y])

        // Sigmoid = 1 / (1 + exp(-x))
        #expect(mlir.contains("stablehlo.negate"))
        #expect(mlir.contains("stablehlo.exponential"))
        #expect(mlir.contains("stablehlo.divide"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("Transpose")
    func transpose() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [3, 4], dtype: .float32))
        let y = builder.transpose(x)

        let mlir = builder.build(name: "transpose", outputs: [y])

        #expect(mlir.contains("stablehlo.transpose"))
        #expect(mlir.contains("tensor<4x3xf32>"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("Compare operation")
    func compare() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let y = builder.argument(TensorType(shape: [2, 3], dtype: .float32))
        let cmp = builder.greater(x, y)

        let mlir = builder.build(name: "compare", outputs: [cmp])

        #expect(mlir.contains("stablehlo.compare GT"))
        #expect(mlir.contains("tensor<2x3xi1>"))  // Bool result

        print("Generated MLIR:\n\(mlir)")
    }
}

@Suite("TensorType Tests")
struct TensorTypeTests {

    @Test("MLIR type string generation")
    func mlirType() {
        let t1 = TensorType(shape: [32, 784], dtype: .float32)
        #expect(t1.mlirType == "tensor<32x784xf32>")

        let t2 = TensorType(shape: [], dtype: .float64)
        #expect(t2.mlirType == "tensor<f64>")

        let t3 = TensorType(shape: [1, 2, 3, 4], dtype: .int32)
        #expect(t3.mlirType == "tensor<1x2x3x4xi32>")
    }

    @Test("Element count")
    func elementCount() {
        let t1 = TensorType(shape: [2, 3, 4], dtype: .float32)
        #expect(t1.elementCount == 24)

        let t2 = TensorType(shape: [], dtype: .float32)
        #expect(t2.elementCount == 1)
    }

    @Test("Matmul result type")
    func matmulResult() {
        let a = TensorType(shape: [32, 784], dtype: .float32)
        let b = TensorType(shape: [784, 256], dtype: .float32)
        let c = a.matmulResult(with: b)

        #expect(c.shape == [32, 256])
        #expect(c.dtype == .float32)
    }

    @Test("Broadcast shapes")
    func broadcast() {
        let shapes = broadcastShapes([1, 3], [2, 1])
        #expect(shapes == [2, 3])

        let shapes2 = broadcastShapes([3], [2, 3])
        #expect(shapes2 == [2, 3])

        let incompatible = broadcastShapes([2, 3], [3, 2])
        #expect(incompatible == nil)
    }
}

@Suite("DType Tests")
struct DTypeTests {

    @Test("Byte size")
    func byteSize() {
        #expect(DType.float32.byteSize == 4)
        #expect(DType.float64.byteSize == 8)
        #expect(DType.float16.byteSize == 2)
        #expect(DType.int8.byteSize == 1)
    }

    @Test("Is floating point")
    func isFloatingPoint() {
        #expect(DType.float32.isFloatingPoint)
        #expect(DType.bfloat16.isFloatingPoint)
        #expect(!DType.int32.isFloatingPoint)
    }
}

// MARK: - Control Flow Tests

@Suite("Control Flow Tests")
struct ControlFlowTests {

    @Test("Bool constant")
    func boolConstant() {
        let builder = MLIRBuilder()
        let trueVal = builder.boolConstant(true)
        let falseVal = builder.boolConstant(false)

        let mlir = builder.build(name: "bool_constants", outputs: [trueVal, falseVal])

        #expect(mlir.contains("dense<true>"))
        #expect(mlir.contains("dense<false>"))
        #expect(mlir.contains("tensor<i1>"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("Compare directions")
    func compareDirections() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [4], dtype: .float32))
        let y = builder.argument(TensorType(shape: [4], dtype: .float32))

        let lt = builder.compare(x, y, direction: .lt)
        let eq = builder.compare(x, y, direction: .eq)
        let gt = builder.compare(x, y, direction: .gt)

        let mlir = builder.build(name: "compare", outputs: [lt, eq, gt])

        #expect(mlir.contains("stablehlo.compare LT"))
        #expect(mlir.contains("stablehlo.compare EQ"))
        #expect(mlir.contains("stablehlo.compare GT"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("Cond basic")
    func condBasic() {
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

        #expect(mlir.contains("stablehlo.if"))
        #expect(mlir.contains("stablehlo.return"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("While loop basic")
    func whileLoopBasic() {
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

        #expect(mlir.contains("stablehlo.while"))
        #expect(mlir.contains("cond {"))
        #expect(mlir.contains("} do {"))
        #expect(mlir.contains("stablehlo.return"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("While loop with tensors")
    func whileLoopWithTensors() {
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

        #expect(results.count == 2)

        let mlir = builder.build(name: "while_tensor", outputs: results)
        #expect(mlir.contains("stablehlo.while"))
        #expect(mlir.contains("tensor<4xf32>"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("Cond with computation")
    func condWithComputation() {
        let builder = MLIRBuilder()
        let x = builder.argument(TensorType(shape: [4], dtype: .float32))
        let threshold = builder.constant(0.5, type: TensorType(shape: [4], dtype: .float32))

        // Check if any element > threshold (simplified as first element)
        let _ = builder.compare(x, threshold, direction: .gt)
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

        #expect(mlir.contains("stablehlo.if"))
        // True branch has relu (implemented as maximum)
        #expect(mlir.contains("stablehlo.maximum") || mlir.contains("stablehlo.return"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("RNG uniform")
    func rngUniform() {
        let builder = MLIRBuilder()
        let low = builder.argument(TensorType(shape: [], dtype: .float32))
        let high = builder.argument(TensorType(shape: [], dtype: .float32))
        let result = builder.rngUniform(low, high, shape: [5, 10])

        let mlir = builder.build(name: "rng_uniform", outputs: [result])

        // Check that the new format is used:
        // 1. Shape constant should be created
        // 2. Generic form "stablehlo.rng" should be used
        // 3. rng_distribution attribute should use #stablehlo<...> format
        #expect(mlir.contains("stablehlo.constant dense<[5, 10]>"))
        #expect(mlir.contains("\"stablehlo.rng\""))
        #expect(mlir.contains("rng_distribution = #stablehlo<rng_distribution UNIFORM>"))

        print("Generated MLIR:\n\(mlir)")
    }

    @Test("RNG normal")
    func rngNormal() {
        let builder = MLIRBuilder()
        let mean = builder.argument(TensorType(shape: [], dtype: .float32))
        let stddev = builder.argument(TensorType(shape: [], dtype: .float32))
        let result = builder.rngNormal(mean, stddev, shape: [3, 4])

        let mlir = builder.build(name: "rng_normal", outputs: [result])

        #expect(mlir.contains("stablehlo.constant dense<[3, 4]>"))
        #expect(mlir.contains("\"stablehlo.rng\""))
        #expect(mlir.contains("rng_distribution = #stablehlo<rng_distribution NORMAL>"))

        print("Generated MLIR:\n\(mlir)")
    }
}
