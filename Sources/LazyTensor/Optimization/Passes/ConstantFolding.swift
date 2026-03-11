// Magma - Constant Folding Pass
// Evaluates operations on constants at compile time
//
// When all inputs to an operation are constants, the operation can be
// computed immediately and replaced with a constant result.

import Foundation

/// Constant Folding Pass
///
/// Evaluates operations where all inputs are known constants at compile time.
/// This eliminates runtime computation for deterministic operations.
///
/// Example:
/// ```
/// a = constant([2.0])
/// b = constant([3.0])
/// c = a * b            // Can be folded to constant([6.0])
/// d = input()
/// e = c + d
/// output(e)
/// ```
/// After constant folding:
/// ```
/// c = constant([6.0])
/// d = input()
/// e = c + d
/// output(e)
/// ```
public final class ConstantFoldingPass: OptimizationPass {

    public let name = "constant-folding"
    public let dependencies: [String] = ["cse"]
    public let enabledByDefault = true

    /// Maximum number of elements for a constant to be folded
    /// Very large constants are not worth folding (compile-time cost)
    public var maxFoldableElements: Int = 10000

    public init() {}

    public func run(on graph: IRGraph) -> IRGraph {
        // Build topological order if not already done
        if graph.nodes.isEmpty {
            graph.buildTopologicalOrder()
        }

        // Track which nodes are constants and their values
        var constants: [UInt64: (values: [Float], shape: [Int])] = [:]

        // Collect initial constants
        for node in graph.nodes {
            if let irNode = node.irNode {
                if case .constant(let values, let shape) = irNode {
                    constants[node.id] = (values, shape)
                }
            }
        }

        // Process nodes in topological order
        var foldedCount = 0
        for node in graph.nodes {
            guard let irNode = node.irNode else { continue }

            switch irNode {
            case .operation(let opKind, let inputs, let attributes):
                // Check if all inputs are constants
                let allInputsConstant = inputs.allSatisfy { constants[$0.id] != nil }

                if allInputsConstant && isFoldable(opKind) {
                    // Get input constant values
                    let inputConstants = inputs.map { constants[$0.id]! }

                    // Check if result would be too large
                    let resultElements = node.shape.isEmpty ? 1 : node.shape.reduce(1, *)
                    if resultElements > maxFoldableElements {
                        continue
                    }

                    // Evaluate the operation
                    if let result = evaluateOnCPU(opKind: opKind, inputs: inputConstants, outputShape: node.shape, attributes: attributes) {
                        // Replace with constant
                        node.irNode = .constant(values: result, shape: node.shape)
                        constants[node.id] = (result, node.shape)
                        foldedCount += 1
                    }
                }

            case .constant, .data, .whileLoopTraced:
                break
            #if os(macOS) && canImport(MetalHLO)
            case .metalData:
                break
            #endif
            }
        }

        // If no folding occurred, return original graph
        if foldedCount == 0 {
            return graph
        }

        return graph
    }

    /// Check if an operation can be folded (evaluated at compile time)
    private func isFoldable(_ opKind: OpKind) -> Bool {
        switch opKind {
        // Elementwise operations are foldable
        case .add, .subtract, .multiply, .divide, .maximum, .minimum, .power:
            return true
        case .negate, .abs, .exponential, .log, .sqrt, .rsqrt, .sine, .cosine, .tanh, .floor, .ceil:
            return true

        // Matrix operations
        case .matmul, .batchedMatmul, .transpose, .reshape, .broadcast:
            return true

        // Activations
        case .relu, .sigmoid, .gelu, .leakyRelu, .elu, .silu:
            return true

        // Clamp
        case .clamp:
            return true

        // Comparisons
        case .equal, .less, .greater:
            return true

        // Reductions
        case .reduceSum, .reduceMax, .reduceMin, .reduceMean:
            return true

        // Not foldable (require runtime data or have side effects)
        case .softmax:  // Could be folded but complex
            return false
        case .select:
            return true
        case .slice, .pad, .gather, .scatter, .concatenate:
            return true
        case .convert:
            return true
        case .conv1d, .conv2d, .convTranspose2d, .maxPool2d, .avgPool2d, .batchNorm, .layerNorm:
            return false  // Too complex for CPU folding
        case .whileLoop, .cond:
            return false  // Control flow not foldable
        case .rngUniform, .rngNormal:
            return false  // Random - not deterministic
        }
    }

    /// Evaluate an operation on CPU
    ///
    /// This is a simplified implementation that handles common cases.
    /// A production implementation would be more comprehensive.
    private func evaluateOnCPU(
        opKind: OpKind,
        inputs: [(values: [Float], shape: [Int])],
        outputShape: [Int],
        attributes: [String: Any]
    ) -> [Float]? {

        switch opKind {
        // Binary elementwise
        case .add:
            return binaryOp(inputs[0], inputs[1]) { $0 + $1 }
        case .subtract:
            return binaryOp(inputs[0], inputs[1]) { $0 - $1 }
        case .multiply:
            return binaryOp(inputs[0], inputs[1]) { $0 * $1 }
        case .divide:
            return binaryOp(inputs[0], inputs[1]) { $0 / $1 }
        case .maximum:
            return binaryOp(inputs[0], inputs[1]) { max($0, $1) }
        case .minimum:
            return binaryOp(inputs[0], inputs[1]) { min($0, $1) }
        case .power:
            return binaryOp(inputs[0], inputs[1]) { pow($0, $1) }

        // Unary elementwise
        case .negate:
            return inputs[0].values.map { -$0 }
        case .abs:
            return inputs[0].values.map { Swift.abs($0) }
        case .exponential:
            return inputs[0].values.map { exp($0) }
        case .log:
            return inputs[0].values.map { Foundation.log($0) }
        case .sqrt:
            return inputs[0].values.map { Foundation.sqrt($0) }
        case .rsqrt:
            return inputs[0].values.map { 1.0 / Foundation.sqrt($0) }
        case .sine:
            return inputs[0].values.map { sin($0) }
        case .cosine:
            return inputs[0].values.map { cos($0) }
        case .tanh:
            return inputs[0].values.map { Foundation.tanh($0) }
        case .floor:
            return inputs[0].values.map { Foundation.floor($0) }
        case .ceil:
            return inputs[0].values.map { Foundation.ceil($0) }

        // Activations
        case .relu:
            return inputs[0].values.map { max(0, $0) }
        case .sigmoid:
            return inputs[0].values.map { 1.0 / (1.0 + exp(-$0)) }
        case .gelu:
            // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let sqrtTwoPi: Float = 0.7978845608
            let coeff: Float = 0.044715
            return inputs[0].values.map { x in
                let x3 = x * x * x
                let inner = sqrtTwoPi * (x + coeff * x3)
                return 0.5 * x * (1 + Foundation.tanh(inner))
            }
        case .leakyRelu:
            let alpha = (attributes["negativeSlope"] as? Float) ?? 0.01
            return inputs[0].values.map { $0 > 0 ? $0 : alpha * $0 }
        case .elu:
            let alpha = (attributes["alpha"] as? Float) ?? 1.0
            return inputs[0].values.map { $0 > 0 ? $0 : alpha * (exp($0) - 1) }
        case .silu:
            return inputs[0].values.map { x in x / (1.0 + exp(-x)) }

        // Reshape - just returns same values with different shape interpretation
        case .reshape:
            return inputs[0].values

        // Transpose - needs actual reordering
        case .transpose:
            return transposeValues(inputs[0].values, shape: inputs[0].shape)

        // Reductions
        case .reduceSum:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].shape.count)
            return reduceOperation(inputs[0], axes: axes, initial: 0) { $0 + $1 }
        case .reduceMax:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].shape.count)
            return reduceOperation(inputs[0], axes: axes, initial: -.infinity) { max($0, $1) }
        case .reduceMin:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].shape.count)
            return reduceOperation(inputs[0], axes: axes, initial: .infinity) { min($0, $1) }
        case .reduceMean:
            let axes = attributes["axes"] as? [Int] ?? Array(0..<inputs[0].shape.count)
            guard let sum = reduceOperation(inputs[0], axes: axes, initial: 0, op: { $0 + $1 }) else {
                return nil
            }
            let count = axes.reduce(1) { $0 * inputs[0].shape[$1] }
            return sum.map { $0 / Float(count) }

        // Comparisons (return 1.0 for true, 0.0 for false)
        case .equal:
            return binaryOp(inputs[0], inputs[1]) { $0 == $1 ? 1.0 : 0.0 }
        case .less:
            return binaryOp(inputs[0], inputs[1]) { $0 < $1 ? 1.0 : 0.0 }
        case .greater:
            return binaryOp(inputs[0], inputs[1]) { $0 > $1 ? 1.0 : 0.0 }

        // Select
        case .select:
            let cond = inputs[0].values
            let onTrue = inputs[1].values
            let onFalse = inputs[2].values
            return zip(cond, zip(onTrue, onFalse)).map { $0.0 != 0 ? $0.1.0 : $0.1.1 }

        // Clamp
        case .clamp:
            let minVal = (attributes["min"] as? Float) ?? -.infinity
            let maxVal = (attributes["max"] as? Float) ?? .infinity
            return inputs[0].values.map { max(minVal, min(maxVal, $0)) }

        // Matrix multiplication
        case .matmul:
            return matmulOnCPU(inputs[0], inputs[1])

        default:
            // Operation not implemented for constant folding
            return nil
        }
    }

    /// Apply binary operation with broadcasting
    private func binaryOp(
        _ lhs: (values: [Float], shape: [Int]),
        _ rhs: (values: [Float], shape: [Int]),
        op: (Float, Float) -> Float
    ) -> [Float]? {
        // Simple case: same shape
        if lhs.shape == rhs.shape {
            return zip(lhs.values, rhs.values).map(op)
        }

        // Scalar broadcasting
        if lhs.values.count == 1 {
            return rhs.values.map { op(lhs.values[0], $0) }
        }
        if rhs.values.count == 1 {
            return lhs.values.map { op($0, rhs.values[0]) }
        }

        // General broadcasting is complex - skip for now
        return nil
    }

    /// Transpose values (for 2D tensors only)
    private func transposeValues(_ values: [Float], shape: [Int]) -> [Float]? {
        guard shape.count == 2 else { return nil }
        let rows = shape[0]
        let cols = shape[1]

        var result = [Float](repeating: 0, count: values.count)
        for i in 0..<rows {
            for j in 0..<cols {
                result[j * rows + i] = values[i * cols + j]
            }
        }
        return result
    }

    /// Reduce operation (simplified - full axis reduction only)
    private func reduceOperation(
        _ input: (values: [Float], shape: [Int]),
        axes: [Int],
        initial: Float,
        op: (Float, Float) -> Float
    ) -> [Float]? {
        // Only handle full reduction for simplicity
        let allAxes = Set(axes.map { $0 < 0 ? input.shape.count + $0 : $0 })
        let isFullReduction = allAxes == Set(0..<input.shape.count)

        if isFullReduction {
            let result = input.values.reduce(initial, op)
            return [result]
        }

        // Partial reduction is more complex - skip for now
        return nil
    }

    /// Matrix multiplication on CPU
    private func matmulOnCPU(
        _ lhs: (values: [Float], shape: [Int]),
        _ rhs: (values: [Float], shape: [Int])
    ) -> [Float]? {
        guard lhs.shape.count == 2, rhs.shape.count == 2 else { return nil }
        guard lhs.shape[1] == rhs.shape[0] else { return nil }

        let m = lhs.shape[0]
        let k = lhs.shape[1]
        let n = rhs.shape[1]

        var result = [Float](repeating: 0, count: m * n)

        for i in 0..<m {
            for j in 0..<n {
                var sum: Float = 0
                for l in 0..<k {
                    sum += lhs.values[i * k + l] * rhs.values[l * n + j]
                }
                result[i * n + j] = sum
            }
        }

        return result
    }
}
