// Magma - Operation Fusion Pass
// Combines multiple operations into fused operations for better performance
//
// This is the most impactful optimization for closing the gap with MLX.
// By fusing operations, we reduce:
// 1. Kernel launch overhead (fewer GPU dispatches)
// 2. Memory bandwidth (intermediate results stay in registers/cache)
// 3. Synchronization points

import Foundation

/// Operation Fusion Pass
///
/// Detects patterns of operations that can be fused and replaces them
/// with single fused operations. Uses fixed-point iteration to find
/// all possible fusions.
///
/// Key fusion patterns (in priority order):
/// 1. Scaled Dot-Product Attention (Q @ K^T / scale → softmax → @ V)
/// 2. Layer Normalization (mean/var/normalize/scale/shift)
/// 3. RMS Normalization (sqrt(mean(x^2)) / scale * weight)
/// 4. MatMul + Bias + Activation (common in FFN layers)
/// 5. Softmax (numerically stable version)
/// 6. GELU (exact or approximate)
public final class OperationFusionPass: OptimizationPass {

    public let name = "operation-fusion"
    public let dependencies: [String] = ["algebraic-simplification"]
    public let enabledByDefault = true

    /// Registered fusion patterns, sorted by priority
    private var patterns: [FusionPattern] = []

    /// Maximum number of fusion iterations (prevents infinite loops)
    public var maxIterations: Int = 100

    public init() {
        // Register patterns in priority order (highest first)
        // Larger patterns should have higher priority to avoid partial matches
        registerDefaultPatterns()
    }

    /// Register default fusion patterns
    private func registerDefaultPatterns() {
        // Phase 2 critical patterns
        patterns.append(ScaledDotProductAttentionPattern())
        patterns.append(LayerNormPattern())
        patterns.append(RMSNormPattern())
        patterns.append(MatMulBiasActivationPattern())
        patterns.append(SoftmaxPattern())
        patterns.append(GeluPattern())

        // Sort by priority (descending)
        patterns.sort { $0.priority > $1.priority }
    }

    /// Register a custom fusion pattern
    public func registerPattern(_ pattern: FusionPattern) {
        patterns.append(pattern)
        patterns.sort { $0.priority > $1.priority }
    }

    public func run(on graph: IRGraph) -> IRGraph {
        // Build topological order if not already done
        if graph.nodes.isEmpty {
            graph.buildTopologicalOrder()
        }

        var currentGraph = graph
        var changed = true
        var iteration = 0

        // Fixed-point iteration: keep applying patterns until no more changes
        while changed && iteration < maxIterations {
            changed = false
            iteration += 1

            for pattern in patterns {
                if let match = pattern.match(in: currentGraph) {
                    // Apply the fusion
                    currentGraph = pattern.apply(match, to: currentGraph)
                    changed = true

                    // Rebuild topological order after transformation
                    currentGraph.buildTopologicalOrder()

                    // Break to restart pattern matching from the beginning
                    // This ensures we don't miss patterns that become available
                    // after a fusion
                    break
                }
            }
        }

        return currentGraph
    }
}

// MARK: - Scaled Dot-Product Attention Pattern

/// Detects and fuses scaled dot-product attention
///
/// Pattern:
/// ```
/// scores = Q @ K^T
/// scaled = scores * scale  (or scores / sqrt(d))
/// masked = scaled + mask   (optional)
/// weights = softmax(scaled_or_masked)
/// output = weights @ V
/// ```
///
/// This is the most impactful fusion for transformer models.
public final class ScaledDotProductAttentionPattern: FusionPattern {

    public let name = "scaled-dot-product-attention"
    public let priority = 100  // Highest priority - largest pattern

    public init() {}

    public func match(in graph: IRGraph) -> FusionMatch? {
        let opMap = PatternMatchingHelpers.buildOperationMap(graph)

        // Look for the final matmul (weights @ V)
        for node in graph.nodes {
            guard let (_, opKind, inputs, _) = opMap[node.id] else { continue }
            guard opKind == .matmul || opKind == .batchedMatmul else { continue }

            // The first input (weights) should come from softmax
            let weightsInput = inputs[0]
            guard let (softmaxNode, softmaxOp, softmaxInputs, _) = opMap[weightsInput.id],
                  softmaxOp == .softmax else { continue }

            // V is the second input to the final matmul
            let v = inputs[1]

            // Trace back from softmax to find scaling and Q @ K^T
            let (scaledScores, mask, maskOp) = extractMaskIfPresent(softmaxInputs[0], opMap: opMap)

            guard let (scaleResult, scaleValue) = extractScaling(scaledScores, opMap: opMap) else {
                continue
            }

            // scaleResult should be Q @ K^T
            guard let (qkNode, qkOp, qkInputs, _) = opMap[scaleResult.id],
                  qkOp == .matmul || qkOp == .batchedMatmul else { continue }

            let q = qkInputs[0]
            let k = qkInputs[1]

            // Validate shapes are compatible for attention
            // Q: [batch, heads, seq, head_dim]
            // K: [batch, heads, seq, head_dim] (will be transposed)
            // V: [batch, heads, seq, head_dim]
            guard validateAttentionShapes(q: q, k: k, v: v) else { continue }

            // Collect matched operations
            var matchedOps: [LazyTensorHandle] = [qkNode]

            // Add scale operation if present
            if scaleResult.id != scaledScores.id {
                if let (scaleOpNode, _, _, _) = opMap[scaledScores.id] {
                    matchedOps.append(scaleOpNode)
                }
            }

            // Add mask operation if present
            if let maskOpNode = maskOp {
                matchedOps.append(maskOpNode)
            }

            // Add softmax and final matmul
            matchedOps.append(softmaxNode)
            matchedOps.append(node)

            // Detect if this is causal attention
            let isCausal = detectCausalMask(mask)

            return FusionMatch(
                operations: matchedOps,
                indices: matchedOps.compactMap { op in graph.nodes.firstIndex(where: { $0.id == op.id }) },
                metadata: [
                    "q": q,
                    "k": k,
                    "v": v,
                    "scale": scaleValue,
                    "mask": mask as Any,
                    "isCausal": isCausal
                ]
            )
        }

        return nil
    }

    public func apply(_ match: FusionMatch, to graph: IRGraph) -> IRGraph {
        let q = match.metadata["q"] as! LazyTensorHandle
        let k = match.metadata["k"] as! LazyTensorHandle
        let v = match.metadata["v"] as! LazyTensorHandle
        let scale = match.metadata["scale"] as! Float
        let mask = match.metadata["mask"] as? LazyTensorHandle
        let isCausal = match.metadata["isCausal"] as! Bool

        // Create the fused attention operation
        let outputHandle = match.output!

        var inputs = [q, k, v]
        if let m = mask {
            inputs.append(m)
        }

        // Create fused operation node
        let fusedHandle = LazyTensorHandle(
            id: outputHandle.id,
            shape: outputHandle.shape,
            dtype: outputHandle.dtype,
            device: outputHandle.device
        )

        fusedHandle.irNode = .operation(
            op: .fusedScaledDotProductAttention,
            inputs: inputs,
            attributes: [
                "scale": scale,
                "isCausal": isCausal,
                "hasMask": mask != nil
            ]
        )

        return graph.replacing(match.operations, with: [fusedHandle])
    }

    // MARK: - Helper Methods

    private func extractMaskIfPresent(
        _ handle: LazyTensorHandle,
        opMap: [UInt64: (LazyTensorHandle, OpKind, [LazyTensorHandle], [String: Any])]
    ) -> (scores: LazyTensorHandle, mask: LazyTensorHandle?, maskOp: LazyTensorHandle?) {
        // Check if handle is an add operation (scores + mask)
        if let (node, op, inputs, _) = opMap[handle.id], op == .add {
            // One input is scores, other is mask
            // Typically mask has large negative values
            return (inputs[0], inputs[1], node)
        }
        return (handle, nil, nil)
    }

    private func extractScaling(
        _ handle: LazyTensorHandle,
        opMap: [UInt64: (LazyTensorHandle, OpKind, [LazyTensorHandle], [String: Any])]
    ) -> (scores: LazyTensorHandle, scale: Float)? {
        guard let (_, op, inputs, _) = opMap[handle.id] else {
            return nil
        }

        switch op {
        case .multiply:
            // scores * scale
            if let scale = PatternMatchingHelpers.getScalarValue(inputs[1]) {
                return (inputs[0], scale)
            }
            if let scale = PatternMatchingHelpers.getScalarValue(inputs[0]) {
                return (inputs[1], scale)
            }

        case .divide:
            // scores / sqrt(d_k)
            if let divisor = PatternMatchingHelpers.getScalarValue(inputs[1]) {
                return (inputs[0], 1.0 / divisor)
            }

        default:
            break
        }

        // No scaling found - assume scale = 1.0
        return (handle, 1.0)
    }

    private func validateAttentionShapes(q: LazyTensorHandle, k: LazyTensorHandle, v: LazyTensorHandle) -> Bool {
        // Basic validation - all should have same rank
        guard q.shape.count == k.shape.count, k.shape.count == v.shape.count else {
            return false
        }

        // For batched attention, expect at least 3D tensors
        guard q.shape.count >= 2 else {
            return false
        }

        return true
    }

    private func detectCausalMask(_ mask: LazyTensorHandle?) -> Bool {
        // TODO: Implement proper causal mask detection
        // For now, return false (assume not causal)
        return false
    }
}

// MARK: - Layer Normalization Pattern

/// Detects and fuses layer normalization
///
/// Pattern:
/// ```
/// mean = reduce_mean(x, axis=-1)
/// variance = reduce_mean((x - mean)^2, axis=-1)
/// normalized = (x - mean) / sqrt(variance + eps)
/// output = normalized * gamma + beta
/// ```
public final class LayerNormPattern: FusionPattern {

    public let name = "layer-norm"
    public let priority = 80

    public init() {}

    public func match(in graph: IRGraph) -> FusionMatch? {
        let opMap = PatternMatchingHelpers.buildOperationMap(graph)

        // Look for the final add (+ beta)
        for node in graph.nodes {
            guard let (_, opKind, inputs, _) = opMap[node.id] else { continue }
            guard opKind == .add else { continue }

            // One input should be multiply (* gamma)
            let (mulResult, gamma) = findMultiplyWithWeight(inputs, opMap: opMap)
            guard let mulNode = mulResult, let gammaHandle = gamma else { continue }

            // Mul input should be divide (normalized / sqrt(var + eps))
            guard let (_, divOp, divInputs, _) = opMap[mulNode.id],
                  divOp == .divide else { continue }

            // Divisor should be sqrt(variance + eps)
            guard let (sqrtNode, sqrtOp, sqrtInputs, _) = opMap[divInputs[1].id],
                  sqrtOp == .sqrt else { continue }

            // Inside sqrt should be add (variance + eps)
            guard let (_, varAddOp, varAddInputs, _) = opMap[sqrtInputs[0].id],
                  varAddOp == .add else { continue }

            // Extract epsilon
            guard let eps = PatternMatchingHelpers.getScalarValue(varAddInputs[1]) else { continue }

            // The numerator (x - mean) should match the variance computation
            // This is complex to fully validate, so we do basic checks

            // Extract the original input x
            // divInputs[0] should be (x - mean)
            guard let (_, subOp, subInputs, _) = opMap[divInputs[0].id],
                  subOp == .subtract else { continue }

            let x = subInputs[0]
            let beta = inputs[0].id == mulNode.id ? inputs[1] : inputs[0]

            // Collect matched operations (simplified - in production, trace all operations)
            let matchedOps = [node]

            return FusionMatch(
                operations: matchedOps,
                indices: matchedOps.compactMap { op in graph.nodes.firstIndex(where: { $0.id == op.id }) },
                metadata: [
                    "input": x,
                    "gamma": gammaHandle,
                    "beta": beta,
                    "eps": eps,
                    "axes": [-1]  // Typically last axis
                ]
            )
        }

        return nil
    }

    public func apply(_ match: FusionMatch, to graph: IRGraph) -> IRGraph {
        let x = match.metadata["input"] as! LazyTensorHandle
        let gamma = match.metadata["gamma"] as! LazyTensorHandle
        let beta = match.metadata["beta"] as! LazyTensorHandle
        let eps = match.metadata["eps"] as! Float
        let axes = match.metadata["axes"] as! [Int]

        let outputHandle = match.output!

        let fusedHandle = LazyTensorHandle(
            id: outputHandle.id,
            shape: outputHandle.shape,
            dtype: outputHandle.dtype,
            device: outputHandle.device
        )

        fusedHandle.irNode = .operation(
            op: .fusedLayerNorm,
            inputs: [x, gamma, beta],
            attributes: [
                "eps": eps,
                "axes": axes
            ]
        )

        return graph.replacing(match.operations, with: [fusedHandle])
    }

    private func findMultiplyWithWeight(
        _ inputs: [LazyTensorHandle],
        opMap: [UInt64: (LazyTensorHandle, OpKind, [LazyTensorHandle], [String: Any])]
    ) -> (LazyTensorHandle?, LazyTensorHandle?) {
        for input in inputs {
            if let (node, op, mulInputs, _) = opMap[input.id], op == .multiply {
                // Return the multiply node and the weight (gamma)
                // Assume weight is the smaller tensor or the second input
                return (node, mulInputs[1])
            }
        }
        return (nil, nil)
    }
}

// MARK: - RMS Normalization Pattern

/// Detects and fuses RMS normalization
///
/// Pattern:
/// ```
/// rms = sqrt(reduce_mean(x^2, axis=-1) + eps)
/// output = (x / rms) * weight
/// ```
public final class RMSNormPattern: FusionPattern {

    public let name = "rms-norm"
    public let priority = 75

    public init() {}

    public func match(in graph: IRGraph) -> FusionMatch? {
        let opMap = PatternMatchingHelpers.buildOperationMap(graph)

        // Look for multiply with weight
        for node in graph.nodes {
            guard let (_, opKind, inputs, _) = opMap[node.id] else { continue }
            guard opKind == .multiply else { continue }

            // One input should be divide (x / rms)
            let divInput = findDivideInput(inputs, opMap: opMap)
            guard let (divNode, divInputs) = divInput else { continue }

            // Divisor should be sqrt(mean(x^2) + eps)
            guard let (_, sqrtOp, sqrtInputs, _) = opMap[divInputs[1].id],
                  sqrtOp == .sqrt else { continue }

            // sqrt input should be add (mean + eps)
            guard let (_, addOp, addInputs, _) = opMap[sqrtInputs[0].id],
                  addOp == .add else { continue }

            guard let eps = PatternMatchingHelpers.getScalarValue(addInputs[1]) else { continue }

            // add input should be reduce_mean
            guard let (_, meanOp, meanInputs, meanAttrs) = opMap[addInputs[0].id],
                  meanOp == .reduceMean else { continue }

            // mean input should be x^2 (multiply x * x or power x 2)
            guard let (_, squareOp, squareInputs, _) = opMap[meanInputs[0].id] else { continue }

            let isSquare: Bool
            switch squareOp {
            case .multiply:
                isSquare = squareInputs[0].id == squareInputs[1].id
            case .power:
                isSquare = PatternMatchingHelpers.isConstantWithValue(squareInputs[1], 2.0)
            default:
                isSquare = false
            }

            guard isSquare else { continue }

            // Verify x in numerator matches x in x^2
            let x = divInputs[0]
            guard x.id == squareInputs[0].id else { continue }

            // Weight is the other input to the outer multiply
            let weight = inputs[0].id == divNode.id ? inputs[1] : inputs[0]

            let axes = meanAttrs["axes"] as? [Int] ?? [-1]

            return FusionMatch(
                operations: [node],
                indices: [graph.nodes.firstIndex(where: { $0.id == node.id })!],
                metadata: [
                    "input": x,
                    "weight": weight,
                    "eps": eps,
                    "axes": axes
                ]
            )
        }

        return nil
    }

    public func apply(_ match: FusionMatch, to graph: IRGraph) -> IRGraph {
        let x = match.metadata["input"] as! LazyTensorHandle
        let weight = match.metadata["weight"] as! LazyTensorHandle
        let eps = match.metadata["eps"] as! Float
        let axes = match.metadata["axes"] as! [Int]

        let outputHandle = match.output!

        let fusedHandle = LazyTensorHandle(
            id: outputHandle.id,
            shape: outputHandle.shape,
            dtype: outputHandle.dtype,
            device: outputHandle.device
        )

        fusedHandle.irNode = .operation(
            op: .fusedRMSNorm,
            inputs: [x, weight],
            attributes: [
                "eps": eps,
                "axes": axes
            ]
        )

        return graph.replacing(match.operations, with: [fusedHandle])
    }

    private func findDivideInput(
        _ inputs: [LazyTensorHandle],
        opMap: [UInt64: (LazyTensorHandle, OpKind, [LazyTensorHandle], [String: Any])]
    ) -> (LazyTensorHandle, [LazyTensorHandle])? {
        for input in inputs {
            if let (node, op, divInputs, _) = opMap[input.id], op == .divide {
                return (node, divInputs)
            }
        }
        return nil
    }
}

// MARK: - MatMul + Bias + Activation Pattern

/// Detects and fuses matmul with bias and optional activation
///
/// Pattern:
/// ```
/// y = matmul(x, w)
/// z = y + bias
/// output = activation(z)  // optional
/// ```
public final class MatMulBiasActivationPattern: FusionPattern {

    public let name = "matmul-bias-activation"
    public let priority = 60

    public init() {}

    public func match(in graph: IRGraph) -> FusionMatch? {
        let opMap = PatternMatchingHelpers.buildOperationMap(graph)

        // Look for activation or add (bias)
        for node in graph.nodes {
            guard let (_, opKind, inputs, attrs) = opMap[node.id] else { continue }

            var activation: ActivationType = .none
            var biasAddNode: LazyTensorHandle?
            var biasAddInputs: [LazyTensorHandle]?

            // Check if this is an activation
            if opKind.isActivation {
                activation = opKind.asActivationType ?? .none

                // Input should be add (bias)
                guard let (addNode, addOp, addInputs, _) = opMap[inputs[0].id],
                      addOp == .add else { continue }

                biasAddNode = addNode
                biasAddInputs = addInputs
            } else if opKind == .add {
                // This might be the bias add directly (no activation)
                biasAddNode = node
                biasAddInputs = inputs
            } else {
                continue
            }

            guard let addInputs = biasAddInputs else { continue }

            // One input of add should be matmul
            let (matmulNode, bias) = identifyMatMulAndBias(addInputs, opMap: opMap)
            guard let mmNode = matmulNode,
                  let biasHandle = bias else { continue }

            guard let (_, mmOp, mmInputs, _) = opMap[mmNode.id],
                  mmOp == .matmul || mmOp == .batchedMatmul else { continue }

            let x = mmInputs[0]
            let w = mmInputs[1]

            // Verify bias is broadcastable (typically 1D or same shape as output)
            guard isBroadcastableBias(biasHandle, toMatmul: mmNode) else { continue }

            var matchedOps = [mmNode]
            if let addNode = biasAddNode, addNode.id != node.id {
                matchedOps.append(addNode)
            }
            matchedOps.append(node)

            return FusionMatch(
                operations: matchedOps,
                indices: matchedOps.compactMap { op in graph.nodes.firstIndex(where: { $0.id == op.id }) },
                metadata: [
                    "x": x,
                    "w": w,
                    "bias": biasHandle,
                    "activation": activation.rawValue,
                    "transA": false,
                    "transB": false
                ]
            )
        }

        return nil
    }

    public func apply(_ match: FusionMatch, to graph: IRGraph) -> IRGraph {
        let x = match.metadata["x"] as! LazyTensorHandle
        let w = match.metadata["w"] as! LazyTensorHandle
        let bias = match.metadata["bias"] as! LazyTensorHandle
        let activationStr = match.metadata["activation"] as! String
        let activation = ActivationType(rawValue: activationStr) ?? .none

        let outputHandle = match.output!

        let fusedHandle = LazyTensorHandle(
            id: outputHandle.id,
            shape: outputHandle.shape,
            dtype: outputHandle.dtype,
            device: outputHandle.device
        )

        fusedHandle.irNode = .operation(
            op: .fusedMatMulBiasActivation,
            inputs: [x, w, bias],
            attributes: [
                "activation": activation.rawValue,
                "transA": false,
                "transB": false
            ]
        )

        return graph.replacing(match.operations, with: [fusedHandle])
    }

    private func identifyMatMulAndBias(
        _ inputs: [LazyTensorHandle],
        opMap: [UInt64: (LazyTensorHandle, OpKind, [LazyTensorHandle], [String: Any])]
    ) -> (LazyTensorHandle?, LazyTensorHandle?) {
        for (i, input) in inputs.enumerated() {
            if let (_, op, _, _) = opMap[input.id],
               op == .matmul || op == .batchedMatmul {
                let biasIndex = i == 0 ? 1 : 0
                return (input, inputs[biasIndex])
            }
        }
        return (nil, nil)
    }

    private func isBroadcastableBias(_ bias: LazyTensorHandle, toMatmul matmul: LazyTensorHandle) -> Bool {
        // Bias should be 1D matching the last dimension of matmul output
        // or have the same shape as the output
        if bias.shape.count == 1 {
            return bias.shape[0] == matmul.shape.last
        }
        return bias.shape == matmul.shape
    }
}

// MARK: - Softmax Pattern

/// Detects and fuses softmax into a numerically stable version
///
/// Pattern:
/// ```
/// max = reduce_max(x, axis)
/// shifted = x - max
/// exp = exp(shifted)
/// sum = reduce_sum(exp, axis)
/// output = exp / sum
/// ```
public final class SoftmaxPattern: FusionPattern {

    public let name = "softmax"
    public let priority = 50

    public init() {}

    public func match(in graph: IRGraph) -> FusionMatch? {
        // Softmax is already an OpKind, but we can fuse the expanded version
        // This pattern matches when softmax is implemented as separate ops
        let opMap = PatternMatchingHelpers.buildOperationMap(graph)

        // Look for divide (final normalization)
        for node in graph.nodes {
            guard let (_, opKind, inputs, _) = opMap[node.id] else { continue }
            guard opKind == .divide else { continue }

            // Numerator should be exp
            guard let (expNode, expOp, expInputs, _) = opMap[inputs[0].id],
                  expOp == .exponential else { continue }

            // Denominator should be reduce_sum of exp
            guard let (_, sumOp, sumInputs, sumAttrs) = opMap[inputs[1].id],
                  sumOp == .reduceSum else { continue }

            // The sum should be over the same exp
            guard sumInputs[0].id == expNode.id else { continue }

            // exp input should be subtract (x - max)
            guard let (_, subOp, subInputs, _) = opMap[expInputs[0].id],
                  subOp == .subtract else { continue }

            let x = subInputs[0]
            let axis = (sumAttrs["axes"] as? [Int])?.first ?? -1

            return FusionMatch(
                operations: [node],
                indices: [graph.nodes.firstIndex(where: { $0.id == node.id })!],
                metadata: [
                    "input": x,
                    "axis": axis
                ]
            )
        }

        return nil
    }

    public func apply(_ match: FusionMatch, to graph: IRGraph) -> IRGraph {
        let x = match.metadata["input"] as! LazyTensorHandle
        let axis = match.metadata["axis"] as! Int

        let outputHandle = match.output!

        let fusedHandle = LazyTensorHandle(
            id: outputHandle.id,
            shape: outputHandle.shape,
            dtype: outputHandle.dtype,
            device: outputHandle.device
        )

        fusedHandle.irNode = .operation(
            op: .fusedSoftmax,
            inputs: [x],
            attributes: ["axis": axis]
        )

        return graph.replacing(match.operations, with: [fusedHandle])
    }
}

// MARK: - GELU Pattern

/// Detects and fuses GELU activation
///
/// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
/// Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
public final class GeluPattern: FusionPattern {

    public let name = "gelu"
    public let priority = 40

    public init() {}

    public func match(in graph: IRGraph) -> FusionMatch? {
        let opMap = PatternMatchingHelpers.buildOperationMap(graph)

        // Look for the characteristic multiply pattern of GELU
        // GELU has form: 0.5 * x * (something with tanh or erf)

        for node in graph.nodes {
            guard let (_, opKind, inputs, _) = opMap[node.id] else { continue }
            guard opKind == .multiply else { continue }

            // Check for 0.5 constant
            if PatternMatchingHelpers.isConstantWithValue(inputs[0], 0.5) ||
               PatternMatchingHelpers.isConstantWithValue(inputs[1], 0.5) {

                // Found potential GELU - need more validation
                // For now, create a basic match
                // A complete implementation would trace the full pattern

                // Get the non-0.5 input
                let otherInput = PatternMatchingHelpers.isConstantWithValue(inputs[0], 0.5) ? inputs[1] : inputs[0]

                // Check if it contains tanh (approximate) or could contain erf (exact)
                if let (_, otherOp, _, _) = opMap[otherInput.id], otherOp == .multiply {
                    // This might be x * (1 + tanh(...)) - potential GELU
                    // For simplicity, skip detailed validation here

                    // Return nil for now - GELU is already an OpKind
                    // This pattern is here for when GELU is decomposed
                    continue
                }
            }
        }

        return nil
    }

    public func apply(_ match: FusionMatch, to graph: IRGraph) -> IRGraph {
        let x = match.metadata["input"] as! LazyTensorHandle
        let approximate = match.metadata["approximate"] as! Bool

        let outputHandle = match.output!

        let fusedHandle = LazyTensorHandle(
            id: outputHandle.id,
            shape: outputHandle.shape,
            dtype: outputHandle.dtype,
            device: outputHandle.device
        )

        fusedHandle.irNode = .operation(
            op: .fusedGelu,
            inputs: [x],
            attributes: ["approximate": approximate]
        )

        return graph.replacing(match.operations, with: [fusedHandle])
    }
}
