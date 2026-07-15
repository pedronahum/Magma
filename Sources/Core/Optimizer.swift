// Magma - Optimizer Module
// Optimization algorithms for training neural networks

import Foundation
import _Differentiation
import LazyTensor
import StableHLO
import XLARuntime

// MARK: - Parameter Group

/// A group of parameters with shared optimization settings.
///
/// Use parameter groups to apply different settings (like weight decay) to
/// different parts of a model. This is common for transformers where you
/// typically don't apply weight decay to biases and LayerNorm parameters.
///
/// Example:
/// ```swift
/// // Separate weight matrices from biases/norms
/// let weightParams = model.parameters().filter { $0.name?.contains("weight") ?? false }
/// let otherParams = model.parameters().filter { !($0.name?.contains("weight") ?? false) }
///
/// let groups = [
///     ParameterGroup(parameters: weightParams, weightDecay: 0.1),
///     ParameterGroup(parameters: otherParams, weightDecay: 0.0),
/// ]
///
/// var optimizer = optim.AdamW(parameterGroups: groups, lr: 6e-4)
/// ```
public struct ParameterGroup {
    /// Parameters in this group
    public let parameters: [Parameter]

    /// Learning rate multiplier for this group (applied on top of base LR)
    public var lrMultiplier: Float

    /// Weight decay for this group
    public var weightDecay: Float

    /// Creates a parameter group.
    ///
    /// - Parameters:
    ///   - parameters: Parameters in this group.
    ///   - weightDecay: Weight decay for this group. Defaults to 0.
    ///   - lrMultiplier: Learning rate multiplier. Defaults to 1.0.
    public init(
        parameters: [Parameter],
        weightDecay: Float = 0,
        lrMultiplier: Float = 1.0
    ) {
        self.parameters = parameters
        self.weightDecay = weightDecay
        self.lrMultiplier = lrMultiplier
    }
}

// MARK: - Optimizer Protocol

/// Protocol for all optimizers.
///
/// Optimizers update model parameters based on computed gradients.
///
/// Example:
/// ```swift
/// var optimizer = optim.SGD(parameters: model.parameters(), lr: 0.01)
/// let grads = computeGradients(model, input, target)
/// optimizer.step(grads)
/// ```
public protocol Optimizer {
    /// The parameters this optimizer updates, in the order that `step(_:)`'s
    /// positional gradient array must follow.
    var parameters: [Parameter] { get }

    /// Update parameters using the given gradients.
    ///
    /// - Parameter gradients: Array of gradient tensors, one per parameter, in the
    ///   same order as `parameters`. Prefer the `[Parameter: Tensor]` overload,
    ///   which matches gradients to parameters by identity and can't be silently
    ///   misordered.
    mutating func step(_ gradients: [Tensor<Float>])

    /// Reset all optimizer state (e.g., momentum buffers).
    mutating func zeroGrad()

    /// The current learning rate.
    var learningRate: Float { get set }
}

extension Optimizer {
    /// Update parameters using gradients matched to parameters **by identity**
    /// rather than by array position.
    ///
    /// This is the safe way to feed autodiff results into an optimizer: the result
    /// does not depend on the order in which you assembled the gradients, only on
    /// which `Parameter` each gradient belongs to — so it cannot be silently
    /// misordered the way the positional `step([Tensor])` can.
    ///
    /// Every parameter the optimizer owns with `requiresGrad == true` must have an
    /// entry in `gradients`; frozen parameters may be omitted. Keys that are not
    /// owned by this optimizer are ignored.
    public mutating func step(_ gradients: [Parameter: Tensor<Float>]) {
        let positional: [Tensor<Float>] = parameters.map { p in
            if let g = gradients[p] { return g }
            precondition(!p.requiresGrad,
                "step(_:): no gradient supplied for trainable parameter"
                    + (p.name.map { " '\($0)'" } ?? ""))
            // Frozen params are skipped inside `step`; a zero placeholder keeps the
            // positional array aligned.
            return Tensor<Float>.zeros(p.value.shape, on: p.value.device)
        }
        step(positional)
    }
}

// MARK: - SGD Optimizer

extension optim {
    /// Stochastic Gradient Descent optimizer.
    ///
    /// Implements SGD with optional momentum and weight decay.
    ///
    /// Update rule (with momentum):
    /// ```
    /// v_t = momentum * v_{t-1} + gradient
    /// parameter = parameter - lr * v_t
    /// ```
    ///
    /// Example:
    /// ```swift
    /// var optimizer = optim.SGD(parameters: model.parameters(), lr: 0.01, momentum: 0.9)
    ///
    /// for batch in dataLoader {
    ///     let loss = model.loss(batch)
    ///     let grads = computeGradients(...)
    ///     optimizer.step(grads)
    /// }
    /// ```
    public struct SGD: Optimizer {
        /// Parameters to optimize
        public let parameters: [Parameter]

        /// Learning rate
        public var learningRate: Float

        /// Momentum factor (0 = no momentum)
        public let momentum: Float

        /// Weight decay (L2 regularization)
        public let weightDecay: Float

        /// Whether to use Nesterov momentum
        public let nesterov: Bool

        /// Velocity buffers for momentum
        private var velocities: [Tensor<Float>?]

        /// Creates an SGD optimizer.
        ///
        /// - Parameters:
        ///   - parameters: Parameters to optimize.
        ///   - lr: Learning rate. Defaults to 0.01.
        ///   - momentum: Momentum factor. Defaults to 0.
        ///   - weightDecay: Weight decay (L2 penalty). Defaults to 0.
        ///   - nesterov: Whether to use Nesterov momentum. Defaults to false.
        public init(
            parameters: [Parameter],
            lr: Float = 0.01,
            momentum: Float = 0,
            weightDecay: Float = 0,
            nesterov: Bool = false
        ) {
            precondition(lr > 0, "Learning rate must be positive")
            precondition(momentum >= 0, "Momentum must be non-negative")
            precondition(weightDecay >= 0, "Weight decay must be non-negative")

            self.parameters = parameters
            self.learningRate = lr
            self.momentum = momentum
            self.weightDecay = weightDecay
            self.nesterov = nesterov
            self.velocities = Array(repeating: nil, count: parameters.count)
        }

        /// Update parameters using gradients.
        ///
        /// - Parameter gradients: Gradient for each parameter.
        public mutating func step(_ gradients: [Tensor<Float>]) {
            precondition(gradients.count == parameters.count,
                        "Number of gradients must match number of parameters")

            for i in 0..<parameters.count {
                let param = parameters[i]
                guard param.requiresGrad else { continue }

                var grad = gradients[i]

                // Apply weight decay (L2 regularization)
                if weightDecay != 0 {
                    grad = grad + param.value * Tensor<Float>.full([], weightDecay, on: grad.device)
                }

                // Apply momentum
                if momentum != 0 {
                    if let vel = velocities[i] {
                        // v = momentum * v + grad
                        let newVel = vel * Tensor<Float>.full([], momentum, on: vel.device) + grad
                        velocities[i] = newVel

                        if nesterov {
                            // Nesterov: use grad + momentum * v
                            grad = grad + newVel * Tensor<Float>.full([], momentum, on: newVel.device)
                        } else {
                            grad = newVel
                        }
                    } else {
                        // First step: initialize velocity
                        velocities[i] = grad
                    }
                }

                // Update parameter: p = p - lr * grad
                let update = grad * Tensor<Float>.full([], learningRate, on: grad.device)
                param.value = param.value - update
            }
        }

        /// Reset optimizer state (velocities).
        public mutating func zeroGrad() {
            velocities = Array(repeating: nil, count: parameters.count)
        }
    }
}

// MARK: - Adam Optimizer

extension optim {
    /// Adam optimizer (Adaptive Moment Estimation).
    ///
    /// Combines momentum with adaptive learning rates per parameter.
    ///
    /// Update rule:
    /// ```
    /// m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    /// v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    /// m_hat = m_t / (1 - beta1^t)
    /// v_hat = v_t / (1 - beta2^t)
    /// p_t = p_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    /// ```
    ///
    /// Example:
    /// ```swift
    /// var optimizer = optim.Adam(parameters: model.parameters(), lr: 0.001)
    /// ```
    public struct Adam: Optimizer {
        /// Parameters to optimize
        public let parameters: [Parameter]

        /// Learning rate
        public var learningRate: Float

        /// First moment decay rate
        public let beta1: Float

        /// Second moment decay rate
        public let beta2: Float

        /// Small constant for numerical stability
        public let eps: Float

        /// Weight decay (AdamW style)
        public let weightDecay: Float

        /// First moment estimates
        private var m: [Tensor<Float>?]

        /// Second moment estimates
        private var v: [Tensor<Float>?]

        /// Current time step
        private var t: Int

        /// Creates an Adam optimizer.
        ///
        /// - Parameters:
        ///   - parameters: Parameters to optimize.
        ///   - lr: Learning rate. Defaults to 0.001.
        ///   - beta1: First moment decay. Defaults to 0.9.
        ///   - beta2: Second moment decay. Defaults to 0.999.
        ///   - eps: Numerical stability constant. Defaults to 1e-8.
        ///   - weightDecay: Weight decay. Defaults to 0.
        public init(
            parameters: [Parameter],
            lr: Float = 0.001,
            beta1: Float = 0.9,
            beta2: Float = 0.999,
            eps: Float = 1e-8,
            weightDecay: Float = 0
        ) {
            precondition(lr > 0, "Learning rate must be positive")
            precondition(beta1 >= 0 && beta1 < 1, "beta1 must be in [0, 1)")
            precondition(beta2 >= 0 && beta2 < 1, "beta2 must be in [0, 1)")

            self.parameters = parameters
            self.learningRate = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.weightDecay = weightDecay
            self.m = Array(repeating: nil, count: parameters.count)
            self.v = Array(repeating: nil, count: parameters.count)
            self.t = 0
        }

        /// Update parameters using gradients.
        public mutating func step(_ gradients: [Tensor<Float>]) {
            precondition(gradients.count == parameters.count,
                        "Number of gradients must match number of parameters")

            t += 1

            guard let device = gradients.first?.device else { return }

            // Hoist parameter-invariant scalar constants out of the loop: these
            // are identical for every parameter, so build them once per step
            // instead of ~8 constant nodes per parameter per step. eps is a
            // scalar broadcast at use rather than a full-shape constant.
            let beta1T = Tensor<Float>.full([], beta1, on: device)
            let oneMinusBeta1 = Tensor<Float>.full([], 1 - beta1, on: device)
            let beta2T = Tensor<Float>.full([], beta2, on: device)
            let oneMinusBeta2 = Tensor<Float>.full([], 1 - beta2, on: device)
            let biasCorr1 = Tensor<Float>.full([], 1 - pow(beta1, Float(t)), on: device)
            let biasCorr2 = Tensor<Float>.full([], 1 - pow(beta2, Float(t)), on: device)
            let epsT = Tensor<Float>.full([], eps, on: device)
            let lrT = Tensor<Float>.full([], learningRate, on: device)
            let lrWdT = weightDecay != 0
                ? Tensor<Float>.full([], learningRate * weightDecay, on: device) : nil

            for i in 0..<parameters.count {
                let param = parameters[i]
                guard param.requiresGrad else { continue }

                let grad = gradients[i]

                // AdamW weight decay (decoupled)
                if let lrWdT = lrWdT {
                    param.value = param.value - param.value * lrWdT
                }

                // Update first moment: m = beta1 * m + (1 - beta1) * g
                if let mPrev = m[i] {
                    m[i] = mPrev * beta1T + grad * oneMinusBeta1
                } else {
                    m[i] = grad * oneMinusBeta1
                }

                // Update second moment: v = beta2 * v + (1 - beta2) * g^2
                let gradSquared = grad * grad
                if let vPrev = v[i] {
                    v[i] = vPrev * beta2T + gradSquared * oneMinusBeta2
                } else {
                    v[i] = gradSquared * oneMinusBeta2
                }

                // Bias correction
                let mHat = m[i]! / biasCorr1
                let vHat = v[i]! / biasCorr2

                // Update: p = p - lr * m_hat / (sqrt(v_hat) + eps)
                let sqrtV = vHat.sqrt()
                let update = mHat / (sqrtV + epsT.broadcast(to: sqrtV.shape))
                param.value = param.value - update * lrT
            }
        }

        /// Reset optimizer state.
        public mutating func zeroGrad() {
            m = Array(repeating: nil, count: parameters.count)
            v = Array(repeating: nil, count: parameters.count)
            t = 0
        }
    }
}

// MARK: - AdamW Optimizer

extension optim {
    /// AdamW optimizer (Adam with decoupled weight decay).
    ///
    /// This is Adam with weight decay applied directly to parameters,
    /// rather than being added to the gradient.
    ///
    /// Example:
    /// ```swift
    /// var optimizer = optim.AdamW(parameters: model.parameters(), lr: 0.001, weightDecay: 0.01)
    /// ```
    public typealias AdamW = Adam  // Adam already implements decoupled weight decay
}

// MARK: - AdamW with Parameter Groups

extension optim {
    /// AdamW optimizer with support for parameter groups.
    ///
    /// This allows applying different weight decay and learning rate multipliers
    /// to different parameters. Essential for transformer training where you
    /// typically don't apply weight decay to biases and LayerNorm.
    ///
    /// Example:
    /// ```swift
    /// // Separate parameters by type
    /// let decayParams = model.parameters().filter { p in
    ///     guard let name = p.name else { return true }
    ///     return !name.contains("bias") && !name.contains("norm")
    /// }
    /// let noDecayParams = model.parameters().filter { p in
    ///     guard let name = p.name else { return false }
    ///     return name.contains("bias") || name.contains("norm")
    /// }
    ///
    /// var optimizer = optim.AdamWGroups(
    ///     parameterGroups: [
    ///         ParameterGroup(parameters: decayParams, weightDecay: 0.1),
    ///         ParameterGroup(parameters: noDecayParams, weightDecay: 0.0),
    ///     ],
    ///     lr: 6e-4
    /// )
    /// ```
    public struct AdamWGroups: Optimizer {
        /// Parameter groups with their settings
        public let parameterGroups: [ParameterGroup]

        /// Base learning rate
        public var learningRate: Float

        /// First moment decay rate
        public let beta1: Float

        /// Second moment decay rate
        public let beta2: Float

        /// Small constant for numerical stability
        public let eps: Float

        /// First moment estimates (flattened across all groups)
        private var m: [Tensor<Float>?]

        /// Second moment estimates (flattened across all groups)
        private var v: [Tensor<Float>?]

        /// Current time step
        private var t: Int

        /// All parameters flattened (for gradient indexing)
        private let allParameters: [Parameter]

        /// All parameters across every group, in positional gradient order.
        public var parameters: [Parameter] { allParameters }

        /// Group index for each parameter
        private let parameterGroupIndex: [Int]

        /// Creates an AdamW optimizer with parameter groups.
        ///
        /// - Parameters:
        ///   - parameterGroups: Parameter groups with per-group settings.
        ///   - lr: Base learning rate. Defaults to 0.001.
        ///   - beta1: First moment decay. Defaults to 0.9.
        ///   - beta2: Second moment decay. Defaults to 0.999.
        ///   - eps: Numerical stability constant. Defaults to 1e-8.
        public init(
            parameterGroups: [ParameterGroup],
            lr: Float = 0.001,
            beta1: Float = 0.9,
            beta2: Float = 0.999,
            eps: Float = 1e-8
        ) {
            precondition(!parameterGroups.isEmpty, "At least one parameter group required")
            precondition(lr > 0, "Learning rate must be positive")
            precondition(beta1 >= 0 && beta1 < 1, "beta1 must be in [0, 1)")
            precondition(beta2 >= 0 && beta2 < 1, "beta2 must be in [0, 1)")

            self.parameterGroups = parameterGroups
            self.learningRate = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps

            // Flatten all parameters and track their group
            var allParams: [Parameter] = []
            var groupIndices: [Int] = []
            for (groupIdx, group) in parameterGroups.enumerated() {
                for param in group.parameters {
                    allParams.append(param)
                    groupIndices.append(groupIdx)
                }
            }
            self.allParameters = allParams
            self.parameterGroupIndex = groupIndices

            self.m = Array(repeating: nil, count: allParams.count)
            self.v = Array(repeating: nil, count: allParams.count)
            self.t = 0
        }

        /// Update parameters using gradients.
        public mutating func step(_ gradients: [Tensor<Float>]) {
            precondition(gradients.count == allParameters.count,
                        "Number of gradients must match total parameters across all groups")

            t += 1

            for i in 0..<allParameters.count {
                let param = allParameters[i]
                guard param.requiresGrad else { continue }

                let groupIdx = parameterGroupIndex[i]
                let group = parameterGroups[groupIdx]
                let grad = gradients[i]

                // Effective learning rate for this group
                let effectiveLR = learningRate * group.lrMultiplier

                // AdamW weight decay (decoupled)
                if group.weightDecay != 0 {
                    param.value = param.value - param.value * Tensor<Float>.full([], effectiveLR * group.weightDecay, on: param.value.device)
                }

                // Update first moment: m = beta1 * m + (1 - beta1) * g
                let beta1Tensor = Tensor<Float>.full([], beta1, on: grad.device)
                let oneMinusBeta1 = Tensor<Float>.full([], 1 - beta1, on: grad.device)
                if let mPrev = m[i] {
                    m[i] = mPrev * beta1Tensor + grad * oneMinusBeta1
                } else {
                    m[i] = grad * oneMinusBeta1
                }

                // Update second moment: v = beta2 * v + (1 - beta2) * g^2
                let beta2Tensor = Tensor<Float>.full([], beta2, on: grad.device)
                let oneMinusBeta2 = Tensor<Float>.full([], 1 - beta2, on: grad.device)
                let gradSquared = grad * grad
                if let vPrev = v[i] {
                    v[i] = vPrev * beta2Tensor + gradSquared * oneMinusBeta2
                } else {
                    v[i] = gradSquared * oneMinusBeta2
                }

                // Bias correction
                let beta1Power = pow(beta1, Float(t))
                let beta2Power = pow(beta2, Float(t))
                let mHat = m[i]! / Tensor<Float>.full([], 1 - beta1Power, on: grad.device)
                let vHat = v[i]! / Tensor<Float>.full([], 1 - beta2Power, on: grad.device)

                // Update: p = p - lr * m_hat / (sqrt(v_hat) + eps)
                let sqrtV = vHat.sqrt()
                let epsTensor = Tensor<Float>.full(sqrtV.shape, eps, on: grad.device)
                let update = mHat / (sqrtV + epsTensor)
                param.value = param.value - update * Tensor<Float>.full([], effectiveLR, on: grad.device)
            }
        }

        /// Reset optimizer state.
        public mutating func zeroGrad() {
            m = Array(repeating: nil, count: allParameters.count)
            v = Array(repeating: nil, count: allParameters.count)
            t = 0
        }
    }
}

// MARK: - Tensor sqrt Extension

extension Tensor where Scalar: TensorScalar & BinaryFloatingPoint {
    /// Element-wise square root
    public func sqrt() -> Tensor {
        let id = TensorRegistry.shared.nextTensorId()
        let handle = LazyTensorHandle(
            id: id,
            shape: shape,
            dtype: dtype,
            device: device
        )
        handle.irNode = .operation(op: .sqrt, inputs: [self.handle], attributes: [:])
        TensorRegistry.shared.registerPending(handle)
        return Tensor(handle: handle)
    }

    /// VJP for sqrt: d(sqrt(x))/dx = 0.5 / sqrt(x)
    @derivative(of: sqrt)
    public func vjpSqrt() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
        let result = self.sqrt()
        return (result, { v in
            let half = Tensor<Scalar>.full([], Scalar(0.5), on: v.device)
            return v * half / result
        })
    }
}

// MARK: - Learning Rate Schedulers

/// Protocol for learning rate schedulers
public protocol LRScheduler {
    /// Update the learning rate based on current step/epoch
    mutating func step()

    /// Get the current learning rate
    var currentLR: Float { get }

    /// Reset the scheduler state
    mutating func reset()
}

extension optim {
    /// Step learning rate scheduler.
    ///
    /// Decays learning rate by gamma every stepSize epochs.
    public struct StepLR: LRScheduler {
        private let baseLR: Float
        private let stepSize: Int
        private let gamma: Float
        private var epoch: Int = 0

        public var currentLR: Float {
            baseLR * pow(gamma, Float(epoch / stepSize))
        }

        public init(baseLR: Float, stepSize: Int, gamma: Float = 0.1) {
            self.baseLR = baseLR
            self.stepSize = stepSize
            self.gamma = gamma
        }

        public mutating func step() {
            epoch += 1
        }

        public mutating func reset() {
            epoch = 0
        }
    }

    /// Exponential learning rate scheduler.
    ///
    /// Decays learning rate by gamma every epoch.
    public struct ExponentialLR: LRScheduler {
        private let baseLR: Float
        private let gamma: Float
        private var epoch: Int = 0

        public var currentLR: Float {
            baseLR * pow(gamma, Float(epoch))
        }

        public init(baseLR: Float, gamma: Float = 0.95) {
            self.baseLR = baseLR
            self.gamma = gamma
        }

        public mutating func step() {
            epoch += 1
        }

        public mutating func reset() {
            epoch = 0
        }
    }

    /// Cosine annealing learning rate scheduler.
    ///
    /// Anneals learning rate following a cosine curve.
    public struct CosineAnnealingLR: LRScheduler {
        private let baseLR: Float
        private let minLR: Float
        private let totalEpochs: Int
        private var epoch: Int = 0

        public var currentLR: Float {
            let progress = Float(epoch) / Float(totalEpochs)
            return minLR + (baseLR - minLR) * (1 + cos(Float.pi * progress)) / 2
        }

        public init(baseLR: Float, totalEpochs: Int, minLR: Float = 0) {
            self.baseLR = baseLR
            self.totalEpochs = totalEpochs
            self.minLR = minLR
        }

        public mutating func step() {
            epoch = Swift.min(epoch + 1, totalEpochs)
        }

        public mutating func reset() {
            epoch = 0
        }
    }

    /// Linear warmup then constant learning rate scheduler.
    public struct WarmupLR: LRScheduler {
        private let baseLR: Float
        private let warmupSteps: Int
        private var step_: Int = 0

        public var currentLR: Float {
            if step_ < warmupSteps {
                return baseLR * Float(step_ + 1) / Float(warmupSteps)
            }
            return baseLR
        }

        public init(baseLR: Float, warmupSteps: Int) {
            self.baseLR = baseLR
            self.warmupSteps = warmupSteps
        }

        public mutating func step() {
            step_ += 1
        }

        public mutating func reset() {
            step_ = 0
        }
    }

    /// Linear warmup then cosine decay scheduler (common for transformers).
    public struct WarmupCosineScheduler: LRScheduler {
        private let baseLR: Float
        private let warmupSteps: Int
        private let totalSteps: Int
        private let minLR: Float
        private var step_: Int = 0

        public var currentLR: Float {
            if step_ < warmupSteps {
                // Linear warmup
                return baseLR * Float(step_ + 1) / Float(warmupSteps)
            }
            // Cosine decay
            let decaySteps = totalSteps - warmupSteps
            let progress = Float(step_ - warmupSteps) / Float(decaySteps)
            return minLR + (baseLR - minLR) * (1 + cos(Float.pi * progress)) / 2
        }

        public init(baseLR: Float, warmupSteps: Int, totalSteps: Int, minLR: Float = 0) {
            self.baseLR = baseLR
            self.warmupSteps = warmupSteps
            self.totalSteps = totalSteps
            self.minLR = minLR
        }

        public mutating func step() {
            step_ = Swift.min(step_ + 1, totalSteps)
        }

        public mutating func reset() {
            step_ = 0
        }
    }
}

// MARK: - RMSProp Optimizer

extension optim {
    /// RMSProp optimizer.
    ///
    /// Implements the RMSProp optimization algorithm. RMSProp divides the learning rate
    /// by an exponentially decaying average of squared gradients.
    ///
    /// Update rule:
    /// ```
    /// v_t = rho * v_{t-1} + (1 - rho) * g_t^2
    /// p_t = p_{t-1} - lr * g_t / (sqrt(v_t) + eps)
    /// ```
    ///
    /// Reference: ["Lecture 6.5 - rmsprop"](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    /// (Tieleman and Hinton, 2012)
    ///
    /// Example:
    /// ```swift
    /// var optimizer = optim.RMSProp(parameters: model.parameters(), lr: 0.01)
    /// ```
    public struct RMSProp: Optimizer {
        /// Parameters to optimize
        public let parameters: [Parameter]

        /// Learning rate
        public var learningRate: Float

        /// Decay rate for running average of squared gradients
        public let rho: Float

        /// Small constant for numerical stability
        public let eps: Float

        /// Weight decay (L2 regularization)
        public let weightDecay: Float

        /// Momentum factor
        public let momentum: Float

        /// Whether to center the gradient (subtract mean)
        public let centered: Bool

        /// Running average of squared gradients
        private var squareAvg: [Tensor<Float>?]

        /// Running average of gradients (for centered mode)
        private var gradAvg: [Tensor<Float>?]

        /// Momentum buffer
        private var momentumBuffer: [Tensor<Float>?]

        /// Creates an RMSProp optimizer.
        ///
        /// - Parameters:
        ///   - parameters: Parameters to optimize.
        ///   - lr: Learning rate. Defaults to 0.01.
        ///   - rho: Decay rate (alpha). Defaults to 0.99.
        ///   - eps: Numerical stability constant. Defaults to 1e-8.
        ///   - weightDecay: Weight decay. Defaults to 0.
        ///   - momentum: Momentum factor. Defaults to 0.
        ///   - centered: If true, normalize gradients by variance estimate. Defaults to false.
        public init(
            parameters: [Parameter],
            lr: Float = 0.01,
            rho: Float = 0.99,
            eps: Float = 1e-8,
            weightDecay: Float = 0,
            momentum: Float = 0,
            centered: Bool = false
        ) {
            precondition(lr > 0, "Learning rate must be positive")
            precondition(rho >= 0 && rho <= 1, "Rho must be in [0, 1]")
            precondition(momentum >= 0, "Momentum must be non-negative")

            self.parameters = parameters
            self.learningRate = lr
            self.rho = rho
            self.eps = eps
            self.weightDecay = weightDecay
            self.momentum = momentum
            self.centered = centered
            self.squareAvg = Array(repeating: nil, count: parameters.count)
            self.gradAvg = Array(repeating: nil, count: parameters.count)
            self.momentumBuffer = Array(repeating: nil, count: parameters.count)
        }

        /// Update parameters using gradients.
        public mutating func step(_ gradients: [Tensor<Float>]) {
            precondition(gradients.count == parameters.count,
                        "Number of gradients must match number of parameters")

            for i in 0..<parameters.count {
                let param = parameters[i]
                guard param.requiresGrad else { continue }

                var grad = gradients[i]

                // Apply weight decay
                if weightDecay != 0 {
                    grad = grad + param.value * Tensor<Float>.full([], weightDecay, on: grad.device)
                }

                let rhoTensor = Tensor<Float>.full([], rho, on: grad.device)
                let oneMinusRho = Tensor<Float>.full([], 1 - rho, on: grad.device)

                // Update running average of squared gradients
                let gradSquared = grad * grad
                if let sq = squareAvg[i] {
                    squareAvg[i] = sq * rhoTensor + gradSquared * oneMinusRho
                } else {
                    squareAvg[i] = gradSquared * oneMinusRho
                }

                var avg: Tensor<Float>
                if centered {
                    // Update running average of gradients
                    if let ga = gradAvg[i] {
                        gradAvg[i] = ga * rhoTensor + grad * oneMinusRho
                    } else {
                        gradAvg[i] = grad * oneMinusRho
                    }
                    // avg = square_avg - grad_avg^2
                    avg = squareAvg[i]! - gradAvg[i]! * gradAvg[i]!
                } else {
                    avg = squareAvg[i]!
                }

                let epsTensor = Tensor<Float>.full(avg.shape, eps, on: grad.device)
                let denom = avg.sqrt() + epsTensor

                var update: Tensor<Float>
                if momentum > 0 {
                    let momTensor = Tensor<Float>.full([], momentum, on: grad.device)
                    if let buf = momentumBuffer[i] {
                        momentumBuffer[i] = buf * momTensor + grad / denom
                    } else {
                        momentumBuffer[i] = grad / denom
                    }
                    update = momentumBuffer[i]!
                } else {
                    update = grad / denom
                }

                param.value = param.value - update * Tensor<Float>.full([], learningRate, on: grad.device)
            }
        }

        /// Reset optimizer state.
        public mutating func zeroGrad() {
            squareAvg = Array(repeating: nil, count: parameters.count)
            gradAvg = Array(repeating: nil, count: parameters.count)
            momentumBuffer = Array(repeating: nil, count: parameters.count)
        }
    }
}

// MARK: - AdaGrad Optimizer

extension optim {
    /// AdaGrad optimizer (Adaptive Gradient).
    ///
    /// Adapts learning rates based on accumulated squared gradients.
    /// Parameters that receive larger gradients have smaller effective learning rates.
    ///
    /// Update rule:
    /// ```
    /// accumulator_t = accumulator_{t-1} + g_t^2
    /// p_t = p_{t-1} - lr * g_t / (sqrt(accumulator_t) + eps)
    /// ```
    ///
    /// Reference: ["Adaptive Subgradient Methods for Online Learning"](http://jmlr.org/papers/v12/duchi11a.html)
    /// (Duchi et al, 2011)
    ///
    /// Example:
    /// ```swift
    /// var optimizer = optim.AdaGrad(parameters: model.parameters(), lr: 0.01)
    /// ```
    public struct AdaGrad: Optimizer {
        /// Parameters to optimize
        public let parameters: [Parameter]

        /// Learning rate
        public var learningRate: Float

        /// Initial accumulator value
        public let initialAccumulatorValue: Float

        /// Small constant for numerical stability
        public let eps: Float

        /// Weight decay (L2 regularization)
        public let weightDecay: Float

        /// Sum of squared gradients
        private var accumulator: [Tensor<Float>?]

        /// Creates an AdaGrad optimizer.
        ///
        /// - Parameters:
        ///   - parameters: Parameters to optimize.
        ///   - lr: Learning rate. Defaults to 0.01.
        ///   - initialAccumulatorValue: Starting accumulator value. Defaults to 0.
        ///   - eps: Numerical stability constant. Defaults to 1e-10.
        ///   - weightDecay: Weight decay. Defaults to 0.
        public init(
            parameters: [Parameter],
            lr: Float = 0.01,
            initialAccumulatorValue: Float = 0,
            eps: Float = 1e-10,
            weightDecay: Float = 0
        ) {
            precondition(lr > 0, "Learning rate must be positive")
            precondition(initialAccumulatorValue >= 0, "Initial accumulator value must be non-negative")

            self.parameters = parameters
            self.learningRate = lr
            self.initialAccumulatorValue = initialAccumulatorValue
            self.eps = eps
            self.weightDecay = weightDecay
            self.accumulator = Array(repeating: nil, count: parameters.count)
        }

        /// Update parameters using gradients.
        public mutating func step(_ gradients: [Tensor<Float>]) {
            precondition(gradients.count == parameters.count,
                        "Number of gradients must match number of parameters")

            for i in 0..<parameters.count {
                let param = parameters[i]
                guard param.requiresGrad else { continue }

                var grad = gradients[i]

                // Apply weight decay
                if weightDecay != 0 {
                    grad = grad + param.value * Tensor<Float>.full([], weightDecay, on: grad.device)
                }

                // Update accumulator: acc = acc + grad^2
                let gradSquared = grad * grad
                if let acc = accumulator[i] {
                    accumulator[i] = acc + gradSquared
                } else {
                    if initialAccumulatorValue > 0 {
                        accumulator[i] = Tensor<Float>.full(grad.shape, initialAccumulatorValue, on: grad.device) + gradSquared
                    } else {
                        accumulator[i] = gradSquared
                    }
                }

                // Update: p = p - lr * grad / (sqrt(acc) + eps)
                let epsTensor = Tensor<Float>.full(grad.shape, eps, on: grad.device)
                let denom = accumulator[i]!.sqrt() + epsTensor
                let update = grad / denom

                param.value = param.value - update * Tensor<Float>.full([], learningRate, on: grad.device)
            }
        }

        /// Reset optimizer state.
        public mutating func zeroGrad() {
            accumulator = Array(repeating: nil, count: parameters.count)
        }
    }
}

// MARK: - AdaDelta Optimizer

extension optim {
    /// AdaDelta optimizer.
    ///
    /// An extension of AdaGrad that seeks to reduce its aggressive, monotonically
    /// decreasing learning rate. Instead of accumulating all past squared gradients,
    /// AdaDelta restricts the window of accumulated past gradients to some fixed size.
    ///
    /// Update rule:
    /// ```
    /// avg_sq_t = rho * avg_sq_{t-1} + (1 - rho) * g_t^2
    /// delta_t = sqrt(acc_delta_{t-1} + eps) / sqrt(avg_sq_t + eps) * g_t
    /// acc_delta_t = rho * acc_delta_{t-1} + (1 - rho) * delta_t^2
    /// p_t = p_{t-1} - lr * delta_t
    /// ```
    ///
    /// Reference: ["ADADELTA: An Adaptive Learning Rate Method"](https://arxiv.org/abs/1212.5701)
    /// (Zeiler, 2012)
    ///
    /// Example:
    /// ```swift
    /// var optimizer = optim.AdaDelta(parameters: model.parameters())
    /// ```
    public struct AdaDelta: Optimizer {
        /// Parameters to optimize
        public let parameters: [Parameter]

        /// Learning rate (scale factor, default 1.0)
        public var learningRate: Float

        /// Decay rate
        public let rho: Float

        /// Small constant for numerical stability
        public let eps: Float

        /// Weight decay (L2 regularization)
        public let weightDecay: Float

        /// Running average of squared gradients
        private var squareAvg: [Tensor<Float>?]

        /// Running average of squared parameter updates
        private var accDelta: [Tensor<Float>?]

        /// Creates an AdaDelta optimizer.
        ///
        /// - Parameters:
        ///   - parameters: Parameters to optimize.
        ///   - lr: Learning rate (scaling factor). Defaults to 1.0.
        ///   - rho: Decay rate. Defaults to 0.9.
        ///   - eps: Numerical stability constant. Defaults to 1e-6.
        ///   - weightDecay: Weight decay. Defaults to 0.
        public init(
            parameters: [Parameter],
            lr: Float = 1.0,
            rho: Float = 0.9,
            eps: Float = 1e-6,
            weightDecay: Float = 0
        ) {
            precondition(lr > 0, "Learning rate must be positive")
            precondition(rho >= 0 && rho <= 1, "Rho must be in [0, 1]")

            self.parameters = parameters
            self.learningRate = lr
            self.rho = rho
            self.eps = eps
            self.weightDecay = weightDecay
            self.squareAvg = Array(repeating: nil, count: parameters.count)
            self.accDelta = Array(repeating: nil, count: parameters.count)
        }

        /// Update parameters using gradients.
        public mutating func step(_ gradients: [Tensor<Float>]) {
            precondition(gradients.count == parameters.count,
                        "Number of gradients must match number of parameters")

            for i in 0..<parameters.count {
                let param = parameters[i]
                guard param.requiresGrad else { continue }

                var grad = gradients[i]

                // Apply weight decay
                if weightDecay != 0 {
                    grad = grad + param.value * Tensor<Float>.full([], weightDecay, on: grad.device)
                }

                let rhoTensor = Tensor<Float>.full([], rho, on: grad.device)
                let oneMinusRho = Tensor<Float>.full([], 1 - rho, on: grad.device)
                let epsTensor = Tensor<Float>.full(grad.shape, eps, on: grad.device)

                // Update running average of squared gradients
                let gradSquared = grad * grad
                if let sq = squareAvg[i] {
                    squareAvg[i] = sq * rhoTensor + gradSquared * oneMinusRho
                } else {
                    squareAvg[i] = gradSquared * oneMinusRho
                }

                // Compute step: delta = sqrt(acc_delta + eps) / sqrt(sq_avg + eps) * grad
                let accDeltaWithEps: Tensor<Float>
                if let acc = accDelta[i] {
                    accDeltaWithEps = acc + epsTensor
                } else {
                    accDeltaWithEps = epsTensor
                }
                let sqAvgWithEps = squareAvg[i]! + epsTensor

                let stdDelta = accDeltaWithEps.sqrt()
                let stdAvg = sqAvgWithEps.sqrt()
                let delta = grad * stdDelta / stdAvg

                // Update accumulated delta
                let deltaSquared = delta * delta
                if let acc = accDelta[i] {
                    accDelta[i] = acc * rhoTensor + deltaSquared * oneMinusRho
                } else {
                    accDelta[i] = deltaSquared * oneMinusRho
                }

                // Update parameter
                param.value = param.value - delta * Tensor<Float>.full([], learningRate, on: grad.device)
            }
        }

        /// Reset optimizer state.
        public mutating func zeroGrad() {
            squareAvg = Array(repeating: nil, count: parameters.count)
            accDelta = Array(repeating: nil, count: parameters.count)
        }
    }
}

// MARK: - Gradient Clipping

extension optim {
    /// Clip gradients by global L2 norm.
    ///
    /// If the total L2 norm of all gradients exceeds `maxNorm`, all gradients
    /// are scaled down proportionally so the total norm equals `maxNorm`.
    ///
    /// This is essential for stable training of transformers and other deep networks.
    ///
    /// - Parameters:
    ///   - gradients: Array of gradient tensors.
    ///   - maxNorm: Maximum allowed L2 norm. Defaults to 1.0.
    /// - Returns: Tuple of (clipped gradients, original total norm).
    ///
    /// Example:
    /// ```swift
    /// let grads = computeGradients(model, input, target)
    /// let (clippedGrads, totalNorm) = optim.clipGradNorm(grads, maxNorm: 1.0)
    /// optimizer.step(clippedGrads)
    /// ```
    public static func clipGradNorm(
        _ gradients: [Tensor<Float>],
        maxNorm: Float = 1.0
    ) -> (clipped: [Tensor<Float>], totalNorm: Tensor<Float>) {
        guard !gradients.isEmpty else {
            return ([], Tensor<Float>.zeros([], on: .default))
        }

        let device = gradients[0].device

        // Compute sum of squared norms for each gradient tensor
        var sumSquaredNorms = Tensor<Float>.zeros([], on: device)
        for grad in gradients {
            let squaredNorm = (grad * grad).sum()
            sumSquaredNorms = sumSquaredNorms + squaredNorm
        }

        // Total norm = sqrt(sum of squared norms)
        let totalNorm = sumSquaredNorms.sqrt()

        // Compute clip coefficient: min(maxNorm / totalNorm, 1.0)
        // If totalNorm <= maxNorm, clipCoeff = 1.0 (no clipping)
        // If totalNorm > maxNorm, clipCoeff = maxNorm / totalNorm
        let maxNormTensor = Tensor<Float>.full([], maxNorm, on: device)
        let one = Tensor<Float>.ones([], on: device)
        let eps = Tensor<Float>.full([], 1e-6, on: device)

        // clipCoeff = maxNorm / (totalNorm + eps)
        let rawClipCoeff = maxNormTensor / (totalNorm + eps)

        // Clamp to max of 1.0: min(rawClipCoeff, 1.0)
        // Since we don't have a direct min op between tensors, use maskedFill
        let needsClipping = rawClipCoeff.lessThan(one)
        let clipCoeff = one.maskedFill(mask: needsClipping, value: rawClipCoeff)

        // Scale all gradients
        let clippedGrads = gradients.map { grad in
            grad * clipCoeff.broadcast(to: grad.shape)
        }

        return (clippedGrads, totalNorm)
    }

    /// Clip gradients by value.
    ///
    /// Clamps each gradient element to be within [-clipValue, clipValue].
    ///
    /// - Parameters:
    ///   - gradients: Array of gradient tensors.
    ///   - clipValue: Maximum absolute value for any gradient element.
    /// - Returns: Clipped gradients.
    ///
    /// Example:
    /// ```swift
    /// let grads = computeGradients(model, input, target)
    /// let clippedGrads = optim.clipGradValue(grads, clipValue: 0.5)
    /// optimizer.step(clippedGrads)
    /// ```
    public static func clipGradValue(
        _ gradients: [Tensor<Float>],
        clipValue: Float
    ) -> [Tensor<Float>] {
        gradients.map { grad in
            grad.clamp(min: -clipValue, max: clipValue)
        }
    }
}

// MARK: - Gradient Accumulation

extension optim {
    /// Accumulates gradients over multiple steps before applying an update.
    ///
    /// This is useful for simulating larger batch sizes when memory is limited.
    ///
    /// Example:
    /// ```swift
    /// var accumulator = optim.GradientAccumulator(accumulationSteps: 4)
    ///
    /// for (step, batch) in dataLoader.enumerated() {
    ///     let grads = computeGradients(model, batch)
    ///     if let accumulatedGrads = accumulator.accumulate(grads) {
    ///         // Only returns non-nil every accumulationSteps
    ///         optimizer.step(accumulatedGrads)
    ///     }
    /// }
    /// ```
    public struct GradientAccumulator {
        /// Number of steps to accumulate before returning gradients
        public let accumulationSteps: Int

        /// Current accumulated gradients
        private var accumulated: [Tensor<Float>]?

        /// Current step count
        private var stepCount: Int = 0

        /// Creates a gradient accumulator.
        ///
        /// - Parameter accumulationSteps: Number of micro-batches to accumulate.
        public init(accumulationSteps: Int) {
            precondition(accumulationSteps > 0, "Accumulation steps must be positive")
            self.accumulationSteps = accumulationSteps
        }

        /// Accumulate gradients and optionally return averaged gradients.
        ///
        /// - Parameter gradients: Gradients from one micro-batch.
        /// - Returns: Averaged gradients if accumulation is complete, nil otherwise.
        public mutating func accumulate(_ gradients: [Tensor<Float>]) -> [Tensor<Float>]? {
            stepCount += 1

            if accumulated == nil {
                // First step: initialize with gradients
                accumulated = gradients
            } else {
                // Add to accumulated gradients
                accumulated = zip(accumulated!, gradients).map { acc, grad in
                    acc + grad
                }
            }

            // Check if we've accumulated enough steps
            if stepCount >= accumulationSteps {
                // Average the accumulated gradients
                let scale = Tensor<Float>.full([], 1.0 / Float(accumulationSteps), on: gradients[0].device)
                let averaged = accumulated!.map { grad in
                    grad * scale.broadcast(to: grad.shape)
                }

                // Reset state
                accumulated = nil
                stepCount = 0

                return averaged
            }

            return nil
        }

        /// Reset accumulator state.
        public mutating func reset() {
            accumulated = nil
            stepCount = 0
        }

        /// Whether an update is ready.
        public var isUpdateReady: Bool {
            stepCount >= accumulationSteps
        }

        /// Current accumulation progress (0.0 to 1.0).
        public var progress: Float {
            Float(stepCount) / Float(accumulationSteps)
        }
    }
}

// MARK: - Training State Checkpointing

/// Complete training state for saving and resuming training.
///
/// Includes model parameters, optimizer state, and training progress.
///
/// Example:
/// ```swift
/// // Save training state
/// let state = TrainingState(
///     modelState: model.stateDict(),
///     optimizerState: optimizer.stateDict(),
///     step: currentStep,
///     epoch: currentEpoch,
///     loss: currentLoss
/// )
/// try state.save(to: URL(fileURLWithPath: "checkpoint.json"))
///
/// // Resume training
/// let loaded = try TrainingState.load(from: URL(fileURLWithPath: "checkpoint.json"))
/// try model.loadStateDict(loaded.modelState)
/// optimizer.loadState(loaded.optimizerState)
/// currentStep = loaded.step
/// ```
public struct TrainingState: Codable {
    /// Model parameter values (name -> values + shape)
    public var modelState: [String: TensorState]

    /// Optimizer state (momentum buffers, etc.)
    public var optimizerState: [String: Any]?

    /// Current training step
    public var step: Int

    /// Current epoch
    public var epoch: Int

    /// Last recorded loss
    public var loss: Float?

    /// Best recorded loss (for early stopping)
    public var bestLoss: Float?

    /// Learning rate at save time
    public var learningRate: Float?

    /// Custom metadata
    public var metadata: [String: String]

    /// Serializable tensor state
    public struct TensorState: Codable {
        public var values: [Float]
        public var shape: [Int]

        public init(values: [Float], shape: [Int]) {
            self.values = values
            self.shape = shape
        }
    }

    /// Creates a training state.
    public init(
        modelState: [String: TensorState] = [:],
        step: Int = 0,
        epoch: Int = 0,
        loss: Float? = nil,
        bestLoss: Float? = nil,
        learningRate: Float? = nil,
        metadata: [String: String] = [:]
    ) {
        self.modelState = modelState
        self.optimizerState = nil
        self.step = step
        self.epoch = epoch
        self.loss = loss
        self.bestLoss = bestLoss
        self.learningRate = learningRate
        self.metadata = metadata
    }

    // Custom coding to handle non-Codable optimizerState
    enum CodingKeys: String, CodingKey {
        case modelState, step, epoch, loss, bestLoss, learningRate, metadata
    }

    /// Save training state to disk.
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: url)
    }

    /// Load training state from disk.
    public static func load(from url: URL) throws -> TrainingState {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        return try decoder.decode(TrainingState.self, from: data)
    }
}

/// Extension to create TrainingState from model state dict
extension TrainingState {
    /// Create training state from a model's state dict.
    public static func from<M: Module>(
        model: M,
        step: Int = 0,
        epoch: Int = 0,
        loss: Float? = nil,
        learningRate: Float? = nil
    ) -> TrainingState {
        let stateDict = model.stateDict()
        var modelState: [String: TensorState] = [:]

        for (name, tensor) in stateDict {
            let values = tensor.scalars()
            modelState[name] = TensorState(values: values, shape: tensor.shape)
        }

        return TrainingState(
            modelState: modelState,
            step: step,
            epoch: epoch,
            loss: loss,
            learningRate: learningRate
        )
    }

    /// Load model state into a model.
    public func loadInto<M: Module>(_ model: inout M, on device: Device = .default) throws {
        var stateDict: [String: Tensor<Float>] = [:]

        for (name, state) in modelState {
            stateDict[name] = Tensor<Float>(state.values, shape: state.shape, on: device)
        }

        try model.loadStateDict(stateDict)
    }
}

// MARK: - Naming Conventions
//
// Magma Optimizers follow PyTorch naming and API conventions:
//
// | Magma       | PyTorch           | TensorFlow/Keras     |
// |-------------------|-------------------|----------------------|
// | optim.SGD         | torch.optim.SGD   | tf.keras.optimizers.SGD |
// | optim.Adam        | torch.optim.Adam  | tf.keras.optimizers.Adam |
// | optim.AdamW       | torch.optim.AdamW | tf.keras.optimizers.AdamW |
// | optim.RMSProp     | torch.optim.RMSprop | tf.keras.optimizers.RMSprop |
// | optim.AdaGrad     | torch.optim.Adagrad | tf.keras.optimizers.Adagrad |
// | optim.AdaDelta    | torch.optim.Adadelta | tf.keras.optimizers.Adadelta |
