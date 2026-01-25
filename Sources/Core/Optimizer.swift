// Magma - Optimizer Module
// Optimization algorithms for training neural networks

import Foundation
import _Differentiation
import LazyTensor
import StableHLO
import XLARuntime

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
    /// Update parameters using the given gradients.
    ///
    /// - Parameter gradients: Array of gradient tensors, one per parameter.
    mutating func step(_ gradients: [Tensor<Float>])

    /// Reset all optimizer state (e.g., momentum buffers).
    mutating func zeroGrad()

    /// The current learning rate.
    var learningRate: Float { get set }
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

            for i in 0..<parameters.count {
                let param = parameters[i]
                guard param.requiresGrad else { continue }

                let grad = gradients[i]

                // AdamW weight decay (decoupled)
                if weightDecay != 0 {
                    param.value = param.value - param.value * Tensor<Float>.full([], learningRate * weightDecay, on: param.value.device)
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
                param.value = param.value - update * Tensor<Float>.full([], learningRate, on: grad.device)
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
