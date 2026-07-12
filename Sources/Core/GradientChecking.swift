// Magma - Gradient Checking
// Numerical gradient verification for autodiff
// API follows PyTorch's torch.autograd.gradcheck

import _Differentiation
import LazyTensor
import StableHLO
import XLARuntime

// MARK: - Gradient Check Result

/// Result of a gradient check comparison
public struct GradcheckResult {
    /// Whether all gradient checks passed
    public let passed: Bool

    /// Maximum absolute difference between analytical and numerical gradients
    public let maxAbsDiff: Float

    /// Maximum relative difference between analytical and numerical gradients
    public let maxRelDiff: Float

    /// Detailed per-element differences (only populated when verbose)
    public let details: [GradcheckDetail]?

    public init(passed: Bool, maxAbsDiff: Float, maxRelDiff: Float, details: [GradcheckDetail]? = nil) {
        self.passed = passed
        self.maxAbsDiff = maxAbsDiff
        self.maxRelDiff = maxRelDiff
        self.details = details
    }
}

/// Detailed information about a single gradient comparison
public struct GradcheckDetail {
    /// Linear index in the tensor
    public let index: Int

    /// Multi-dimensional index
    public let multiIndex: [Int]

    /// Analytical gradient value
    public let analyticalGrad: Float

    /// Numerical gradient value
    public let numericalGrad: Float

    /// Absolute difference
    public let absDiff: Float

    /// Relative difference
    public let relDiff: Float
}

// MARK: - Gradient Check Functions

/// Check gradients of a function using numerical finite differences.
///
/// This function compares the analytical gradients computed by autodiff
/// against numerical gradients computed using centered finite differences.
///
/// - Parameters:
///   - func: The function to check gradients for (must return a scalar)
///   - input: The input tensor at which to check gradients
///   - eps: Step size for finite difference calculation (default: 1e-4)
///   - atol: Absolute tolerance for comparison (default: 1e-5)
///   - rtol: Relative tolerance for comparison (default: 1e-3)
///   - verbose: If true, return detailed per-element information
/// - Returns: GradcheckResult with pass/fail status and difference metrics
///
/// Example:
/// ```swift
/// let x = Tensor<Float>.randn([2, 3])
/// let result = gradcheck({ $0.sum() }, input: x)
/// print("Gradcheck passed: \(result.passed)")
/// ```
public func gradcheck(
    _ f: @differentiable(reverse) (Tensor<Float>) -> Tensor<Float>,
    input: Tensor<Float>,
    eps: Float = 1e-2,
    atol: Float = 1e-5,
    rtol: Float = 1e-3,
    verbose: Bool = false
) -> GradcheckResult {
    // Compute analytical gradient using autodiff
    let analyticalGrad = gradient(at: input, of: f)

    // Compute numerical gradient using centered finite differences
    let numericalGrad = numericalGradient(of: { tensor in
        f(tensor)
    }, at: input, eps: eps)

    // Compare gradients
    return compareGradients(
        analytical: analyticalGrad,
        numerical: numericalGrad,
        atol: atol,
        rtol: rtol,
        verbose: verbose
    )
}

/// Check gradients of a function with two inputs using numerical finite differences.
///
/// - Parameters:
///   - func: The function to check gradients for (must return a scalar)
///   - input1: The first input tensor
///   - input2: The second input tensor
///   - eps: Step size for finite difference calculation
///   - atol: Absolute tolerance for comparison
///   - rtol: Relative tolerance for comparison
/// - Returns: Tuple of GradcheckResults for each input
public func gradcheck(
    _ f: @differentiable(reverse) (Tensor<Float>, Tensor<Float>) -> Tensor<Float>,
    input1: Tensor<Float>,
    input2: Tensor<Float>,
    eps: Float = 1e-2,
    atol: Float = 1e-5,
    rtol: Float = 1e-3
) -> (GradcheckResult, GradcheckResult) {
    // Compute analytical gradients
    let (grad1, grad2) = gradient(at: input1, input2, of: f)

    // Compute numerical gradients
    let numGrad1 = numericalGradient(of: { x in f(x, input2) }, at: input1, eps: eps)
    let numGrad2 = numericalGradient(of: { x in f(input1, x) }, at: input2, eps: eps)

    // Compare
    let result1 = compareGradients(analytical: grad1, numerical: numGrad1, atol: atol, rtol: rtol)
    let result2 = compareGradients(analytical: grad2, numerical: numGrad2, atol: atol, rtol: rtol)

    return (result1, result2)
}

// MARK: - Numerical Gradient Computation

/// Compute numerical gradient using centered finite differences.
///
/// For each element x_i, computes:
///   df/dx_i ≈ (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
///
/// - Parameters:
///   - f: The function to differentiate (must return a scalar tensor)
///   - x: The point at which to compute the gradient
///   - eps: Step size for finite differences
/// - Returns: Tensor of gradients with same shape as input
public func numericalGradient(
    of f: (Tensor<Float>) -> Tensor<Float>,
    at x: Tensor<Float>,
    eps: Float = 1e-2
) -> Tensor<Float> {
    // Get input values (scalars() triggers barrier internally)
    let xValues = x.scalars()
    let n = xValues.count

    // Compute gradient for each element
    var gradValues = [Float](repeating: 0, count: n)

    for i in 0..<n {
        // Create perturbed inputs
        var xPlusValues = xValues
        var xMinusValues = xValues
        xPlusValues[i] += eps
        xMinusValues[i] -= eps

        let xPlus = Tensor<Float>(xPlusValues, shape: x.shape, on: x.device)
        let xMinus = Tensor<Float>(xMinusValues, shape: x.shape, on: x.device)

        // Evaluate function at perturbed points
        let fPlus = f(xPlus)
        let fMinus = f(xMinus)

        // Get scalar outputs (scalars() forces evaluation)
        let fPlusVal = fPlus.scalars()[0]
        let fMinusVal = fMinus.scalars()[0]

        // Centered finite difference
        gradValues[i] = (fPlusVal - fMinusVal) / (2 * eps)
    }

    return Tensor<Float>(gradValues, shape: x.shape, on: x.device)
}

/// Compute numerical gradient using forward finite differences (one-sided).
/// Less accurate but faster than centered differences.
///
/// For each element x_i, computes:
///   df/dx_i ≈ (f(x + eps*e_i) - f(x)) / eps
public func numericalGradientForward(
    of f: (Tensor<Float>) -> Tensor<Float>,
    at x: Tensor<Float>,
    eps: Float = 1e-2
) -> Tensor<Float> {
    // Get input values and base function value
    let xValues = x.scalars()
    let n = xValues.count

    let fBase = f(x)
    let fBaseVal = fBase.scalars()[0]

    // Compute gradient for each element
    var gradValues = [Float](repeating: 0, count: n)

    for i in 0..<n {
        var xPlusValues = xValues
        xPlusValues[i] += eps

        let xPlus = Tensor<Float>(xPlusValues, shape: x.shape, on: x.device)
        let fPlus = f(xPlus)
        let fPlusVal = fPlus.scalars()[0]

        gradValues[i] = (fPlusVal - fBaseVal) / eps
    }

    return Tensor<Float>(gradValues, shape: x.shape, on: x.device)
}

// MARK: - Gradient Comparison

/// Compare analytical and numerical gradients.
private func compareGradients(
    analytical: Tensor<Float>,
    numerical: Tensor<Float>,
    atol: Float,
    rtol: Float,
    verbose: Bool = false
) -> GradcheckResult {
    // scalars() triggers evaluation internally
    let analyticalValues = analytical.scalars()
    let numericalValues = numerical.scalars()

    precondition(analyticalValues.count == numericalValues.count,
                 "Gradient shapes must match: analytical has \(analyticalValues.count) elements, numerical has \(numericalValues.count)")

    var maxAbsDiff: Float = 0
    var maxRelDiff: Float = 0
    var allPassed = true
    var details: [GradcheckDetail]? = verbose ? [] : nil

    for i in 0..<analyticalValues.count {
        let a = analyticalValues[i]
        let n = numericalValues[i]

        let absDiff = Swift.abs(a - n)
        let relDiff = absDiff / (Swift.max(Swift.abs(a), Swift.abs(n)) + 1e-12)

        maxAbsDiff = Swift.max(maxAbsDiff, absDiff)
        maxRelDiff = Swift.max(maxRelDiff, relDiff)

        // Check if this element passes: |a - n| <= atol + rtol * |n|
        let tolerance = atol + rtol * Swift.abs(n)
        if absDiff > tolerance {
            allPassed = false
        }

        if verbose {
            let multiIndex = linearToMultiIndex(i, shape: analytical.shape)
            details?.append(GradcheckDetail(
                index: i,
                multiIndex: multiIndex,
                analyticalGrad: a,
                numericalGrad: n,
                absDiff: absDiff,
                relDiff: relDiff
            ))
        }
    }

    return GradcheckResult(
        passed: allPassed,
        maxAbsDiff: maxAbsDiff,
        maxRelDiff: maxRelDiff,
        details: details
    )
}

/// Convert linear index to multi-dimensional index
private func linearToMultiIndex(_ linear: Int, shape: [Int]) -> [Int] {
    guard !shape.isEmpty else { return [] }

    var result = [Int](repeating: 0, count: shape.count)
    var remaining = linear

    for i in stride(from: shape.count - 1, through: 0, by: -1) {
        result[i] = remaining % shape[i]
        remaining /= shape[i]
    }

    return result
}

// MARK: - Convenience Functions

/// Quick gradient check that returns just pass/fail.
///
/// Example:
/// ```swift
/// assert(gradcheckPasses({ $0.relu().sum() }, input: x))
/// ```
public func gradcheckPasses(
    _ f: @differentiable(reverse) (Tensor<Float>) -> Tensor<Float>,
    input: Tensor<Float>,
    eps: Float = 1e-2,
    atol: Float = 1e-5,
    rtol: Float = 1e-3
) -> Bool {
    return gradcheck(f, input: input, eps: eps, atol: atol, rtol: rtol).passed
}

/// Assert that gradients are correct, throwing an error if not.
///
/// Example:
/// ```swift
/// try assertGradcheck({ $0.sum() }, input: x)
/// ```
public func assertGradcheck(
    _ f: @differentiable(reverse) (Tensor<Float>) -> Tensor<Float>,
    input: Tensor<Float>,
    eps: Float = 1e-2,
    atol: Float = 1e-5,
    rtol: Float = 1e-3,
    file: StaticString = #file,
    line: UInt = #line
) throws {
    let result = gradcheck(f, input: input, eps: eps, atol: atol, rtol: rtol, verbose: true)

    if !result.passed {
        var message = "Gradient check failed!\n"
        message += "Max absolute difference: \(result.maxAbsDiff)\n"
        message += "Max relative difference: \(result.maxRelDiff)\n"

        if let details = result.details {
            let failing = details.filter { detail in
                detail.absDiff > atol + rtol * Swift.abs(detail.numericalGrad)
            }.prefix(10)

            message += "First \(failing.count) failing elements:\n"
            for detail in failing {
                message += "  Index \(detail.multiIndex): analytical=\(detail.analyticalGrad), numerical=\(detail.numericalGrad), diff=\(detail.absDiff)\n"
            }
        }

        throw GradcheckError.gradientMismatch(message)
    }
}

/// Errors that can occur during gradient checking
public enum GradcheckError: Error, CustomStringConvertible {
    case gradientMismatch(String)

    public var description: String {
        switch self {
        case .gradientMismatch(let message):
            return message
        }
    }
}

// MARK: - Jacobian Computation

/// Compute the full Jacobian matrix of a vector-valued function.
///
/// For a function f: R^n -> R^m, the Jacobian is an m×n matrix where
/// J[i,j] = df_i/dx_j
///
/// - Parameters:
///   - f: The function to differentiate
///   - x: The point at which to compute the Jacobian
///   - eps: Step size for finite differences
/// - Returns: Jacobian matrix as a 2D tensor [output_size, input_size]
public func numericalJacobian(
    of f: (Tensor<Float>) -> Tensor<Float>,
    at x: Tensor<Float>,
    eps: Float = 1e-2
) -> Tensor<Float> {
    let xValues = x.scalars()
    let n = xValues.count  // input dimension

    // Get output dimension
    let fBase = f(x)
    let fBaseValues = fBase.scalars()
    let m = fBaseValues.count  // output dimension

    // Compute Jacobian column by column
    var jacobianValues = [[Float]](repeating: [Float](repeating: 0, count: n), count: m)

    for j in 0..<n {
        // Perturb input j
        var xPlusValues = xValues
        var xMinusValues = xValues
        xPlusValues[j] += eps
        xMinusValues[j] -= eps

        let xPlus = Tensor<Float>(xPlusValues, shape: x.shape, on: x.device)
        let xMinus = Tensor<Float>(xMinusValues, shape: x.shape, on: x.device)

        let fPlus = f(xPlus)
        let fMinus = f(xMinus)

        let fPlusValues = fPlus.scalars()
        let fMinusValues = fMinus.scalars()

        // Compute column j of Jacobian
        for i in 0..<m {
            jacobianValues[i][j] = (fPlusValues[i] - fMinusValues[i]) / (2 * eps)
        }
    }

    // Flatten and create tensor
    let flatJacobian = jacobianValues.flatMap { $0 }
    return Tensor<Float>(flatJacobian, shape: [m, n], on: x.device)
}

// MARK: - Higher-Order Gradient Checking

/// Check the gradient of a gradient using numerical approximation only.
///
/// This computes the Hessian (second derivatives) numerically to verify
/// gradient computation is correct. Compares numerical gradient of the
/// gradient function against a reference numerical computation.
///
/// Note: True analytical second-order gradients (gradgradcheck) would require
/// the `gradient` function itself to be differentiable, which is not currently
/// supported by Swift's differentiation.
///
/// - Parameters:
///   - f: The function whose second derivative to check
///   - input: The input tensor
///   - eps: Step size for numerical gradient computation
///   - atol: Absolute tolerance
///   - rtol: Relative tolerance
/// - Returns: GradcheckResult comparing two numerical gradient approaches
public func numericalGradgradcheck(
    _ f: @differentiable(reverse) (Tensor<Float>) -> Tensor<Float>,
    input: Tensor<Float>,
    eps: Float = 1e-3,
    atol: Float = 1e-3,
    rtol: Float = 1e-2
) -> GradcheckResult {
    // Compute numerical gradient of the analytical gradient function
    let numGradOfAnalytical = numericalGradient(of: { x in
        let grad = gradient(at: x, of: f)
        return grad.sum()
    }, at: input, eps: eps)

    // Compute using double numerical differentiation (reference)
    let doubleNumericalGrad = numericalGradient(of: { x in
        let innerGrad = numericalGradient(of: { inner in f(inner) }, at: x, eps: eps * 0.1)
        return innerGrad.sum()
    }, at: input, eps: eps)

    return compareGradients(
        analytical: numGradOfAnalytical,
        numerical: doubleNumericalGrad,
        atol: atol,
        rtol: rtol
    )
}
