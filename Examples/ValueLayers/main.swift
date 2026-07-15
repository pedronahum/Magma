// Magma - Value-Semantic Layers Example
//
// Trains a small MLP to fit a nonlinear function, showcasing the value-semantic
// ("Design A") layer API:
//   - a model built with `sequential { ... }` and held as `some Layer`,
//   - trained with Swift-native reverse-mode autodiff (no tape, no tracer),
//   - by a generic `Adam` that discovers the model's tensors via reflection —
//     no parameter list, no per-layer optimizer code.
//
// The ReLU units learn a piecewise-linear approximation of y = x², which a plain
// linear model cannot represent — the hidden layer + nonlinearity + autodiff are
// doing real work.
//
// Run:  MAGMA_XLA_PATH=... swift run ValueLayersExample

import Foundation
import Magma
import LazyTensor
import _Differentiation

// A 1 -> 16 -> 1 ReLU MLP. Returned as `some Layer`: the caller never spells the
// concrete (nested Sequential2) type. Biases are spread so each ReLU unit has a
// different "knee", giving the network a piecewise-linear basis to work with.
func makeMLP() -> some Layer {
    let hidden = 16
    let knees = (0..<hidden).map { Float($0) / Float(hidden - 1) * 4 - 2 }   // -2 ... 2
    return sequential {
        Linear(weight: Tensor<Float>([Float](repeating: 1, count: hidden), shape: [1, hidden]),
               bias: Tensor<Float>(knees.map { -$0 }, shape: [hidden]))
        ReLU()
        Linear(weight: Tensor<Float>((0..<hidden).map { Float($0 % 3) * 0.1 - 0.1 }, shape: [hidden, 1]),
               bias: Tensor<Float>.zeros([1]))
    }
}

@main
struct ValueLayersExample {
    static func main() {
        print("Magma — Value-Semantic Layers")
        print("=============================\n")

        // Dataset: y = x² on 9 points in [-2, 2].
        let xs = stride(from: Float(-2), through: 2, by: 0.5).map { $0 }
        let x = Tensor<Float>(xs, shape: [xs.count, 1])
        let y = Tensor<Float>(xs.map { $0 * $0 }, shape: [xs.count, 1])
        let n = Tensor<Float>.full([], Float(xs.count))

        // Mean-squared error, typed over concrete tensors (pred, target).
        let mse: @differentiable(reverse) (Tensor<Float>, Tensor<Float>) -> Tensor<Float> = { pred, target in
            let r = pred - target
            return (r * r).sum() / n
        }

        var model = makeMLP()                       // held as `some Layer`
        var optimizer = Adam(learningRate: 0.05)

        print("Fitting y = x²  (1→16→1 ReLU MLP, Adam)\n")
        let initialLoss = modelValueWithGradient(of: model, input: x, target: y, lossFn: mse)
            .value.scalars()[0]
        print(String(format: "  step    0   loss = %.4f", initialLoss))

        for step in 1...2000 {
            // Differentiate the opaque model through the generic helper, then let
            // the generic Adam update every tensor slot it finds by reflection.
            let grad = modelGradient(of: model, input: x, target: y, lossFn: mse)
            optimizer.update(&model, gradient: grad)

            if step % 400 == 0 {
                let loss = modelValueWithGradient(of: model, input: x, target: y, lossFn: mse)
                    .value.scalars()[0]
                print(String(format: "  step %4d   loss = %.4f", step, loss))
            }
        }

        // Show the learned fit.
        let preds = model(x).scalars()
        print("\n  x     target   model")
        for (i, xv) in xs.enumerated() {
            print(String(format: "  %+.1f    %5.2f   %5.2f", xv, xv * xv, preds[i]))
        }
        print("\nThe MLP learned a nonlinear function via Swift-native autodiff.")
    }
}
