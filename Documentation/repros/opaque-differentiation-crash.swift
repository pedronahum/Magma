// Minimal reproducer: differentiating through an opaque return type (`some P`,
// where `P: Differentiable`) crashes the Swift compiler's Differentiation SIL
// pass. Self-contained — only imports the standard `_Differentiation` module.
//
// Repro:   swiftc opaque-differentiation-crash.swift -o /tmp/repro   (-Onone or -O)
// Observed: swift-frontend crashes in
//           swift::autodiff::VJPCloner::Implementation::createEmptyPullback(),
//           while building the differentiability witness for the closure whose
//           argument type is `@_opaqueReturnTypeOf("...make...", 0)`.
//           - Release toolchain (6.3.3): signal 11 (segfault, asserts compiled out).
//           - Asserts-enabled 6.5-dev snapshot: aborts on
//               AbstractionPattern.h:535:
//               Assertion `signature || !origType->hasTypeParameter()' failed
//             i.e. the opaque archetype's underlying type is lowered without a
//             generic signature though it still has a type parameter.
//
// Also crashes: `any P` (existential) return type; and `valueWithGradient`.
//
// Control:  changing `make()`'s return type from `some P` to the concrete `A`
//           compiles and runs, printing `TangentVector(w: 1.0)`.
//
// Workaround: do the differentiation inside a generic function so the value is an
//   ordinary generic archetype rather than the opaque one — this compiles & runs:
//       func lossGrad<M: P>(_ m: M) -> M.TangentVector { gradient(at: m) { $0.f(1.0) } }
//       print(lossGrad(make()))   // TangentVector(w: 1.0)
//   (In Magma this is Sources/Core/LayerGradient.swift `modelGradient`.)
//
// Reproduced on: Swift 6.3.3 (swift-6.3.3-RELEASE) and Swift 6.5-dev
//                (main-snapshot-2026-07-05), aarch64-unknown-linux-gnu.

import _Differentiation

protocol P: Differentiable {
    @differentiable(reverse)
    func f(_ x: Float) -> Float
}

struct A: P {
    var w: Float
    @differentiable(reverse)
    func f(_ x: Float) -> Float { x * w }
}

// The trigger is the opaque return type. Return `A` instead and it compiles.
func make() -> some P { A(w: 2) }

let m = make()
let g = gradient(at: m) { model in model.f(1.0) }
print(g)
