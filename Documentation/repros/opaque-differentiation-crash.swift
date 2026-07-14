// Minimal reproducer: differentiating through an opaque return type (`some P`,
// where `P: Differentiable`) crashes the Swift compiler's Differentiation SIL
// pass. Self-contained — only imports the standard `_Differentiation` module.
//
// Repro:   swiftc -O opaque-differentiation-crash.swift -o /tmp/repro
// Observed: swift-frontend crashes with signal 11 (segfault) in
//           swift::autodiff::VJPCloner::Implementation::createEmptyPullback(),
//           while building the differentiability witness for the closure whose
//           argument type is `@_opaqueReturnTypeOf("...make...", 0)`.
//
// Control:  changing `make()`'s return type from `some P` to the concrete `A`
//           compiles and runs, printing `TangentVector(w: 1.0)`.
//
// Toolchain where observed: Swift 6.3.3 (swift-6.3.3-RELEASE),
//                           aarch64-unknown-linux-gnu.

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
