# Known Compiler Issues

Swift toolchain bugs that affect Magma, with minimal reproducers and the
workarounds currently in the codebase.

---

## 1. Differentiating an opaque return type (`some P`) crashes the SIL Differentiation pass

**Status:** open · **Toolchain:** Swift 6.3.3 (`swift-6.3.3-RELEASE`), `aarch64-unknown-linux-gnu`
**Reproducer:** [`repros/opaque-differentiation-crash.swift`](repros/opaque-differentiation-crash.swift)

### Symptom

Reverse-mode differentiation of a closure that operates on a value whose type is
an **opaque return type** (`some P` where `P: Differentiable`) crashes
`swift-frontend` with signal 11 (segfault). The concrete-typed equivalent
compiles and runs correctly.

```swift
protocol P: Differentiable {
    @differentiable(reverse) func f(_ x: Float) -> Float
}
struct A: P {
    var w: Float
    @differentiable(reverse) func f(_ x: Float) -> Float { x * w }
}

func make() -> some P { A(w: 2) }        // <-- opaque return type is the trigger
let m = make()
let g = gradient(at: m) { $0.f(1.0) }    // crashes the compiler
```

Changing `make()`'s return type from `some P` to the concrete `A` makes the
program compile and print `TangentVector(w: 1.0)`.

### Where it crashes

```
While running pass #462 SILModuleTransform "Differentiation".
While processing differentiability witness for closure ...
    ... for <@_opaqueReturnTypeOf("$s...make...", 0) __>
swift::autodiff::VJPCloner::Implementation::createEmptyPullback()
```

The pass fails while building the differentiability witness for a closure whose
parameter type is the `@_opaqueReturnTypeOf(...)` archetype produced by the
opaque return type.

### Impact on Magma & workaround

The value-semantic layer API (`Sources/Core/ValueLayers.swift`) composes layers
into a nested `Sequential2<...>` via a result builder. The ergonomic spelling
would let a model be held as `some Layer`, but any `gradient(at:)` over such a
value hits this crash. **Workaround:** spell models with their concrete
(nested `Sequential2`) type, e.g.

```swift
typealias MLP = Sequential2<Sequential2<Linear, ReLU>, Linear>
```

This is why the layer tests (`ValueLayerMLPTests`, `ConvLayerTests`) declare
concrete model types rather than returning `some Layer`. Training works
identically; only the opaque spelling is unavailable until the compiler is fixed.
