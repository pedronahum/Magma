# Known Compiler Issues

Swift toolchain bugs that affect Magma, with minimal reproducers and the
workarounds currently in the codebase.

---

## 1. Differentiating a value of opaque/existential type crashes the SIL Differentiation pass

**Status:** open (still reproduces on Swift 6.5-dev) · **Reproducer:** [`repros/opaque-differentiation-crash.swift`](repros/opaque-differentiation-crash.swift)
**There is a full workaround** (see below) — the ergonomic `some Layer` spelling is usable.

### Symptom

Emitting a reverse-mode differentiability witness for a closure whose **parameter
type is an opaque return type** (`some P` where `P: Differentiable`) crashes
`swift-frontend`. The concrete-typed equivalent compiles and runs correctly.

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

### Crash site

The Differentiation pass fails in `VJPCloner::createEmptyPullback()` while
building the witness for a closure over the `@_opaqueReturnTypeOf(...)` archetype:

```
While running pass SILModuleTransform "Differentiation".
While canonicalizing `differentiable_function` ... for <A>
While processing differentiability witness for closure
    ... for <@_opaqueReturnTypeOf("$s...make...", 0) __>
swift::autodiff::VJPCloner::Implementation::createEmptyPullback()
```

On an asserts-enabled build (Swift 6.5-dev, `main-snapshot-2026-07-05`) it aborts
on a specific assertion, which pinpoints the root cause — the opaque archetype's
underlying type is lowered **without a generic signature** even though it still
carries a type parameter:

```
AbstractionPattern.h:535:
  void swift::Lowering::AbstractionPattern::initSwiftType(...):
  Assertion `signature || !origType->hasTypeParameter()' failed.
```

Release toolchains (assertions compiled out) turn this into a signal-11 segfault.

### Which forms crash (measured, `-Onone` and `-O`)

| Form | Result |
|------|--------|
| `gradient(at: m) { … }` where `m: some P` | **crash** |
| `valueWithGradient(at: m) { … }`, `m: some P` | **crash** |
| `m: any P` (existential) instead of `some P` | **crash** (should be a clean diagnostic) |
| closure passed *into* a generic `<M>` helper but written at the opaque site | **crash** |
| return type is the concrete `A` | ok |
| **differentiated closure written *inside* a generic `<M: P>` helper** | **ok** ← workaround |

Verified on Swift 6.3.3 (release) and Swift 6.5-dev (`main-snapshot-2026-07-05`):
same crash, and the workaround holds on both.

### Root-cause reading

The crash is **not** "opaque types can't be differentiated". The trigger is
narrow: a `differentiable_function` / differentiability witness whose parameter
is the *opaque archetype itself*. When the same value is opened into an ordinary
**generic** archetype — by passing it to a generic function whose body does the
differentiation — the witness is emitted against the generic signature and
lowering succeeds. (This is also why Magma's generic reflection optimizers, which
never differentiate at an opaque type, were unaffected all along.)

### Workaround & impact on Magma

Keep the differentiated closure inside a generic `<M: Layer>` function, and type
any user-supplied loss over **concrete tensors** (`pred`, `target`) rather than
over the model `M`. `Sources/Core/LayerGradient.swift` provides exactly this:

```swift
public func modelGradient<M: Layer>(
    of model: M, input: Tensor<Float>, target: Tensor<Float>,
    lossFn: @differentiable(reverse) (Tensor<Float>, Tensor<Float>) -> Tensor<Float>
) -> M.TangentVector {
    gradient(at: model) { m in lossFn(m(input), target) }   // closure param is generic M
}
```

With it, a model may be built by a `-> some Layer` factory, held opaque, and
trained end to end (`OpaqueLayerTrainingTests`):

```swift
var model = makeMLP()                       // some Layer
let g = modelGradient(of: model, input: x, target: y) { p, t in ((p - t) * (p - t)).sum() }
opt.update(&model, gradient: g)             // Adam over the opaque type (reflection only)
```

So both spellings work: the concrete nested `Sequential2<...>` type (used by
`ValueLayerMLPTests`/`ConvLayerTests`, and required if you call `gradient(at:)`
directly), or the opaque `some Layer` type routed through `modelGradient`.
