// Magma - Re-exports
//
// `import Magma` is all a user needs: the lower layers define types that
// appear throughout the public API (Device, Backend, DType, TensorScalar,
// MaterializationError, LazyTensorBarrier, ...), and autodiff needs the
// `_Differentiation` module for `@differentiable`, `gradient(at:)` and friends.

@_exported import _Differentiation
@_exported import LazyTensor
@_exported import StableHLO
@_exported import XLARuntime
