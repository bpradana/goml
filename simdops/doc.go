// Package simdops provides allocation-free float32 slice kernels.
//
// Inputs must have equal lengths, including dst. Validation precedes writes.
// Exact in-place aliasing is supported; partial overlap is unsupported.
// Empty Sum and Dot return zero; empty Max returns an error.
// Arithmetic follows IEEE 754, including infinities and NaNs. Max propagates
// NaN and prefers positive zero. Reduction order may differ with SIMD enabled;
// compare reductions using absolute and relative tolerances, not bit equality.
//
// The default build is scalar. GOEXPERIMENT=simd enables portable SIMD kernels.
package simdops
