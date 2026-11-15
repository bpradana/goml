// Package tensor provides immutable float32 arrays with trailing-dimension
// broadcasting and strided views. Constructors and Data copy their inputs or
// outputs; Transpose, Slice, and contiguous Reshape share read-only storage.
// Arithmetic allocates independent outputs. Contiguous always materializes a copy.
//
// Scalars have rank zero and one element. Zero dimensions are supported.
// Use constructors; the zero Tensor value is uninitialized. Nil tensors are
// invalid operation inputs. Numerical functions follow IEEE 754 and propagate
// NaN. Mean and Max reject empty reduction groups.
package tensor
