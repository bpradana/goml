package backend

// Numerical capabilities write freshly allocated, nonoverlapping destinations.
// Each family is optional; unknown operation values wrap ErrUnsupported.
// Unsupported layouts also wrap ErrUnsupported rather than copying to CPU.
type UnaryBackend interface {
	Unary(op UnaryOp, dst, src View) error
}

// BinaryBackend writes a broadcast result of dst.Shape from a and b.
type BinaryBackend interface {
	Binary(op BinaryOp, dst, a, b View) error
}

// ScalarBackend writes one scalar operation without transferring tensor data.
type ScalarBackend interface {
	Scalar(op ScalarOp, dst, src View, value float32) error
}

// ReductionBackend reduces the validated, distinct axes of src into dst.
type ReductionBackend interface {
	Reduce(op ReduceOp, dst, src View, axes []int) error
}

// MatMulBackend writes [m,k] x [k,n] into [m,n], including empty dimensions.
type MatMulBackend interface {
	MatMul(dst, a, b View) error
}

// FillBackend initializes a nonoverlapping destination.
type FillBackend interface {
	Fill(dst View, value float32) error
}

// CopyBackend copies equally shaped logical values into a nonoverlapping view.
// Source and destination must not alias unless explicitly documented.
type CopyBackend interface {
	Copy(dst, src View) error
}

// HostAccess is an opt-in capability for host-native execution contexts. Reads
// do not perform a device transfer or wait for accelerator work. The destination
// belongs to the caller and must have src.Numel() elements. At is bounds-checked.
// Hardware accelerators must not claim this capability merely because their
// allocation can be CPU-mapped. HostData remains available through Download.
type HostAccess interface {
	Read(src View, values []float32) error
	At(src View, indices []int) (float32, error)
}

// FiniteBackend validates values without downloading the tensor. Accelerator
// implementations may synchronize a small status flag and return a numerical error.
type FiniteBackend interface {
	CheckFinite(src View) error
}

// SubScaledBackend preserves the fused immutable a-factor*b CPU optimization.
// Multiplication rounds to Float32 before subtraction.
type SubScaledBackend interface {
	SubScaled(dst, a, b View, factor float32) error
}

// AdamOptions defines the existing bias-corrected Adam update.
type AdamOptions struct {
	Rate, Beta1, Beta2, Epsilon float32
	Step                        uint64
}

// HostAdamBackend preserves synchronous, fused updates with optimizer-owned
// host moment arrays. Implementations fail before writing on validation errors.
// This capability is only for host-native contexts, not accelerator transfers.
// Devices without it use ordinary device-resident Tensor moment operations.
type HostAdamBackend interface {
	Adam(dst, values, gradient View, first, second []float32, options AdamOptions) error
}

// ConcatBackend preserves allocation-efficient joining of validated operands.
// Inputs have equal ranks and dimensions outside axis; dst has the inferred shape.
type ConcatBackend interface {
	Concat(dst View, axis int, inputs []View) error
}
