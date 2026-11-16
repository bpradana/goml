package backend

// Pool2DOptions describes floor-size NCHW maximum pooling. Padding is negative
// infinity; finite ties and NaNs select the first source in row-major order.
type Pool2DOptions struct {
	Kernel, Stride, Padding int
}

// PoolBackend writes spatial maxima on the selected execution context.
type PoolBackend interface {
	Pool(dst, input View, options Pool2DOptions) error
}

// PoolBackwardBackend gathers overlapping-window derivatives into input shape.
type PoolBackwardBackend interface {
	PoolBackward(dst, input, gradient View, options Pool2DOptions) error
}

// ResizeOptions describes existing integer-scale spatial interpolation.
type ResizeOptions struct {
	Scale int
	Mode  ResizeMode
}

// ResizeBackend writes NCHW interpolation into [n,c,h*scale,w*scale].
type ResizeBackend interface {
	Resize(dst, input View, options ResizeOptions) error
}

// ResizeBackwardBackend writes interpolation derivatives into the input shape.
type ResizeBackwardBackend interface {
	ResizeBackward(dst, gradient View, options ResizeOptions) error
}
