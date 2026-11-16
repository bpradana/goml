package tensor

// Buffer is an opaque device allocation. Close is idempotent. Implementations
// retain native memory until pending commands finish, even after Close.
type Buffer interface {
	Len() int
	DeviceID() string
	Valid() error
	Close() error
}

// Device owns allocations and an ordered execution queue. ID uniquely identifies
// this device instance, rather than just the physical GPU. Upload and Download
// copy values; Download synchronizes prior writes. Close synchronizes and
// invalidates live buffers. CPU execution is represented by a nil Device.
type Device interface {
	Name() string
	ID() string
	Allocate(elements int) (Buffer, error)
	Upload(Buffer, []float32) error
	Download(Buffer, []float32) error
	Synchronize() error
	Close() error
}

// DeviceView describes read-only storage in elements. Shapes and strides are
// validated by tensor; backends must still validate buffer ownership and bounds.
type DeviceView struct {
	Buffer  Buffer
	Shape   []int
	Strides []int
	Offset  int
}

// ArithmeticBackend supplies Float32 kernels on a device. Operations write only
// their newly allocated destination. Unsupported operations return errors.
type ArithmeticBackend interface {
	Unary(op string, dst Buffer, a DeviceView, scalar float32) error
	Binary(op string, dst Buffer, a, b DeviceView, shape []int) error
	MatMul(dst Buffer, a, b DeviceView) error
	Reduce(op string, dst Buffer, a DeviceView, axes []int) error
	Fill(dst Buffer, value float32) error
}

// LayoutBackend materializes and scatters device views without host access.
// Copy writes a source view into an equally shaped destination view.
type LayoutBackend interface {
	Copy(dst, src DeviceView) error
}

// ConvolutionBackend executes native NCHW convolution and both derivatives.
// Weights follow goml's [in*kernel*kernel,out] layout; bias is added separately.
type ConvolutionBackend interface {
	Conv2D(dst Buffer, input, weight DeviceView, kernel, stride, padding int) error
	Conv2DBackward(dx, dw Buffer, input, weight, gradient DeviceView, kernel, stride, padding int) error
}

// SpatialBackend supplies NCHW pooling and interpolation and their derivatives.
type SpatialBackend interface {
	Pool(dst Buffer, input DeviceView, kernel, stride, padding int) error
	PoolBackward(dst Buffer, input, gradient DeviceView, kernel, stride, padding int) error
	Resize(dst Buffer, input DeviceView, scale int, mode string) error
	ResizeBackward(dst Buffer, gradient DeviceView, shape []int, scale int, mode string) error
}
