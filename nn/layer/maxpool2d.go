package layer

import (
	"fmt"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

// MaxPool2D selects spatial maxima independently for each NCHW channel.
// Padding is negative infinity. Finite ties select the first input in row-major
// order; the first NaN in a window propagates and receives its gradient.
type MaxPool2D struct {
	kernel  int
	stride  int
	padding int
}

// NewMaxPool2D creates square pooling with equal spatial strides and symmetric
// padding. Kernel and stride must be positive; padding must be in [0,kernel/2].
func NewMaxPool2D(kernel, stride, padding int) (*MaxPool2D, error) {
	if kernel <= 0 || stride <= 0 || padding < 0 || padding > kernel/2 {
		return nil, fmt.Errorf("nn: maxpool2d: positive kernel and stride, padding in [0,kernel/2] required")
	}

	return &MaxPool2D{kernel: kernel, stride: stride, padding: padding}, nil
}

// Forward allocates [batch,channels,outH,outW], where outH is
// floor((height+2*padding-kernel)/stride)+1, and similarly for outW. Input must
// have rank four and positive spatial dimensions. Invalid geometry or overflow
// returns an error. Overlapping windows accumulate gradients; inference omits
// saved argmax indices. Empty batches and channels are supported.
func (p *MaxPool2D) Forward(input *nn.Value) (*nn.Value, error) {
	if p == nil || p.kernel <= 0 || input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn: maxpool2d: uninitialized layer or input")
	}

	saved := input.Data()

	data, err := tensor.MaxPool2D(saved, p.kernel, p.stride, p.padding)
	if err != nil {
		return nil, err
	}

	return nn.NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.MaxPool2DBackward(saved, g, p.kernel, p.stride, p.padding)
	})
}

// Parameters returns nil because pooling has no trainable parameters.
func (*MaxPool2D) Parameters() []*nn.Parameter {
	return nil
}
