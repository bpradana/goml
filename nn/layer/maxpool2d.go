package layer

import (
	"fmt"
	"math"

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

	shape := input.Data().Shape()
	if len(shape) != 4 || shape[2] <= 0 || shape[3] <= 0 {
		return nil, fmt.Errorf("nn: maxpool2d: expected [batch,channels,height>0,width>0], got %v", shape)
	}

	maxInt := int(^uint(0) >> 1)
	if p.padding > (maxInt-shape[2])/2 || p.padding > (maxInt-shape[3])/2 {
		return nil, fmt.Errorf("nn: maxpool2d: padded shape overflow")
	}

	height, width := shape[2]+2*p.padding, shape[3]+2*p.padding
	if height < p.kernel || width < p.kernel {
		return nil, fmt.Errorf("nn: maxpool2d: kernel exceeds padded shape %v", shape)
	}

	outH, outW := (height-p.kernel)/p.stride+1, (width-p.kernel)/p.stride+1
	outputShape := []int{shape[0], shape[1], outH, outW}

	storage, err := tensor.Zeros(outputShape...)
	if err != nil {
		return nil, err
	}

	values, output := input.Data().Data(), storage.Data()

	var selected []int
	if input.RequiresGrad() {
		selected = make([]int, len(output))
	}

	for plane := range shape[0] * shape[1] {
		for row := range outH {
			for column := range outW {
				best, index := float32(math.Inf(-1)), -1

				for ky := range p.kernel {
					for kx := range p.kernel {
						y, x := row*p.stride-p.padding+ky, column*p.stride-p.padding+kx
						if y < 0 || y >= shape[2] || x < 0 || x >= shape[3] {
							continue
						}

						source := (plane*shape[2]+y)*shape[3] + x

						value := values[source]
						if !math.IsNaN(float64(best)) && (index < 0 || value > best || math.IsNaN(float64(value))) {
							best, index = value, source
						}
					}
				}

				target := (plane*outH+row)*outW + column

				output[target] = best
				if selected != nil {
					selected[target] = index
				}
			}
		}
	}

	data, err := tensor.From(output, outputShape...)
	if err != nil {
		return nil, err
	}

	if !input.RequiresGrad() {
		return nn.NewOperation(data, input, nil)
	}

	return nn.NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		gradient := make([]float32, len(values))

		for i, value := range g.Data() {
			if selected[i] >= 0 {
				gradient[selected[i]] += value
			}
		}

		return tensor.From(gradient, shape...)
	})
}

// Parameters returns nil because pooling has no trainable parameters.
func (*MaxPool2D) Parameters() []*nn.Parameter {
	return nil
}
