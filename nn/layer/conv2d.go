package layer

import (
	"fmt"
	"math/rand/v2"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

// Conv2D performs 2D cross-correlation on NCHW input, with square kernels,
// equal spatial strides, symmetric zero padding, and optional per-channel bias.
// Dilation and grouped convolution are not supported.
type Conv2D struct {
	inChannels  int
	outChannels int
	kernel      int
	stride      int
	padding     int
	projection  *Dense
}

// NewConv2D creates trainable kernels using a caller-owned seeded generator.
// Channels, kernel, and stride must be positive; padding must be nonnegative.
// The flattened kernel [inChannels*kernel*kernel,outChannels] uses Dense's Xavier
// initialization. Parameters returns this weight followed by optional zero bias.
func NewConv2D(inChannels, outChannels, kernel, stride, padding int, bias bool, rng *rand.Rand) (*Conv2D, error) {
	if inChannels <= 0 || outChannels <= 0 || kernel <= 0 || stride <= 0 || padding < 0 || rng == nil {
		return nil, fmt.Errorf("nn: conv2d: positive channels, kernel, stride, nonnegative padding, and generator required")
	}

	maxInt := int(^uint(0) >> 1)
	if kernel > maxInt/inChannels || kernel > maxInt/(inChannels*kernel) {
		return nil, fmt.Errorf("nn: conv2d: kernel size overflow")
	}

	projection, err := NewDense(inChannels*kernel*kernel, outChannels, bias, rng)
	if err != nil {
		return nil, err
	}

	return &Conv2D{
		inChannels: inChannels, outChannels: outChannels, kernel: kernel,
		stride: stride, padding: padding, projection: projection,
	}, nil
}

// Forward maps [batch,inChannels,height,width] to [batch,outChannels,outH,outW],
// where outH=floor((height+2*padding-kernel)/stride)+1, and similarly for outW.
// It allocates a patch matrix, computes a matrix product, and materializes NCHW
// output. Invalid shapes and size overflow fail before indexing. Inference is
// propagated through the patch extraction and projection.
func (c *Conv2D) Forward(input *nn.Value) (*nn.Value, error) {
	if c == nil || c.projection == nil || input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn: conv2d: uninitialized layer or input")
	}

	shape := input.Data().Shape()
	if len(shape) != 4 || shape[1] != c.inChannels || shape[2] <= 0 || shape[3] <= 0 {
		return nil, fmt.Errorf("nn: conv2d: expected [batch,%d,height>0,width>0], got %v", c.inChannels, shape)
	}

	maxInt := int(^uint(0) >> 1)
	if c.padding > (maxInt-shape[2])/2 || c.padding > (maxInt-shape[3])/2 {
		return nil, fmt.Errorf("nn: conv2d: padded shape overflow")
	}

	height, width := shape[2]+2*c.padding, shape[3]+2*c.padding
	if height < c.kernel || width < c.kernel {
		return nil, fmt.Errorf("nn: conv2d: kernel %d exceeds padded spatial shape [%d %d]", c.kernel, height, width)
	}

	outH, outW := (height-c.kernel)/c.stride+1, (width-c.kernel)/c.stride+1
	features := c.inChannels * c.kernel * c.kernel

	storage, err := tensor.Zeros(shape[0], outH, outW, features)
	if err != nil {
		return nil, err
	}

	patches := storage.Data()
	values := input.Data().Data()

	for batch := range shape[0] {
		for row := range outH {
			for column := range outW {
				base := ((batch*outH+row)*outW + column) * features

				for channel := range c.inChannels {
					for ky := range c.kernel {
						for kx := range c.kernel {
							y, x := row*c.stride-c.padding+ky, column*c.stride-c.padding+kx
							if y >= 0 && y < shape[2] && x >= 0 && x < shape[3] {
								patches[base+(channel*c.kernel+ky)*c.kernel+kx] = values[((batch*c.inChannels+channel)*shape[2]+y)*shape[3]+x]
							}
						}
					}
				}
			}
		}
	}

	matrix, err := tensor.From(patches, shape[0]*outH*outW, features)
	if err != nil {
		return nil, err
	}

	size := len(values)

	patchValue, err := nn.NewOperation(matrix, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		gradient := make([]float32, size)
		patchGrad := g.Data()

		for batch := range shape[0] {
			for row := range outH {
				for column := range outW {
					base := ((batch*outH+row)*outW + column) * features

					for channel := range c.inChannels {
						for ky := range c.kernel {
							for kx := range c.kernel {
								y, x := row*c.stride-c.padding+ky, column*c.stride-c.padding+kx
								if y >= 0 && y < shape[2] && x >= 0 && x < shape[3] {
									gradient[((batch*c.inChannels+channel)*shape[2]+y)*shape[3]+x] += patchGrad[base+(channel*c.kernel+ky)*c.kernel+kx]
								}
							}
						}
					}
				}
			}
		}

		return tensor.From(gradient, shape...)
	})
	if err != nil {
		return nil, err
	}

	output, err := c.projection.Forward(patchValue)
	if err != nil {
		return nil, err
	}

	output, err = nn.Reshape(output, shape[0], outH, outW, c.outChannels)
	if err != nil {
		return nil, err
	}

	output, err = nn.Transpose(output, 0, 3, 1, 2)
	if err != nil {
		return nil, err
	}

	return nn.Contiguous(output)
}

// Parameters returns the flattened kernel weight and optional bias.
func (c *Conv2D) Parameters() []*nn.Parameter {
	return c.projection.Parameters()
}
