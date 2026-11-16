package activation

import (
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func deviceActivation(input *nn.Value, kind string) (*nn.Value, error) {
	source := input.Data()

	data, err := tensor.Activation(kind, source)
	if err != nil {
		return nil, err
	}

	return nn.NewOperation(data, input, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.ActivationBackward(kind, source, g)
	})
}
