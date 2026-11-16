package layer

import (
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func (c *Conv2D) deviceForward(input *nn.Value) (*nn.Value, error) {
	weight := c.projection.weight.Value()
	inputs := []*nn.Value{input, weight}

	var bias *tensor.Tensor
	if c.projection.bias != nil {
		inputs = append(inputs, c.projection.bias.Value())
		bias = c.projection.bias.Value().Data()
	}

	savedInput, savedWeight := input.Data(), weight.Data()

	data, err := tensor.Conv2D(savedInput, savedWeight, bias, c.kernel, c.stride, c.padding)
	if err != nil {
		return nil, err
	}

	return nn.NewMultiOperation(data, inputs, func(g *tensor.Tensor) ([]*tensor.Tensor, error) {
		dx, dw, err := tensor.Conv2DBackward(savedInput, savedWeight, g, c.kernel, c.stride, c.padding)
		if err != nil {
			return nil, err
		}

		gradients := []*tensor.Tensor{dx, dw}

		if bias != nil {
			db, err := tensor.Sum(g, 0, 2, 3)
			if err != nil {
				return nil, err
			}

			gradients = append(gradients, db)
		}

		return gradients, nil
	})
}
