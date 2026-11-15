package layer

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/bpradana/goml/nn"

	"github.com/bpradana/goml/tensor"
)

// Dense computes XW+b for 2D input [batch,in] and weight [in,out].
type Dense struct {
	weight *nn.Parameter
	bias   *nn.Parameter
}

// NewDense constructs a dense layer with positive dimensions and optional bias.
// Weights use Xavier normal initialization with variance 2/(in+out); bias is zero.
// The caller supplies a seeded generator. Invalid dimensions or a nil generator
// return an error.
func NewDense(in, out int, bias bool, rng *rand.Rand) (*Dense, error) {
	if in <= 0 || out <= 0 || rng == nil {
		return nil, fmt.Errorf("nn: dense: positive dimensions and a generator required, got in=%d out=%d", in, out)
	}

	weights, err := tensor.Randn(rng, in, out)
	if err != nil {
		return nil, err
	}

	weights, err = tensor.Scale(weights, float32(math.Sqrt(2/(float64(in)+float64(out)))))
	if err != nil {
		return nil, err
	}

	weight, err := nn.NewParameter(weights)
	if err != nil {
		return nil, err
	}

	dense := &Dense{weight: weight}

	if bias {
		zeros, err := tensor.Zeros(out)
		if err != nil {
			return nil, err
		}

		dense.bias, err = nn.NewParameter(zeros)
		if err != nil {
			return nil, err
		}
	}

	return dense, nil
}

// Forward allocates XW+b and records gradients unless input carries inference
// mode. Incompatible input shapes or uninitialized layers return an error.
func (d *Dense) Forward(input *nn.Value) (*nn.Value, error) {
	if d == nil || d.weight == nil {
		return nil, fmt.Errorf("nn: dense: uninitialized layer")
	}

	output, err := nn.MatMul(input, d.weight.Value())
	if err != nil {
		return nil, err
	}

	if d.bias != nil {
		return nn.Add(output, d.bias.Value())
	}

	return output, nil
}

// Parameters returns a fresh slice containing weight followed by optional bias.
func (d *Dense) Parameters() []*nn.Parameter {
	parameters := []*nn.Parameter{d.weight}
	if d.bias != nil {
		parameters = append(parameters, d.bias)
	}

	return parameters
}
