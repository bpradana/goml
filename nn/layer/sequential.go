package layer

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/nn"

	"github.com/bpradana/goml/tensor"
)

// Sequential applies a fixed sequence of layers in order.
type Sequential struct {
	layers []Layer
}

// NewSequential copies the layer list. Nil layers return an error.
// An empty sequence is the identity operation.
func NewSequential(layers ...Layer) (*Sequential, error) {
	for i, layer := range layers {
		if layer == nil {
			return nil, fmt.Errorf("nn: sequential: nil layer at index %d", i)
		}
	}

	return &Sequential{layers: slices.Clone(layers)}, nil
}

// Forward applies all layers, preserving inference mode.
// A failed layer returns an error that includes its index.
func (s *Sequential) Forward(input *nn.Value) (*nn.Value, error) {
	if input == nil || input.Data() == nil {
		return nil, fmt.Errorf("nn: sequential: nil or uninitialized input")
	}

	output := input

	for i, layer := range s.layers {
		var err error

		output, err = layer.Forward(output)
		if err != nil {
			return nil, fmt.Errorf("nn: sequential: layer %d: %w", i, err)
		}
	}

	return output, nil
}

// Parameters returns a fresh list of distinct parameters in layer order.
// Shared layers and shared parameters appear once.
func (s *Sequential) Parameters() []*nn.Parameter {
	var parameters []*nn.Parameter

	seen := make(map[*nn.Parameter]bool)

	for _, layer := range s.layers {
		for _, parameter := range layer.Parameters() {
			if !seen[parameter] {
				seen[parameter] = true
				parameters = append(parameters, parameter)
			}
		}
	}

	return parameters
}

// Predict runs a forward pass with graph recording disabled.
// The result is immutable and contains no graph references.
func (s *Sequential) Predict(input *tensor.Tensor) (*tensor.Tensor, error) {
	value, err := nn.Inference(input)
	if err != nil {
		return nil, err
	}

	output, err := s.Forward(value)
	if err != nil {
		return nil, err
	}

	return output.Data(), nil
}
