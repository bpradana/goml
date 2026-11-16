package yolo

import (
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/layer"
)

type bottleneck struct {
	first, second *layer.Sequential
	shortcut      bool
}

type c2f struct {
	first, last *layer.Sequential
	hidden      int
	blocks      []bottleneck
}

func (c *c2f) Forward(input *nn.Value) (*nn.Value, error) {
	projected, err := c.first.Forward(input)
	if err != nil {
		return nil, err
	}

	first, err := nn.Slice(projected, 1, 0, c.hidden)
	if err != nil {
		return nil, err
	}

	current, err := nn.Slice(projected, 1, c.hidden, 2*c.hidden)
	if err != nil {
		return nil, err
	}

	parts := []*nn.Value{first, current}
	for _, block := range c.blocks {
		value, err := block.first.Forward(current)
		if err != nil {
			return nil, err
		}

		value, err = block.second.Forward(value)
		if err != nil {
			return nil, err
		}

		if block.shortcut {
			value, err = nn.Add(current, value)
			if err != nil {
				return nil, err
			}
		}

		current = value
		parts = append(parts, current)
	}

	combined, err := nn.Concat(1, parts...)
	if err != nil {
		return nil, err
	}

	return c.last.Forward(combined)
}

func (c *c2f) Parameters() []*nn.Parameter {
	parameters := append(c.first.Parameters(), c.last.Parameters()...)
	for _, block := range c.blocks {
		parameters = append(parameters, block.first.Parameters()...)
		parameters = append(parameters, block.second.Parameters()...)
	}

	return parameters
}

type sppf struct {
	first, last *layer.Sequential
	pool        *layer.MaxPool2D
}

func (s *sppf) Forward(input *nn.Value) (*nn.Value, error) {
	current, err := s.first.Forward(input)
	if err != nil {
		return nil, err
	}

	parts := []*nn.Value{current}
	for range 3 {
		current, err = s.pool.Forward(current)
		if err != nil {
			return nil, err
		}

		parts = append(parts, current)
	}

	combined, err := nn.Concat(1, parts...)
	if err != nil {
		return nil, err
	}

	return s.last.Forward(combined)
}

func (s *sppf) Parameters() []*nn.Parameter {
	return append(s.first.Parameters(), s.last.Parameters()...)
}
