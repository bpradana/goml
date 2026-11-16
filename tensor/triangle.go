package tensor

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

// MaskUpperTriangle copies a matrix and replaces positions above its main
// diagonal with fill. GPU execution remains on-device. Rank must be two.
func MaskUpperTriangle(input *Tensor, fill float32) (*Tensor, error) {
	if err := valid("triangle", input); err != nil {
		return nil, err
	}

	if input.Rank() != 2 {
		return nil, fmt.Errorf("tensor: triangle: expected rank two, got %v", input.shape)
	}

	return scalar(backend.MaskUpperTriangle, input, fill)
}
