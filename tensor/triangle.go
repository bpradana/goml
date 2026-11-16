package tensor

import "fmt"

// MaskUpperTriangle copies a matrix and replaces positions above its main
// diagonal with fill. GPU execution remains on-device. Rank must be two.
func MaskUpperTriangle(input *Tensor, fill float32) (*Tensor, error) {
	if err := valid("triangle", input); err != nil {
		return nil, err
	}

	if input.Rank() != 2 {
		return nil, fmt.Errorf("tensor: triangle: expected rank two, got %v", input.shape)
	}

	if input.gpu != nil {
		return gpuUnary("causal", input, fill)
	}

	output := input.Contiguous()
	for row := range input.shape[0] {
		for column := row + 1; column < input.shape[1]; column++ {
			output.data[row*input.shape[1]+column] = fill
		}
	}

	return output, nil
}
