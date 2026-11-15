package shape

import "fmt"

// Layout validates dimensions and computes element count and row-major strides.
func Layout(shape []int) (int, []int, error) {
	strides := make([]int, len(shape))
	n := 1

	for i := len(shape) - 1; i >= 0; i-- {
		d := shape[i]
		if d < 0 {
			return 0, nil, fmt.Errorf("tensor: shape: negative dimension %d in %v", d, shape)
		}

		strides[i] = n
		if d != 0 && n > int(^uint(0)>>1)/d {
			return 0, nil, fmt.Errorf("tensor: shape: element count or strides overflow for %v", shape)
		}

		n *= d
	}

	return n, strides, nil
}
