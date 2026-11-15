package tensor

import (
	"fmt"
	"math"
)

func reduction(op string, a *Tensor, axes []int) (*Tensor, error) {
	if err := valid(op, a); err != nil {
		return nil, err
	}

	reduce := make([]bool, a.Rank())
	if len(axes) == 0 {
		for i := range reduce {
			reduce[i] = true
		}
	} else {
		for _, axis := range axes {
			if axis < 0 || axis >= a.Rank() || reduce[axis] {
				return nil, fmt.Errorf("tensor: %s: invalid or duplicate axes %v for shape %v", op, axes, a.shape)
			}

			reduce[axis] = true
		}
	}

	shape := make([]int, 0, a.Rank())
	emptyGroup := false

	for i, d := range a.shape {
		if reduce[i] {
			emptyGroup = emptyGroup || d == 0
		} else {
			shape = append(shape, d)
		}
	}

	if emptyGroup && op != "sum" {
		return nil, fmt.Errorf("tensor: %s: empty reduction in shape %v", op, a.shape)
	}

	out, err := allocated(shape)
	if err != nil {
		return nil, err
	}

	if out.size == 0 {
		return out, nil
	}

	count := a.size / out.size

	if op == "max" {
		for i := range out.data {
			out.data[i] = float32(math.Inf(-1))
		}
	}

	coords := make([]int, a.Rank())
	for i := 0; i < a.size; i++ {
		rest := i
		for ax := a.Rank() - 1; ax >= 0; ax-- {
			coords[ax] = rest % a.shape[ax]
			rest /= a.shape[ax]
		}

		dest := 0

		for ax, d := range a.shape {
			if !reduce[ax] {
				dest = dest*d + coords[ax]
			}
		}

		v := a.data[a.index(i)]
		if op == "max" {
			out.data[dest] = float32(math.Max(float64(out.data[dest]), float64(v)))
		} else {
			out.data[dest] += v
		}
	}

	if op == "mean" {
		for i := range out.data {
			out.data[i] /= float32(count)
		}
	}

	return out, nil
}

// Sum allocates a sum over axes, removing reduced dimensions.
// No axes means all dimensions. Empty sums are zero; invalid axes return an error.
func Sum(a *Tensor, axes ...int) (*Tensor, error) { return reduction("sum", a, axes) }

// Mean allocates a mean over axes, removing them; no axes means all elements.
// Empty reduction groups and invalid or duplicate axes return errors.
func Mean(a *Tensor, axes ...int) (*Tensor, error) { return reduction("mean", a, axes) }

// Max allocates maxima over axes, removing them; no axes means all elements.
// NaNs propagate. Empty reduction groups and invalid axes return errors.
func Max(a *Tensor, axes ...int) (*Tensor, error) { return reduction("max", a, axes) }
