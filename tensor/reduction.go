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

	// A contiguous reduced suffix consists of independent contiguous groups.
	// Keep the original element order so Float32 sums do not change grouping.
	suffix, started := true, false
	for _, reduced := range reduce {
		started = started || reduced
		if started && !reduced {
			suffix = false
		}
	}

	if suffix && a.IsContiguous() {
		for dest := range out.data {
			values := a.data[a.offset+dest*count : a.offset+(dest+1)*count]
			if op == "max" {
				for _, value := range values {
					out.data[dest] = float32(math.Max(float64(out.data[dest]), float64(value)))
				}
			} else {
				var sum float32
				for _, value := range values {
					sum += value
				}

				out.data[dest] = sum
			}
		}
	} else {
		var local [16]int

		work := local[:]
		if a.Rank() > 8 {
			work = make([]int, 2*a.Rank())
		}

		coords, destStrides := work[:a.Rank()], work[a.Rank():2*a.Rank()]
		axis := 0

		for i, reduced := range reduce {
			if !reduced {
				destStrides[i] = out.strides[axis]
				axis++
			}
		}

		source, dest := a.offset, 0
		for range a.size {
			value := a.data[source]
			if op == "max" {
				out.data[dest] = float32(math.Max(float64(out.data[dest]), float64(value)))
			} else {
				out.data[dest] += value
			}

			for axis := a.Rank() - 1; axis >= 0; axis-- {
				coords[axis]++
				source += a.strides[axis]

				dest += destStrides[axis]
				if coords[axis] < a.shape[axis] {
					break
				}

				source -= coords[axis] * a.strides[axis]
				dest -= coords[axis] * destStrides[axis]
				coords[axis] = 0
			}
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
