package cpu

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/tensor/backend"
)

// Reduce removes selected axes, preserving accumulation order and NaN behavior.
func (Device) Reduce(op backend.ReduceOp, dst, src backend.View, axes []int) error {
	if op < backend.Sum || op > backend.Max {
		return fmt.Errorf("%w: reduction %d", backend.ErrUnsupported, op)
	}

	out, err := writeView(dst, true)
	if err != nil {
		return err
	}

	source, err := readView(src)
	if err != nil {
		return err
	}

	if dst.Buffer == src.Buffer {
		return fmt.Errorf("%w: aliased reduction output", backend.ErrInvalidView)
	}

	a := &source

	var localReduced [8]bool

	reduce := localReduced[:]
	if a.Rank() > len(localReduced) {
		reduce = make([]bool, a.Rank())
	} else {
		reduce = reduce[:a.Rank()]
	}

	if len(axes) == 0 {
		for axis := range reduce {
			reduce[axis] = true
		}
	} else {
		for _, axis := range axes {
			if axis < 0 || axis >= a.Rank() || reduce[axis] {
				return fmt.Errorf("%w: invalid reduction axes", backend.ErrInvalidView)
			}

			reduce[axis] = true
		}
	}

	expected := 1
	outputAxis := 0

	for axis, size := range a.shape {
		if reduce[axis] {
			if size == 0 && op != backend.Sum {
				return fmt.Errorf("%w: empty reduction group", backend.ErrInvalidView)
			}
		} else {
			if outputAxis >= out.Rank() || out.shape[outputAxis] != size {
				return backend.ErrInvalidView
			}

			outputAxis++
			expected *= size
		}
	}

	if expected != out.size || outputAxis != out.Rank() {
		return fmt.Errorf("%w: reduction output size", backend.ErrInvalidView)
	}

	clear(out.data)

	if out.size == 0 {
		return nil
	}

	count := a.size / out.size
	if count == 0 {
		return nil
	}

	if op == backend.Max {
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
			if op == backend.Max {
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
			if op == backend.Max {
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

	if op == backend.Mean {
		for i := range out.data {
			out.data[i] /= float32(count)
		}
	}

	return nil
}
