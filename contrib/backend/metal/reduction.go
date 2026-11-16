package metal

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

// Reduce encodes a per-output reduction over selected axes.
func (d *Device) Reduce(op backend.ReduceOp, dst, a backend.View, axes []int) error {
	code := uint32(op - 1)
	if op < backend.Sum || op > backend.Max {
		return fmt.Errorf("%w: reduction %d", backend.ErrUnsupported, op)
	}

	if len(axes) == 0 {
		axes = make([]int, len(a.Shape))
		for i := range axes {
			axes[i] = i
		}
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	input, err := d.view(a)
	if err != nil {
		return err
	}

	output, err := d.output(dst, a)
	if err != nil {
		return err
	}

	p := viewParams(a.Shape, a, backend.View{})
	p.n = uint32(output.length)

	p.op = code

	count, reduced := 1, uint32(0)
	for _, axis := range axes {
		if axis < 0 || axis >= len(a.Shape) || reduced&(1<<axis) != 0 {
			return fmt.Errorf("%w: metal: reduction: invalid or repeated axis", backend.ErrInvalidView)
		}

		reduced |= 1 << axis
		p.flags |= 1 << axis
	}

	for axis, size := range a.Shape {
		if reduced&(1<<axis) == 0 {
			count *= size
		} else if size == 0 && op != backend.Sum {
			return fmt.Errorf("%w: metal: reduction: empty mean or max", backend.ErrInvalidView)
		}
	}

	if count != output.length {
		return fmt.Errorf("%w: metal: reduction: output size mismatch", backend.ErrInvalidView)
	}

	return nativeCompute(d.handle, "reduce_values", input.handle, nil, output.handle, p)
}
