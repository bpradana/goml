package webgpu

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

// Reduce writes Sum, Mean, or Max with selected dimensions removed. No axes
// reduces every dimension. Empty sums are zero; empty mean/max groups fail.
// NaNs propagate. Inputs may be strided and no values are downloaded.
func (d *Device) Reduce(op backend.ReduceOp, dst, src backend.View, axes []int) error {
	if op < backend.Sum || op > backend.Max {
		return fmt.Errorf("webgpu: %w: reduction %v", backend.ErrUnsupported, op)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	handles, err := d.bindings(dst, src)
	if err != nil {
		return err
	}

	p := params(dst, src, backend.View{}, backend.View{})
	p.op = uint32(op)

	p.rank = uint32(len(src.Shape))
	for i, size := range src.Shape {
		p.shape[i] = uint32(size)
	}

	if len(axes) == 0 {
		p.flags = (1 << len(src.Shape)) - 1
	} else {
		for _, axis := range axes {
			if axis < 0 || axis >= len(src.Shape) || p.flags&(1<<axis) != 0 {
				return fmt.Errorf("webgpu: reduce: %w: invalid or repeated axis", backend.ErrInvalidView)
			}

			p.flags |= 1 << axis
		}
	}

	shape := make([]int, 0, len(src.Shape))
	for i, size := range src.Shape {
		if p.flags&(1<<i) == 0 {
			shape = append(shape, size)
		} else if size == 0 && op != backend.Sum {
			return fmt.Errorf("webgpu: reduce: %w: empty mean or max", backend.ErrInvalidView)
		}
	}

	if !slices.Equal(shape, dst.Shape) {
		return fmt.Errorf("webgpu: reduce: %w: output shape mismatch", backend.ErrInvalidView)
	}

	return d.dispatch("reduce_values", p, handles)
}
