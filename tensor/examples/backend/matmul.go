package examplebackend

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

// MatMul implements rectangular matrices and strided inputs. It accepts only
// complete contiguous destinations; other output layouts return ErrUnsupported.
// This scalar loop is a tutorial reference, not an optimized CPU replacement.
func (d *Device) MatMul(dst, a, b backend.View) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	out, err := d.checked(dst.Buffer)
	if err != nil {
		return err
	}

	left, err := d.checked(a.Buffer)
	if err != nil {
		return err
	}

	right, err := d.checked(b.Buffer)
	if err != nil {
		return err
	}

	for _, value := range []backend.View{dst, a, b} {
		if err := value.ValidateLayout(value.Buffer.Len(), value.Buffer == dst.Buffer); err != nil {
			return err
		}
	}

	if !dst.IsContiguous() || dst.Offset != 0 {
		return fmt.Errorf("%w: example destination layout", backend.ErrUnsupported)
	}

	if len(a.Shape) != 2 || len(b.Shape) != 2 || len(dst.Shape) != 2 {
		return backend.ErrInvalidView
	}

	m, k, n := a.Shape[0], a.Shape[1], b.Shape[1]
	if b.Shape[0] != k || dst.Shape[0] != m || dst.Shape[1] != n || out.length != m*n || out == left || out == right {
		return backend.ErrInvalidView
	}

	clear(out.data)

	for row := range m {
		for column := range n {
			for inner := range k {
				ai := a.Offset + row*a.Strides[0] + inner*a.Strides[1]
				bi := b.Offset + inner*b.Strides[0] + column*b.Strides[1]
				out.data[row*n+column] += left.data[ai] * right.data[bi]
			}
		}
	}

	return nil
}

var _ backend.MatMulBackend = (*Device)(nil)
