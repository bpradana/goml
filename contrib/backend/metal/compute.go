package metal

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

type kernelParams struct {
	n, rank, op, flags uint32
	shape, as, bs      [8]uint32
	ao, bo             uint32
	ints               [8]uint32
	scalars            [4]float32
}

func elements(shape []int) (int, error) {
	if len(shape) > 8 {
		return 0, fmt.Errorf("%w: Metal rank exceeds eight", backend.ErrUnsupported)
	}

	empty := false

	for _, size := range shape {
		if size < 0 {
			return 0, fmt.Errorf("%w: negative Metal dimension", backend.ErrInvalidView)
		}

		if uint64(size) > uint64(^uint32(0)) {
			return 0, fmt.Errorf("%w: Metal dimension %d", backend.ErrUnsupported, size)
		}

		empty = empty || size == 0
	}

	if empty {
		return 0, nil
	}

	count := uint64(1)
	for _, size := range shape {
		if count > uint64(^uint32(0))/uint64(size) {
			return 0, fmt.Errorf("%w: Metal indexing exceeds 32 bits", backend.ErrUnsupported)
		}

		count *= uint64(size)
	}

	return int(count), nil
}

func (d *Device) view(v backend.View) (*buffer, error) {
	b, err := d.checked(v.Buffer)
	if err != nil {
		return nil, err
	}

	if err := v.ValidateLayout(b.length, false); err != nil {
		return nil, err
	}

	if _, err := elements(v.Shape); err != nil {
		return nil, err
	}

	if uint64(v.Offset) > uint64(^uint32(0)) {
		return nil, fmt.Errorf("%w: Metal offset exceeds 32 bits", backend.ErrUnsupported)
	}

	for _, stride := range v.Strides {
		if uint64(stride) > uint64(^uint32(0)) {
			return nil, fmt.Errorf("%w: Metal stride exceeds 32 bits", backend.ErrUnsupported)
		}
	}

	return b, nil
}

func viewParams(shape []int, a, b backend.View) kernelParams {
	p := kernelParams{n: 1, rank: uint32(len(shape)), ao: uint32(a.Offset), bo: uint32(b.Offset)}
	for axis, size := range shape {
		p.n *= uint32(size)
		p.shape[axis] = uint32(size)

		ai := axis - len(shape) + len(a.Shape)
		if ai >= 0 && a.Shape[ai] != 1 {
			p.as[axis] = uint32(a.Strides[ai])
		}

		bi := axis - len(shape) + len(b.Shape)
		if bi >= 0 && b.Shape[bi] != 1 {
			p.bs[axis] = uint32(b.Strides[bi])
		}
	}

	return p
}

// output validates native write constraints under the device lock.
func (d *Device) output(v backend.View, inputs ...backend.View) (*buffer, error) {
	b, err := d.view(v)
	if err != nil {
		return nil, err
	}

	if err := v.ValidateLayout(b.length, true); err != nil {
		return nil, err
	}

	for _, input := range inputs {
		if input.Buffer == v.Buffer {
			return nil, fmt.Errorf("%w: aliased Metal destination", backend.ErrInvalidView)
		}
	}

	count, _ := v.Numel()
	if !v.IsContiguous() || v.Offset != 0 || count != b.length {
		return nil, fmt.Errorf("%w: Metal destination requires a complete contiguous allocation", backend.ErrUnsupported)
	}

	return b, nil
}
