package tensor

import (
	"fmt"
	"runtime"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

// ScatterSlice reverses a slice by writing gradient into a new zero tensor.
// Shape and axis follow Slice; all execution stays on gradient's device.
func ScatterSlice(gradient *Tensor, shape []int, axis, start int) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{gradient})

	if err := valid("scatter", gradient); err != nil {
		return nil, err
	}

	if axis < 0 || axis >= len(shape) || gradient.Rank() != len(shape) || start < 0 {
		return nil, fmt.Errorf("tensor: scatter: invalid axis, rank, or start")
	}

	out, err := FullLike(gradient, 0, shape...)
	if err != nil {
		return nil, err
	}

	target, err := out.Slice(axis, start, start+gradient.shape[axis])
	if err != nil {
		_ = out.Close()
		return nil, err
	}

	defer func() { _ = target.Close() }()

	if !slices.Equal(target.shape, gradient.shape) {
		_ = out.Close()
		return nil, fmt.Errorf("tensor: scatter: gradient shape %v must match slice shape %v", gradient.shape, target.shape)
	}

	compute, ok := gradient.device.(backend.CopyBackend)
	if !ok {
		_ = out.Close()
		return nil, unsupported("scatter", gradient.device)
	}

	if err := compute.Copy(target.deviceView(), gradient.deviceView()); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}
