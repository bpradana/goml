package tensor

import (
	"fmt"
	"runtime"

	"github.com/bpradana/goml/tensor/backend"
)

func reduction(op backend.ReduceOp, a *Tensor, axes []int) (*Tensor, error) {
	defer runtime.KeepAlive(a)

	if err := valid("reduce", a); err != nil {
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
				return nil, fmt.Errorf("tensor: %d: invalid or duplicate axes %v for shape %v", op, axes, a.shape)
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

	if emptyGroup && op != backend.Sum {
		return nil, fmt.Errorf("tensor: %d: empty reduction in shape %v", op, a.shape)
	}

	compute, ok := a.device.(backend.ReductionBackend)
	if !ok {
		return nil, unsupported("reduce", a.device)
	}

	out, err := allocatedOn(shape, a)
	if err != nil {
		return nil, err
	}

	if err := compute.Reduce(op, out.deviceView(), a.deviceView(), axes); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

// Sum allocates a sum over axes, removing reduced dimensions.
// No axes means all dimensions. Empty sums are zero; invalid axes return an error.
func Sum(a *Tensor, axes ...int) (*Tensor, error) { return reduction(backend.Sum, a, axes) }

// Mean allocates a mean over axes, removing them; no axes means all elements.
// Empty reduction groups and invalid or duplicate axes return errors.
func Mean(a *Tensor, axes ...int) (*Tensor, error) { return reduction(backend.Mean, a, axes) }

// Max allocates maxima over axes, removing them; no axes means all elements.
// NaNs propagate. Empty reduction groups and invalid axes return errors.
func Max(a *Tensor, axes ...int) (*Tensor, error) { return reduction(backend.Max, a, axes) }
