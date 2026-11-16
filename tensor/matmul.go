package tensor

import (
	"fmt"
	"runtime"

	"github.com/bpradana/goml/tensor/backend"
)

// MatMul multiplies two matrices on their shared device, allocating [m,n].
func MatMul(a, b *Tensor) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{a, b})

	if err := valid("matmul", a, b); err != nil {
		return nil, err
	}

	if a.Rank() != 2 || b.Rank() != 2 || a.shape[1] != b.shape[0] {
		return nil, fmt.Errorf("tensor: matmul: incompatible shapes %v and %v", a.shape, b.shape)
	}

	if err := compatible("matmul", a, b); err != nil {
		return nil, err
	}

	compute, ok := a.device.(backend.MatMulBackend)
	if !ok {
		return nil, unsupported("matmul", a.device)
	}

	out, err := allocatedOn([]int{a.shape[0], b.shape[1]}, a, b)
	if err != nil {
		return nil, err
	}

	if err := compute.MatMul(out.deviceView(), a.deviceView(), b.deviceView()); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}
