package metal

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

// MatMul uses MPSMatrixMultiplication. Strided operands are first materialized
// on-device; native matrix inputs persist across calls without host transfers.
func (d *Device) MatMul(dst, a, b backend.View) error {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		return fmt.Errorf("%w: metal: matmul: expected two matrices", backend.ErrInvalidView)
	}

	av, releaseA, err := d.contiguous(a)
	if err != nil {
		return err
	}
	defer releaseA()

	bv, releaseB, err := d.contiguous(b)
	if err != nil {
		return err
	}
	defer releaseB()

	d.mu.Lock()
	defer d.mu.Unlock()

	aa, err := d.view(av)
	if err != nil {
		return err
	}

	bb, err := d.view(bv)
	if err != nil {
		return err
	}

	out, err := d.output(dst, a, b)
	if err != nil {
		return err
	}

	m, k, n := a.Shape[0], a.Shape[1], b.Shape[1]
	if out.length != m*n || b.Shape[0] != k {
		return fmt.Errorf("%w: metal: matmul shape mismatch", backend.ErrInvalidView)
	}

	if m == 0 || n == 0 || k == 0 {
		return nativeCompute(d.handle, "fill_values", nil, nil, out.handle, kernelParams{n: uint32(out.length)})
	}

	args := [9]uint32{uint32(m), uint32(k), uint32(n), uint32(av.Offset), uint32(bv.Offset), uint32(k), uint32(n)}

	return nativeMatMul(d.handle, aa.handle, bb.handle, out.handle, args, false, false)
}
