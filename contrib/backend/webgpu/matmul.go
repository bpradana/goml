package webgpu

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

// MatMul writes [m,k] × [k,n] into [m,n] using a 16×16 workgroup-tiled WGSL
// kernel. Inputs may be strided; results and operands remain on the device.
// Empty inner dimensions write zeros. Invalid shapes or aliasing return errors.
func (d *Device) MatMul(dst, a, b backend.View) error {
	if len(a.Shape) != 2 || len(b.Shape) != 2 || len(dst.Shape) != 2 ||
		a.Shape[1] != b.Shape[0] || dst.Shape[0] != a.Shape[0] || dst.Shape[1] != b.Shape[1] {
		return fmt.Errorf("webgpu: matmul: %w: incompatible matrix shapes", backend.ErrInvalidView)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	handles, err := d.bindings(dst, a, b)
	if err != nil {
		return err
	}

	p := params(dst, a, b, backend.View{})
	if p.n == 0 {
		return nil
	}

	p.ints[0], p.ints[1], p.ints[2] = uint32(a.Shape[0]), uint32(a.Shape[1]), uint32(b.Shape[1])

	return d.native.dispatch("matmul_values", p, handles, (p.ints[2]+15)/16, (p.ints[0]+15)/16)
}
