package tensor

import (
	"fmt"
	"runtime"
	"slices"
)

func gpuUnary(op string, a *Tensor, scalar float32) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{a})

	backend, ok := a.Device().(ArithmeticBackend)
	if !ok {
		return nil, fmt.Errorf("tensor: device lacks unary kernels")
	}

	out, err := allocatedOn(a.shape, a)
	if err != nil {
		return nil, err
	}

	if err := backend.Unary(op, out.gpu.storage.buffer, a.deviceView(), scalar); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

func gpuBinary(op string, a, b *Tensor, shape []int) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{a, b})

	backend, ok := a.Device().(ArithmeticBackend)
	if !ok {
		return nil, fmt.Errorf("tensor: device lacks binary kernels")
	}

	out, err := allocatedOn(shape, a, b)
	if err != nil {
		return nil, err
	}

	if err := backend.Binary(op, out.gpu.storage.buffer, a.deviceView(), b.deviceView(), shape); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

func gpuMatMul(a, b *Tensor) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{a, b})

	backend, ok := a.Device().(ArithmeticBackend)
	if !ok {
		return nil, fmt.Errorf("tensor: device lacks matrix multiplication")
	}

	out, err := allocatedOn([]int{a.shape[0], b.shape[1]}, a, b)
	if err != nil {
		return nil, err
	}

	if err := backend.MatMul(out.gpu.storage.buffer, a.deviceView(), b.deviceView()); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

func gpuReduce(op string, a *Tensor, shape []int, reduced []bool) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{a})

	backend, ok := a.Device().(ArithmeticBackend)
	if !ok {
		return nil, fmt.Errorf("tensor: device lacks reductions")
	}

	axes := make([]int, 0, len(reduced))
	for axis, yes := range reduced {
		if yes {
			axes = append(axes, axis)
		}
	}

	out, err := allocatedOn(shape, a)
	if err != nil {
		return nil, err
	}

	if err := backend.Reduce(op, out.gpu.storage.buffer, a.deviceView(), axes); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

func gpuConcat(axis int, shape []int, inputs []*Tensor) (*Tensor, error) {
	defer runtime.KeepAlive(inputs)

	backend, ok := inputs[0].Device().(LayoutBackend)
	if !ok {
		return nil, fmt.Errorf("tensor: device lacks concatenation")
	}

	out, err := allocatedOn(shape, inputs...)
	if err != nil {
		return nil, err
	}

	start := 0
	for _, input := range inputs {
		destination, err := out.Slice(axis, start, start+input.shape[axis])
		if err != nil {
			_ = out.Close()
			return nil, err
		}

		err = backend.Copy(destination.deviceView(), input.deviceView())

		_ = destination.Close()
		if err != nil {
			_ = out.Close()
			return nil, err
		}

		start += input.shape[axis]
	}

	return out, nil
}

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

	if gradient.gpu != nil {
		backend, ok := gradient.Device().(LayoutBackend)
		if !ok {
			_ = out.Close()
			return nil, fmt.Errorf("tensor: device lacks scatter")
		}

		if err := backend.Copy(target.deviceView(), gradient.deviceView()); err != nil {
			_ = out.Close()
			return nil, err
		}
	} else {
		for i, value := range gradient.Data() {
			out.data[target.index(i)] = value
		}
	}

	return out, nil
}
