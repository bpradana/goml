//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/bpradana/goml/tensor"
)

func TestCompute_InvalidBackendViewsReturnErrorsBeforeEncoding(t *testing.T) {
	device := openDevice(t)

	storage, err := device.Allocate(4)
	if err != nil {
		t.Fatal(err)
	}

	valid := tensor.DeviceView{Buffer: storage, Shape: []int{2, 2}, Strides: []int{2, 1}}
	badBounds := tensor.DeviceView{Buffer: storage, Shape: []int{2, 2}, Strides: []int{4, 1}}
	badRank := tensor.DeviceView{Buffer: storage, Shape: []int{4}, Strides: []int{1}}

	checks := []struct {
		name string
		run  func() error
	}{
		{"bounds", func() error { return device.Unary("exp", storage, badBounds, 0) }},
		{"matmul_rank", func() error { return device.MatMul(storage, badRank, valid) }},
		{"broadcast", func() error { return device.Binary("add", storage, valid, valid, []int{4}) }},
		{"copy_shape", func() error { return device.Copy(valid, badRank) }},
		{"axis", func() error { return device.Reduce("sum", storage, valid, []int{2}) }},
		{"pool_geometry", func() error { return device.Pool(storage, valid, 2, 0, 0) }},
		{"resize_scale", func() error { return device.Resize(storage, valid, 0, "nearest") }},
		{"convolution_rank", func() error { return device.Conv2D(storage, valid, valid, 1, 1, 0) }},
	}

	for _, check := range checks {
		t.Run(check.name, func(t *testing.T) {
			if err := check.run(); err == nil {
				t.Fatal("invalid backend input accepted")
			}
		})
	}
}

func TestCompute_BroadcastArithmeticAndReductionsStayOnDevice(t *testing.T) {
	device := openDevice(t)

	scope, err := tensor.NewScope(device)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = scope.Close() }()

	source, err := tensor.From([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	a, err := scope.To(source)
	if err != nil {
		t.Fatal(err)
	}

	b, err := scope.To(tensor.Scalar(2))
	if err != nil {
		t.Fatal(err)
	}

	product, err := tensor.Mul(a, b)
	if err != nil {
		t.Fatal(err)
	}

	reduced, err := tensor.Sum(product, 0)
	if err != nil {
		t.Fatal(err)
	}

	if reduced.Device() != device {
		t.Fatal("result left the GPU")
	}

	assertValues(t, reduced, []float32{10, 14, 18})

	if _, err := tensor.Add(a, source); err == nil {
		t.Fatal("mixed CPU and GPU arithmetic accepted")
	}
}

func TestMatMul_RectangularIrregularTransposedAndZeroDimensions(t *testing.T) {
	device := openDevice(t)

	for _, dimensions := range [][3]int{{2, 1, 3}, {2, 3, 5}, {17, 31, 13}, {1, 9, 1}, {3, 0, 4}, {0, 3, 4}} {
		m, k, n := dimensions[0], dimensions[1], dimensions[2]
		av, bv := make([]float32, m*k), make([]float32, n*k)

		for i := range av {
			av[i] = float32(i%11-5) / 7
		}

		for i := range bv {
			bv[i] = float32(i%7-3) / 5
		}

		a, err := tensor.From(av, m, k)
		if err != nil {
			t.Fatal(err)
		}

		b, err := tensor.From(bv, n, k)
		if err != nil {
			t.Fatal(err)
		}

		b, err = b.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		expected, err := tensor.MatMul(a, b)
		if err != nil {
			t.Fatal(err)
		}

		scope, err := tensor.NewScope(device)
		if err != nil {
			t.Fatal(err)
		}

		ga, err := scope.To(a)
		if err != nil {
			t.Fatal(err)
		}

		storedB, err := tensor.From(bv, n, k)
		if err != nil {
			t.Fatal(err)
		}

		gb, err := scope.To(storedB)
		if err != nil {
			t.Fatal(err)
		}

		gb, err = gb.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		output, err := tensor.MatMul(ga, gb)
		if err != nil {
			t.Fatal(err)
		}

		assertValues(t, output, expected.Data())

		if err := scope.Close(); err != nil {
			t.Fatal(err)
		}
	}
}

func TestConv2D_NativeMPSMatchesCPUReference(t *testing.T) {
	device := openDevice(t)

	scope, err := tensor.NewScope(device)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = scope.Close() }()

	input, err := tensor.From([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)
	if err != nil {
		t.Fatal(err)
	}

	weights, err := tensor.Ones(9, 2)
	if err != nil {
		t.Fatal(err)
	}

	expected, err := tensor.Conv2D(input, weights, nil, 3, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	x, err := scope.To(input)
	if err != nil {
		t.Fatal(err)
	}

	w, err := scope.To(weights)
	if err != nil {
		t.Fatal(err)
	}

	output, err := tensor.Conv2D(x, w, nil, 3, 1, 1)
	if err != nil {
		t.Fatal(err)
	}

	assertValues(t, output, expected.Data())
}
