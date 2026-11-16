//go:build darwin && cgo

package metal_test

import (
	"errors"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

func TestCompute_InvalidBackendViewsReturnErrorsBeforeEncoding(t *testing.T) {
	device := openDevice(t)

	storage, err := device.Allocate(4)
	if err != nil {
		t.Fatal(err)
	}

	valid := backend.View{Buffer: storage, Shape: []int{2, 2}, Strides: []int{2, 1}}
	badBounds := backend.View{Buffer: storage, Shape: []int{2, 2}, Strides: []int{4, 1}}
	badRank := backend.View{Buffer: storage, Shape: []int{4}, Strides: []int{1}}

	checks := []struct {
		name string
		run  func() error
	}{
		{"bounds", func() error { return device.Unary(backend.Exp, valid, badBounds) }},
		{"matmul_rank", func() error { return device.MatMul(valid, badRank, valid) }},
		{"broadcast", func() error { return device.Binary(backend.Add, badRank, valid, valid) }},
		{"copy_shape", func() error { return device.Copy(valid, badRank) }},
		{"axis", func() error { return device.Reduce(backend.Sum, valid, valid, []int{2}) }},
		{"pool_geometry", func() error { return device.Pool(valid, valid, backend.Pool2DOptions{Kernel: 2}) }},
		{"resize_scale", func() error { return device.Resize(valid, valid, backend.ResizeOptions{Mode: backend.Nearest}) }},
		{"convolution_rank", func() error {
			return device.Conv2D(valid, valid, valid, backend.View{}, backend.Conv2DOptions{Kernel: 1, Stride: 1})
		}},
	}

	for _, check := range checks {
		t.Run(check.name, func(t *testing.T) {
			if err := check.run(); err == nil {
				t.Fatal("invalid backend input accepted")
			}
		})
	}
}

func TestCompute_UnsupportedRankHasStandardErrorCategory(t *testing.T) {
	device := openDevice(t)

	buffer, err := device.Allocate(1)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = buffer.Close() }()

	input := backend.View{Buffer: buffer, Shape: []int{1, 1, 1, 1, 1, 1, 1, 1, 1}, Strides: []int{1, 1, 1, 1, 1, 1, 1, 1, 1}}
	if err := device.Unary(backend.Exp, input, input); !errors.Is(err, backend.ErrUnsupported) {
		t.Fatalf("rank limit: %v", err)
	}
}
