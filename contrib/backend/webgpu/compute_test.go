package webgpu

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

func TestCompute_ParameterEncodingMatchesWGSL(t *testing.T) {
	p := kernelParams{n: 17, rank: 2, ao: 3, offset: 7}
	p.shape[0], p.as[0], p.ds[0] = 4, 6, 8
	p.ints[0], p.scalars[0] = 11, 0.5

	data, err := p.bytes()
	if err != nil {
		t.Fatal(err)
	}

	if len(data) != 304 || binary.Size(p) != 304 {
		t.Fatalf("parameter size %d; expected WGSL size 304", len(data))
	}

	for _, test := range []struct {
		word int
		want uint32
	}{
		{0, 17}, {1, 2}, {4, 4}, {12, 6}, {36, 8}, {44, 3}, {47, 7}, {48, 11}, {72, math.Float32bits(0.5)},
	} {
		if got := binary.LittleEndian.Uint32(data[test.word*4:]); got != test.want {
			t.Errorf("word %d: got %d, want %d", test.word, got, test.want)
		}
	}
}

func TestCompute_ElementCountsRejectOverflowAndHardwareLimits(t *testing.T) {
	for _, shape := range [][]int{{-1}, {int(^uint(0) >> 1), 2}, {1, 1, 1, 1, 1, 1, 1, 1, 1}} {
		if _, err := elements(shape); err == nil {
			t.Fatalf("invalid shader shape accepted: %v", shape)
		}
	}

	if n, err := elements([]int{0, 1 << 20}); err != nil || n != 0 {
		t.Fatalf("empty shape: %d, %v", n, err)
	}
}

func TestCompute_InvalidViewsFailBeforeEncoding(t *testing.T) {
	device := openDevice(t)

	a, err := device.Allocate(4)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = a.Close() }()

	b, err := device.Allocate(4)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = b.Close() }()

	input := backend.View{Buffer: a, Shape: []int{2, 2}, Strides: []int{2, 1}}
	output := backend.View{Buffer: b, Shape: []int{2, 2}, Strides: []int{2, 1}}
	badBounds, overlap := output, output
	badBounds.Strides, overlap.Strides = []int{8, 1}, []int{0, 1}
	missing := input
	missing.Buffer = nil

	for _, test := range []struct {
		name string
		run  func() error
	}{
		{"bounds", func() error { return device.Unary(backend.Exp, badBounds, input) }},
		{"overlap", func() error { return device.Fill(overlap, 1) }},
		{"alias", func() error { return device.Copy(input, input) }},
		{"axis", func() error { return device.Reduce(backend.Sum, output, input, []int{2}) }},
		{"matrix_shape", func() error { return device.MatMul(backend.View{Buffer: b}, input, input) }},
		{"missing_unary_input", func() error { return device.Unary(backend.Exp, output, missing) }},
		{"missing_binary_input", func() error { return device.Binary(backend.Add, output, input, missing) }},
		{"missing_scalar_input", func() error { return device.Scalar(backend.Scale, output, missing, 2) }},
		{"missing_copy_input", func() error { return device.Copy(output, missing) }},
		{"missing_reduction_input", func() error { return device.Reduce(backend.Sum, output, missing, nil) }},
		{"missing_finite_input", func() error { return device.CheckFinite(missing) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			if err := test.run(); !errors.Is(err, backend.ErrInvalidView) {
				t.Fatalf("got %v, want invalid view", err)
			}
		})
	}
}
