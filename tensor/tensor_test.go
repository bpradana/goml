package tensor

import (
	"errors"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

func fromTest(t *testing.T, data []float32, shape ...int) *Tensor {
	t.Helper()

	x, err := From(data, shape...)
	if err != nil {
		t.Fatal(err)
	}

	return x
}

func assertData(t *testing.T, x *Tensor, shape []int, want []float32) {
	t.Helper()

	if !slices.Equal(x.Shape(), shape) || !slices.Equal(hostData(t, x), want) {
		t.Fatalf("got %v, want shape=%v data=%v", x, shape, want)
	}
}

func TestConstructorsAndInvalidInputs(t *testing.T) {
	for _, shape := range [][]int{{-1}, {0, -1}, {int(^uint(0) >> 1), 2}} {
		if _, err := Zeros(shape...); err == nil {
			t.Fatalf("expected shape error for %v", shape)
		}
	}

	if _, err := From([]float32{1}, 2); err == nil {
		t.Fatal("expected count error")
	}

	a := fromTest(t, []float32{1, 2, 3, 4}, 2, 2)

	for _, axes := range [][]int{{0, 0}, {0}, {0, 2}, {-1, 1}} {
		if _, err := a.Transpose(axes...); err == nil {
			t.Fatalf("expected axes error for %v", axes)
		}
	}

	for _, idx := range [][]int{{}, {0}, {-1, 0}, {2, 0}} {
		if _, err := a.At(idx...); err == nil {
			t.Fatalf("expected indexing error for %v", idx)
		}
	}

	if _, err := a.Slice(0, 2, 1); err == nil {
		t.Fatal("expected slice error")
	}

	if _, err := a.Reshape(3); err == nil {
		t.Fatal("expected reshape error")
	}

	if _, err := Add(nil, a); err == nil {
		t.Fatal("expected nil error")
	}

	if _, err := Add(&Tensor{}, a); err == nil {
		t.Fatal("expected uninitialized error")
	}

	if _, err := Rand(nil, 1); err == nil {
		t.Fatal("expected random generator error")
	}

	if _, err := Randn(nil, 1); err == nil {
		t.Fatal("expected random generator error")
	}

	for _, fn := range []func(*rand.Rand, ...int) (*Tensor, error){Rand, Randn} {
		x, err := fn(rand.New(rand.NewPCG(1, 2)), 2, 3)
		if err != nil {
			t.Fatal(err)
		}

		y, err := fn(rand.New(rand.NewPCG(1, 2)), 2, 3)
		if err != nil || !slices.Equal(hostData(t, x), hostData(t, y)) {
			t.Fatal("random initialization must be reproducible")
		}
	}

	s := Scalar(3)

	v, err := s.At()
	if err != nil || v != 3 || s.Rank() != 0 || s.Numel() != 1 {
		t.Fatalf("scalar: %v %v", s, err)
	}

	ones, err := Ones(2)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, ones, []int{2}, []float32{1, 1})
}

func TestTensor_StringBoundsHighRankMetadata(t *testing.T) {
	shape := make([]int, 1000)
	for i := range shape {
		shape[i] = 1
	}

	a, err := Zeros(shape...)
	if err != nil {
		t.Fatal(err)
	}

	if text := a.String(); len(text) > 150 || !strings.Contains(text, "...") {
		t.Fatalf("unbounded high-rank preview: %d bytes", len(text))
	}
}

func TestConstructors_RejectFloat32ByteCountOverflow(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	maximum := int(^uint(0) >> 1)

	for _, constructor := range []func() (*Tensor, error){
		func() (*Tensor, error) { return Zeros(maximum) },
		func() (*Tensor, error) { return Rand(rng, maximum) },
		func() (*Tensor, error) { return Randn(rng, maximum) },
	} {
		if _, err := constructor(); !errors.Is(err, backend.ErrAllocation) {
			t.Fatalf("got %v, want allocation failure", err)
		}
	}
}
