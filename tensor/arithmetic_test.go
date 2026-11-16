package tensor

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestArithmetic_BroadcastLayoutsMatchReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	a, _ := Randn(rng, 2, 3, 4)
	a, _ = a.Transpose(2, 0, 1)
	a, _ = a.Slice(0, 1, 4)
	b, _ := Rand(rng, 2, 1)

	b, _ = Add(b, Scalar(1))
	for _, pair := range [][2]*Tensor{{a, b}, {b, a}, {a, Scalar(2)}, {Scalar(2), a}} {
		for _, operation := range []struct {
			name string
			fn   func(*Tensor, *Tensor) (*Tensor, error)
		}{{"add", Add}, {"sub", Sub}, {"mul", Mul}, {"div", Div}} {
			got, err := operation.fn(pair[0], pair[1])
			if err != nil {
				t.Fatal(err)
			}

			for i, value := range hostData(t, got) {
				x := broadcastValue(t, pair[0], got.shape, i)
				y := broadcastValue(t, pair[1], got.shape, i)

				var want float32

				switch operation.name {
				case "add":
					want = x + y
				case "sub":
					want = x - y
				case "mul":
					want = x * y
				case "div":
					want = x / y
				}

				if value != want {
					t.Fatal(operation.name, i, value, want)
				}
			}
		}
	}

	wide, _ := Ones(2, 1, 2, 1, 2, 1, 2, 1, 2)
	wide, _ = wide.Transpose()

	got, err := Add(wide, Scalar(3))
	if err != nil {
		t.Fatal(err)
	}

	for _, value := range hostData(t, got) {
		if value != 4 {
			t.Fatal("high-rank broadcast", got)
		}
	}
}

func TestArithmeticAndMath_KnownValues(t *testing.T) {
	a := fromTest(t, []float32{1, 4}, 2)

	tests := []struct {
		name string
		fn   func(*Tensor) (*Tensor, error)
		want []float32
	}{
		{"sub", func(a *Tensor) (*Tensor, error) { return Sub(a, Scalar(1)) }, []float32{0, 3}},
		{"mul", func(a *Tensor) (*Tensor, error) { return Mul(a, Scalar(2)) }, []float32{2, 8}},
		{"div", func(a *Tensor) (*Tensor, error) { return Div(a, Scalar(2)) }, []float32{0.5, 2}},
		{"scale", func(a *Tensor) (*Tensor, error) { return Scale(a, 2) }, []float32{2, 8}},
		{"neg", Neg, []float32{-1, -4}},
		{"sqrt", Sqrt, []float32{1, 2}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.fn(a)
			if err != nil {
				t.Fatal(err)
			}

			assertData(t, got, []int{2}, tt.want)
		})
	}

	log, err := Log(Scalar(0))
	if err != nil || !math.IsInf(float64(hostData(t, log)[0]), -1) {
		t.Fatal("log zero must be -Inf")
	}

	exp, err := Exp(Scalar(0))
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, exp, nil, []float32{1})
}

func broadcastValue(t *testing.T, input *Tensor, shape []int, index int) float32 {
	t.Helper()

	indices := make([]int, input.Rank())

	for axis := len(shape) - 1; axis >= 0; axis-- {
		coordinate := index % shape[axis]
		index /= shape[axis]

		target := axis - len(shape) + input.Rank()
		if target >= 0 && input.shape[target] != 1 {
			indices[target] = coordinate
		}
	}

	value, err := input.At(indices...)
	if err != nil {
		t.Fatal(err)
	}

	return value
}
