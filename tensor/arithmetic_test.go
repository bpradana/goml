package tensor

import (
	"math"
	"testing"
)

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
	if err != nil || !math.IsInf(float64(log.Data()[0]), -1) {
		t.Fatal("log zero must be -Inf")
	}

	exp, err := Exp(Scalar(0))
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, exp, nil, []float32{1})
}
