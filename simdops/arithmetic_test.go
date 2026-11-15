package simdops

import (
	"math/rand/v2"
	"slices"
	"testing"
)

func TestArithmetic_ScalarParityAndAliasing(t *testing.T) {
	tests := []struct {
		name string
		fn   func([]float32, []float32, []float32) error
		ref  func([]float32, []float32, []float32)
	}{
		{"add", Add, func(dst, a, b []float32) {
			for i := range dst {
				dst[i] = a[i] + b[i]
			}
		}},
		{"sub", Sub, func(dst, a, b []float32) {
			for i := range dst {
				dst[i] = a[i] - b[i]
			}
		}},
		{"mul", Mul, func(dst, a, b []float32) {
			for i := range dst {
				dst[i] = a[i] * b[i]
			}
		}},
		{"div", Div, func(dst, a, b []float32) {
			for i := range dst {
				dst[i] = a[i] / b[i]
			}
		}},
		{"axpy", func(dst, a, b []float32) error {
			return Axpy(dst, a, b, 0.25)
		}, func(dst, a, b []float32) {
			for i := range dst {
				dst[i] = 0.25*a[i] + b[i]
			}
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for _, n := range []int{0, 1, 3, 8, 17, 64, 1025, 65537} {
				rng := rand.New(rand.NewPCG(1, 2))
				a, b := make([]float32, n), make([]float32, n)

				for i := range a {
					a[i] = rng.Float32()*4 - 2
					b[i] = rng.Float32() + 0.1
				}

				want, got := make([]float32, n), make([]float32, n)
				tt.ref(want, a, b)

				if err := tt.fn(got, a, b); err != nil {
					t.Fatal(err)
				}

				if !slices.Equal(got, want) {
					t.Fatalf("size %d: scalar parity failed", n)
				}

				inPlace := slices.Clone(a)
				if err := tt.fn(inPlace, inPlace, b); err != nil {
					t.Fatal(err)
				}

				if !slices.Equal(inPlace, want) {
					t.Fatalf("size %d: aliasing a failed", n)
				}

				inPlace = slices.Clone(b)
				if err := tt.fn(inPlace, a, inPlace); err != nil {
					t.Fatal(err)
				}

				if !slices.Equal(inPlace, want) {
					t.Fatalf("size %d: aliasing b failed", n)
				}
			}

			dst := []float32{9, 9}
			if err := tt.fn(dst, []float32{1}, []float32{2}); err == nil {
				t.Fatal("expected length error")
			}

			if !slices.Equal(dst, []float32{9, 9}) {
				t.Fatal("validation wrote to destination")
			}
		})
	}
}

func TestScale_InPlaceAndLengthValidation(t *testing.T) {
	a := []float32{1, -2, 3, 4, 5}
	if err := Scale(a, a, 2); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(a, []float32{2, -4, 6, 8, 10}) {
		t.Fatal(a)
	}

	if err := Scale(a[:1], a, 2); err == nil {
		t.Fatal("expected length error")
	}
}

func TestAdd_NoAllocations(t *testing.T) {
	a, b, dst := make([]float32, 1024), make([]float32, 1024), make([]float32, 1024)
	allocs := testing.AllocsPerRun(100, func() {
		if err := Add(dst, a, b); err != nil {
			panic(err)
		}
	})

	if allocs != 0 {
		t.Fatalf("Add allocated %g times", allocs)
	}
}
