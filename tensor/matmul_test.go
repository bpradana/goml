package tensor

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestMatMul_StridedAndEmptyInputsMatchReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))

	for _, dims := range [][3]int{{2, 3, 4}, {7, 5, 3}, {1, 8, 2}, {0, 3, 2}, {2, 0, 4}} {
		m, k, n := dims[0], dims[1], dims[2]

		a, err := Rand(rng, k, m)
		if err != nil {
			t.Fatal(err)
		}

		a, err = a.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		b, err := Rand(rng, n, k)
		if err != nil {
			t.Fatal(err)
		}

		b, err = b.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		got, err := MatMul(a, b)
		if err != nil {
			t.Fatal(err)
		}

		if !slices.Equal(got.Shape(), []int{m, n}) {
			t.Fatalf("got shape %v, want [%d %d]", got.Shape(), m, n)
		}

		for i := 0; i < m; i++ {
			for j := 0; j < n; j++ {
				var want float64

				for p := 0; p < k; p++ {
					x, err := a.At(i, p)
					if err != nil {
						t.Fatal(err)
					}

					y, err := b.At(p, j)
					if err != nil {
						t.Fatal(err)
					}

					want += float64(x) * float64(y)
				}

				v, err := got.At(i, j)
				if err != nil || math.Abs(float64(v)-want) > 1e-5 {
					t.Fatalf("dims %v element [%d,%d]: got %v want %v: %v", dims, i, j, v, want, err)
				}
			}
		}
	}

	if _, err := MatMul(Scalar(1), Scalar(2)); err == nil {
		t.Fatal("expected rank error")
	}
}
