package tensor

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

func TestMatMul_StridedAndEmptyInputsMatchReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))

	for _, dims := range [][3]int{{2, 3, 4}, {7, 5, 3}, {1, 8, 2}, {0, 3, 2}, {2, 0, 4}, {2, 3, 0}} {
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

// referenceMatMul retains the original Float32 accumulation, independently of
// kernel dispatch. The Float64 reference below also checks mathematical error.
func referenceMatMul(a, b *Tensor) []float32 {
	m, k, n := a.shape[0], a.shape[1], b.shape[1]
	out := make([]float32, m*n)

	for i := range m {
		for p := range k {
			x, _ := a.At(i, p)
			for j := range n {
				y, _ := b.At(p, j)
				out[i*n+j] += x * y
			}
		}
	}

	return out
}

func TestMatMul_TileBoundariesAndLayouts(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))

	for _, dims := range [][3]int{{3, 63, 65}, {4, 64, 64}, {5, 65, 63}, {33, 129, 127}, {65, 31, 129}, {129, 67, 131}} {
		for _, transpose := range [][2]bool{{false, false}, {true, false}, {false, true}, {true, true}} {
			m, k, n := dims[0], dims[1], dims[2]

			a, _ := Randn(rng, m+1, k+1)
			if transpose[0] {
				a, _ = Randn(rng, k+1, m+1)
				a, _ = a.Transpose()
			}

			a, _ = a.Slice(0, 1, m+1)
			a, _ = a.Slice(1, 1, k+1)

			b, _ := Randn(rng, k+1, n+1)
			if transpose[1] {
				b, _ = Randn(rng, n+1, k+1)
				b, _ = b.Transpose()
			}

			b, _ = b.Slice(0, 1, k+1)
			b, _ = b.Slice(1, 1, n+1)

			got, err := MatMul(a, b)
			if err != nil {
				t.Fatal(err)
			}

			original := referenceMatMul(a, b)

			for i, value := range hostData(t, got) {
				var want, magnitude float64

				for p := range k {
					x, _ := a.At(i/n, p)
					y, _ := b.At(p, i%n)
					product := float64(x) * float64(y)
					want += product
					magnitude += math.Abs(product)
				}

				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value)-want) > 1e-6*magnitude || math.Float32bits(value) != math.Float32bits(original[i]) {
					t.Fatalf("dims=%v transpose=%v element=%d: got %g reference32 %g reference64 %g", dims, transpose, i, value, original[i], want)
				}
			}
		}
	}
}

func TestMatMul_NonfiniteProductsAreNotSkipped(t *testing.T) {
	a, _ := From([]float32{0, 1, 0, 1}, 2, 2)
	b, _ := From([]float32{float32(math.Inf(1)), 1, 1, float32(math.NaN())}, 2, 2)

	for _, transpose := range []bool{false, true} {
		if transpose {
			b, _ = b.Transpose()
		}

		got, err := MatMul(a, b)
		if err != nil {
			t.Fatal(err)
		}

		want := referenceMatMul(a, b)
		for i, v := range hostData(t, got) {
			if v != want[i] && !(math.IsNaN(float64(v)) && math.IsNaN(float64(want[i]))) {
				t.Fatal(i, v, want[i])
			}
		}
	}
}
