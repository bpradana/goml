package shape

import (
	"math/big"
	"slices"
	"testing"
)

func TestLayout_RowMajorStrides(t *testing.T) {
	count, strides, err := Layout([]int{2, 3, 4})
	if err != nil || count != 24 || !slices.Equal(strides, []int{12, 4, 1}) {
		t.Fatalf("got count=%d strides=%v, want 24 and [12 4 1]: %v", count, strides, err)
	}
}

func TestLayout_ScalarHasOneElement(t *testing.T) {
	count, strides, err := Layout(nil)
	if err != nil || count != 1 || len(strides) != 0 {
		t.Fatalf("got count=%d strides=%v, want scalar: %v", count, strides, err)
	}
}

func TestLayout_ZeroDimensionProducesEmptyArray(t *testing.T) {
	count, strides, err := Layout([]int{2, 0, 4})
	if err != nil || count != 0 || !slices.Equal(strides, []int{0, 4, 1}) {
		t.Fatalf("got count=%d strides=%v, want empty layout: %v", count, strides, err)
	}
}

func TestLayout_RejectsNegativeDimensionsAndOverflow(t *testing.T) {
	for _, tt := range []struct {
		name  string
		shape []int
	}{
		{"NegativeDimension", []int{2, -1}},
		{"NegativeAfterZero", []int{0, -1}},
		{"ElementCountOverflow", []int{int(^uint(0) >> 1), 2}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, _, err := Layout(tt.shape); err == nil {
				t.Fatal("expected layout validation error")
			}
		})
	}
}

func FuzzShapes(f *testing.F) {
	f.Add(2, 3)
	f.Add(-1, 0)
	f.Add(int(^uint(0)>>1), 2)

	f.Fuzz(func(t *testing.T, a, b int) {
		n, _, err := Layout([]int{a, b})

		if a < 0 || b < 0 {
			if err == nil {
				t.Fatal("negative dimension accepted")
			}

			return
		}

		want := new(big.Int).Mul(big.NewInt(int64(a)), big.NewInt(int64(b)))
		maximum := big.NewInt(int64(int(^uint(0) >> 1)))

		if want.Cmp(maximum) > 0 {
			if err == nil {
				t.Fatal("overflowing element count accepted")
			}

			return
		}

		if err != nil || int64(n) != want.Int64() {
			t.Fatalf("shape [%d %d]: got %d, want %v: %v", a, b, n, want, err)
		}
	})
}
