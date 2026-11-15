package tensor

import (
	"testing"
)

func TestViews_PreserveValuesAndStorageOwnership(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6}
	a := fromTest(t, input, 2, 3)
	input[0] = 99
	a.Shape()[0] = 99
	a.Strides()[0] = 99
	a.Data()[0] = 99
	assertData(t, a, []int{2, 3}, []float32{1, 2, 3, 4, 5, 6})

	transposed, err := a.Transpose()
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, transposed, []int{3, 2}, []float32{1, 4, 2, 5, 3, 6})

	if transposed.IsContiguous() {
		t.Fatal("transpose unexpectedly contiguous")
	}

	if _, err := transposed.Reshape(6); err == nil {
		t.Fatal("noncontiguous reshape must fail")
	}

	contiguous := transposed.Contiguous()
	assertData(t, contiguous, []int{3, 2}, transposed.Data())

	if &contiguous.data[0] == &a.data[0] {
		t.Fatal("contiguous did not copy")
	}

	sliced, err := transposed.Slice(0, 1, 3)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, sliced, []int{2, 2}, []float32{2, 5, 3, 6})

	if &sliced.data[0] != &a.data[0] {
		t.Fatal("slice must share storage")
	}

	reshaped, err := a.Reshape(3, 2)
	if err != nil {
		t.Fatal(err)
	}

	if &reshaped.data[0] != &a.data[0] {
		t.Fatal("reshape must share storage")
	}

	v, err := sliced.At(1, 1)
	if err != nil || v != 6 {
		t.Fatalf("index: %v %v", v, err)
	}
}

func FuzzViews(f *testing.F) {
	f.Add(uint8(2), uint8(3), uint8(1), uint8(2))

	f.Fuzz(func(t *testing.T, m8, n8, start8, end8 uint8) {
		m, n := int(m8%12), int(n8%12)
		data := make([]float32, m*n)

		for i := range data {
			data[i] = float32(i)
		}

		a := fromTest(t, data, m, n)

		flat, err := a.Reshape(m * n)
		if err != nil {
			t.Fatal(err)
		}

		assertData(t, flat, []int{m * n}, data)

		i, j := int(start8)-1, int(end8)-1
		element, err := a.At(i, j)

		if i >= 0 && i < m && j >= 0 && j < n {
			if err != nil || element != data[i*n+j] {
				t.Fatalf("valid index [%d,%d] failed: %v", i, j, err)
			}
		} else if err == nil {
			t.Fatalf("invalid index [%d,%d] succeeded for [%d,%d]", i, j, m, n)
		}

		transposed, err := a.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		roundTrip, err := transposed.Transpose()
		if err != nil {
			t.Fatal(err)
		}

		assertData(t, roundTrip, []int{m, n}, data)
		assertData(t, transposed.Contiguous(), []int{n, m}, transposed.Data())

		start, end := int(start8)-1, int(end8)-1
		slice, err := transposed.Slice(0, start, end)

		if start < 0 || end < start || end > n {
			if err == nil {
				t.Fatal("invalid slice succeeded")
			}

			return
		}

		if err != nil || len(slice.Data()) != (end-start)*m {
			t.Fatalf("valid slice failed: %v", err)
		}
	})
}
