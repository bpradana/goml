package tensor

import "testing"

func TestTransfer_CPUCopiesAndLikeConstructorsPreserveOwnership(t *testing.T) {
	source := fromTest(t, []float32{1, 2, 3, 4}, 2, 2)

	view, err := source.Transpose()
	if err != nil {
		t.Fatal(err)
	}

	copy, err := view.To(nil)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, copy, []int{2, 2}, []float32{1, 3, 2, 4})

	if !copy.IsCPU() || copy.Device() != nil || copy.IsClosed() || !copy.IsContiguous() {
		t.Fatal("unexpected CPU transfer metadata")
	}

	data, err := view.HostData()
	if err != nil {
		t.Fatal(err)
	}

	data[0] = 90

	assertData(t, source, []int{2, 2}, []float32{1, 2, 3, 4})

	full, err := FullLike(source, 7, 2)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, full, []int{2}, []float32{7, 7})

	if _, err := FromLike(source, []float32{1}, 2); err == nil {
		t.Fatal("invalid constant shape accepted")
	}

	if err := source.Close(); err != nil || source.IsClosed() {
		t.Fatal("CPU Close changed tensor validity", err)
	}
}
