package tensor

import "testing"

func TestBroadcastIndex_StridedOperandPreservesLogicalValues(t *testing.T) {
	a := fromTest(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)

	view, err := a.Transpose()
	if err != nil {
		t.Fatal(err)
	}

	row := fromTest(t, []float32{10, 20}, 2)

	result, err := Add(view, row)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, result, []int{3, 2}, []float32{11, 24, 12, 25, 13, 26})
}

func TestBroadcastIndex_EmptyDimensionProducesNoElements(t *testing.T) {
	empty := fromTest(t, nil, 0, 3)
	row := fromTest(t, []float32{1, 2, 3}, 3)

	result, err := Add(empty, row)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, result, []int{0, 3}, nil)
}
