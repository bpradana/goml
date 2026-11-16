package layer

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn"
)

func TestLastStep_SelectsEachBatchAndRoutesGradient(t *testing.T) {
	input := layerValue(t, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 2}, true)

	output, err := NewLastStep().Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, output.Data()), []float32{3, 4, 7, 8}) || !slices.Equal(output.Data().Shape(), []int{2, 2}) {
		t.Fatal(output.Data())
	}

	objective, err := nn.Sum(output)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, input.Grad()), []float32{0, 0, 1, 1, 0, 0, 1, 1}) {
		t.Fatal(input.Grad())
	}
}

func TestLastStep_RejectsEmptyTimeAndInvalidRank(t *testing.T) {
	for _, input := range []*nn.Value{nil, layerValue(t, nil, []int{1, 0, 2}, false), layerValue(t, []float32{1}, []int{1}, false)} {
		if _, err := NewLastStep().Forward(input); err == nil {
			t.Fatal("accepted invalid sequence")
		}
	}
}
