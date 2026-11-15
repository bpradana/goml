package layer

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
)

func TestFlatten_NoncontiguousInputPreservesBatchAndGradient(t *testing.T) {
	input := layerValue(t, []float32{1, 2, 3, 4, 5, 6}, []int{1, 2, 3}, true)

	transposed, err := nn.Transpose(input, 0, 2, 1)
	if err != nil {
		t.Fatal(err)
	}

	output, err := NewFlatten().Forward(transposed)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(output.Data().Shape(), []int{1, 6}) || !slices.Equal(output.Data().Data(), []float32{1, 4, 2, 5, 3, 6}) {
		t.Fatal(output.Data())
	}

	objective, err := nn.Sum(output)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(input.Grad().Data(), []float32{1, 1, 1, 1, 1, 1}) || len(NewFlatten().Parameters()) != 0 {
		t.Fatal("flatten changed gradients or exposed parameters")
	}
}

func TestFlatten_EmptyBatchAndInvalidRank(t *testing.T) {
	output, err := NewFlatten().Forward(layerValue(t, nil, []int{0, 2, 3}, false))
	if err != nil || !slices.Equal(output.Data().Shape(), []int{0, 6}) {
		t.Fatalf("empty batch: %v %v", output, err)
	}

	for _, input := range []*nn.Value{nil, layerValue(t, []float32{1}, nil, false)} {
		if _, err := NewFlatten().Forward(input); err == nil {
			t.Fatal("accepted invalid input")
		}
	}
}
