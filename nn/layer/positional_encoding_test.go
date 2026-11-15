package layer

import (
	"math"
	"testing"

	"github.com/bpradana/goml/nn"
)

func TestPositionalEncoding_ReferenceSineCosineIncludingOddWidth(t *testing.T) {
	input := layerValue(t, make([]float32, 6), []int{1, 2, 3}, true)
	encoding := NewPositionalEncoding()

	output, err := encoding.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	want := []float64{0, 1, 0, math.Sin(1), math.Cos(1), math.Sin(1 / math.Pow(10000, 2.0/3))}
	for i, got := range output.Data().Data() {
		if math.Abs(float64(got)-want[i]) > 1e-6 {
			t.Fatalf("element %d: got %g, want %g", i, got, want[i])
		}
	}

	objective, err := nn.Sum(output)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	for _, gradient := range input.Grad().Data() {
		if gradient != 1 {
			t.Fatal("encoding changed input derivative")
		}
	}

	if len(encoding.Parameters()) != 0 {
		t.Fatal("encoding exposed parameters")
	}
}

func TestPositionalEncoding_RejectsInvalidInputs(t *testing.T) {
	for _, input := range []*nn.Value{nil, layerValue(t, nil, []int{1, 0, 3}, false), layerValue(t, []float32{1}, []int{1}, false)} {
		if _, err := NewPositionalEncoding().Forward(input); err == nil {
			t.Fatal("accepted invalid input")
		}
	}
}
