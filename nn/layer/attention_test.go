package layer

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
)

func TestMultiHeadAttention_HeadWidthScalesLogitsBeforeSoftmax(t *testing.T) {
	attention, err := NewMultiHeadAttention(2, 1, false, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	for _, projection := range []*Dense{attention.query, attention.key, attention.value, attention.output} {
		setParameter(t, projection.weight, []float32{1, 0, 0, 1})
		setParameter(t, projection.bias, []float32{0, 0})
	}

	output, err := attention.Forward(layerValue(t, []float32{1, 0, 0, 1}, []int{1, 2, 2}, false))
	if err != nil {
		t.Fatal(err)
	}

	probability := math.Exp(1/math.Sqrt(2)) / (1 + math.Exp(1/math.Sqrt(2)))
	for i, want := range []float64{probability, 1 - probability, 1 - probability, probability} {
		if math.Abs(float64(output.Data().Data()[i])-want) > 1e-6 {
			t.Fatalf("element %d: got %v, want %g", i, output.Data(), want)
		}
	}
}

func TestMultiHeadAttention_IdentityProjectionsMatchIndependentHeads(t *testing.T) {
	attention, err := NewMultiHeadAttention(2, 2, false, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	for _, projection := range []*Dense{attention.query, attention.key, attention.value, attention.output} {
		setParameter(t, projection.weight, []float32{1, 0, 0, 1})
		setParameter(t, projection.bias, []float32{0, 0})
	}

	values := []float32{0.1, 0.4, 0.3, -0.2, 0.2, -0.1, -0.4, 0.3}

	output, err := attention.Forward(layerValue(t, values, []int{2, 2, 2}, false))
	if err != nil {
		t.Fatal(err)
	}

	for batch := range 2 {
		for position := range 2 {
			for feature := range 2 {
				query := float64(values[(batch*2+position)*2+feature])
				a := float64(values[batch*4+feature])
				b := float64(values[batch*4+2+feature])
				weightA, weightB := math.Exp(query*a), math.Exp(query*b)
				want := (weightA*a + weightB*b) / (weightA + weightB)

				got := output.Data().Data()[(batch*2+position)*2+feature]
				if math.Abs(float64(got)-want) > 1e-6 {
					t.Fatalf("batch %d position %d head %d: got %g, want %g", batch, position, feature, got, want)
				}
			}
		}
	}
}

func TestMultiHeadAttention_InputAndAllProjectionsMatchFiniteDifferences(t *testing.T) {
	attention, err := NewMultiHeadAttention(4, 2, false, rand.New(rand.NewPCG(3, 4)))
	if err != nil {
		t.Fatal(err)
	}

	checkLayerGradients(t, attention, []float32{0.2, 0.1, -0.4, 0.6, 0.3, -0.2, 0.7, -0.1}, []int{1, 2, 4})
}

func TestMultiHeadAttention_CausalMaskBlocksFutureValuesAndGradients(t *testing.T) {
	attention, err := NewMultiHeadAttention(2, 1, true, rand.New(rand.NewPCG(5, 6)))
	if err != nil {
		t.Fatal(err)
	}

	input := layerValue(t, []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int{1, 3, 2}, true)

	output, err := attention.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	changed, err := attention.Forward(layerValue(t, []float32{0.1, 0.2, 30, 40, 50, 60}, []int{1, 3, 2}, false))
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(output.Data().Data()[:2], changed.Data().Data()[:2]) {
		t.Fatal("future values changed the first position")
	}

	first, err := nn.Slice(output, 1, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	objective, err := nn.Sum(first)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	for _, gradient := range input.Grad().Data()[2:] {
		if gradient != 0 {
			t.Fatal("gradient leaked into a future position")
		}
	}
}

func TestNewMultiHeadAttention_RejectsInvalidHeadsAndInput(t *testing.T) {
	for _, dimensions := range [][2]int{{0, 1}, {2, 0}, {3, 2}} {
		if _, err := NewMultiHeadAttention(dimensions[0], dimensions[1], false, rand.New(rand.NewPCG(1, 2))); err == nil {
			t.Fatalf("accepted dimensions %v", dimensions)
		}
	}

	attention, err := NewMultiHeadAttention(2, 1, false, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	for _, input := range []*nn.Value{nil, layerValue(t, nil, []int{0, 2, 2}, false), layerValue(t, nil, []int{1, 0, 2}, false)} {
		if _, err := attention.Forward(input); err == nil {
			t.Fatal("accepted invalid sequence")
		}
	}
}
