package layer

import (
	"math"
	"testing"

	"github.com/bpradana/goml/nn"
)

func TestLayerNorm_ReferencePopulationVarianceAndAffineTransform(t *testing.T) {
	normalization, err := NewLayerNorm(3, 1e-5)
	if err != nil {
		t.Fatal(err)
	}

	setParameter(t, normalization.Parameters()[0], []float32{2, 3, 4})
	setParameter(t, normalization.Parameters()[1], []float32{0.1, 0.2, 0.3})

	output, err := normalization.Forward(layerValue(t, []float32{1, 2, 3, 5, 5, 5}, []int{2, 3}, false))
	if err != nil {
		t.Fatal(err)
	}

	inverse := 1 / math.Sqrt(2.0/3+1e-5)
	for i, want := range []float64{-2*inverse + 0.1, 0.2, 4*inverse + 0.3, 0.1, 0.2, 0.3} {
		if math.Abs(float64(output.Data().Data()[i])-want) > 1e-6 {
			t.Fatalf("element %d: %v, want %g", i, output.Data(), want)
		}
	}
}

func TestLayerNorm_InputScaleAndBiasMatchFiniteDifferences(t *testing.T) {
	normalization, err := NewLayerNorm(3, 1e-4)
	if err != nil {
		t.Fatal(err)
	}

	setParameter(t, normalization.Parameters()[0], []float32{0.7, 1.1, -0.4})
	checkLayerGradients(t, normalization, []float32{0.2, -0.3, 0.8, 1.2, 0.4, -0.1}, []int{1, 2, 3})
}

func TestLayerNorm_RejectsInvalidConfigurationAndShape(t *testing.T) {
	for _, epsilon := range []float32{0, -1, float32(math.NaN()), float32(math.Inf(1))} {
		if _, err := NewLayerNorm(2, epsilon); err == nil {
			t.Fatalf("accepted epsilon %g", epsilon)
		}
	}

	if _, err := NewLayerNorm(0, 1e-5); err == nil {
		t.Fatal("accepted zero features")
	}

	normalization, err := NewLayerNorm(2, 1e-5)
	if err != nil {
		t.Fatal(err)
	}

	for _, input := range []*nn.Value{nil, {}, layerValue(t, []float32{1}, nil, false)} {
		if _, err := normalization.Forward(input); err == nil {
			t.Fatal("accepted invalid input")
		}
	}
}
