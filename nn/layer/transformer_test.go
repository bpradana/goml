package layer

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
)

func TestTransformerEncoder_ResidualPathsAndAllParametersMatchFiniteDifferences(t *testing.T) {
	encoder, err := NewTransformerEncoder(4, 2, 5, rand.New(rand.NewPCG(11, 12)))
	if err != nil {
		t.Fatal(err)
	}

	checkLayerGradients(t, encoder, []float32{0.1, -0.4, 0.7, 0.3, 0.8, 0.2, -0.5, 0.4}, []int{1, 2, 4})
}

func TestTransformerEncoder_ZeroProjectionsLeaveResidualInputAndCompose(t *testing.T) {
	encoder, err := NewTransformerEncoder(4, 2, 6, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	for _, parameter := range encoder.attention.output.Parameters() {
		setParameter(t, parameter, make([]float32, parameter.Value().Data().Numel()))
	}

	for _, parameter := range encoder.feedForwardOut.Parameters() {
		setParameter(t, parameter, make([]float32, parameter.Value().Data().Numel()))
	}

	model, err := NewSequential(NewPositionalEncoding(), encoder)
	if err != nil {
		t.Fatal(err)
	}

	input := layerValue(t, []float32{0.1, 0.2, 0.3, 0.4}, []int{1, 1, 4}, false)

	output, err := model.Predict(input.Data())
	if err != nil {
		t.Fatal(err)
	}

	positioned, err := NewPositionalEncoding().Forward(input)
	if err != nil || !slices.Equal(output.Data(), positioned.Data().Data()) {
		t.Fatalf("residual changed input: %v %v", output, err)
	}

	if len(model.Parameters()) != 16 {
		t.Fatalf("got %d parameters, want 16 tensors", len(model.Parameters()))
	}

	for _, parameter := range model.Parameters() {
		if parameter.Value().Grad() != nil {
			t.Fatal("prediction retained a parameter gradient")
		}
	}
}

func TestTransformerEncoder_RejectsInvalidDimensionsAndUninitializedLayer(t *testing.T) {
	if _, err := NewTransformerEncoder(4, 2, 0, rand.New(rand.NewPCG(1, 2))); err == nil {
		t.Fatal("accepted zero feed-forward width")
	}

	if _, err := (&TransformerEncoder{}).Forward(&nn.Value{}); err == nil {
		t.Fatal("accepted uninitialized encoder")
	}
}
