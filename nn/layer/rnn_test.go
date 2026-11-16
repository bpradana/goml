package layer

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn"
)

func TestRNN_ReferenceRecurrenceAndInitialStateGradient(t *testing.T) {
	recurrent, err := NewRNN(1, 1, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	setParameter(t, recurrent.Parameters()[0], []float32{0.5})
	setParameter(t, recurrent.Parameters()[1], []float32{0.1})
	setParameter(t, recurrent.Parameters()[2], []float32{0.3})
	input := layerValue(t, []float32{0.2, -0.4}, []int{1, 2, 1}, true)
	initial := layerValue(t, []float32{0.7}, []int{1, 1}, true)

	output, last, err := recurrent.ForwardState(input, initial)
	if err != nil {
		t.Fatal(err)
	}

	first := math.Tanh(0.2*0.5 + 0.7*0.3 + 0.1)

	second := math.Tanh(-0.4*0.5 + first*0.3 + 0.1)
	for i, want := range []float64{first, second} {
		if math.Abs(float64(backendtest.HostData(t, output.Data())[i])-want) > 1e-6 {
			t.Fatalf("step %d: %v, want %g", i, output.Data(), want)
		}
	}

	objective, err := nn.Sum(last)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	want := (1 - second*second) * 0.3 * (1 - first*first) * 0.3
	assertDerivative(t, "initial hidden", 0, backendtest.HostData(t, initial.Grad())[0], want)
}

func TestRNN_BatchedSequenceAndAllParametersMatchFiniteDifferences(t *testing.T) {
	recurrent, err := NewRNN(2, 2, rand.New(rand.NewPCG(5, 6)))
	if err != nil {
		t.Fatal(err)
	}

	checkLayerGradients(t, recurrent, []float32{0.1, 0.3, -0.2, 0.4, 0.2, -0.1, -0.3, 0.2}, []int{2, 2, 2})
}

func TestRNN_ExplicitStateMatchesUnbrokenSequenceAndForwardResetsState(t *testing.T) {
	recurrent, err := NewRNN(1, 2, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	input, err := nn.Inference(layerValue(t, []float32{0.1, 0.2, 0.3}, []int{1, 3, 1}, false).Data())
	if err != nil {
		t.Fatal(err)
	}

	full, err := recurrent.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	first, err := nn.Slice(input, 1, 0, 1)
	if err != nil {
		t.Fatal(err)
	}

	_, state, err := recurrent.ForwardState(first, nil)
	if err != nil {
		t.Fatal(err)
	}

	rest, err := nn.Slice(input, 1, 1, 3)
	if err != nil {
		t.Fatal(err)
	}

	chunk, _, err := recurrent.ForwardState(rest, state)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, chunk.Data()), backendtest.HostData(t, full.Data())[2:]) {
		t.Fatal("chunked state differs from uninterrupted sequence")
	}

	again, err := recurrent.Forward(input)
	if err != nil || !slices.Equal(backendtest.HostData(t, again.Data()), backendtest.HostData(t, full.Data())) {
		t.Fatalf("forward retained state: %v", err)
	}
}

func TestRNN_RejectsInvalidConfigurationAndInput(t *testing.T) {
	if _, err := NewRNN(0, 2, rand.New(rand.NewPCG(1, 2))); err == nil {
		t.Fatal("accepted zero input size")
	}

	if _, err := NewRNN(1, 2, nil); err == nil {
		t.Fatal("accepted nil generator")
	}

	if _, err := (&RNN{}).Forward(nil); err == nil {
		t.Fatal("accepted uninitialized layer")
	}
}
