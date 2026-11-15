package layer

import (
	"testing"

	"github.com/bpradana/goml/nn"
)

func TestSequenceShape_RejectsInvalidSequenceDimensions(t *testing.T) {
	for _, input := range []*nn.Value{nil, {}, layerValue(t, nil, []int{1, 0, 2}, false), layerValue(t, []float32{1}, []int{1, 1, 1}, false)} {
		if _, err := sequenceShape("test", input, 2); err == nil {
			t.Fatal("accepted invalid sequence")
		}
	}
}

func TestRecurrentState_InferenceDropsSuppliedStateGraph(t *testing.T) {
	input, err := nn.Inference(layerValue(t, []float32{1}, []int{1, 1, 1}, false).Data())
	if err != nil {
		t.Fatal(err)
	}

	for _, initial := range []*nn.Value{nil, layerValue(t, []float32{1}, []int{1, 1}, true)} {
		state, err := recurrentState(input, initial, 1, 1)
		if err != nil {
			t.Fatal(err)
		}

		if state.RequiresGrad() || !state.IsInference() {
			t.Fatal("inference retained initial state tracking")
		}
	}

	if _, err := recurrentState(input, &nn.Value{}, 1, 1); err == nil {
		t.Fatal("accepted uninitialized state")
	}
}
