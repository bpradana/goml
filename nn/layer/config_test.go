package layer

import (
	"encoding/json"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
)

func TestMarshalConfig_PreservesSharedLayerAndParameterReferences(t *testing.T) {
	dense, err := NewDense(2, 2, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := NewSequential(dense, activation.NewReLU(), dense)
	if err != nil {
		t.Fatal(err)
	}

	data, err := MarshalConfig(model)
	if err != nil {
		t.Fatal(err)
	}

	var config architecture
	if err := json.Unmarshal(data, &config); err != nil {
		t.Fatal(err)
	}

	if config.Root != 0 || len(config.Layers) != 3 || !slices.Equal(config.Layers[0].Children, []int{1, 2, 1}) || !slices.Equal(config.Layers[1].Parameters, []int{0, 1}) {
		t.Fatalf("unexpected references: %s", data)
	}
}

type unsupportedConfigLayer struct {
	values []int
}

func (unsupportedConfigLayer) Forward(input *nn.Value) (*nn.Value, error) {
	return input, nil
}

func (unsupportedConfigLayer) Parameters() []*nn.Parameter {
	return nil
}

func TestMarshalConfig_RejectsUnsupportedNilAndUninitializedLayers(t *testing.T) {
	for _, stage := range []Layer{
		unsupportedConfigLayer{values: []int{1}},
		&Dense{}, &Conv2D{}, &RNN{}, &LSTM{}, &LayerNorm{}, &MultiHeadAttention{}, &TransformerEncoder{},
		(*Sequential)(nil), (*activation.ReLU)(nil), (*activation.Sigmoid)(nil), (*activation.Tanh)(nil), (*activation.Softmax)(nil),
		(*Flatten)(nil), (*LastStep)(nil), (*PositionalEncoding)(nil), activation.NewSoftmax(-1),
	} {
		model, err := NewSequential(stage)
		if err != nil {
			t.Fatal(err)
		}

		if _, err := MarshalConfig(model); err == nil {
			t.Fatalf("accepted layer %T", stage)
		}
	}

	if _, err := MarshalConfig(nil); err == nil {
		t.Fatal("accepted nil model")
	}
}

func TestMarshalConfig_RejectsCyclesAndExcessiveDepth(t *testing.T) {
	cyclic := &Sequential{}

	cyclic.layers = []Layer{cyclic}
	if _, err := MarshalConfig(cyclic); err == nil {
		t.Fatal("accepted cycle")
	}

	model, err := NewSequential()
	if err != nil {
		t.Fatal(err)
	}

	for range maxConfigDepth + 1 {
		model, err = NewSequential(model)
		if err != nil {
			t.Fatal(err)
		}
	}

	if _, err := MarshalConfig(model); err == nil {
		t.Fatal("accepted excessive depth")
	}
}
