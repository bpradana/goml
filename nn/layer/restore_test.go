package layer

import (
	"encoding/json"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/tensor"
)

func TestRestore_RoundTripsSiLUPoolingAndUpsampling(t *testing.T) {
	for _, mode := range []UpsampleMode{Nearest, Bilinear, Bicubic} {
		pool, _ := NewMaxPool2D(3, 1, 1)
		upsample, _ := NewUpsample2D(2, mode)
		model, _ := NewSequential(activation.NewSiLU(), pool, upsample)

		config, err := MarshalConfig(model)
		if err != nil {
			t.Fatal(err)
		}

		restored, err := Restore(config, nil)
		if err != nil {
			t.Fatal(err)
		}

		input, _ := tensor.From([]float32{-1, 2, 3, -4}, 1, 1, 2, 2)

		want, err := model.Predict(input)
		if err != nil {
			t.Fatal(err)
		}

		got, err := restored.Predict(input)
		if err != nil || !slices.Equal(backendtest.HostData(t, got), backendtest.HostData(t, want)) || !slices.Equal(got.Shape(), want.Shape()) {
			t.Fatalf("round-trip mismatch: %v %v", got, err)
		}
	}

	for _, stage := range []string{`{"type":"maxpool2d","config":{"kernel":3,"stride":0,"padding":1}}`, `{"type":"upsample2d","config":{"scale":-1}}`, `{"type":"silu","config":{"unknown":1}}`} {
		data := []byte(`{"root":0,"layers":[{"type":"sequential","children":[1]},` + stage + `]}`)
		if _, err := Restore(data, nil); err == nil {
			t.Fatal("invalid config accepted")
		}
	}
}

func TestRestore_PreservesSharingAndUsesSuppliedTrainableParameters(t *testing.T) {
	dense, err := NewDense(2, 2, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	nested, err := NewSequential(dense)
	if err != nil {
		t.Fatal(err)
	}

	model, err := NewSequential(nested, dense)
	if err != nil {
		t.Fatal(err)
	}

	config, err := MarshalConfig(model)
	if err != nil {
		t.Fatal(err)
	}

	parameters := make([]*nn.Parameter, len(model.Parameters()))
	for i, parameter := range model.Parameters() {
		data, err := parameter.Value().Data().MakeContiguous()
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = data.Close() }()

		parameters[i], err = nn.NewParameter(data)
		if err != nil {
			t.Fatal(err)
		}
	}

	restored, err := Restore(config, parameters)
	if err != nil {
		t.Fatal(err)
	}

	child := restored.layers[0].(*Sequential)
	if child.layers[0] != restored.layers[1] || !slices.Equal(restored.Parameters(), parameters) {
		t.Fatal("restored architecture lost shared identity")
	}

	input, err := tensor.From([]float32{0.1, -0.2, 0.3, 0.4}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	want, err := model.Predict(input)
	if err != nil {
		t.Fatal(err)
	}

	got, err := restored.Predict(input)
	if err != nil || !slices.Equal(backendtest.HostData(t, got), backendtest.HostData(t, want)) {
		t.Fatalf("prediction mismatch: %v %v", got, err)
	}

	value, err := nn.NewValue(input, false)
	if err != nil {
		t.Fatal(err)
	}

	output, err := restored.Forward(value)
	if err != nil {
		t.Fatal(err)
	}

	objective, err := nn.Sum(output)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	for i, parameter := range parameters {
		if parameter.Value().Grad() == nil || model.Parameters()[i].Value().Grad() != nil {
			t.Fatal("restored training did not isolate gradients")
		}
	}
}

func TestRestore_RejectsMalformedTopologyAndMetadata(t *testing.T) {
	for _, data := range []string{
		`null`, `{}`, `{"root":0,"layers":[],"extra":1}`,
		`{"root":1,"layers":[{"type":"sequential"}]}`,
		`{"root":0,"layers":[{"type":"sequential","children":[0]}]}`,
		`{"root":0,"layers":[{"type":"sequential","children":[-1]}]}`,
		`{"root":0,"layers":[{"type":"sequential"},{"type":"relu"}]}`,
		`{"root":0,"layers":[{"type":"relu"}]}`,
		`{"root":0,"layers":[{"type":"sequential","children":[1]},{"type":"unknown"}]}`,
		`{"root":0,"layers":[{"type":"sequential","children":[1]},{"type":"dense","parameters":[0]}]}`,
		`{"root":0,"layers":[{"type":"sequential","config":{"unknown":1}}]}`,
		`{"root":0,"layers":[{"type":"sequential","children":[1]},{"type":"softmax","config":{"axis":-1}}]}`,
		`{"root":0,"layers":[{"type":"sequential"}]} {}`,
	} {
		if _, err := Restore([]byte(data), nil); err == nil {
			t.Fatalf("accepted %s", data)
		}
	}

	config, err := json.Marshal(architecture{Layers: make([]layerDefinition, maxConfigLayers+1)})
	if err != nil {
		t.Fatal(err)
	}

	if _, err := Restore(config, nil); err == nil {
		t.Fatal("accepted excessive layer count")
	}
}

func TestRestore_RejectsUnusedRepeatedAndMismatchedParameters(t *testing.T) {
	weight, err := tensor.Ones(2, 2)
	if err != nil {
		t.Fatal(err)
	}

	parameter, err := nn.NewParameter(weight)
	if err != nil {
		t.Fatal(err)
	}

	empty := []byte(`{"root":0,"layers":[{"type":"sequential"}]}`)
	for _, parameters := range [][]*nn.Parameter{{parameter}, {nil}, {{}}, {parameter, parameter}} {
		if _, err := Restore(empty, parameters); err == nil {
			t.Fatal("accepted unused, invalid, or repeated parameters")
		}
	}

	wrongBias, err := nn.NewParameter(tensor.Scalar(1))
	if err != nil {
		t.Fatal(err)
	}

	config := []byte(`{"root":0,"layers":[{"type":"sequential","children":[1]},{"type":"dense","parameters":[0,1]}]}`)
	if _, err := Restore(config, []*nn.Parameter{parameter, wrongBias}); err == nil {
		t.Fatal("accepted bias with wrong shape")
	}
}

func FuzzRestore(f *testing.F) {
	weight, err := tensor.Ones(2, 2)
	if err != nil {
		f.Fatal(err)
	}

	bias, err := tensor.Zeros(2)
	if err != nil {
		f.Fatal(err)
	}

	w, err := nn.NewParameter(weight)
	if err != nil {
		f.Fatal(err)
	}

	b, err := nn.NewParameter(bias)
	if err != nil {
		f.Fatal(err)
	}

	f.Add([]byte(`{"root":0,"layers":[{"type":"sequential"}]}`))
	f.Add([]byte(`{"root":0,"layers":[{"type":"sequential","children":[1]},{"type":"dense","parameters":[0,1]}]}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 32768 {
			return
		}

		_, _ = Restore(data, []*nn.Parameter{w, b})
	})
}
