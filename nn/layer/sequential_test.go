package layer

import (
	"math/rand/v2"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/tensor"
)

func TestSequential_SharedLayersDeduplicateParameters(t *testing.T) {
	dense, err := NewDense(2, 2, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := NewSequential(dense, dense)
	if err != nil {
		t.Fatal(err)
	}

	if len(model.Parameters()) != 2 {
		t.Fatal("shared parameters were duplicated")
	}
}

func TestSequential_InferenceConstructsNoGradientGraph(t *testing.T) {
	dense, err := NewDense(2, 1, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := NewSequential(dense, activation.NewSigmoid())
	if err != nil {
		t.Fatal(err)
	}

	data, err := tensor.Ones(1, 2)
	if err != nil {
		t.Fatal(err)
	}

	input, err := nn.Inference(data)
	if err != nil {
		t.Fatal(err)
	}

	output, err := model.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if output.RequiresGrad() || output.Grad() != nil {
		t.Fatal("inference retained gradient tracking")
	}

	for _, p := range model.Parameters() {
		if p.Value().Grad() != nil {
			t.Fatal("inference changed a parameter gradient")
		}
	}
}

func TestNewSequential_RejectsNilLayer(t *testing.T) {
	if _, err := NewSequential(nil); err == nil {
		t.Fatal("expected nil layer error")
	}
}

func TestSequential_ActivationConstructorsImplementLayer(t *testing.T) {
	for _, tt := range []struct {
		name  string
		stage Layer
		want  float32
	}{
		{"ReLU", activation.NewReLU(), 0},
		{"Sigmoid", activation.NewSigmoid(), 0.5},
		{"Tanh", activation.NewTanh(), 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			input, err := nn.NewValue(tensor.Scalar(0), true)
			if err != nil {
				t.Fatal(err)
			}

			model, err := NewSequential(tt.stage)
			if err != nil {
				t.Fatal(err)
			}

			output, err := model.Forward(input)
			if err != nil || output.Data().Data()[0] != tt.want {
				t.Fatalf("got %v, want %g: %v", output, tt.want, err)
			}

			if len(model.Parameters()) != 0 || !output.RequiresGrad() {
				t.Fatal("activation composition exposed parameters or dropped tracking")
			}
		})
	}
}
