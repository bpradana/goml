package optimizer

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/tensor"
)

func TestAdam_MultipleStepsMatchIndependentReference(t *testing.T) {
	parameter := testParameter(t, []float32{1})

	opt, err := NewAdam([]*nn.Parameter{parameter}, 0.01, 0.9, 0.999, 1e-8)
	if err != nil {
		t.Fatal(err)
	}

	want, first, second := 1.0, 0.0, 0.0

	for step, g := range []float32{2, -1, 0.5} {
		opt.ZeroGrad()
		accumulateGradient(t, parameter, []float32{g})

		if err := opt.Step(); err != nil {
			t.Fatal(err)
		}

		first = 0.9*first + 0.1*float64(g)
		second = 0.999*second + 0.001*float64(g)*float64(g)
		m := first / (1 - math.Pow(0.9, float64(step+1)))
		v := second / (1 - math.Pow(0.999, float64(step+1)))
		want -= 0.01 * m / (math.Sqrt(v) + 1e-8)

		got := parameter.Value().Data().Data()[0]
		if math.Abs(float64(got)-want) > 2e-6 {
			t.Fatalf("step %d: got %g, want %g", step+1, got, want)
		}
	}
}

func TestAdam_MissingGradientsDoNotAdvanceState(t *testing.T) {
	parameter := testParameter(t, []float32{1})

	opt, err := NewAdam([]*nn.Parameter{parameter}, 0.01, 0.9, 0.999, 1e-8)
	if err != nil {
		t.Fatal(err)
	}

	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}

	if len(opt.state) != 0 {
		t.Fatal("missing gradient allocated moment state")
	}

	leaf, previous := parameter.Value(), parameter.Value().Data()
	accumulateGradient(t, parameter, []float32{2})

	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}

	state := opt.state[parameter]
	first, second := slices.Clone(state.first), slices.Clone(state.second)

	opt.ZeroGrad()

	if err := opt.Step(); err != nil {
		t.Fatal(err)
	}

	if state.steps != 1 || !slices.Equal(first, state.first) || !slices.Equal(second, state.second) {
		t.Fatal("missing gradient advanced existing state")
	}

	if parameter.Value() != leaf || previous.Data()[0] != 1 {
		t.Fatal("update changed parameter identity or previous tensor")
	}
}

func TestAdam_RejectsInvalidHyperparameters(t *testing.T) {
	for _, tt := range []struct {
		name    string
		beta1   float32
		beta2   float32
		epsilon float32
	}{
		{"Beta1AtOne", 1, 0.999, 1e-8},
		{"Beta2Negative", 0.9, -1, 1e-8},
		{"ZeroEpsilon", 0.9, 0.999, 0},
		{"NaNBeta", float32(math.NaN()), 0.999, 1e-8},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := NewAdam(nil, 0.01, tt.beta1, tt.beta2, tt.epsilon); err == nil {
				t.Fatal("expected hyperparameter error")
			}
		})
	}
}

func TestAdam_DenseNetworkLearnsXOR(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))

	hidden, err := layer.NewDense(2, 8, true, rng)
	if err != nil {
		t.Fatal(err)
	}

	output, err := layer.NewDense(8, 1, true, rng)
	if err != nil {
		t.Fatal(err)
	}

	model, err := layer.NewSequential(hidden, activation.NewTanh(), output, activation.NewSigmoid())
	if err != nil {
		t.Fatal(err)
	}

	x, err := tensor.From([]float32{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
	if err != nil {
		t.Fatal(err)
	}

	targetData, err := tensor.From([]float32{0, 1, 1, 0}, 4, 1)
	if err != nil {
		t.Fatal(err)
	}

	input, err := nn.NewValue(x, false)
	if err != nil {
		t.Fatal(err)
	}

	target, err := nn.NewValue(targetData, false)
	if err != nil {
		t.Fatal(err)
	}

	opt, err := NewAdam(model.Parameters(), 0.03, 0.9, 0.999, 1e-8)
	if err != nil {
		t.Fatal(err)
	}

	var criterion loss.Loss[*nn.Value] = loss.NewMSE()

	for iteration := 0; iteration < 1000; iteration++ {
		prediction, err := model.Forward(input)
		if err != nil {
			t.Fatal(err)
		}

		objective, err := criterion.Forward(prediction, target)
		if err != nil {
			t.Fatal(err)
		}

		if err := objective.Backward(); err != nil {
			t.Fatal(err)
		}

		if err := opt.Step(); err != nil {
			t.Fatal(err)
		}

		opt.ZeroGrad()
	}

	prediction, err := model.Predict(x)
	if err != nil {
		t.Fatal(err)
	}

	for i, got := range prediction.Data() {
		want := targetData.Data()[i]
		if math.Abs(float64(got-want)) > 0.1 {
			t.Fatalf("XOR example %d: got %g, want %g", i, got, want)
		}
	}
}
