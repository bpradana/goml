package main

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/tensor"
)

func TestNewModel_NCHWInputProducesLogitsAndParameterGradients(t *testing.T) {
	for _, name := range []string{"dense", "cnn"} {
		t.Run(name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(1, 2))

			model, err := newModel(name, rng)
			if err != nil {
				t.Fatal(err)
			}

			data, err := tensor.Rand(rng, 2, 1, 28, 28)
			if err != nil {
				t.Fatal(err)
			}

			input, err := nn.NewValue(data, false)
			if err != nil {
				t.Fatal(err)
			}

			logits, err := model.Forward(input)
			if err != nil {
				t.Fatal(err)
			}

			if !slices.Equal(logits.Data().Shape(), []int{2, 10}) {
				t.Fatal(logits.Data())
			}

			objective, err := loss.NewCrossEntropy().Forward(logits, []int{0, 9})
			if err != nil {
				t.Fatal(err)
			}

			if err := objective.Backward(); err != nil {
				t.Fatal(err)
			}

			for index, parameter := range model.Parameters() {
				if parameter.Value().Grad() == nil {
					t.Fatalf("parameter %d received no gradient", index)
				}

				for _, gradient := range backendtest.HostData(t, parameter.Value().Grad()) {
					if math.IsNaN(float64(gradient)) || math.IsInf(float64(gradient), 0) {
						t.Fatalf("parameter %d has nonfinite gradient %g", index, gradient)
					}
				}
			}
		})
	}
}

func TestNewModel_SeededPredictionsAreReproducible(t *testing.T) {
	data, err := tensor.Ones(1, 1, 28, 28)
	if err != nil {
		t.Fatal(err)
	}

	for _, name := range []string{"dense", "cnn"} {
		t.Run(name, func(t *testing.T) {
			var previous []float32

			for iteration := range 2 {
				model, err := newModel(name, rand.New(rand.NewPCG(1, 2)))
				if err != nil {
					t.Fatal(err)
				}

				prediction, err := model.Predict(data)
				if err != nil {
					t.Fatal(err)
				}

				if iteration > 0 && !slices.Equal(previous, backendtest.HostData(t, prediction)) {
					t.Fatal("identical seeds produced different predictions")
				}

				previous = backendtest.HostData(t, prediction)

				for _, parameter := range model.Parameters() {
					if parameter.Value().Grad() != nil {
						t.Fatal("prediction created a parameter gradient")
					}
				}
			}
		})
	}
}

func TestNewModel_RejectsInvalidNameAndMissingGenerator(t *testing.T) {
	if _, err := newModel("unknown", rand.New(rand.NewPCG(1, 2))); err == nil {
		t.Fatal("accepted unknown model")
	}

	for _, name := range []string{"dense", "cnn"} {
		if _, err := newModel(name, nil); err == nil {
			t.Fatalf("accepted nil generator for %s", name)
		}
	}
}
