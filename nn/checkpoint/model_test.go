package checkpoint

import (
	"math/rand/v2"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/tensor"
)

func TestMarshalUnmarshal_AllBuiltInModelsPreservePredictionsAndTraining(t *testing.T) {
	for _, tt := range []struct {
		name  string
		shape []int
		build func(*rand.Rand) (*layer.Sequential, error)
	}{
		{"Dense", []int{2, 3}, func(rng *rand.Rand) (*layer.Sequential, error) {
			stage, err := layer.NewDense(3, 2, true, rng)
			if err != nil {
				return nil, err
			}

			return layer.NewSequential(stage, activation.NewTanh())
		}},
		{"DenseWithoutBias", []int{2, 3}, func(rng *rand.Rand) (*layer.Sequential, error) {
			stage, err := layer.NewDense(3, 2, false, rng)
			if err != nil {
				return nil, err
			}

			return layer.NewSequential(stage)
		}},
		{"Convolution", []int{2, 1, 5, 5}, func(rng *rand.Rand) (*layer.Sequential, error) {
			stage, err := layer.NewConv2D(1, 2, 3, 2, 1, false, rng)
			if err != nil {
				return nil, err
			}

			return layer.NewSequential(stage, activation.NewReLU(), layer.NewFlatten())
		}},
		{"RNN", []int{2, 3, 2}, func(rng *rand.Rand) (*layer.Sequential, error) {
			stage, err := layer.NewRNN(2, 3, rng)
			if err != nil {
				return nil, err
			}

			return layer.NewSequential(stage, layer.NewLastStep())
		}},
		{"LSTM", []int{2, 3, 2}, func(rng *rand.Rand) (*layer.Sequential, error) {
			stage, err := layer.NewLSTM(2, 3, rng)
			if err != nil {
				return nil, err
			}

			return layer.NewSequential(stage)
		}},
		{"CausalAttention", []int{2, 3, 4}, func(rng *rand.Rand) (*layer.Sequential, error) {
			normalization, err := layer.NewLayerNorm(4, 1e-3)
			if err != nil {
				return nil, err
			}

			attention, err := layer.NewMultiHeadAttention(4, 2, true, rng)
			if err != nil {
				return nil, err
			}

			return layer.NewSequential(normalization, attention, activation.NewSigmoid(), activation.NewSoftmax(2))
		}},
		{"Transformer", []int{2, 3, 4}, func(rng *rand.Rand) (*layer.Sequential, error) {
			stage, err := layer.NewTransformerEncoder(4, 2, 6, rng)
			if err != nil {
				return nil, err
			}

			return layer.NewSequential(layer.NewPositionalEncoding(), stage, layer.NewLastStep())
		}},
		{"NestedSharedLayers", []int{2, 3}, func(rng *rand.Rand) (*layer.Sequential, error) {
			shared, err := layer.NewDense(3, 3, true, rng)
			if err != nil {
				return nil, err
			}

			nested, err := layer.NewSequential(shared)
			if err != nil {
				return nil, err
			}

			return layer.NewSequential(nested, activation.NewReLU(), shared)
		}},
		{"EmptySequential", []int{2, 3}, func(*rand.Rand) (*layer.Sequential, error) {
			return layer.NewSequential()
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(1, 2))

			model, err := tt.build(rng)
			if err != nil {
				t.Fatal(err)
			}

			input, err := tensor.Randn(rng, tt.shape...)
			if err != nil {
				t.Fatal(err)
			}

			want, err := model.Predict(input)
			if err != nil {
				t.Fatal(err)
			}

			if len(model.Parameters()) > 0 {
				seedGradients(t, model.Parameters())
			}

			encoded, err := Marshal(model)
			if err != nil {
				t.Fatal(err)
			}

			restored, err := Unmarshal(encoded)
			if err != nil {
				t.Fatal(err)
			}

			clear(encoded)

			got, err := restored.Predict(input)
			if err != nil || !slices.Equal(got.Shape(), want.Shape()) || !slices.Equal(got.Data(), want.Data()) {
				t.Fatalf("restored prediction differs: %v %v", got, err)
			}

			if len(restored.Parameters()) != len(model.Parameters()) {
				t.Fatal("parameter sharing was lost")
			}

			for i, parameter := range restored.Parameters() {
				if parameter == model.Parameters()[i] || parameter.Value().Grad() != nil || model.Parameters()[i].Value().Grad() == nil {
					t.Fatal("serialization shared parameter identity or saved gradient state")
				}
			}

			if len(restored.Parameters()) == 0 {
				return
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

			for _, parameter := range restored.Parameters() {
				if parameter.Value().Grad() == nil {
					t.Fatal("restored model is not trainable")
				}
			}
		})
	}
}

func TestSaveLoad_NetworkFilesAndWeightsFilesAreDistinct(t *testing.T) {
	dense, err := layer.NewDense(2, 1, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := layer.NewSequential(dense)
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "model.goml")
	if err := Save(path, model); err != nil {
		t.Fatal(err)
	}

	restored, err := Load(path)
	if err != nil || len(restored.Parameters()) != 2 {
		t.Fatalf("load: %v", err)
	}

	if err := LoadWeights(path, model.Parameters()); err == nil {
		t.Fatal("accepted network as weights-only file")
	}

	if err := SaveWeights(path, model.Parameters()); err != nil {
		t.Fatal(err)
	}

	if _, err := Load(path); err == nil {
		t.Fatal("accepted weights-only file as complete network")
	}
}

func TestSave_InvalidNetworkPreservesExistingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.goml")
	if err := os.WriteFile(path, []byte("original"), 0o600); err != nil {
		t.Fatal(err)
	}

	if err := Save(path, nil); err == nil {
		t.Fatal("accepted nil model")
	}

	data, err := os.ReadFile(path)
	if err != nil || string(data) != "original" {
		t.Fatalf("failed save changed original: %q %v", data, err)
	}
}
