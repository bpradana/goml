package layer

import (
	"math/rand/v2"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
)

func BenchmarkSequential(b *testing.B) {
	for _, mode := range []string{"Forward", "Inference", "TrainingIteration"} {
		b.Run(mode, func(b *testing.B) {
			dense, err := NewDense(16, 8, true, rand.New(rand.NewPCG(1, 2)))
			if err != nil {
				b.Fatal(err)
			}

			model, err := NewSequential(dense, activation.NewTanh())
			if err != nil {
				b.Fatal(err)
			}

			data, err := tensor.Ones(8, 16)
			if err != nil {
				b.Fatal(err)
			}

			input, err := nn.NewValue(data, false)
			if err != nil {
				b.Fatal(err)
			}

			if mode == "Inference" {
				input, err = nn.Inference(data)
				if err != nil {
					b.Fatal(err)
				}
			}

			opt, err := optimizer.NewSGD(model.Parameters(), 0.001)
			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()

			for b.Loop() {
				output, err := model.Forward(input)
				if err != nil {
					b.Fatal(err)
				}

				if mode != "TrainingIteration" {
					continue
				}

				loss, err := nn.Mean(output)
				if err != nil {
					b.Fatal(err)
				}

				if err := loss.Backward(); err != nil {
					b.Fatal(err)
				}

				if err := opt.Step(); err != nil {
					b.Fatal(err)
				}

				opt.ZeroGrad()
			}
		})
	}
}

func BenchmarkAdvancedLayers(b *testing.B) {
	for _, configuration := range []struct {
		name  string
		shape []int
		build func(*rand.Rand) (Layer, error)
	}{
		{"Conv2D", []int{4, 1, 8, 8}, func(rng *rand.Rand) (Layer, error) {
			return NewConv2D(1, 4, 3, 1, 1, true, rng)
		}},
		{"RNN", []int{4, 8, 8}, func(rng *rand.Rand) (Layer, error) {
			return NewRNN(8, 16, rng)
		}},
		{"LSTM", []int{4, 8, 8}, func(rng *rand.Rand) (Layer, error) {
			return NewLSTM(8, 16, rng)
		}},
		{"TransformerEncoder", []int{2, 8, 16}, func(rng *rand.Rand) (Layer, error) {
			return NewTransformerEncoder(16, 4, 32, rng)
		}},
	} {
		b.Run(configuration.name, func(b *testing.B) {
			for _, mode := range []string{"Forward", "Inference", "ForwardBackward"} {
				b.Run(mode, func(b *testing.B) {
					rng := rand.New(rand.NewPCG(1, 2))

					stage, err := configuration.build(rng)
					if err != nil {
						b.Fatal(err)
					}

					data, err := tensor.Randn(rng, configuration.shape...)
					if err != nil {
						b.Fatal(err)
					}

					input, err := nn.NewValue(data, false)
					if err != nil {
						b.Fatal(err)
					}

					if mode == "Inference" {
						input, err = nn.Inference(data)
						if err != nil {
							b.Fatal(err)
						}
					}

					b.ReportAllocs()

					for b.Loop() {
						output, err := stage.Forward(input)
						if err != nil {
							b.Fatal(err)
						}

						if mode != "ForwardBackward" {
							continue
						}

						objective, err := nn.Mean(output)
						if err != nil {
							b.Fatal(err)
						}

						if err := objective.Backward(); err != nil {
							b.Fatal(err)
						}

						for _, parameter := range stage.Parameters() {
							parameter.ZeroGrad()
						}
					}
				})
			}
		})
	}
}

func BenchmarkSpatialTransforms(b *testing.B) {
	data, err := tensor.Ones(1, 16, 32, 32)
	if err != nil {
		b.Fatal(err)
	}

	input, err := nn.Inference(data)
	if err != nil {
		b.Fatal(err)
	}

	pool, err := NewMaxPool2D(3, 1, 1)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("MaxPool2D", func(b *testing.B) {
		b.ReportAllocs()

		for b.Loop() {
			if _, err := pool.Forward(input); err != nil {
				b.Fatal(err)
			}
		}
	})

	for _, mode := range []UpsampleMode{Nearest, Bilinear, Bicubic} {
		b.Run(string(mode), func(b *testing.B) {
			stage, err := NewUpsample2D(2, mode)
			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()

			for b.Loop() {
				if _, err := stage.Forward(input); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
