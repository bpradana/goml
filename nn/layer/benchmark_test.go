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
