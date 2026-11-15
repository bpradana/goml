package nn_test

import (
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"

	"github.com/bpradana/goml/tensor"
)

func BenchmarkForwardBackward(b *testing.B) {
	data, err := tensor.Ones(32, 32)
	if err != nil {
		b.Fatal(err)
	}

	value, err := nn.NewValue(data, true)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()

	for b.Loop() {
		square, err := nn.Mul(value, value)
		if err != nil {
			b.Fatal(err)
		}

		loss, err := nn.Mean(square)
		if err != nil {
			b.Fatal(err)
		}

		if err := loss.Backward(); err != nil {
			b.Fatal(err)
		}

		value.ZeroGrad()
	}
}

func BenchmarkInference(b *testing.B) {
	data, err := tensor.Ones(32, 32)
	if err != nil {
		b.Fatal(err)
	}

	value, err := nn.Inference(data)
	if err != nil {
		b.Fatal(err)
	}

	sigmoid := activation.NewSigmoid()

	b.ReportAllocs()

	for b.Loop() {
		if _, err := sigmoid.Forward(value); err != nil {
			b.Fatal(err)
		}
	}
}
