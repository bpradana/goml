//go:build darwin && cgo

package examples

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
)

func TestTraining_MetalCNNConvergesWithResidentAdamState(t *testing.T) {
	device := deviceForTest(t)
	rng := rand.New(rand.NewPCG(17, 23))

	conv, err := layer.NewConv2D(1, 2, 3, 1, 1, true, rng)
	if err != nil {
		t.Fatal(err)
	}

	dense, err := layer.NewDense(8, 2, true, rng)
	if err != nil {
		t.Fatal(err)
	}

	pool, err := layer.NewMaxPool2D(2, 2, 0)
	if err != nil {
		t.Fatal(err)
	}

	model, err := layer.NewSequential(conv, activation.NewTanh(), pool, layer.NewFlatten(), dense)
	if err != nil {
		t.Fatal(err)
	}

	if err := nn.ToDevice(model.Parameters(), device); err != nil {
		t.Fatal(err)
	}

	adam, err := optimizer.NewAdam(model.Parameters(), 0.03, 0.9, 0.999, 1e-8)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = adam.Close() }()

	values := make([]float32, 32)
	for index := range values {
		values[index] = -1
		if index >= 16 {
			values[index] = 1
		}
	}

	host, err := tensor.From(values, 2, 1, 4, 4)
	if err != nil {
		t.Fatal(err)
	}

	input, err := host.To(device)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = input.Close() }()

	criterion := loss.NewCrossEntropy()

	var (
		first, last float32
		baseline    uint64
	)

	for iteration := range 60 {
		scope, err := tensor.NewScope(device)
		if err != nil {
			t.Fatal(err)
		}

		batch, err := scope.To(input)
		if err != nil {
			t.Fatal(err)
		}

		value, err := nn.NewValue(batch, false)
		if err != nil {
			t.Fatal(err)
		}

		prediction, err := model.Forward(value)
		if err != nil {
			t.Fatal(err)
		}

		objective, err := criterion.Forward(prediction, []int{0, 1})
		if err != nil {
			t.Fatal(err)
		}

		if iteration == 0 || iteration == 59 {
			measured, err := objective.Data().HostData()
			if err != nil {
				t.Fatal(err)
			}

			if iteration == 0 {
				first = measured[0]
			} else {
				last = measured[0]
			}
		}

		if err := objective.Backward(); err != nil {
			t.Fatal(err)
		}

		if err := adam.Step(); err != nil {
			t.Fatal(err)
		}

		adam.ZeroGrad()

		if err := scope.Close(); err != nil {
			t.Fatal(err)
		}

		if err := device.Synchronize(); err != nil {
			t.Fatal(err)
		}

		live := device.Stats().LiveBytes
		if iteration == 0 {
			baseline = live
		} else if live != baseline {
			t.Fatalf("iteration %d retained native allocations: %d, baseline %d, stats %+v", iteration, live, baseline, device.Stats())
		}
	}

	if math.IsNaN(float64(last)) || last >= 0.02 || last >= first/10 {
		t.Fatalf("training failed to converge: first=%g last=%g", first, last)
	}

	t.Logf("CNN cross-entropy: %g -> %g; stable resident bytes: %d", first, last, baseline)
}

func TestBackward_MetalLeafGradientsSurviveScopesAndAccumulate(t *testing.T) {
	device := deviceForTest(t)

	parameter, err := nn.NewParameter(tensor.Scalar(3))
	if err != nil {
		t.Fatal(err)
	}

	if err := nn.ToDevice([]*nn.Parameter{parameter}, device); err != nil {
		t.Fatal(err)
	}

	for range 2 {
		scope, err := tensor.NewScope(device)
		if err != nil {
			t.Fatal(err)
		}

		factor, err := scope.To(tensor.Scalar(2))
		if err != nil {
			t.Fatal(err)
		}

		constant, err := nn.NewValue(factor, false)
		if err != nil {
			t.Fatal(err)
		}

		product, err := nn.Mul(parameter.Value(), constant)
		if err != nil {
			t.Fatal(err)
		}

		if err := product.Backward(); err != nil {
			t.Fatal(err)
		}

		if err := scope.Close(); err != nil {
			t.Fatal(err)
		}
	}

	compareTensor(t, parameter.Value().Grad(), tensor.Scalar(4), 0)
	parameter.ZeroGrad()

	if err := device.Synchronize(); err != nil {
		t.Fatal(err)
	}

	if device.Stats().LiveBuffers != 1 {
		t.Fatalf("only the parameter should remain, got %+v", device.Stats())
	}
}

func TestCrossEntropy_MetalPreservesLossForLargeCommonLogitOffsets(t *testing.T) {
	device := deviceForTest(t)

	scope, err := tensor.NewScope(device)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = scope.Close() }()

	for _, offset := range []float32{1000, 1e30, -1e30} {
		host, err := tensor.From([]float32{offset, offset}, 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		data, err := scope.To(host)
		if err != nil {
			t.Fatal(err)
		}

		prediction, err := nn.NewValue(data, true)
		if err != nil {
			t.Fatal(err)
		}

		objective, err := loss.NewCrossEntropy().Forward(prediction, []int{1})
		if err != nil {
			t.Fatal(err)
		}

		compareTensor(t, objective.Data(), tensor.Scalar(float32(math.Log(2))), 1e-6)

		if err := objective.Backward(); err != nil {
			t.Fatal(err)
		}

		expected, err := tensor.From([]float32{0.5, -0.5}, 1, 2)
		if err != nil {
			t.Fatal(err)
		}

		compareTensor(t, prediction.Grad(), expected, 1e-6)
		prediction.ZeroGrad()
	}
}
