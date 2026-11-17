//go:build darwin && cgo

package examples

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/bpradana/goml/contrib/backend/metal"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/tensor"
)

func deviceForTest(t *testing.T) *metal.Device {
	t.Helper()

	device, err := metal.Open()
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		if err := device.Close(); err != nil {
			t.Error(err)
		}
	})

	return device
}

func compareTensor(t *testing.T, got, expected *tensor.Tensor, tolerance float64) {
	t.Helper()

	actual, err := got.HostData()
	if err != nil {
		t.Fatal(err)
	}

	wanted, err := expected.HostData()
	if err != nil {
		t.Fatal(err)
	}

	if len(actual) != len(wanted) {
		t.Fatal("tensor lengths differ")
	}

	for index, value := range actual {
		difference := math.Abs(float64(value) - float64(wanted[index]))
		if math.IsNaN(difference) || difference > tolerance+tolerance*math.Abs(float64(wanted[index])) {
			t.Fatalf("element %d: got %g, expected %g", index, value, wanted[index])
		}
	}
}

func checkModelGradient(t *testing.T, cpu *layer.Sequential, input *tensor.Tensor) {
	t.Helper()

	encoded, err := checkpoint.Marshal(cpu)
	if err != nil {
		t.Fatal(err)
	}

	gpu, err := checkpoint.Unmarshal(encoded)
	if err != nil {
		t.Fatal(err)
	}

	device := deviceForTest(t)
	if err := nn.ToDevice(gpu.Parameters(), device); err != nil {
		t.Fatal(err)
	}

	scope, err := tensor.NewScope(device)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = scope.Close() }()

	gpuInput, err := scope.To(input)
	if err != nil {
		t.Fatal(err)
	}

	cpuValue, err := nn.NewValue(input, true)
	if err != nil {
		t.Fatal(err)
	}

	gpuValue, err := nn.NewValue(gpuInput, true)
	if err != nil {
		t.Fatal(err)
	}

	var outputs [2]*nn.Value

	for index, current := range []*layer.Sequential{cpu, gpu} {
		value := cpuValue
		if index == 1 {
			value = gpuValue
		}

		output, err := current.Forward(value)
		if err != nil {
			t.Fatal(err)
		}

		outputs[index] = output

		squared, err := nn.Mul(output, output)
		if err != nil {
			t.Fatal(err)
		}

		objective, err := nn.Mean(squared)
		if err != nil {
			t.Fatal(err)
		}

		if err := objective.Backward(); err != nil {
			t.Fatal(err)
		}

		if err := objective.Backward(); err == nil {
			t.Fatal("consumed graph accepted a second backward pass")
		}
	}

	compareTensor(t, outputs[1].Data(), outputs[0].Data(), 0.001)
	compareTensor(t, gpuValue.Grad(), cpuValue.Grad(), 0.003)

	cpuParameters, gpuParameters := cpu.Parameters(), gpu.Parameters()
	for index, parameter := range gpuParameters {
		gradient := parameter.Value().Grad()
		if gradient == nil || gradient.Device() != device {
			t.Fatalf("parameter %d gradient left the device", index)
		}

		compareTensor(t, gradient, cpuParameters[index].Value().Grad(), 0.003)
	}
}

func TestDense_MetalForwardAndBackwardMatchCPU(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))

	first, err := layer.NewDense(2, 3, true, rng)
	if err != nil {
		t.Fatal(err)
	}

	last, err := layer.NewDense(3, 1, true, rng)
	if err != nil {
		t.Fatal(err)
	}

	model, err := layer.NewSequential(first, activation.NewTanh(), last)
	if err != nil {
		t.Fatal(err)
	}

	input, err := tensor.From([]float32{0.2, -0.7, 0.4, 0.9}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}

	checkModelGradient(t, model, input)
}

func TestConv2D_MetalNativeGradientsMatchCPU(t *testing.T) {
	conv, err := layer.NewConv2D(2, 3, 3, 1, 1, true, rand.New(rand.NewPCG(5, 8)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := layer.NewSequential(conv, activation.NewSiLU())
	if err != nil {
		t.Fatal(err)
	}

	values := make([]float32, 2*2*3*4)
	for index := range values {
		values[index] = float32(index%13-6) / 9
	}

	input, err := tensor.From(values, 2, 2, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	checkModelGradient(t, model, input)
}

func TestTransformer_MetalNormalizationAttentionAndGradientsMatchCPU(t *testing.T) {
	encoder, err := layer.NewTransformerEncoder(4, 2, 8, rand.New(rand.NewPCG(3, 7)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := layer.NewSequential(layer.NewPositionalEncoding(), encoder, activation.NewSoftmax(2))
	if err != nil {
		t.Fatal(err)
	}

	values := make([]float32, 12)
	for index := range values {
		values[index] = float32(index%7-3) / 5
	}

	input, err := tensor.From(values, 1, 3, 4)
	if err != nil {
		t.Fatal(err)
	}

	checkModelGradient(t, model, input)
}

func TestAttention_MetalCausalMaskAndGradientsMatchCPU(t *testing.T) {
	attention, err := layer.NewMultiHeadAttention(4, 2, true, rand.New(rand.NewPCG(11, 13)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := layer.NewSequential(attention)
	if err != nil {
		t.Fatal(err)
	}

	input, err := tensor.From([]float32{0.2, 0.1, -0.3, 0.4, 0.5, 0.3, 0.8, 0.2}, 1, 2, 4)
	if err != nil {
		t.Fatal(err)
	}

	checkModelGradient(t, model, input)
}

func TestRecurrent_MetalRNNAndLSTMGradientsMatchCPU(t *testing.T) {
	for _, kind := range []string{"rnn", "lstm"} {
		t.Run(kind, func(t *testing.T) {
			rng := rand.New(rand.NewPCG(19, 29))

			var (
				recurrent layer.Layer
				err       error
			)
			if kind == "rnn" {
				recurrent, err = layer.NewRNN(2, 3, rng)
			} else {
				recurrent, err = layer.NewLSTM(2, 3, rng)
			}

			if err != nil {
				t.Fatal(err)
			}

			model, err := layer.NewSequential(recurrent)
			if err != nil {
				t.Fatal(err)
			}

			input, err := tensor.From([]float32{0.2, -0.1, 0.5, 0.3, -0.4, 0.6}, 1, 3, 2)
			if err != nil {
				t.Fatal(err)
			}

			checkModelGradient(t, model, input)
		})
	}
}

func TestBackward_MetalRetainsForwardParameterGeneration(t *testing.T) {
	device := deviceForTest(t)

	initial, err := tensor.FromOn(device, []float32{2, 3}, 2)
	if err != nil {
		t.Fatal(err)
	}

	parameter, err := nn.NewParameter(initial)
	if err != nil {
		t.Fatal(err)
	}

	if err := initial.Close(); err != nil {
		t.Fatal(err)
	}

	borrowed := parameter.Value().Data()

	squared, err := nn.Mul(parameter.Value(), parameter.Value())
	if err != nil {
		t.Fatal(err)
	}

	objective, err := nn.Sum(squared)
	if err != nil {
		t.Fatal(err)
	}

	replacement, err := tensor.FromOn(device, []float32{20, 30}, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = replacement.Close() }()

	if err := parameter.Update(replacement); err != nil {
		t.Fatal(err)
	}

	if borrowed.IsClosed() {
		t.Fatal("live graph lost its saved parameter generation")
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	expected, err := tensor.From([]float32{4, 6}, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = expected.Close() }()

	compareTensor(t, parameter.Value().Grad(), expected, 1e-5)

	if !borrowed.IsClosed() {
		t.Fatal("consumed graph retained retired parameter storage")
	}
}

func TestDetach_MetalOwnsLeaseBeforeParameterUpdate(t *testing.T) {
	device := deviceForTest(t)

	initial, err := tensor.ScalarOn(device, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = initial.Close() }()

	parameter, err := nn.NewParameter(initial)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = parameter.Value().Data().Close() }()

	if err := initial.Close(); err != nil {
		t.Fatal(err)
	}

	borrowed := parameter.Value().Data()
	before := device.Stats()

	detached, err := parameter.Value().Detach()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = detached.Data().Close() }()

	if after := device.Stats(); after.LiveBuffers != before.LiveBuffers || after.LiveBytes != before.LiveBytes {
		t.Fatal("detach copied device storage instead of retaining a lease")
	}

	replacement, err := tensor.ScalarOn(device, 20)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = replacement.Close() }()

	if err := parameter.Update(replacement); err != nil {
		t.Fatal(err)
	}

	if !borrowed.IsClosed() {
		t.Fatal("detach retained the parameter generation instead of an independent handle")
	}

	values, err := detached.Data().HostData()
	if err != nil {
		t.Fatal(err)
	}

	if len(values) != 1 || values[0] != 2 || detached.Data().Device() != device {
		t.Fatalf("detached values %v, want original Metal value [2]", values)
	}

	for _, data := range []*tensor.Tensor{detached.Data(), replacement, parameter.Value().Data()} {
		if err := data.Close(); err != nil {
			t.Fatal(err)
		}
	}

	if err := device.Synchronize(); err != nil {
		t.Fatal(err)
	}

	if stats := device.Stats(); stats.LiveBuffers != 0 || stats.LiveBytes != 0 {
		t.Fatalf("detach cleanup retained native storage: %+v", stats)
	}
}
