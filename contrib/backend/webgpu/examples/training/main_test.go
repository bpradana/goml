package main

import (
	"errors"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/bpradana/goml/contrib/backend/webgpu"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/backendtest"
)

func deviceForTest(t *testing.T) *webgpu.Device {
	t.Helper()

	device, err := webgpu.Open()
	if err != nil {
		if errors.Is(err, backend.ErrUnsupported) && os.Getenv("GOML_WEBGPU_REQUIRED") != "1" {
			t.Skip(err)
		}

		t.Fatal(err)
	}

	t.Cleanup(func() {
		if err := device.Close(); err != nil {
			t.Error(err)
		}
	})

	return device
}

func TestTraining_ConvergesAndRestoresPortableCheckpoint(t *testing.T) {
	device := deviceForTest(t)
	path := filepath.Join(t.TempDir(), "linear.goml")

	first, last, err := train(device, path)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("linear regression: %g -> %g", first, last)

	if !(last < first*0.01) {
		t.Fatalf("training failed to improve: %g -> %g", first, last)
	}

	if err := device.Synchronize(); err != nil {
		t.Fatal(err)
	}

	if stats := device.Stats(); stats.LiveBuffers != 0 || stats.LiveBytes != 0 {
		t.Fatalf("training retained native storage: %+v", stats)
	}

	if err := device.Close(); err != nil {
		t.Fatal(err)
	}

	model, err := checkpoint.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		for _, parameter := range model.Parameters() {
			_ = parameter.Value().Data().Close()
		}
	}()

	for _, parameter := range model.Parameters() {
		if !parameter.Value().Data().IsCPU() {
			t.Fatal("portable checkpoint restored native handles")
		}
	}

	input, err := tensor.From([]float32{-1, 0, 1, 2}, 4, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = input.Close() }()

	prediction, err := model.Predict(input)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = prediction.Close() }()

	for i, value := range backendtest.HostData(t, prediction) {
		if math.Abs(float64(value-float32(2*i-1))) > 0.02 {
			t.Fatalf("restored prediction %d: %g", i, value)
		}
	}
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
	defer func() {
		for _, parameter := range gpu.Parameters() {
			parameter.ZeroGrad()
			_ = parameter.Value().Data().Close()
		}

		if err := device.Synchronize(); err != nil {
			t.Error(err)
		}

		if stats := device.Stats(); stats.LiveBuffers != 0 || stats.LiveBytes != 0 {
			t.Errorf("gradient pass retained native buffers: %+v", stats)
		}
	}()

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
	defer gpuValue.ZeroGrad()

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

func TestDense_WebGPUForwardAndBackwardMatchCPU(t *testing.T) {
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

func TestConv2D_WebGPUNativeGradientsMatchCPU(t *testing.T) {
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

func TestTransformer_WebGPUNormalizationAttentionAndGradientsMatchCPU(t *testing.T) {
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

func TestAttention_WebGPUCausalMaskAndGradientsMatchCPU(t *testing.T) {
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

func TestRecurrent_WebGPURNNAndLSTMGradientsMatchCPU(t *testing.T) {
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

func TestOptimizers_WebGPUSGDAndAdamMatchCPUUpdates(t *testing.T) {
	for _, kind := range []string{"sgd", "adam"} {
		t.Run(kind, func(t *testing.T) {
			device := deviceForTest(t)

			data, err := tensor.From([]float32{1, -2}, 2)
			if err != nil {
				t.Fatal(err)
			}

			cpu, err := nn.NewParameter(data)
			if err != nil {
				t.Fatal(err)
			}

			gpu, err := nn.NewParameter(data)
			if err != nil {
				t.Fatal(err)
			}

			if err := nn.ToDevice([]*nn.Parameter{gpu}, device); err != nil {
				t.Fatal(err)
			}

			var cpuOptimizer, gpuOptimizer optimizer.Optimizer
			if kind == "sgd" {
				cpuOptimizer, err = optimizer.NewSGD([]*nn.Parameter{cpu}, 0.1)
				if err != nil {
					t.Fatal(err)
				}

				gpuOptimizer, err = optimizer.NewSGD([]*nn.Parameter{gpu}, 0.1)
			} else {
				cpuOptimizer, err = optimizer.NewAdam([]*nn.Parameter{cpu}, 0.1, 0.9, 0.999, 1e-8)
				if err != nil {
					t.Fatal(err)
				}

				adam, createErr := optimizer.NewAdam([]*nn.Parameter{gpu}, 0.1, 0.9, 0.999, 1e-8)

				err = createErr
				if createErr == nil {
					defer func() { _ = adam.Close() }()
				}

				gpuOptimizer = adam
			}

			if err != nil {
				t.Fatal(err)
			}

			identity := gpu.Value()

			var baseline uint64

			for step := range 4 {
				scope, err := tensor.NewScope(device)
				if err != nil {
					t.Fatal(err)
				}

				scalar, err := scope.To(tensor.Scalar(float32(step + 1)))
				if err != nil {
					t.Fatal(err)
				}

				multiplier, err := nn.NewValue(scalar, false)
				if err != nil {
					t.Fatal(err)
				}

				cpuMultiplier, err := nn.NewValue(tensor.Scalar(float32(step+1)), false)
				if err != nil {
					t.Fatal(err)
				}

				for index, parameter := range []*nn.Parameter{cpu, gpu} {
					factor := cpuMultiplier
					if index == 1 {
						factor = multiplier
					}

					product, err := nn.Mul(parameter.Value(), factor)
					if err != nil {
						t.Fatal(err)
					}

					objective, err := nn.Sum(product)
					if err != nil {
						t.Fatal(err)
					}

					if err := objective.Backward(); err != nil {
						t.Fatal(err)
					}
				}

				if err := cpuOptimizer.Step(); err != nil {
					t.Fatal(err)
				}

				if err := gpuOptimizer.Step(); err != nil {
					t.Fatal(err)
				}

				compareTensor(t, gpu.Value().Data(), cpu.Value().Data(), 1e-5)

				if gpu.Value() != identity || gpu.Value().Data().Device() != device {
					t.Fatal("optimizer changed parameter identity or device")
				}

				cpuOptimizer.ZeroGrad()
				gpuOptimizer.ZeroGrad()

				if err := scope.Close(); err != nil {
					t.Fatal(err)
				}

				if err := device.Synchronize(); err != nil {
					t.Fatal(err)
				}

				live := device.Stats().LiveBytes
				if step == 0 {
					baseline = live
				} else if live != baseline {
					t.Fatalf("optimizer step retained storage: %d, baseline %d", live, baseline)
				}
			}
		})
	}
}
