//go:build darwin && cgo

package main

import (
	_ "image/jpeg"
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/bpradana/goml/contrib/backend/metal"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/nn/optimizer"
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

func TestCheckpoint_MetalWeightsRestoreOnCPUAndGPUWithoutRetainingStagingBuffers(t *testing.T) {
	dense, err := layer.NewDense(2, 3, true, rand.New(rand.NewPCG(1, 3)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := layer.NewSequential(dense)
	if err != nil {
		t.Fatal(err)
	}

	device := deviceForTest(t)
	if err := nn.ToDevice(model.Parameters(), device); err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "weights.goml")
	if err := checkpoint.SaveWeights(path, model.Parameters()); err != nil {
		t.Fatal(err)
	}

	encoded, err := checkpoint.Marshal(model)
	if err != nil {
		t.Fatal(err)
	}

	cpu, err := checkpoint.Unmarshal(encoded)
	if err != nil {
		t.Fatal(err)
	}

	baseline := device.Stats().LiveBytes

	identity := model.Parameters()[0].Value()
	for range 3 {
		if err := checkpoint.LoadWeights(path, model.Parameters()); err != nil {
			t.Fatal(err)
		}

		if err := device.Synchronize(); err != nil {
			t.Fatal(err)
		}

		if device.Stats().LiveBytes != baseline {
			t.Fatalf("checkpoint retained staging allocations: %+v", device.Stats())
		}
	}

	if model.Parameters()[0].Value() != identity {
		t.Fatal("restoration changed parameter identity")
	}

	for index, parameter := range cpu.Parameters() {
		if !parameter.Value().Data().IsCPU() {
			t.Fatal("portable checkpoint did not restore onto CPU")
		}

		compareTensor(t, model.Parameters()[index].Value().Data(), parameter.Value().Data(), 0)
	}
}

func TestOptimizers_MetalSGDAndAdamMatchCPUUpdates(t *testing.T) {
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
			}
		})
	}
}

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

func BenchmarkOptimizerStep(b *testing.B) {
	device, err := metal.Open()
	if err != nil {
		b.Fatal(err)
	}

	b.Cleanup(func() { _ = device.Close() })

	for _, kind := range []string{"sgd", "adam"} {
		for _, location := range []string{"cpu", "metal"} {
			b.Run(kind+"/"+location, func(b *testing.B) {
				data, err := tensor.Ones(32768)
				if err != nil {
					b.Fatal(err)
				}

				parameters := make([]*nn.Parameter, 4)
				for index := range parameters {
					parameters[index], err = nn.NewParameter(data)
					if err != nil {
						b.Fatal(err)
					}
				}

				if location == "metal" {
					if err := nn.ToDevice(parameters, device); err != nil {
						b.Fatal(err)
					}
				}

				var trainer optimizer.Optimizer
				if kind == "sgd" {
					trainer, err = optimizer.NewSGD(parameters, 0.01)
				} else {
					adam, createErr := optimizer.NewAdam(parameters, 0.01, 0.9, 0.999, 1e-8)

					trainer, err = adam, createErr
					if createErr == nil {
						defer func() { _ = adam.Close() }()
					}
				}

				if err != nil {
					b.Fatal(err)
				}

				var scope *tensor.Scope

				constant := tensor.Scalar(1)

				if location == "metal" {
					scope, err = tensor.NewScope(device)
					if err != nil {
						b.Fatal(err)
					}

					constant, err = scope.To(constant)
					if err != nil {
						b.Fatal(err)
					}
				}

				factor, err := nn.NewValue(constant, false)
				if err != nil {
					b.Fatal(err)
				}

				for _, parameter := range parameters {
					product, err := nn.Mul(parameter.Value(), factor)
					if err != nil {
						b.Fatal(err)
					}

					objective, err := nn.Sum(product)
					if err != nil {
						b.Fatal(err)
					}

					if err := objective.Backward(); err != nil {
						b.Fatal(err)
					}
				}

				if scope != nil {
					_ = scope.Close()
				}

				defer func() {
					trainer.ZeroGrad()

					for _, parameter := range parameters {
						_ = parameter.Value().Data().Close()
					}
				}()

				if err := trainer.Step(); err != nil {
					b.Fatal(err)
				}

				b.ReportAllocs()

				for b.Loop() {
					if err := trainer.Step(); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func BenchmarkDenseBackwardResident(b *testing.B) {
	device, err := metal.Open()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	model, err := layer.NewDense(64, 32, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		b.Fatal(err)
	}

	if err := nn.ToDevice(model.Parameters(), device); err != nil {
		b.Fatal(err)
	}
	defer func() {
		for _, parameter := range model.Parameters() {
			_ = parameter.Value().Data().Close()
		}
	}()

	host, err := tensor.Ones(32, 64)
	if err != nil {
		b.Fatal(err)
	}

	resident, err := host.To(device)
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = resident.Close() }()

	b.ReportAllocs()

	for b.Loop() {
		scope, err := tensor.NewScope(device)
		if err != nil {
			b.Fatal(err)
		}

		data, err := scope.To(resident)
		if err != nil {
			b.Fatal(err)
		}

		input, err := nn.NewValue(data, false)
		if err != nil {
			b.Fatal(err)
		}

		output, err := model.Forward(input)
		if err != nil {
			b.Fatal(err)
		}

		squared, err := nn.Mul(output, output)
		if err != nil {
			b.Fatal(err)
		}

		objective, err := nn.Mean(squared)
		if err != nil {
			b.Fatal(err)
		}

		if err := objective.Backward(); err != nil {
			b.Fatal(err)
		}

		for _, parameter := range model.Parameters() {
			parameter.ZeroGrad()
		}

		if err := scope.Close(); err != nil {
			b.Fatal(err)
		}
	}
}
