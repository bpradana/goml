//go:build darwin && cgo

package examples

import (
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
)

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
