//go:build darwin && cgo

package examples

import (
	"fmt"
	"image"
	_ "image/jpeg"
	"os"
	"path/filepath"
	"testing"

	"github.com/bpradana/goml/contrib/backend/metal"
	"github.com/bpradana/goml/contrib/nn/yolo"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
)

func BenchmarkYOLOStartup(b *testing.B) {
	path := os.Getenv("GOML_YOLO_MODEL")
	if path == "" {
		b.Skip("set GOML_YOLO_MODEL to a trusted local converted archive")
	}

	device, err := metal.Open()
	if err != nil {
		b.Fatal(err)
	}

	b.Cleanup(func() { _ = device.Close() })

	b.Run("model_load", func(b *testing.B) {
		b.ReportAllocs()

		for b.Loop() {
			model, err := yolo.Load(path)
			if err != nil {
				b.Fatal(err)
			}

			_ = model.Close()
		}
	})

	b.Run("weight_upload", func(b *testing.B) {
		model, err := yolo.Load(path)
		if err != nil {
			b.Fatal(err)
		}
		defer func() { _ = model.Close() }()

		b.ReportAllocs()
		b.ResetTimer()

		for range b.N {
			if err := model.ToDevice(device); err != nil {
				b.Fatal(err)
			}

			b.StopTimer()

			if err := model.ToDevice(nil); err != nil {
				b.Fatal(err)
			}

			b.StartTimer()
		}

		b.StopTimer()
	})
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

func BenchmarkYOLO(b *testing.B) {
	path := os.Getenv("GOML_YOLO_MODEL")
	if path == "" {
		b.Skip("set GOML_YOLO_MODEL to a trusted local converted archive")
	}

	device, err := metal.Open()
	if err != nil {
		b.Fatal(err)
	}

	b.Cleanup(func() { _ = device.Close() })

	file, err := os.Open(filepath.Join(filepath.Dir(path), "bus.jpg"))
	if err != nil {
		b.Fatal(err)
	}

	source, _, err := image.Decode(file)
	_ = file.Close()

	if err != nil {
		b.Fatal(err)
	}

	for _, size := range []int{320, 640} {
		b.Run(fmt.Sprint(size), func(b *testing.B) {
			options := yolo.DefaultOptions()
			options.Size = size

			cpu, err := yolo.Load(path)
			if err != nil {
				b.Fatal(err)
			}

			gpu, err := yolo.Load(path)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = gpu.Close() }()

			if err := gpu.ToDevice(device); err != nil {
				b.Fatal(err)
			}

			input := referenceInput(b, path, size)

			resident, err := input.To(device)
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = resident.Close() }()

			heads, err := gpu.Forward(resident)
			if err != nil {
				b.Fatal(err)
			}
			defer func() {
				for _, head := range heads {
					_ = head.Close()
				}
			}()

			var hostHeads [3]*tensor.Tensor
			for index, head := range heads {
				hostHeads[index], err = head.ToCPU()
				if err != nil {
					b.Fatal(err)
				}
			}

			b.Run("input_upload", func(b *testing.B) {
				b.ReportAllocs()

				for b.Loop() {
					value, err := input.To(device)
					if err != nil {
						b.Fatal(err)
					}

					_ = value.Close()
				}
			})

			for _, target := range []string{"cpu", "metal"} {
				b.Run("forward_"+target, func(b *testing.B) {
					model, value := cpu, input
					if target == "metal" {
						model, value = gpu, resident
					}

					b.ReportAllocs()

					for b.Loop() {
						result, err := model.Forward(value)
						if err != nil {
							b.Fatal(err)
						}

						if target == "metal" {
							if err := device.Synchronize(); err != nil {
								b.Fatal(err)
							}
						}

						for _, head := range result {
							_ = head.Close()
						}
					}
				})

				b.Run("detect_"+target, func(b *testing.B) {
					model := cpu
					if target == "metal" {
						model = gpu
					}

					b.ReportAllocs()

					for b.Loop() {
						if _, err := model.Detect(source, options); err != nil {
							b.Fatal(err)
						}
					}
				})
			}

			b.Run("output_download", func(b *testing.B) {
				b.ReportAllocs()

				for b.Loop() {
					for _, head := range heads {
						if _, err := head.HostData(); err != nil {
							b.Fatal(err)
						}
					}
				}
			})

			b.Run("postprocess_cpu", func(b *testing.B) {
				b.ReportAllocs()

				for b.Loop() {
					if _, err := gpu.Postprocess(hostHeads, source.Bounds().Size(), options); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}
