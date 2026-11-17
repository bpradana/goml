package main

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"os"

	"github.com/bpradana/goml/contrib/backend/webgpu"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	path := flag.String("checkpoint", "", "optional portable network checkpoint destination")

	flag.Parse()

	device, err := webgpu.Open()
	if err != nil {
		return err
	}
	defer func() { _ = device.Close() }()

	first, last, err := train(device, *path)
	if err != nil {
		return err
	}

	fmt.Printf("%s: first_loss=%g final_loss=%g\n", device.Name(), first, last)

	return nil
}

func train(device backend.Device, path string) (float32, float32, error) {
	dense, err := layer.NewDense(1, 1, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		return 0, 0, err
	}

	model, err := layer.NewSequential(dense)
	if err != nil {
		return 0, 0, err
	}

	if err := nn.ToDevice(model.Parameters(), device); err != nil {
		return 0, 0, err
	}
	defer func() {
		for _, parameter := range model.Parameters() {
			parameter.ZeroGrad()
			_ = parameter.Value().Data().Close()
		}
	}()

	trainer, err := optimizer.NewSGD(model.Parameters(), 0.05)
	if err != nil {
		return 0, 0, err
	}

	input, err := tensor.FromOn(device, []float32{-1, 0, 1, 2}, 4, 1)
	if err != nil {
		return 0, 0, err
	}
	defer func() { _ = input.Close() }()

	target, err := tensor.FromOn(device, []float32{-1, 1, 3, 5}, 4, 1)
	if err != nil {
		return 0, 0, err
	}
	defer func() { _ = target.Close() }()

	var first, last float32

	for step := range 100 {
		value, err := trainStep(model, trainer, input, target, step == 0 || step == 99)
		if err != nil {
			return 0, 0, err
		}

		switch step {
		case 0:
			first = value
		case 99:
			last = value
		}
	}

	if last >= 0.0001 {
		return first, last, fmt.Errorf("training did not converge: loss %g", last)
	}

	if path != "" {
		if err := checkpoint.Save(path, model); err != nil {
			return first, last, err
		}
	}

	return first, last, nil
}

func trainStep(model *layer.Sequential, trainer *optimizer.SGD, input, target *tensor.Tensor, measure bool) (float32, error) {
	scope, err := tensor.NewScope(input.Device())
	if err != nil {
		return 0, err
	}
	defer func() { _ = scope.Close() }()

	batch, err := scope.To(input)
	if err != nil {
		return 0, err
	}

	labels, err := scope.To(target)
	if err != nil {
		return 0, err
	}

	x, err := nn.NewValue(batch, false)
	if err != nil {
		return 0, err
	}

	y, err := nn.NewValue(labels, false)
	if err != nil {
		return 0, err
	}

	prediction, err := model.Forward(x)
	if err != nil {
		return 0, err
	}

	objective, err := loss.NewMSE().Forward(prediction, y)
	if err != nil {
		return 0, err
	}

	var measured float32

	if measure {
		values, err := objective.Data().HostData()
		if err != nil {
			return 0, err
		}

		measured = values[0]
	}

	if err := objective.Backward(); err != nil {
		return 0, err
	}

	if err := trainer.Step(); err != nil {
		return 0, err
	}

	trainer.ZeroGrad()

	return measured, nil
}
