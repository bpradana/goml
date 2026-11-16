package main

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"os"

	"github.com/bpradana/goml/contrib/backend/metal"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
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

	device, err := metal.Open()
	if err != nil {
		return err
	}
	defer func() { _ = device.Close() }()

	dense, err := layer.NewDense(1, 1, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		return err
	}

	model, err := layer.NewSequential(dense)
	if err != nil {
		return err
	}

	if err := nn.ToDevice(model.Parameters(), device); err != nil {
		return err
	}

	trainer, err := optimizer.NewSGD(model.Parameters(), 0.05)
	if err != nil {
		return err
	}

	hostInput, err := tensor.From([]float32{-1, 0, 1, 2}, 4, 1)
	if err != nil {
		return err
	}

	hostTarget, err := tensor.From([]float32{-1, 1, 3, 5}, 4, 1)
	if err != nil {
		return err
	}

	input, err := hostInput.To(device)
	if err != nil {
		return err
	}
	defer func() { _ = input.Close() }()

	target, err := hostTarget.To(device)
	if err != nil {
		return err
	}
	defer func() { _ = target.Close() }()

	criterion := loss.NewMSE()

	for step := range 100 {
		if err := trainStep(model, trainer, criterion, input, target, step); err != nil {
			return err
		}
	}

	if *path != "" {
		if err := checkpoint.Save(*path, model); err != nil {
			return err
		}
	}

	for index, parameter := range model.Parameters() {
		values, err := parameter.Value().Data().HostData()
		if err != nil {
			return err
		}

		fmt.Printf("parameter %d: %v\n", index, values)
	}

	return nil
}

func trainStep(model *layer.Sequential, trainer *optimizer.SGD, criterion *loss.MSE, input, target *tensor.Tensor, step int) error {
	scope, err := tensor.NewScope(input.Device())
	if err != nil {
		return err
	}
	defer func() { _ = scope.Close() }()

	batch, err := scope.To(input)
	if err != nil {
		return err
	}

	labels, err := scope.To(target)
	if err != nil {
		return err
	}

	x, err := nn.NewValue(batch, false)
	if err != nil {
		return err
	}

	y, err := nn.NewValue(labels, false)
	if err != nil {
		return err
	}

	prediction, err := model.Forward(x)
	if err != nil {
		return err
	}

	objective, err := criterion.Forward(prediction, y)
	if err != nil {
		return err
	}

	if step == 0 || step == 99 {
		value, err := objective.Data().HostData()
		if err != nil {
			return err
		}

		fmt.Printf("step %d: loss=%g\n", step, value[0])
	}

	if err := objective.Backward(); err != nil {
		return err
	}

	if err := trainer.Step(); err != nil {
		return err
	}

	trainer.ZeroGrad()

	return nil
}
