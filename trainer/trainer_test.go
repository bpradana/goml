package trainer_test

import (
	"context"
	"errors"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/trainer"
)

type failingModel struct {
	trainer.Model
	failure   error
	backward  bool
	temporary *tensor.Tensor
}

func (m *failingModel) Forward(input *nn.Value) (*nn.Value, error) {
	output, err := m.Model.Forward(input)
	if err != nil {
		return nil, err
	}

	m.temporary = output.Data()
	if !m.backward {
		return nil, m.failure
	}

	return nn.NewOperation(output.Data(), output, func(*tensor.Tensor) (*tensor.Tensor, error) { return nil, m.failure })
}

type watchedDevice struct {
	backend.Device
	closed bool
}

func (d *watchedDevice) Close() error { d.closed = true; return nil }

func TestTrainer_ForwardBackwardAndLossFailuresReleaseOwnedResources(t *testing.T) {
	for _, kind := range []string{"forward", "backward", "loss"} {
		t.Run(kind, func(t *testing.T) {
			base := denseModel(t, 1, 2)
			wanted := errors.New(kind)
			model := trainer.Model(base)

			failing := &failingModel{Model: base, failure: wanted, backward: kind == "backward"}
			if kind != "loss" {
				model = failing
			}

			opt, _ := optimizer.NewSGD(base.Parameters(), 0.1)
			device := &watchedDevice{Device: tensor.CPU()}

			runner, err := trainer.NewClassifier(model, opt, device, 2)
			if err != nil {
				t.Fatal(err)
			}

			loader := classLoader(t, 3)

			if kind == "loss" {
				x, _ := tensor.Zeros(1, 1)
				defer func() { _ = x.Close() }()

				ds, _ := data.NewClassificationDataset(x, []int{99})
				loader, _ = data.NewLoader(ds, data.CollateClassification, data.LoaderOptions{BatchSize: 1})
			}

			source := &observedClasses{source: loader}

			_, err = runner.Fit(context.Background(), source, nil, trainer.FitOptions{Epochs: 1})
			if err == nil || (kind != "loss" && !errors.Is(err, wanted)) {
				t.Fatal(err)
			}

			if device.closed {
				t.Fatal("closed borrowed device")
			}

			if !source.batches[0].Inputs.IsClosed() {
				t.Fatal("batch leaked")
			}

			if failing.temporary != nil && !failing.temporary.IsClosed() {
				t.Fatal("forward temporary leaked")
			}

			for _, p := range base.Parameters() {
				if p.Value().Grad() != nil || p.Value().Data().IsClosed() {
					t.Fatal("gradient retained or model closed")
				}
			}
		})
	}
}

func TestTrainer_RejectsInvalidDependenciesAndSources(t *testing.T) {
	model := denseModel(t, 1, 2)

	opt, _ := optimizer.NewSGD(model.Parameters(), 0.1)
	if _, err := trainer.NewClassifier(nil, opt, tensor.CPU(), 2); err == nil {
		t.Fatal("nil model accepted")
	}

	if _, err := trainer.NewClassifier(model, nil, tensor.CPU(), 2); err == nil {
		t.Fatal("nil optimizer accepted")
	}

	if _, err := trainer.NewClassifier(model, opt, nil, 2); err == nil {
		t.Fatal("nil device accepted")
	}

	if _, err := trainer.NewClassifier(model, opt, tensor.CPU(), 0); err == nil {
		t.Fatal("invalid classes accepted")
	}

	runner, _ := trainer.NewClassifier(model, opt, tensor.CPU(), 2)
	if _, err := runner.Fit(context.Background(), nil, nil, trainer.FitOptions{Epochs: 1}); err == nil {
		t.Fatal("nil training accepted")
	}

	if _, err := runner.Evaluate(context.Background(), nil); err == nil {
		t.Fatal("nil evaluation accepted")
	}
}
