package trainer_test

import (
	"context"
	"errors"
	"iter"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/trainer"
)

func denseModel(t testing.TB, in, out int) *layer.Sequential {
	t.Helper()

	dense, err := layer.NewDense(in, out, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := layer.NewSequential(dense)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		for _, p := range model.Parameters() {
			p.ZeroGrad()
			_ = p.Value().Data().Close()
		}
	})

	return model
}

func classLoader(t testing.TB, size int) *data.Loader[data.ClassificationSample, *data.ClassificationBatch] {
	t.Helper()

	inputs, err := tensor.From([]float32{-2, -1, 1, 2}, 4, 1)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = inputs.Close() }()

	dataset, err := data.NewClassificationDataset(inputs, []int{0, 0, 1, 1})
	if err != nil {
		t.Fatal(err)
	}

	loader, err := data.NewLoader(dataset, data.CollateClassification, data.LoaderOptions{BatchSize: size, Shuffle: true, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}

	return loader
}

type countingOptimizer struct {
	optimizer.Optimizer
	steps, clears int
	failure       error
	closed        bool
}

func (o *countingOptimizer) Step() error {
	o.steps++
	if o.failure != nil {
		return o.failure
	}

	return o.Optimizer.Step()
}

func (o *countingOptimizer) ZeroGrad() { o.clears++; o.Optimizer.ZeroGrad() }

func (o *countingOptimizer) Close() error { o.closed = true; return nil }

type observedClasses struct {
	source  trainer.BatchSource[*data.ClassificationBatch]
	batches []*data.ClassificationBatch
}

func (s *observedClasses) Batches(ctx context.Context, epoch int) iter.Seq2[*data.ClassificationBatch, error] {
	return func(yield func(*data.ClassificationBatch, error) bool) {
		for batch, err := range s.source.Batches(ctx, epoch) {
			if err == nil {
				s.batches = append(s.batches, batch)
			}

			if !yield(batch, err) {
				return
			}
		}
	}
}

func TestClassifier_RealTrainingConvergesAndBorrowedObjectsSurvive(t *testing.T) {
	for _, size := range []int{1, 3, 4} {
		model := denseModel(t, 1, 2)

		sgd, err := optimizer.NewSGD(model.Parameters(), 0.1)
		if err != nil {
			t.Fatal(err)
		}

		opt := &countingOptimizer{Optimizer: sgd}

		runner, err := trainer.NewClassifier(model, opt, tensor.CPU(), 2)
		if err != nil {
			t.Fatal(err)
		}

		loader := classLoader(t, size)
		source := &observedClasses{source: loader}

		before, err := runner.Evaluate(context.Background(), loader)
		if err != nil {
			t.Fatal(err)
		}

		history, err := runner.Fit(context.Background(), source, loader, trainer.FitOptions{Epochs: 20})
		if err != nil {
			t.Fatal(err)
		}

		last := history.Epochs[len(history.Epochs)-1]
		if len(history.Epochs) != 20 || last.Classification.Accuracy != 1 || last.Validation.MeanLoss >= before.MeanLoss*0.5 || last.Train.Samples != 4 {
			t.Fatalf("training did not converge: before=%+v after=%+v", before, last)
		}

		if opt.steps != 20*((4+size-1)/size) || opt.closed {
			t.Fatal("optimizer ownership/call count", opt.steps)
		}

		for _, batch := range source.batches {
			if !batch.Inputs.IsClosed() {
				t.Fatal("batch remained open")
			}
		}

		for _, parameter := range model.Parameters() {
			if parameter.Value().Data().IsClosed() || parameter.Value().Grad() != nil {
				t.Fatal("model closed or gradients retained")
			}
		}

		weights, _ := model.Parameters()[0].Value().Data().HostData()
		steps := opt.steps

		if _, err := runner.Evaluate(context.Background(), loader); err != nil {
			t.Fatal(err)
		}

		after, _ := model.Parameters()[0].Value().Data().HostData()
		if !slices.Equal(weights, after) || opt.steps != steps {
			t.Fatal("evaluation updated model")
		}
	}
}

func TestClassifier_FailedOptimizerClearsGradientsAndClosesBatch(t *testing.T) {
	model := denseModel(t, 1, 2)
	sgd, _ := optimizer.NewSGD(model.Parameters(), 0.1)
	wanted := errors.New("optimizer failed")
	opt := &countingOptimizer{Optimizer: sgd, failure: wanted}
	runner, _ := trainer.NewClassifier(model, opt, tensor.CPU(), 2)
	source := &observedClasses{source: classLoader(t, 3)}

	history, err := runner.Fit(context.Background(), source, nil, trainer.FitOptions{Epochs: 2})
	if !errors.Is(err, wanted) || len(history.Epochs) != 0 || opt.steps != 1 {
		t.Fatal(history, err, opt.steps)
	}

	for _, parameter := range model.Parameters() {
		if parameter.Value().Grad() != nil || parameter.Value().Data().IsClosed() {
			t.Fatal("failed step retained gradient or closed weights")
		}
	}

	if len(source.batches) != 1 || !source.batches[0].Inputs.IsClosed() {
		t.Fatal("failed batch not closed")
	}
}

type readFailureDataset struct {
	failure error
}

func (d readFailureDataset) Len() int {
	return 1
}

func (d readFailureDataset) Get(ctx context.Context, index int) (data.ClassificationSample, error) {
	return data.ClassificationSample{Input: []float32{1}, Shape: []int{1}, Label: 0}, d.failure
}

func TestClassifier_DatasetAndCollationFailuresDoNotStep(t *testing.T) {
	for _, kind := range []string{"read", "collate"} {
		t.Run(kind, func(t *testing.T) {
			wanted := errors.New(kind)
			dataset := readFailureDataset{}
			collate := data.CollateClassification

			if kind == "read" {
				dataset.failure = wanted
			} else {
				collate = func([]data.ClassificationSample) (*data.ClassificationBatch, error) {
					return nil, wanted
				}
			}

			loader, err := data.NewLoader(dataset, collate, data.LoaderOptions{BatchSize: 1})
			if err != nil {
				t.Fatal(err)
			}

			model := denseModel(t, 1, 2)

			sgd, err := optimizer.NewSGD(model.Parameters(), 0.1)
			if err != nil {
				t.Fatal(err)
			}

			opt := &countingOptimizer{Optimizer: sgd}

			runner, err := trainer.NewClassifier(model, opt, tensor.CPU(), 2)
			if err != nil {
				t.Fatal(err)
			}

			history, err := runner.Fit(context.Background(), loader, nil, trainer.FitOptions{Epochs: 2})
			if !errors.Is(err, wanted) || opt.steps != 0 || len(history.Epochs) != 0 {
				t.Fatal(history, err, opt.steps)
			}
		})
	}
}

func BenchmarkClassifier_Epoch(b *testing.B) {
	for _, evaluation := range []bool{false, true} {
		name := "Train"
		if evaluation {
			name = "Evaluate"
		}

		b.Run(name, func(b *testing.B) {
			model := denseModel(b, 1, 2)

			opt, err := optimizer.NewSGD(model.Parameters(), 0.1)
			if err != nil {
				b.Fatal(err)
			}

			runner, err := trainer.NewClassifier(model, opt, tensor.CPU(), 2)
			if err != nil {
				b.Fatal(err)
			}

			loader := classLoader(b, 3)
			b.ReportAllocs()
			b.ResetTimer()

			for range b.N {
				if evaluation {
					_, err = runner.Evaluate(context.Background(), loader)
				} else {
					_, err = runner.Fit(context.Background(), loader, nil, trainer.FitOptions{Epochs: 1})
				}

				if err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
