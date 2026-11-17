package data_test

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/tensor"
)

func TestArray_SourceClosureAndConcurrentCopies(t *testing.T) {
	source, _ := tensor.From([]float32{1, 2, 3, 4}, 2, 2)
	labels := []int{1, 0}

	ds, err := data.NewClassificationDataset(source, labels)
	if err != nil {
		t.Fatal(err)
	}

	_ = source.Close()
	labels[0] = 99

	var workers sync.WaitGroup
	for range 8 {
		workers.Go(func() {
			for range 10 {
				sample, err := ds.Get(context.Background(), 0)
				if err != nil || sample.Input[0] != 1 || sample.Label != 1 {
					t.Errorf("bad copied sample: %+v %v", sample, err)
					return
				}

				sample.Input[0] = 99
				sample.Shape[0] = 0
			}
		})
	}

	workers.Wait()

	for _, i := range []int{-1, 2} {
		if _, err := ds.Get(context.Background(), i); err == nil {
			t.Fatal("invalid index accepted")
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := ds.Get(ctx, 0); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}

func TestArray_RegressionAndInvalidSampleCounts(t *testing.T) {
	inputs, _ := tensor.Zeros(3, 2)
	targets, _ := tensor.From([]float32{1, 2, 3}, 3)

	defer func() {
		_ = inputs.Close()
		_ = targets.Close()
	}()

	ds, err := data.NewRegressionDataset(inputs, targets)
	if err != nil {
		t.Fatal(err)
	}

	sample, err := ds.Get(context.Background(), 2)
	if err != nil || sample.Target[0] != 3 || len(sample.TargetShape) != 0 {
		t.Fatal("incorrect regression sample", err)
	}

	if _, err := data.NewClassificationDataset(inputs, []int{1}); err == nil {
		t.Fatal("accepted mismatched labels")
	}

	if _, err := data.NewRegressionDataset(inputs, tensor.Scalar(1)); err == nil {
		t.Fatal("accepted scalar dataset")
	}

	if _, err := data.NewClassificationDataset(nil, nil); err == nil {
		t.Fatal("accepted nil tensor")
	}
}

func TestArray_NoncontiguousInputsAndEmptyLeadingDimension(t *testing.T) {
	original, err := tensor.From([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = original.Close() }()

	view, err := original.Transpose()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = view.Close() }()

	ds, err := data.NewClassificationDataset(view, []int{0, 1, 2})
	if err != nil {
		t.Fatal(err)
	}

	sample, err := ds.Get(context.Background(), 1)
	if err != nil || !slices.Equal(sample.Input, []float32{2, 5}) {
		t.Fatal(sample, err)
	}

	empty, err := tensor.Zeros(0, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = empty.Close() }()

	emptyDataset, err := data.NewClassificationDataset(empty, nil)
	if err != nil || emptyDataset.Len() != 0 {
		t.Fatal(emptyDataset, err)
	}
}
