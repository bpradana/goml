package metrics_test

import (
	"math"
	"reflect"
	"testing"

	"github.com/bpradana/goml/metrics"
)

func TestClassification_IndependentMulticlassAveragesAndOrientation(t *testing.T) {
	metric, err := metrics.NewClassification(3)
	if err != nil {
		t.Fatal(err)
	}

	if err := metric.Update([]int{0, 0, 0, 1, 1, 2}, []int{0, 0, 0, 0, 1, 1}); err != nil {
		t.Fatal(err)
	}

	got, err := metric.Result()
	if err != nil {
		t.Fatal(err)
	}

	for name, pair := range map[string][2]float64{
		"accuracy": {got.Accuracy, 2.0 / 3}, "micro_precision": {got.PrecisionMicro, 2.0 / 3},
		"micro_recall": {got.RecallMicro, 2.0 / 3}, "micro_f1": {got.F1Micro, 2.0 / 3},
		"macro_precision": {got.PrecisionMacro, 0.5}, "macro_recall": {got.RecallMacro, 5.0 / 12},
		"macro_f1": {got.F1Macro, 19.0 / 42}, "weighted_precision": {got.PrecisionWeighted, 5.0 / 6},
		"weighted_recall": {got.RecallWeighted, 2.0 / 3}, "weighted_f1": {got.F1Weighted, 31.0 / 42},
	} {
		if math.Abs(pair[0]-pair[1]) > 1e-14 {
			t.Errorf("%s: got %g want %g", name, pair[0], pair[1])
		}
	}

	if got.Count != 6 || !reflect.DeepEqual(got.Confusion, [][]uint64{{3, 1, 0}, {0, 1, 1}, {0, 0, 0}}) {
		t.Fatal(got)
	}

	if got.PerClass[0].Support != 4 || got.PerClass[2].Precision != 0 || got.PerClass[2].Recall != 0 || got.PerClass[2].F1 != 0 {
		t.Fatal(got.PerClass)
	}
}

func TestClassification_PartitionsSnapshotsResetAndBinaryExtremes(t *testing.T) {
	for _, predictions := range [][]int{{0, 1, 0, 1}, {1, 0, 1, 0}} {
		whole, _ := metrics.NewClassification(2)
		stream, _ := metrics.NewClassification(2)

		targets := []int{0, 1, 0, 1}
		if err := whole.Update(predictions, targets); err != nil {
			t.Fatal(err)
		}

		for i := range predictions {
			if err := stream.Update(predictions[i:i+1], targets[i:i+1]); err != nil {
				t.Fatal(err)
			}
		}

		want, _ := whole.Result()

		got, _ := stream.Result()
		if !reflect.DeepEqual(got, want) {
			t.Fatal("partition-dependent report")
		}

		got.Confusion[0][0] = 99
		got.PerClass[0].Support = 99

		unchanged, _ := stream.Result()
		if !reflect.DeepEqual(unchanged, want) {
			t.Fatal("snapshot mutated state")
		}

		stream.Reset()

		if _, err := stream.Result(); err == nil {
			t.Fatal("empty reset succeeded")
		}

		if want.Count != 4 {
			t.Fatal("reset changed report")
		}
	}
}

func TestClassification_InvalidUpdatesAreAtomicAndLogitTiesAreLowestIndex(t *testing.T) {
	metric, _ := metrics.NewClassification(3)
	if _, err := metric.Result(); err == nil {
		t.Fatal("empty result succeeded")
	}

	if err := metric.UpdateLogits([]float32{1, 1, 0, 0, 2, 2}, []int{0, 1}); err != nil {
		t.Fatal(err)
	}

	before, _ := metric.Result()
	if before.Accuracy != 1 {
		t.Fatal("unstable ties", before)
	}

	checks := []func() error{
		func() error { return metric.Update([]int{0}, nil) },
		func() error { return metric.Update([]int{0, 3}, []int{0, 1}) },
		func() error { return metric.Update([]int{0, 1}, []int{0, -1}) },
		func() error { return metric.UpdateLogits([]float32{1}, []int{0}) },
	}
	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		checks = append(checks, func() error { return metric.UpdateLogits([]float32{1, 0, 0, 0, 1, value}, []int{0, 1}) })
	}

	for _, check := range checks {
		if err := check(); err == nil {
			t.Fatal("invalid update succeeded")
		}

		after, _ := metric.Result()
		if !reflect.DeepEqual(before, after) {
			t.Fatal("failed update changed state")
		}
	}

	for _, classes := range []int{0, -1, int(^uint(0) >> 1)} {
		if _, err := metrics.NewClassification(classes); err == nil {
			t.Fatal("invalid class count accepted")
		}
	}
}

func BenchmarkClassification_UpdateLogits(b *testing.B) {
	metric, _ := metrics.NewClassification(10)
	logits, targets := make([]float32, 256*10), make([]int, 256)

	b.ReportAllocs()
	b.SetBytes(int64(len(logits) * 4))

	for b.Loop() {
		if err := metric.UpdateLogits(logits, targets); err != nil {
			b.Fatal(err)
		}
	}
}
