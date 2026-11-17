package meanbias_test

import (
	"math"
	"testing"

	"github.com/bpradana/goml/metrics/examples/custom/meanbias"
)

func TestMetric_StreamingResultsAtomicFailureAndReset(t *testing.T) {
	metric := meanbias.New()
	if err := metric.Update([]float32{2, 4}, []float32{1, 3}); err != nil {
		t.Fatal(err)
	}

	if err := metric.Update([]float32{2.5}, []float32{3}); err != nil {
		t.Fatal(err)
	}

	before, err := metric.Result()
	if err != nil || before.Count != 3 || before.MeanBias != 0.5 {
		t.Fatal(before, err)
	}

	if err := metric.Update([]float32{1, float32(math.NaN())}, []float32{0, 1}); err == nil {
		t.Fatal("nonfinite batch accepted")
	}

	after, _ := metric.Result()
	if before != after {
		t.Fatal("failed update changed state")
	}

	metric.Reset()

	if _, err := metric.Result(); err == nil {
		t.Fatal("reset failed")
	}

	if before.Count != 3 {
		t.Fatal("snapshot changed")
	}
}
