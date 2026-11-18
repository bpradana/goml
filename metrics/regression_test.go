package metrics_test

import (
	"bytes"
	"encoding/json"
	"math"
	"testing"

	"github.com/bpradana/goml/metrics"
)

func TestRegression_IndependentReferenceAndStreamingPartitions(t *testing.T) {
	predictions, targets := []float32{2.5, 0, 2, 8}, []float32{3, -0.5, 2, 7}

	for _, partition := range []int{1, 2, 4} {
		metric := metrics.NewRegression()
		for start := 0; start < len(targets); start += partition {
			if err := metric.Update(predictions[start:start+partition], targets[start:start+partition]); err != nil {
				t.Fatal(err)
			}
		}

		got, err := metric.Result()
		if err != nil || got.Count != 4 || !got.R2Valid {
			t.Fatal(got, err)
		}

		for _, pair := range [][2]float64{{got.MAE, 0.5}, {got.MSE, 0.375}, {got.RMSE, math.Sqrt(0.375)}, {got.R2, 443.0 / 467}} {
			if math.Abs(pair[0]-pair[1]) > 1e-14 {
				t.Fatalf("got %g want %g", pair[0], pair[1])
			}
		}

		metric.Reset()

		if _, err := metric.Result(); err == nil {
			t.Fatal("empty result succeeded")
		}

		if got.Count != 4 {
			t.Fatal("reset altered snapshot")
		}
	}
}

func TestRegression_WelfordLargeOffsetAndUndefinedR2(t *testing.T) {
	metric := metrics.NewRegression()
	if err := metric.Update([]float32{100000008, 100000016, 100000024}, []float32{100000000, 100000008, 100000016}); err != nil {
		t.Fatal(err)
	}

	got, _ := metric.Result()
	if !got.R2Valid || got.MSE != 64 || got.R2 != -0.5 {
		t.Fatal(got)
	}

	for _, targets := range [][]float32{{1}, {2, 2}, {-3, -3, -3}} {
		metric.Reset()

		if err := metric.Update(targets, targets); err != nil {
			t.Fatal(err)
		}

		got, _ := metric.Result()
		if got.MSE != 0 || got.MAE != 0 || got.R2Valid || !math.IsNaN(got.R2) {
			t.Fatal(got)
		}
	}

	metric.Reset()

	if err := metric.Update([]float32{math.MaxFloat32}, []float32{-math.MaxFloat32}); err != nil {
		t.Fatal(err)
	}

	got, _ = metric.Result()
	if math.IsInf(got.MSE, 0) || got.MSE <= 0 {
		t.Fatal(got)
	}

	metric.Reset()

	if err := metric.Update([]float32{1e-20}, []float32{0}); err != nil {
		t.Fatal(err)
	}

	got, _ = metric.Result()
	if got.MSE <= 0 {
		t.Fatal("small errors underflowed in Float32")
	}
}

func TestRegression_FailedUpdatesPreserveAllState(t *testing.T) {
	metric := metrics.NewRegression()
	if err := metric.Update([]float32{1, 2}, []float32{0, 3}); err != nil {
		t.Fatal(err)
	}

	before, _ := metric.Result()

	for _, invalid := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		for _, targetInvalid := range []bool{false, true} {
			p, y := []float32{2, 3}, []float32{1, 2}
			if targetInvalid {
				y[1] = invalid
			} else {
				p[1] = invalid
			}

			if err := metric.Update(p, y); err == nil {
				t.Fatal("nonfinite update accepted")
			}

			after, _ := metric.Result()
			if before != after {
				t.Fatal("failed update changed state")
			}
		}
	}

	if err := metric.Update(nil, []float32{1}); err == nil {
		t.Fatal("length mismatch accepted")
	}
}

func BenchmarkRegression_Update(b *testing.B) {
	metric := metrics.NewRegression()

	predictions, targets := make([]float32, 1024), make([]float32, 1024)
	for i := range targets {
		targets[i] = float32(i % 17)
	}

	b.ReportAllocs()
	b.SetBytes(int64(len(targets) * 8))

	for b.Loop() {
		if err := metric.Update(predictions, targets); err != nil {
			b.Fatal(err)
		}
	}
}

func TestRegressionResult_JSONPreservesNumericsAndNullUndefinedR2(t *testing.T) {
	for _, tc := range []struct {
		name                 string
		predictions, targets []float32
		valid                bool
		r2                   float64
	}{
		{"perfect", []float32{1, 2}, []float32{1, 2}, true, 1},
		{"negative", []float32{0, 0}, []float32{1, 2}, true, -9},
		{"constant", []float32{2, 3}, []float32{1, 1}, false, 0},
		{"single", []float32{2}, []float32{1}, false, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			metric := metrics.NewRegression()
			if err := metric.Update(tc.predictions, tc.targets); err != nil {
				t.Fatal(err)
			}

			result, err := metric.Result()
			if err != nil {
				t.Fatal(err)
			}

			encoded, err := json.Marshal(result)
			if err != nil {
				t.Fatal(err)
			}

			var decoded struct {
				Count          uint64
				MAE, MSE, RMSE float64
				R2             *float64
				R2Valid        bool
			}
			if err := json.Unmarshal(encoded, &decoded); err != nil {
				t.Fatal(err)
			}

			if decoded.Count != result.Count || decoded.MAE != result.MAE || decoded.MSE != result.MSE || decoded.RMSE != result.RMSE || decoded.R2Valid != tc.valid {
				t.Fatal("changed report fields", string(encoded))
			}

			if tc.valid {
				if decoded.R2 == nil || *decoded.R2 != tc.r2 || result.R2 != tc.r2 {
					t.Fatal(string(encoded), result)
				}
			} else if decoded.R2 != nil || !math.IsNaN(result.R2) || !bytes.Contains(encoded, []byte(`"R2":null`)) {
				t.Fatal("undefined R2 must remain NaN in Go and null in JSON", string(encoded), result)
			}
		})
	}
}

func TestRegressionResult_JSONRejectsOtherNonfiniteFields(t *testing.T) {
	for _, result := range []metrics.RegressionResult{
		{Count: 1, MSE: math.Inf(1)},
		{Count: 1, MAE: math.NaN()},
		{Count: 2, R2Valid: true, R2: math.NaN()},
	} {
		if _, err := json.Marshal(result); err == nil {
			t.Fatal("invalid finite metric marked serializable", result)
		}
	}
}
