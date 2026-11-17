// Package meanbias demonstrates an external, typed metric implementation.
// Positive bias means overprediction; negative bias means underprediction.
package meanbias

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/metrics"
)

// Metric accumulates a Float64 mean signed error with constant memory.
// Updates require caller synchronization and never retain sample slices.
type Metric struct {
	count uint64
	mean  float64
}

// Result is this custom metric's report, independent of goml's built-in reports.
type Result struct {
	Count    uint64
	MeanBias float64
}

// New constructs an empty custom metric. The zero value is usable too.
func New() *Metric { return &Metric{} }

// Update validates the complete batch before updating an online Float64 mean.
// Invalid lengths, nonfinite values, and counter overflow leave state unchanged.
func (m *Metric) Update(predictions, targets []float32) error {
	if len(predictions) != len(targets) || uint64(len(targets)) > ^uint64(0)-m.count {
		return fmt.Errorf("meanbias: length mismatch or counter overflow")
	}

	for index, prediction := range predictions {
		if math.IsNaN(float64(prediction)) || math.IsInf(float64(prediction), 0) || math.IsNaN(float64(targets[index])) || math.IsInf(float64(targets[index]), 0) {
			return fmt.Errorf("meanbias: nonfinite observation %d", index)
		}
	}

	for index, prediction := range predictions {
		m.count++
		difference := float64(prediction) - float64(targets[index])
		m.mean += (difference - m.mean) / float64(m.count)
	}

	return nil
}

// Result returns a concrete typed snapshot or an empty-result error.
func (m *Metric) Result() (Result, error) {
	if m.count == 0 {
		return Result{}, fmt.Errorf("meanbias: no observations")
	}

	return Result{Count: m.count, MeanBias: m.mean}, nil
}

// Reset clears observations without invalidating previous result values.
func (m *Metric) Reset() { *m = Metric{} }

var _ metrics.Metric[float32, float32, Result] = (*Metric)(nil)
