package metrics

import (
	"encoding/json"
	"fmt"
	"math"
)

// Regression accumulates scalar observations using compensated Float64 error
// sums and Welford target variance. The zero value is ready to use. Multi-output
// values are flattened: every scalar has equal weight, not every output column.
type Regression struct {
	count             uint64
	absolute, squared compensated
	mean, m2          float64
}

type compensated struct {
	value, correction float64
}

func (s *compensated) add(value float64) {
	y := value - s.correction
	total := s.value + y
	s.correction = (total - s.value) - y
	s.value = total
}

// RegressionResult reports mean errors over Count scalar observations. R2 is
// NaN and R2Valid false with fewer than two observations or zero target variance.
type RegressionResult struct {
	Count              uint64
	MAE, MSE, RMSE, R2 float64
	R2Valid            bool
}

// MarshalJSON preserves the existing field names and emits R2 as null when
// R2Valid is false. The Go value remains unchanged, including its undefined NaN.
// Other nonfinite fields, or a nonfinite R2 marked valid, still return JSON errors.
func (r RegressionResult) MarshalJSON() ([]byte, error) {
	type report RegressionResult

	var r2 *float64
	if r.R2Valid {
		r2 = &r.R2
	}

	return json.Marshal(struct {
		report
		R2 *float64
	}{report: report(r), R2: r2})
}

// NewRegression constructs an empty, reusable accumulator without external state.
func NewRegression() *Regression {
	return &Regression{}
}

// Update validates matching lengths and all finite values before changing state.
// It retains no input slices. Empty updates are valid. Counts cannot overflow.
func (m *Regression) Update(predictions, targets []float32) error {
	if len(predictions) != len(targets) || uint64(len(targets)) > ^uint64(0)-m.count {
		return fmt.Errorf("metrics: regression: count mismatch or counter overflow")
	}

	for index, prediction := range predictions {
		if !finite(prediction) || !finite(targets[index]) {
			return fmt.Errorf("metrics: regression: nonfinite value at observation %d", index)
		}
	}

	for index, prediction := range predictions {
		target := float64(targets[index])
		difference := float64(prediction) - target
		m.absolute.add(math.Abs(difference))
		m.squared.add(difference * difference)
		m.count++
		delta := target - m.mean
		m.mean += delta / float64(m.count)
		m.m2 += delta * (target - m.mean)
	}

	return nil
}

// Result returns mean errors or an explicit empty-result error. R2 validity is
// independent of MAE/MSE/RMSE validity. It never changes accumulator state.
func (m *Regression) Result() (RegressionResult, error) {
	if m.count == 0 {
		return RegressionResult{}, fmt.Errorf("metrics: regression: no observations")
	}

	result := RegressionResult{Count: m.count, MAE: m.absolute.value / float64(m.count), MSE: m.squared.value / float64(m.count), R2: math.NaN()}

	result.RMSE = math.Sqrt(result.MSE)
	if m.count >= 2 && m.m2 > 0 {
		result.R2Valid = true
		result.R2 = 1 - m.squared.value/m.m2
	}

	return result, nil
}

// Reset clears numerical state without invalidating previous results.
func (m *Regression) Reset() {
	*m = Regression{}
}
