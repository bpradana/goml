package nn

import "testing"

func TestSum_AxisGradientMatchesFiniteDifferences(t *testing.T) {
	checkGradient(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, func(a *Value) (*Value, error) {
		return Sum(a, 0)
	})
}

func TestMean_AxisGradientMatchesFiniteDifferences(t *testing.T) {
	checkGradient(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, func(a *Value) (*Value, error) {
		return Mean(a, 1)
	})
}

func TestMean_MultipleAxesGradientMatchesFiniteDifferences(t *testing.T) {
	checkGradient(t, []float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{2, 2, 2}, func(a *Value) (*Value, error) {
		return Mean(a, 2, 0)
	})
}

func TestMean_RejectsEmptyReduction(t *testing.T) {
	if _, err := Mean(testValue(t, nil, []int{0}, true)); err == nil {
		t.Fatal("expected empty mean error")
	}
}
