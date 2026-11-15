package nn

import "testing"

func TestScale_GradientMatchesFiniteDifferences(t *testing.T) {
	checkGradient(t, []float32{.2, -.4, .8, 1}, []int{2, 2}, func(x *Value) (*Value, error) { return Scale(x, -2.5) })

	if _, err := Scale(nil, 1); err == nil {
		t.Fatal("accepted nil")
	}
}
