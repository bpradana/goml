package nn

import "testing"

func TestReshape_GradientMatchesFiniteDifferences(t *testing.T) {
	checkGradient(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, func(a *Value) (*Value, error) {
		return Reshape(a, 3, 2)
	})
}

func TestTranspose_InversePermutationGradient(t *testing.T) {
	checkGradient(t, []float32{1, 2, 3, 4, 5, 6}, []int{1, 2, 3}, func(a *Value) (*Value, error) {
		transposed, err := Transpose(a, 2, 0, 1)
		if err != nil {
			return nil, err
		}

		weights := testValue(t, []float32{1, 2, 3, 4, 5, 6}, []int{3, 1, 2}, false)

		return Mul(transposed, weights)
	})
}
