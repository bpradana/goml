package tensor

import (
	"math"
	"slices"
	"testing"
)

func TestReductions_StridedAndContiguousGroups(t *testing.T) {
	a := fromTest(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 2, 2, 3)

	for _, strided := range []bool{false, true} {
		input := a
		if strided {
			input, _ = input.Transpose(2, 0, 1)
		}

		for _, axes := range [][]int{nil, {0}, {1}, {2}, {0, 2}, {1, 2}} {
			reduced := make([]bool, 3)
			for _, axis := range axes {
				reduced[axis] = true
			}

			if len(axes) == 0 {
				reduced = []bool{true, true, true}
			}

			for _, operation := range []struct {
				name string
				fn   func(*Tensor, ...int) (*Tensor, error)
			}{{"sum", Sum}, {"mean", Mean}, {"max", Max}} {
				got, err := operation.fn(input, axes...)
				if err != nil {
					t.Fatal(err)
				}

				want, counts := make([]float32, got.Numel()), make([]int, got.Numel())

				for i := range input.shape[0] {
					for j := range input.shape[1] {
						for k := range input.shape[2] {
							index := 0

							for axis, coordinate := range []int{i, j, k} {
								if !reduced[axis] {
									index = index*input.shape[axis] + coordinate
								}
							}

							value, _ := input.At(i, j, k)
							if operation.name == "max" {
								want[index] = max(want[index], value)
							} else {
								want[index] += value
							}

							counts[index]++
						}
					}
				}

				if operation.name == "mean" {
					for i := range want {
						want[i] /= float32(counts[i])
					}
				}

				if !slices.Equal(got.Data(), want) {
					t.Fatal(strided, axes, operation.name, got, want)
				}
			}
		}
	}
}

func TestSum_ReducesSelectedAxis(t *testing.T) {
	a := fromTest(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)

	result, err := Sum(a, 0)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, result, []int{3}, []float32{5, 7, 9})
}

func TestMean_ReducesSelectedAxis(t *testing.T) {
	a := fromTest(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)

	result, err := Mean(a, 1)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, result, []int{2}, []float32{2, 5})
}

func TestMax_NoAxesReducesAllElements(t *testing.T) {
	a := fromTest(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)

	result, err := Max(a)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, result, nil, []float32{6})
}

func TestSum_EmptyGroupsYieldZero(t *testing.T) {
	a := fromTest(t, nil, 0, 3)

	result, err := Sum(a, 0)
	if err != nil {
		t.Fatal(err)
	}

	assertData(t, result, []int{3}, []float32{0, 0, 0})
}

func TestMean_RejectsEmptyGroups(t *testing.T) {
	if _, err := Mean(fromTest(t, nil, 0, 3), 0); err == nil {
		t.Fatal("expected empty reduction error")
	}
}

func TestMax_RejectsEmptyGroups(t *testing.T) {
	if _, err := Max(fromTest(t, nil, 0, 3), 0); err == nil {
		t.Fatal("expected empty reduction error")
	}
}

func TestSum_RejectsInvalidAxes(t *testing.T) {
	a := fromTest(t, []float32{1, 2, 3, 4, 5, 6}, 2, 3)

	for _, tt := range []struct {
		name string
		axes []int
	}{
		{"DuplicateAxis", []int{0, 0}},
		{"NegativeAxis", []int{-1}},
		{"AxisOutsideRank", []int{2}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := Sum(a, tt.axes...); err == nil {
				t.Fatal("expected invalid axes error")
			}
		})
	}
}

func TestMax_PropagatesNaN(t *testing.T) {
	a := fromTest(t, []float32{1, float32(math.NaN())}, 2)

	result, err := Max(a)
	if err != nil || !math.IsNaN(float64(result.Data()[0])) {
		t.Fatalf("got %v, want NaN: %v", result, err)
	}
}

func TestReductions_EmptyOutputDoesNotOverflowGroupSize(t *testing.T) {
	large := int(^uint(0)>>1)/2 + 1
	a := fromTest(t, nil, large, 0, large)

	for _, tt := range []struct {
		name      string
		operation func(*Tensor, ...int) (*Tensor, error)
	}{
		{"Sum", Sum},
		{"Mean", Mean},
		{"Max", Max},
	} {
		t.Run(tt.name, func(t *testing.T) {
			result, err := tt.operation(a, 0, 2)
			if err != nil {
				t.Fatal(err)
			}

			assertData(t, result, []int{0}, nil)
		})
	}
}
