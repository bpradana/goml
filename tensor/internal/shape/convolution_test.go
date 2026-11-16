package shape

import "testing"

func TestConvolution_RectangularPaddedGeometry(t *testing.T) {
	h, w, features, err := Convolution([]int{2, 3, 5, 7}, 3, 2, 1)
	if err != nil || h != 3 || w != 4 || features != 27 {
		t.Fatalf("got %d %d %d: %v", h, w, features, err)
	}
}

func TestConvolution_RejectsInvalidAndOverflowingGeometry(t *testing.T) {
	for _, test := range []struct {
		shape                   []int
		kernel, stride, padding int
	}{
		{[]int{1, 0, 5, 7}, 3, 1, 1},
		{[]int{1, 3, 0, 7}, 3, 1, 1},
		{[]int{1, 3, 5, 7}, 0, 1, 1},
		{[]int{1, 3, 5, 7}, 3, 0, 1},
		{[]int{1, 3, 5, 7}, 3, 1, -1},
		{[]int{1, 3, 5, 7}, 3, 1, int(^uint(0) >> 1)},
	} {
		if _, _, _, err := Convolution(test.shape, test.kernel, test.stride, test.padding); err == nil {
			t.Fatalf("accepted %+v", test)
		}
	}
}
