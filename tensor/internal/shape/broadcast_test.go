package shape

import (
	"slices"
	"testing"
)

func TestBroadcastShapes(t *testing.T) {
	tests := []struct {
		a, b, want []int
	}{
		{[]int{32, 128}, []int{128}, []int{32, 128}},
		{[]int{10, 1, 64}, []int{1, 64}, []int{10, 1, 64}},
		{[]int{4, 8}, nil, []int{4, 8}},
		{[]int{0, 3}, []int{1, 3}, []int{0, 3}},
	}

	for _, tt := range tests {
		got, err := Broadcast(tt.a, tt.b)
		if err != nil || !slices.Equal(got, tt.want) {
			t.Fatalf("%v and %v: got %v, want %v: %v", tt.a, tt.b, got, tt.want, err)
		}
	}

	if _, err := Broadcast([]int{0}, []int{2}); err == nil {
		t.Fatal("zero and two cannot broadcast")
	}
}

func FuzzBroadcast(f *testing.F) {
	f.Add(uint8(2), uint8(3), uint8(1), uint8(3))

	f.Fuzz(func(t *testing.T, a, b, c, d uint8) {
		left, right := []int{int(a % 8), int(b % 8)}, []int{int(c % 8), int(d % 8)}

		shape, err := Broadcast(left, right)
		if err != nil {
			return
		}

		for i, n := range shape {
			if n != left[i] && left[i] != 1 {
				t.Fatal("invalid left broadcast")
			}

			if n != right[i] && right[i] != 1 {
				t.Fatal("invalid right broadcast")
			}
		}
	})
}
