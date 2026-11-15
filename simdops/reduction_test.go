package simdops

import (
	"math"
	"testing"
)

func TestDot_KnownValues(t *testing.T) {
	a := []float32{1, -2, 3}

	got, err := Dot(a, a)
	if err != nil || got != 14 {
		t.Fatalf("got %v, want 14: %v", got, err)
	}
}

func TestDot_EmptyInput(t *testing.T) {
	got, err := Dot(nil, nil)
	if err != nil || got != 0 {
		t.Fatalf("got %v, want zero: %v", got, err)
	}
}

func TestDot_LengthMismatch(t *testing.T) {
	if _, err := Dot([]float32{1}, nil); err == nil {
		t.Fatal("expected length error")
	}
}

func TestSum_KnownValues(t *testing.T) {
	if got := Sum([]float32{1, -2, 3}); got != 2 {
		t.Fatalf("got %v, want 2", got)
	}
}

func TestSum_EmptyInput(t *testing.T) {
	if got := Sum(nil); got != 0 {
		t.Fatalf("got %v, want zero", got)
	}
}

func TestMax_KnownValues(t *testing.T) {
	got, err := Max([]float32{1, -2, 3})
	if err != nil || got != 3 {
		t.Fatalf("got %v, want 3: %v", got, err)
	}
}

func TestMax_EmptyInput(t *testing.T) {
	if _, err := Max(nil); err == nil {
		t.Fatal("expected empty input error")
	}
}

func TestMax_PropagatesNaN(t *testing.T) {
	for _, a := range [][]float32{
		{float32(math.NaN()), 1, 2},
		{1, float32(math.NaN()), 2},
		{1, 2, float32(math.NaN())},
	} {
		got, err := Max(a)
		if err != nil || !math.IsNaN(float64(got)) {
			t.Fatalf("got %v, want NaN: %v", got, err)
		}
	}
}

func TestMax_PrefersPositiveZero(t *testing.T) {
	got, err := Max([]float32{float32(math.Copysign(0, -1)), 0})
	if err != nil || got != 0 || math.Signbit(float64(got)) {
		t.Fatalf("got %v, want positive zero: %v", got, err)
	}
}
