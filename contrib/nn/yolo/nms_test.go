package yolo

import (
	"math"
	"testing"
)

func TestSuppress_ClassAwareStableOrderingAndLimit(t *testing.T) {
	box := Box{Left: 0, Top: 0, Right: 10, Bottom: 10}
	candidates := []Detection{
		{Box: box, ClassID: 0, Confidence: 0.8, Label: "first"},
		{Box: box, ClassID: 0, Confidence: 0.7, Label: "duplicate"},
		{Box: box, ClassID: 1, Confidence: 0.8, Label: "other class"},
		{Box: Box{Left: 20, Top: 20, Right: 30, Bottom: 30}, ClassID: 0, Confidence: 0.6, Label: "separate"},
	}

	got := suppress(candidates, 0.45, 2)
	if len(got) != 2 || got[0].Label != "first" || got[1].Label != "other class" {
		t.Fatal(got)
	}

	if got := suppress(nil, 0.45, 300); got == nil || len(got) != 0 {
		t.Fatal("empty result must be non-nil")
	}
}

func TestIntersectionOverUnion_OverlapDisjointAndDegenerate(t *testing.T) {
	for _, test := range []struct {
		a, b Box
		want float32
	}{
		{Box{0, 0, 2, 2}, Box{1, 1, 3, 3}, 1.0 / 7},
		{Box{0, 0, 2, 2}, Box{0, 0, 2, 2}, 1},
		{Box{0, 0, 1, 1}, Box{1, 1, 2, 2}, 0},
		{Box{}, Box{}, 0},
	} {
		if got := intersectionOverUnion(test.a, test.b); math.Abs(float64(got-test.want)) > 1e-6 {
			t.Fatal(got, test.want)
		}
	}
}
