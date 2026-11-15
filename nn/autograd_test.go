package nn

import (
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor"
)

func TestBackward_AccumulatesSharedDependencies(t *testing.T) {
	input := testValue(t, []float32{2}, nil, true)

	square, err := Mul(input, input)
	if err != nil {
		t.Fatal(err)
	}

	output, err := Add(square, square)
	if err != nil {
		t.Fatal(err)
	}

	if err := output.Backward(); err != nil {
		t.Fatal(err)
	}

	if got := input.Grad().Data()[0]; got != 8 {
		t.Fatalf("got %v, want 8", got)
	}

	if output.parents != nil || output.backward != nil || square.parents != nil {
		t.Fatal("backward retained saved graph resources")
	}

	if err := output.Backward(); err == nil {
		t.Fatal("expected consumed graph error")
	}
}

func TestBackward_AccumulatesAcrossForwardPassesUntilZeroGrad(t *testing.T) {
	input := testValue(t, []float32{2}, nil, true)

	for iteration := 0; iteration < 2; iteration++ {
		output, err := Mul(input, input)
		if err != nil {
			t.Fatal(err)
		}

		if err := output.Backward(); err != nil {
			t.Fatal(err)
		}
	}

	if got := input.Grad().Data()[0]; got != 8 {
		t.Fatalf("got %v, want accumulated gradient 8", got)
	}

	input.ZeroGrad()

	if input.Grad() != nil {
		t.Fatal("ZeroGrad retained a gradient")
	}
}

func TestBackward_RejectsConsumedSharedGraph(t *testing.T) {
	input := testValue(t, []float32{2}, nil, true)

	shared, err := Mul(input, input)
	if err != nil {
		t.Fatal(err)
	}

	first, err := Add(shared, input)
	if err != nil {
		t.Fatal(err)
	}

	second, err := Mul(shared, input)
	if err != nil {
		t.Fatal(err)
	}

	if err := first.Backward(); err != nil {
		t.Fatal(err)
	}

	before := input.Grad().Data()

	if err := second.Backward(); err == nil {
		t.Fatal("expected consumed dependency error")
	}

	if !slices.Equal(before, input.Grad().Data()) {
		t.Fatal("failed backward changed accumulated gradients")
	}
}

func TestBackwardWith_ValidatesSeedAndSupportsVectorOutputs(t *testing.T) {
	input := testValue(t, []float32{2, 3}, []int{2}, true)

	output, err := Mul(input, input)
	if err != nil {
		t.Fatal(err)
	}

	if err := output.Backward(); err == nil {
		t.Fatal("expected non-scalar output error")
	}

	if err := output.BackwardWith(tensor.Scalar(1)); err == nil {
		t.Fatal("expected seed shape error")
	}

	seed, err := tensor.From([]float32{2, 1}, 2)
	if err != nil {
		t.Fatal(err)
	}

	if err := output.BackwardWith(seed); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(input.Grad().Data(), []float32{8, 6}) {
		t.Fatal(input.Grad())
	}
}
