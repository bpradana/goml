package nn

import (
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
)

func testBinaryGradients(t *testing.T, operation func(*Value, *Value) (*Value, error)) {
	t.Helper()

	t.Run("LeftOperandBroadcastGradient", func(t *testing.T) {
		checkGradient(t, []float32{0.5, 1.5, 2.5, 3.5}, []int{2, 2}, func(a *Value) (*Value, error) {
			b := testValue(t, []float32{1.2, 2.3}, []int{2}, false)
			return operation(a, b)
		})
	})

	t.Run("RightOperandBroadcastGradient", func(t *testing.T) {
		checkGradient(t, []float32{1.2, 2.3}, []int{2}, func(b *Value) (*Value, error) {
			a := testValue(t, []float32{0.5, 1.5, 2.5, 3.5}, []int{2, 2}, false)
			return operation(a, b)
		})
	})

	t.Run("RejectsNilInput", func(t *testing.T) {
		if _, err := operation(nil, nil); err == nil {
			t.Fatal("expected invalid input error")
		}
	})
}

func TestAdd_Gradients(t *testing.T) {
	testBinaryGradients(t, Add)
}

func TestSub_Gradients(t *testing.T) {
	testBinaryGradients(t, Sub)
}

func TestMul_Gradients(t *testing.T) {
	testBinaryGradients(t, Mul)
}

func TestDiv_Gradients(t *testing.T) {
	testBinaryGradients(t, Div)
}

func TestMatMul_LeftGradientMatchesFiniteDifferences(t *testing.T) {
	checkGradient(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, func(a *Value) (*Value, error) {
		b := testValue(t, []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int{3, 2}, false)
		return MatMul(a, b)
	})
}

func TestMatMul_RightGradientMatchesFiniteDifferences(t *testing.T) {
	checkGradient(t, []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int{3, 2}, func(b *Value) (*Value, error) {
		a := testValue(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, false)
		return MatMul(a, b)
	})
}

func TestAdd_EmptyBroadcastGradient(t *testing.T) {
	a := testValue(t, nil, []int{0, 3}, true)
	b := testValue(t, []float32{1, 2, 3}, []int{3}, true)

	output, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}

	loss, err := Sum(output)
	if err != nil {
		t.Fatal(err)
	}

	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	if a.Grad().Numel() != 0 || backendtest.HostData(t, b.Grad())[0] != 0 {
		t.Fatal("empty broadcast produced invalid gradients")
	}
}

func TestAdd_GradientsReduceMultipleSingletonAndLeadingAxes(t *testing.T) {
	a := testValue(t, []float32{1, 2, 3, 4, 5, 6}, []int{2, 1, 3}, true)
	b := testValue(t, []float32{1, 2, 3, 4}, []int{1, 4, 1}, true)

	result, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}

	loss, err := Sum(result)
	if err != nil {
		t.Fatal(err)
	}

	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	for _, gradient := range backendtest.HostData(t, a.Grad()) {
		if gradient != 4 {
			t.Fatalf("left gradient got %v, want 4", gradient)
		}
	}

	for _, gradient := range backendtest.HostData(t, b.Grad()) {
		if gradient != 6 {
			t.Fatalf("right gradient got %v, want 6", gradient)
		}
	}
}
