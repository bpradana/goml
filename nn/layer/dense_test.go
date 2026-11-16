package layer

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func TestNewDense_SeededInitializationIsReproducible(t *testing.T) {
	a, err := NewDense(3, 2, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	b, err := NewDense(3, 2, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	for i, p := range a.Parameters() {
		if !slices.Equal(backendtest.HostData(t, p.Value().Data()), backendtest.HostData(t, b.Parameters()[i].Value().Data())) {
			t.Fatal("seeded parameters differ")
		}
	}
}

func TestDense_ForwardAndParameterGradients(t *testing.T) {
	dense, err := NewDense(2, 1, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	weights, err := tensor.From([]float32{2, 3}, 2, 1)
	if err != nil {
		t.Fatal(err)
	}

	bias, err := tensor.From([]float32{1}, 1)
	if err != nil {
		t.Fatal(err)
	}

	if err := dense.weight.Update(weights); err != nil {
		t.Fatal(err)
	}

	if err := dense.bias.Update(bias); err != nil {
		t.Fatal(err)
	}

	data, err := tensor.From([]float32{4, 5}, 1, 2)
	if err != nil {
		t.Fatal(err)
	}

	input, err := nn.NewValue(data, true)
	if err != nil {
		t.Fatal(err)
	}

	output, err := dense.Forward(input)
	if err != nil || backendtest.HostData(t, output.Data())[0] != 24 {
		t.Fatalf("forward: %v %v", output, err)
	}

	loss, err := nn.Sum(output)
	if err != nil {
		t.Fatal(err)
	}

	if err := loss.Backward(); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, dense.weight.Value().Grad()), []float32{4, 5}) ||
		backendtest.HostData(t, dense.bias.Value().Grad())[0] != 1 ||
		!slices.Equal(backendtest.HostData(t, input.Grad()), []float32{2, 3}) {
		t.Fatal("incorrect dense gradients")
	}
}

func TestNewDense_RejectsInvalidDimensionsAndGenerator(t *testing.T) {
	if _, err := NewDense(0, 1, true, rand.New(rand.NewPCG(1, 2))); err == nil {
		t.Fatal("expected dimension error")
	}

	if _, err := NewDense(1, 1, false, nil); err == nil {
		t.Fatal("expected generator error")
	}
}
