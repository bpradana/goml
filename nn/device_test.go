package nn

import (
	"testing"

	"github.com/bpradana/goml/tensor"
	examplebackend "github.com/bpradana/goml/tensor/examples/backend"
)

func TestToDevice_RejectsCrossDeviceTransferOfAnActiveGraph(t *testing.T) {
	device, err := examplebackend.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	parameter, err := NewParameter(tensor.Scalar(3))
	if err != nil {
		t.Fatal(err)
	}

	product, err := Mul(parameter.Value(), parameter.Value())
	if err != nil {
		t.Fatal(err)
	}

	if err := ToDevice([]*Parameter{parameter}, device); err == nil {
		t.Fatal("active parameter graph moved to another context")
	}

	if !parameter.Value().Data().IsCPU() {
		t.Fatal("failed transfer changed parameter storage")
	}

	if err := product.Backward(); err != nil {
		t.Fatal(err)
	}

	if err := ToDevice([]*Parameter{parameter}, device); err != nil {
		t.Fatal(err)
	}

	if parameter.Value().Data().Device().ID() != device.ID() || parameter.Value().Grad() != nil {
		t.Fatal("successful transfer lost identity or retained old-device gradients")
	}
}
