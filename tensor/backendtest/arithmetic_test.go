package backendtest_test

import (
	"testing"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/tensor/backendtest"
)

type expOnly struct{ backend.Device }

func (d expOnly) Unary(op backend.UnaryOp, dst, src backend.View) error {
	if op != backend.Exp {
		return backend.ErrUnsupported
	}

	return d.Device.(backend.UnaryBackend).Unary(op, dst, src)
}

func TestRunUnary_PartialFamilyCanDeclareSupportedOperations(t *testing.T) {
	device := expOnly{Device: tensor.CPU()}
	backendtest.RunDevice(t, device)
	backendtest.RunUnary(t, device, backend.Exp)
}
