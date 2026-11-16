package tensor_test

import (
	"testing"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backendtest"
)

func TestCPU_Conformance(t *testing.T) {
	device := tensor.CPU()
	backendtest.RunTensor(t, device)
}
