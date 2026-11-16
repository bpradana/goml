package cpu_test

import (
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
	"github.com/bpradana/goml/tensor/internal/cpu"
)

func TestConv2DBackward_Conformance(t *testing.T) {
	device := cpu.Device{}
	backendtest.RunConvolutionBackward(t, device)
}
