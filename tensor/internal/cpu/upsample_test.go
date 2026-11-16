package cpu_test

import (
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
	"github.com/bpradana/goml/tensor/internal/cpu"
)

func TestResize_Conformance(t *testing.T) {
	device := cpu.Device{}
	backendtest.RunResize(t, device)
	backendtest.RunResizeBackward(t, device)
}
