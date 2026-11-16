package cpu_test

import (
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
	"github.com/bpradana/goml/tensor/internal/cpu"
)

func TestReduce_Conformance(t *testing.T) {
	device := cpu.Device{}
	backendtest.RunReduction(t, device)
}
