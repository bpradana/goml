package cpu_test

import (
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
	"github.com/bpradana/goml/tensor/internal/cpu"
)

func TestCopy_Conformance(t *testing.T) {
	device := cpu.Device{}
	backendtest.RunCopy(t, device)
	backendtest.RunFill(t, device)
}

func TestConcat_Conformance(t *testing.T) { backendtest.RunConcat(t, cpu.Device{}) }

func TestCheckFinite_Conformance(t *testing.T) { backendtest.RunFinite(t, cpu.Device{}) }
