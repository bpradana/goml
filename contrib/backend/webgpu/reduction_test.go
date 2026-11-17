package webgpu

import (
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
)

func TestReduction_AxesEmptySumsAndNaNPropagation(t *testing.T) {
	backendtest.RunReduction(t, openDevice(t))
}
