package webgpu

import (
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
)

func TestCheckFinite_LogicalViewsAndNonfiniteValues(t *testing.T) {
	backendtest.RunFinite(t, openDevice(t))
}
