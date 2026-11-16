//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
)

func TestReduce_Conformance(t *testing.T) {
	device := openDevice(t)
	backendtest.RunReduction(t, device)
}
