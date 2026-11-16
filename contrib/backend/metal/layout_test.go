//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
)

func TestLayout_Conformance(t *testing.T) {
	device := openDevice(t)
	backendtest.RunFill(t, device)
	backendtest.RunCopy(t, device)
}
