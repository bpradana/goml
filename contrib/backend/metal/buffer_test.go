//go:build darwin && cgo

package metal_test

import (
	"testing"

	"github.com/bpradana/goml/tensor"
)

func TestBuffer_RejectsForeignBuffersAndClosedDevice(t *testing.T) {
	first := openDevice(t)
	second := openDevice(t)

	buffer, err := first.Allocate(1)
	if err != nil {
		t.Fatal(err)
	}

	if err := second.Upload(buffer, []float32{1}); err == nil {
		t.Fatal("foreign buffer accepted")
	}

	value, err := tensor.Scalar(1).To(first)
	if err != nil {
		t.Fatal(err)
	}

	if err := first.Close(); err != nil {
		t.Fatal(err)
	}

	if _, err := value.HostData(); err == nil {
		t.Fatal("closed device remained readable")
	}
}
