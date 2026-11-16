//go:build darwin && cgo

package metal_test

import (
	"math"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/tensor"
)

func TestCheckFinite_RejectsNaNInfinityAndLeavesDeviceUsable(t *testing.T) {
	device := openDevice(t)

	for _, value := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
		gpu, err := tensor.Scalar(value).To(device)
		if err != nil {
			t.Fatal(err)
		}

		if err := tensor.CheckFinite(gpu); err == nil {
			t.Fatal("nonfinite value accepted")
		}

		if err := gpu.Close(); err != nil {
			t.Fatal(err)
		}
	}

	valid, err := tensor.Scalar(3).To(device)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = valid.Close() }()

	if err := tensor.CheckFinite(valid); err != nil {
		t.Fatal(err)
	}
}

func TestCheckFinite_Conformance(t *testing.T) { backendtest.RunFinite(t, openDevice(t)) }
