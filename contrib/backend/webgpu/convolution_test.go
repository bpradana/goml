package webgpu

import (
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
)

func TestConv2D_NCHWBiasStrideAndPadding(t *testing.T) {
	backendtest.RunConvolution(t, openDevice(t))
}

func TestConv2DBackward_InputAndWeightDerivatives(t *testing.T) {
	backendtest.RunConvolutionBackward(t, openDevice(t))
}
