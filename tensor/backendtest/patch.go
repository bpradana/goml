package backendtest

import (
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

// RunPatch checks extraction and accumulation of overlapping NCHW patches.
func RunPatch(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.PatchBackend)
	if !ok {
		t.Fatal("patch suite requires PatchBackend")
	}

	t.Run("Patch/ExtractAndScatter", func(t *testing.T) {
		input := view(t, device, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)
		patches := view(t, device, nil, 4, 4)

		options := backend.Conv2DOptions{Kernel: 2, Stride: 1}
		if err := compute.Im2Col(patches, input, options); err != nil {
			t.Fatal(err)
		}

		compare(t, device, patches, []float32{1, 2, 4, 5, 2, 3, 5, 6, 4, 5, 7, 8, 5, 6, 8, 9}, 0)

		output := view(t, device, nil, 1, 1, 3, 3)
		if err := compute.Col2Im(output, patches, options); err != nil {
			t.Fatal(err)
		}

		compare(t, device, output, []float32{1, 4, 3, 8, 20, 12, 7, 16, 9}, 0)
	})
}
