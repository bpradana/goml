package tensor

import (
	"runtime"

	"github.com/bpradana/goml/tensor/backend"
)

// CheckFinite validates every logical value without a full tensor download.
// Accelerator implementations may synchronize a small status flag.
func CheckFinite(input *Tensor) error {
	defer runtime.KeepAlive(input)

	if err := valid("finite", input); err != nil {
		return err
	}

	compute, ok := input.device.(backend.FiniteBackend)
	if !ok {
		return unsupported("finite", input.device)
	}

	return compute.CheckFinite(input.deviceView())
}
