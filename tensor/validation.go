package tensor

import (
	"fmt"
	"math"
	"runtime"
)

// ValidationBackend checks numerical constraints on-device. Small validation
// status transfers are permitted; input tensor values must not be downloaded.
type ValidationBackend interface {
	CheckFinite(input DeviceView) error
}

// CheckFinite returns an error if any logical element is NaN or infinity.
// GPU implementations synchronize a validation status, without downloading the
// tensor. This explicit validation barrier preserves fallible loss contracts.
func CheckFinite(input *Tensor) error {
	defer runtime.KeepAlive(input)

	if err := valid("finite", input); err != nil {
		return err
	}

	if input.gpu != nil {
		backend, ok := input.Device().(ValidationBackend)
		if !ok {
			return fmt.Errorf("tensor: device lacks finite-value validation")
		}

		return backend.CheckFinite(input.deviceView())
	}

	for index := range input.size {
		value := float64(input.data[input.index(index)])
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return fmt.Errorf("tensor: nonfinite value at logical element %d", index)
		}
	}

	return nil
}
