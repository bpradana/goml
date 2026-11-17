//go:build !cgo || android || ios || (!darwin && !linux && !windows) || (!amd64 && !arm64)

package webgpu

import (
	"fmt"

	"github.com/bpradana/goml/tensor/backend"
)

type nativeBuffer struct{}

type nativeDevice struct {
	failure failureState
}

func openNative(int) (*nativeDevice, string, error) {
	return nil, "", fmt.Errorf("webgpu: %w: native execution requires CGo and desktop amd64 or arm64", backend.ErrUnsupported)
}

func (*nativeDevice) allocate(uint64) (*nativeBuffer, error) {
	return nil, backend.ErrUnsupported
}

func (*nativeDevice) upload(*nativeBuffer, []float32) error {
	return backend.ErrUnsupported
}

func (*nativeDevice) download(*nativeBuffer, []float32) error {
	return backend.ErrUnsupported
}

func (*nativeDevice) dispatch(string, kernelParams, [4]*nativeBuffer, uint32, uint32) error {
	return backend.ErrUnsupported
}

func (*nativeDevice) synchronize() error {
	return backend.ErrUnsupported
}

func (*nativeDevice) retire(*nativeBuffer) {}

func (*nativeDevice) stats() Stats {
	return Stats{}
}

func (*nativeDevice) close() {}
