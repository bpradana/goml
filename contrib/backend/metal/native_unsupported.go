//go:build !darwin || !cgo

package metal

import (
	"fmt"
	"unsafe"

	"github.com/bpradana/goml/tensor/backend"
)

func nativeOpen(int) (unsafe.Pointer, string, error) {
	return nil, "", fmt.Errorf("%w: Metal requires macOS and CGo", backend.ErrUnsupported)
}

func nativeClose(unsafe.Pointer) {}

func nativeAllocate(unsafe.Pointer, uint64) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("%w: Metal platform", backend.ErrUnsupported)
}

func nativeRelease(unsafe.Pointer) {}

func nativeUpload(unsafe.Pointer, unsafe.Pointer, []float32) error {
	return fmt.Errorf("%w: Metal platform", backend.ErrUnsupported)
}

func nativeDownload(unsafe.Pointer, unsafe.Pointer, []float32) error {
	return fmt.Errorf("%w: Metal platform", backend.ErrUnsupported)
}

func nativeSync(unsafe.Pointer) error {
	return fmt.Errorf("%w: Metal platform", backend.ErrUnsupported)
}

func nativeStats(unsafe.Pointer) (uint64, uint64) {
	return 0, 0
}

func nativeCompute(unsafe.Pointer, string, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, kernelParams) error {
	return fmt.Errorf("%w: Metal platform", backend.ErrUnsupported)
}

func nativeMatMul(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, [9]uint32, bool, bool) error {
	return fmt.Errorf("%w: Metal platform", backend.ErrUnsupported)
}

func nativeConv(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, kernelParams, bool) error {
	return fmt.Errorf("%w: Metal platform", backend.ErrUnsupported)
}
