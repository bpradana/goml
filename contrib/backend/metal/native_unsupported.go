//go:build !darwin || !cgo

package metal

import (
	"fmt"
	"unsafe"
)

func nativeOpen(int) (unsafe.Pointer, string, error) {
	return nil, "", fmt.Errorf("metal: requires macOS and CGo")
}

func nativeClose(unsafe.Pointer) {}

func nativeAllocate(unsafe.Pointer, uint64) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("metal: unsupported platform")
}

func nativeRelease(unsafe.Pointer) {}

func nativeUpload(unsafe.Pointer, unsafe.Pointer, []float32) error {
	return fmt.Errorf("metal: unsupported platform")
}

func nativeDownload(unsafe.Pointer, unsafe.Pointer, []float32) error {
	return fmt.Errorf("metal: unsupported platform")
}

func nativeSync(unsafe.Pointer) error {
	return fmt.Errorf("metal: unsupported platform")
}

func nativeStats(unsafe.Pointer) (uint64, uint64) {
	return 0, 0
}

func nativeCompute(unsafe.Pointer, string, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, kernelParams) error {
	return fmt.Errorf("metal: unsupported platform")
}

func nativeMatMul(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, [9]uint32, bool, bool) error {
	return fmt.Errorf("metal: unsupported platform")
}

func nativeConv(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, kernelParams, bool) error {
	return fmt.Errorf("metal: unsupported platform")
}
