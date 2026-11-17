//go:build darwin && cgo

package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph
#include "metal.h"
*/
import "C"

import (
	"embed"
	"fmt"
	"strings"
	"unsafe"
)

//go:embed kernels/*.metal
var shaders embed.FS

func nativeError(status C.int, message *C.char) error {
	if message != nil {
		defer C.free(unsafe.Pointer(message))

		return fmt.Errorf("metal: %s", C.GoString(message))
	}

	if status == 0 {
		return fmt.Errorf("metal: native operation failed")
	}

	return nil
}

func nativeOpen(index int) (unsafe.Pointer, string, error) {
	var library strings.Builder

	for _, name := range []string{"common", "layout", "arithmetic", "reduction", "spatial"} {
		source, err := shaders.ReadFile("kernels/" + name + ".metal")
		if err != nil {
			return nil, "", fmt.Errorf("metal: load %s shader: %w", name, err)
		}

		library.Write(source)
		library.WriteByte('\n')
	}

	source := C.CString(library.String())
	defer C.free(unsafe.Pointer(source))

	var message *C.char

	handle := C.gm_open(C.int(index), source, &message)
	if handle == nil {
		return nil, "", nativeError(0, message)
	}

	return handle, C.GoString(C.gm_name(handle)), nil
}

func nativeClose(p unsafe.Pointer) {
	C.gm_close(p)
}

func nativeAllocate(d unsafe.Pointer, n uint64) (unsafe.Pointer, error) {
	var message *C.char

	b := C.gm_allocate(d, C.uint64_t(n), &message)
	if b == nil {
		return nil, nativeError(0, message)
	}

	return b, nil
}

func nativeRelease(p unsafe.Pointer) {
	C.gm_release_buffer(p)
}

func nativeUpload(d, b unsafe.Pointer, values []float32) error {
	var message *C.char

	status := C.gm_upload(d, b, (*C.float)(unsafe.Pointer(unsafe.SliceData(values))), C.uint64_t(len(values))*4, &message)

	return nativeError(status, message)
}

func nativeDownload(d, b unsafe.Pointer, values []float32) error {
	var message *C.char

	status := C.gm_download(d, b, (*C.float)(unsafe.Pointer(unsafe.SliceData(values))), C.uint64_t(len(values))*4, &message)

	return nativeError(status, message)
}

func nativeSync(d unsafe.Pointer) error {
	var message *C.char

	status := C.gm_sync(d, &message)

	return nativeError(status, message)
}

func nativeStats(d unsafe.Pointer) (uint64, uint64) {
	var live, peak C.uint64_t

	C.gm_stats(d, &live, &peak)

	return uint64(live), uint64(peak)
}

func cParams(p kernelParams) C.GMParams {
	var c C.GMParams

	c.n = C.uint32_t(p.n)
	c.rank = C.uint32_t(p.rank)
	c.op = C.uint32_t(p.op)
	c.flags = C.uint32_t(p.flags)
	c.ao = C.uint32_t(p.ao)
	c.bo = C.uint32_t(p.bo)

	for i := range 8 {
		c.shape[i] = C.uint32_t(p.shape[i])
		c.as[i] = C.uint32_t(p.as[i])
		c.bs[i] = C.uint32_t(p.bs[i])
		c.ints[i] = C.uint32_t(p.ints[i])
	}

	for i := range 4 {
		c.scalars[i] = C.float(p.scalars[i])
	}

	return c
}

func nativeCompute(d unsafe.Pointer, name string, a, b, out unsafe.Pointer, p kernelParams) error {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	cp := cParams(p)

	var message *C.char

	status := C.gm_compute(d, cname, a, b, out, &cp, &message)

	return nativeError(status, message)
}

func nativeMatMul(d, a, b, out unsafe.Pointer, args [9]uint32, ta, tb bool) error {
	var ca, cb C.int

	if ta {
		ca = 1
	}

	if tb {
		cb = 1
	}

	var message *C.char

	status := C.gm_matmul(
		d, a, b, out,
		C.uint32_t(args[0]), C.uint32_t(args[1]), C.uint32_t(args[2]),
		C.uint32_t(args[3]), C.uint32_t(args[4]),
		C.uint32_t(args[5]), C.uint32_t(args[6]),
		ca, cb, &message,
	)

	return nativeError(status, message)
}

func nativeConv(d, a, w, g, out, dw unsafe.Pointer, p kernelParams, backward bool) error {
	cp := cParams(p)
	var mode C.int

	if backward {
		mode = 1
	}

	var message *C.char

	status := C.gm_conv(d, a, w, g, out, dw, &cp, mode, &message)

	return nativeError(status, message)
}
