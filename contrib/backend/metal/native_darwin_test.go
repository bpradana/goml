//go:build darwin && cgo

package metal

import (
	"testing"
	"unsafe"
)

func TestNativeParams_PreservesShaderFieldLayoutAndValues(t *testing.T) {
	params := kernelParams{n: 7, rank: 3, op: 2, flags: 5, ao: 11, bo: 13}
	for index := range 8 {
		params.shape[index] = uint32(index + 17)
		params.as[index] = uint32(index + 31)
		params.bs[index] = uint32(index + 47)
		params.ints[index] = uint32(index + 61)
	}

	params.scalars = [4]float32{0.5, -2, 3.25, 4}
	converted := cParams(params)

	if unsafe.Sizeof(converted) != 168 ||
		unsafe.Offsetof(converted.shape) != 16 ||
		unsafe.Offsetof(converted.as) != 48 ||
		unsafe.Offsetof(converted.bs) != 80 ||
		unsafe.Offsetof(converted.ao) != 112 ||
		unsafe.Offsetof(converted.bo) != 116 ||
		unsafe.Offsetof(converted.ints) != 120 ||
		unsafe.Offsetof(converted.scalars) != 152 {
		t.Fatal("native parameter layout differs from kernels/common.metal")
	}

	if uint32(converted.n) != params.n || uint32(converted.rank) != params.rank ||
		uint32(converted.op) != params.op || uint32(converted.flags) != params.flags ||
		uint32(converted.ao) != params.ao || uint32(converted.bo) != params.bo {
		t.Fatal("native parameter conversion changed scalar metadata")
	}

	for index := range 8 {
		if uint32(converted.shape[index]) != params.shape[index] ||
			uint32(converted.as[index]) != params.as[index] ||
			uint32(converted.bs[index]) != params.bs[index] ||
			uint32(converted.ints[index]) != params.ints[index] {
			t.Fatalf("native parameter conversion changed metadata at index %d", index)
		}
	}

	for index, expected := range params.scalars {
		if float32(converted.scalars[index]) != expected {
			t.Fatalf("native parameter conversion changed numerical scalar %d", index)
		}
	}
}
