//go:build !cgo || android || ios || (!darwin && !linux && !windows) || (!amd64 && !arm64)

package webgpu

import (
	"errors"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

func TestOpen_UnsupportedBuildReturnsExplicitError(t *testing.T) {
	device, err := Open()
	if device != nil || !errors.Is(err, backend.ErrUnsupported) {
		t.Fatalf("Open() = %v, %v", device, err)
	}
}
