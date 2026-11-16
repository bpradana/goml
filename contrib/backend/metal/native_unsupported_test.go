//go:build !darwin || !cgo

package metal_test

import (
	"testing"

	"github.com/bpradana/goml/contrib/backend/metal"
)

func TestOpen_ReportsUnsupportedPlatform(t *testing.T) {
	if device, err := metal.Open(); err == nil || device != nil {
		t.Fatal("unsupported platform advertised a working Metal device")
	}
}
