//go:build !webgpu

package main

import (
	"errors"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

func TestDevice_webgpu_RequiresExplicitBuildTag(t *testing.T) {
	if _, err := openDevice("webgpu"); !errors.Is(err, backend.ErrUnsupported) {
		t.Fatal(err)
	}
}
