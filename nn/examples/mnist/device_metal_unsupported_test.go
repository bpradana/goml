//go:build !metal

package main

import (
	"errors"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

func TestDevice_metal_RequiresExplicitBuildTag(t *testing.T) {
	if _, err := openDevice("metal"); !errors.Is(err, backend.ErrUnsupported) {
		t.Fatal(err)
	}
}
