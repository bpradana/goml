package main

import (
	"errors"
	"os"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

func TestMain_RectangularProductOnHardware(t *testing.T) {
	if err := run(); err != nil {
		if errors.Is(err, backend.ErrUnsupported) && os.Getenv("GOML_WEBGPU_REQUIRED") != "1" {
			t.Skip(err)
		}

		t.Fatal(err)
	}
}
