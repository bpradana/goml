package examplebackend_test

import (
	"errors"
	"testing"

	"github.com/bpradana/goml/tensor/backend"
	examplebackend "github.com/bpradana/goml/tensor/examples/backend"
)

func TestBuffer_ForeignContextAndIdempotentRelease(t *testing.T) {
	first, err := examplebackend.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = first.Close() }()

	second, err := examplebackend.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = second.Close() }()

	buffer, err := first.Allocate(1)
	if err != nil {
		t.Fatal(err)
	}

	if err := second.Upload(buffer, []float32{1}); !errors.Is(err, backend.ErrDeviceMismatch) {
		t.Fatalf("got %v", err)
	}

	if err := buffer.Close(); err != nil {
		t.Fatal(err)
	}

	if err := buffer.Close(); err != nil {
		t.Fatal(err)
	}

	if err := buffer.Valid(); !errors.Is(err, backend.ErrClosed) {
		t.Fatalf("got %v", err)
	}
}
