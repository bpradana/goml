package examplebackend_test

import (
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"
	examplebackend "github.com/bpradana/goml/tensor/examples/backend"
)

func TestDevice_Conformance(t *testing.T) {
	device, err := examplebackend.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	backendtest.RunDevice(t, device)
	backendtest.RunShutdown(t, device)
}
