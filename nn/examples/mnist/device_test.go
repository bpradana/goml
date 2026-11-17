//go:build !metal && !webgpu

package main

import "testing"

func TestDevice_DefaultCPUAndExplicitUnsupportedSelection(t *testing.T) {
	device, err := openDevice("cpu")
	if err != nil || device == nil {
		t.Fatal(device, err)
	}

	if _, err := openDevice("metal"); err == nil {
		t.Fatal("initialized accelerator in ordinary CPU build")
	}
}
