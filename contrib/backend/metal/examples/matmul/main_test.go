//go:build darwin && cgo

package main

import "testing"

func TestRun_RectangularMPSProduct(t *testing.T) {
	if err := run(); err != nil {
		t.Fatal(err)
	}
}
