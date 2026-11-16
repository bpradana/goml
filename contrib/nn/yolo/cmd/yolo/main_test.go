package main

import (
	"io"
	"testing"
)

func TestRun_RequiresModelAndImagePaths(t *testing.T) {
	for _, args := range [][]string{nil, {"-model", "missing.goml"}, {"-unknown"}, {"-model", "missing.goml", "-image", "missing.jpg"}} {
		if err := run(args, io.Discard); err == nil {
			t.Fatal(args)
		}
	}
}
