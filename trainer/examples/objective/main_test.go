package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestRun_DenoisingWithConfiguredCompositeLoss(t *testing.T) {
	if err := run(""); err != nil {
		t.Fatal(err)
	}
}

func TestRun_CustomBestWeightsSaver(t *testing.T) {
	path := filepath.Join(t.TempDir(), "best-weights.goml")
	if err := run(path); err != nil {
		t.Fatal(err)
	}

	file, err := os.Stat(path)
	if err != nil || file.Size() == 0 {
		t.Fatal(file, err)
	}
}
