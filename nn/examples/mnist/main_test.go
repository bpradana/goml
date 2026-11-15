package main

import (
	"compress/gzip"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeIDX(t *testing.T, header []uint32, data []byte) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "fixture.gz")

	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}

	writer := gzip.NewWriter(file)
	if err := binary.Write(writer, binary.BigEndian, header); err != nil {
		t.Fatal(err)
	}

	if _, err := writer.Write(data); err != nil {
		t.Fatal(err)
	}

	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	if err := file.Close(); err != nil {
		t.Fatal(err)
	}

	return path
}

func TestLoad_AcceptsValidImageAndLabelIDX(t *testing.T) {
	for _, tt := range []struct {
		name   string
		header []uint32
		size   int
	}{
		{"Images", []uint32{2051, 1, 28, 28}, 784},
		{"Labels", []uint32{2049, 1}, 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := writeIDX(t, tt.header, make([]byte, tt.size))
			header, data, err := load(path, len(tt.header))

			if err != nil || len(header) != len(tt.header) || len(data) != tt.size {
				t.Fatalf("unexpected IDX result: %v %d %v", header, len(data), err)
			}
		})
	}
}

func TestLoad_RejectsMalformedIDX(t *testing.T) {
	for _, tt := range []struct {
		name   string
		header []uint32
		size   int
	}{
		{"BadImageMagic", []uint32{42, 1, 28, 28}, 784},
		{"BadDimensions", []uint32{2051, 1, 1, 784}, 784},
		{"ZeroItems", []uint32{2049, 0}, 0},
		{"TruncatedData", []uint32{2051, 1, 28, 28}, 10},
		{"ExtraData", []uint32{2049, 1}, 2},
	} {
		t.Run(tt.name, func(t *testing.T) {
			path := writeIDX(t, tt.header, make([]byte, tt.size))
			if _, _, err := load(path, len(tt.header)); err == nil {
				t.Fatal("expected IDX validation error")
			}
		})
	}
}

func TestRun_TrainsOnLocalIDXFixtures(t *testing.T) {
	pixels := make([]byte, 3*784)
	for i := 784; i < len(pixels); i++ {
		pixels[i] = 255
	}

	images := writeIDX(t, []uint32{2051, 3, 28, 28}, pixels)
	labels := writeIDX(t, []uint32{2049, 3}, []byte{0, 1, 2})

	for _, model := range []string{"dense", "cnn"} {
		t.Run(model, func(t *testing.T) {
			if err := run(images, labels, model, 3, 2, 2); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRun_RejectsUnknownModelBeforeOpeningDataset(t *testing.T) {
	err := run("missing-images.gz", "missing-labels.gz", "unknown", 1, 1, 1)
	if err == nil || !strings.Contains(err.Error(), "unknown model") {
		t.Fatalf("expected model validation error, got %v", err)
	}
}

func TestRun_RejectsInvalidTrainingLimits(t *testing.T) {
	for _, arguments := range [][3]int{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {-1, 1, 1}} {
		if err := run("missing-images.gz", "missing-labels.gz", "cnn", arguments[0], arguments[1], arguments[2]); err == nil || !strings.Contains(err.Error(), "must be positive") {
			t.Fatalf("accepted invalid arguments %v: %v", arguments, err)
		}
	}
}
