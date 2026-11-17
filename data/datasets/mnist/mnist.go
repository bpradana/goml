// Package mnist opens validated local MNIST IDX files without network access.
// Images stay as bytes in host memory and are converted per sample to [1,28,28]
// Float32 values in [0,1]. No file handles remain open after construction.
package mnist

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/bpradana/goml/data"
)

// Split selects the standard training or held-out test files.
type Split uint8

const (
	// Train selects train-images and train-labels (at most 60,000 samples).
	Train Split = iota
	// Test selects t10k-images and t10k-labels (at most 10,000 samples).
	Test
)

// Dataset owns validated image bytes and labels. Concurrent Get calls are safe.
type Dataset struct {
	pixels []byte
	labels []byte
}

// Open reads plain or gzip IDX files in directory. It never downloads data.
func Open(directory string, split Split) (*Dataset, error) {
	return OpenContext(context.Background(), directory, split)
}

// OpenContext is Open with cancellation during file reads and decompression.
// Plain standard filenames take precedence over their .gz counterparts.
func OpenContext(ctx context.Context, directory string, split Split) (*Dataset, error) {
	prefix, maximum := "train", uint32(60000)

	switch split {
	case Train:
	case Test:
		prefix, maximum = "t10k", 10000
	default:
		return nil, fmt.Errorf("mnist: invalid split %d", split)
	}

	paths := make([]string, 2)

	for i, suffix := range []string{"-images-idx3-ubyte", "-labels-idx1-ubyte"} {
		path := filepath.Join(directory, prefix+suffix)
		if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
			path += ".gz"
		} else if err != nil {
			return nil, err
		}

		paths[i] = path
	}

	return openFiles(ctx, paths[0], paths[1], maximum)
}

// OpenFiles opens an explicit image/label pair, plain or .gz, with a 60,000-item
// limit. Use Open with Test for the stricter held-out split size limit.
func OpenFiles(ctx context.Context, images, labels string) (*Dataset, error) {
	return openFiles(ctx, images, labels, 60000)
}

func openFiles(ctx context.Context, images, labels string, maximum uint32) (*Dataset, error) {
	pixels, count, err := readIDX(ctx, images, true, maximum)
	if err != nil {
		return nil, err
	}

	targets, labelCount, err := readIDX(ctx, labels, false, maximum)
	if err != nil {
		return nil, err
	}

	if count != labelCount {
		return nil, fmt.Errorf("mnist: image count %d differs from label count %d", count, labelCount)
	}

	for index, label := range targets {
		if label >= 10 {
			return nil, fmt.Errorf("mnist: invalid label %d at sample %d", label, index)
		}
	}

	return &Dataset{pixels: pixels, labels: targets}, nil
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r contextReader) Read(dst []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}

	return r.reader.Read(dst)
}

func readIDX(ctx context.Context, path string, images bool, maximum uint32) ([]byte, uint32, error) {
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}

	file, err := os.Open(path)
	if err != nil {
		return nil, 0, fmt.Errorf("mnist: open local IDX file %s (plain or .gz files are required; no automatic download): %w", path, err)
	}
	defer func() { _ = file.Close() }()

	var reader io.Reader = contextReader{ctx: ctx, reader: file}
	if filepath.Ext(path) == ".gz" {
		compressed, err := gzip.NewReader(reader)
		if err != nil {
			return nil, 0, fmt.Errorf("mnist: gzip %s: %w", path, err)
		}
		defer func() { _ = compressed.Close() }()

		reader = contextReader{ctx: ctx, reader: compressed}
	}

	words, magic := 2, uint32(2049)
	if images {
		words, magic = 4, 2051
	}

	header := make([]uint32, words)
	if err := binary.Read(reader, binary.BigEndian, header); err != nil {
		return nil, 0, fmt.Errorf("mnist: header %s: %w", path, err)
	}

	if header[0] != magic || header[1] == 0 || header[1] > maximum {
		return nil, 0, fmt.Errorf("mnist: %s: invalid IDX magic or item count %v", path, header)
	}

	size := int(header[1])
	if images {
		if header[2] != 28 || header[3] != 28 {
			return nil, 0, fmt.Errorf("mnist: %s: images must be 28x28", path)
		}

		size *= 784
	}

	values, err := io.ReadAll(io.LimitReader(reader, int64(size)+1))
	if err != nil {
		return nil, 0, fmt.Errorf("mnist: read %s: %w", path, err)
	}

	if len(values) != size {
		return nil, 0, fmt.Errorf("mnist: %s: expected exactly %d bytes, got %d", path, size, len(values))
	}

	return values, header[1], nil
}

// Len returns the validated image count.
func (d *Dataset) Len() int {
	return len(d.labels)
}

// Get returns an independent normalized sample and integer label.
func (d *Dataset) Get(ctx context.Context, index int) (data.ClassificationSample, error) {
	if err := ctx.Err(); err != nil {
		return data.ClassificationSample{}, err
	}

	if index < 0 || index >= d.Len() {
		return data.ClassificationSample{}, fmt.Errorf("mnist: index %d out of range", index)
	}

	values := make([]float32, 784)
	for i, pixel := range d.pixels[index*784 : (index+1)*784] {
		values[i] = float32(pixel) / 255
	}

	return data.ClassificationSample{Input: values, Shape: []int{1, 28, 28}, Label: int(d.labels[index])}, nil
}
