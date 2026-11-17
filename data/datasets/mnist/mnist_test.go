package mnist_test

import (
	"compress/gzip"
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/data/datasets/mnist"
)

func fixture(t testing.TB, directory, name string, header []uint32, payload []byte, compressed bool) string {
	t.Helper()

	if compressed {
		name += ".gz"
	}

	path := filepath.Join(directory, name)

	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = file.Close() }()

	if compressed {
		writer := gzip.NewWriter(file)
		if err := binary.Write(writer, binary.BigEndian, header); err != nil {
			t.Fatal(err)
		}

		if _, err := writer.Write(payload); err != nil {
			t.Fatal(err)
		}

		if err := writer.Close(); err != nil {
			t.Fatal(err)
		}
	} else {
		if err := binary.Write(file, binary.BigEndian, header); err != nil {
			t.Fatal(err)
		}

		if _, err := file.Write(payload); err != nil {
			t.Fatal(err)
		}
	}

	return path
}

func TestOpen_SplitsCompressionNormalizationAndConcurrentGet(t *testing.T) {
	for _, compressed := range []bool{false, true} {
		directory := t.TempDir()
		pixels := make([]byte, 2*784)

		pixels[0], pixels[1], pixels[784] = 255, 127, 64
		for _, prefix := range []string{"train", "t10k"} {
			fixture(t, directory, prefix+"-images-idx3-ubyte", []uint32{2051, 2, 28, 28}, pixels, compressed)
			fixture(t, directory, prefix+"-labels-idx1-ubyte", []uint32{2049, 2}, []byte{4, 7}, compressed)
		}

		for _, split := range []mnist.Split{mnist.Train, mnist.Test} {
			ds, err := mnist.Open(directory, split)
			if err != nil {
				t.Fatal(err)
			}

			var workers sync.WaitGroup
			for range 4 {
				workers.Go(func() {
					sample, err := ds.Get(context.Background(), 0)
					if err != nil || sample.Label != 4 || sample.Input[0] != 1 || sample.Input[1] != float32(127)/255 || !slices.Equal(sample.Shape, []int{1, 28, 28}) {
						t.Errorf("invalid sample: %+v %v", sample, err)
						return
					}

					sample.Input[0] = 99
				})
			}

			workers.Wait()

			for _, index := range []int{-1, 2} {
				if _, err := ds.Get(context.Background(), index); err == nil {
					t.Fatal("accepted invalid index")
				}
			}

			loader, _ := data.NewLoader(ds, data.CollateClassification, data.LoaderOptions{BatchSize: 2})
			for batch, err := range loader.Batches(context.Background(), 0) {
				if err != nil || !slices.Equal(batch.Inputs.Shape(), []int{2, 1, 28, 28}) {
					t.Fatal("incorrect batch", err)
				}

				_ = batch.Close()
			}
		}
	}
}

func TestOpenFiles_RejectsMalformedAndTruncatedInputs(t *testing.T) {
	for _, tc := range []struct {
		name            string
		images, labels  []uint32
		pixels, targets []byte
	}{
		{"magic", []uint32{1, 1, 28, 28}, []uint32{2049, 1}, make([]byte, 784), []byte{0}},
		{"dimension", []uint32{2051, 1, 1, 784}, []uint32{2049, 1}, make([]byte, 784), []byte{0}},
		{"truncated", []uint32{2051, 1, 28, 28}, []uint32{2049, 1}, make([]byte, 2), []byte{0}},
		{"extra", []uint32{2051, 1, 28, 28}, []uint32{2049, 1}, make([]byte, 785), []byte{0}},
		{"count", []uint32{2051, 1, 28, 28}, []uint32{2049, 2}, make([]byte, 784), []byte{0, 1}},
		{"label", []uint32{2051, 1, 28, 28}, []uint32{2049, 1}, make([]byte, 784), []byte{10}},
		{"excessive", []uint32{2051, ^uint32(0), 28, 28}, []uint32{2049, 1}, nil, []byte{0}},
		{"zero", []uint32{2051, 0, 28, 28}, []uint32{2049, 0}, nil, nil},
		{"header", []uint32{2051}, []uint32{2049}, nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, compressed := range []bool{false, true} {
				directory := t.TempDir()
				images := fixture(t, directory, "images", tc.images, tc.pixels, compressed)

				labels := fixture(t, directory, "labels", tc.labels, tc.targets, compressed)
				if _, err := mnist.OpenFiles(context.Background(), images, labels); err == nil {
					t.Fatal("accepted malformed IDX")
				}
			}
		})
	}
}

func TestOpen_CancellationMissingFilesAndGzipChecksum(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := mnist.OpenContext(ctx, t.TempDir(), mnist.Train); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}

	if _, err := mnist.Open(t.TempDir(), mnist.Train); !errors.Is(err, os.ErrNotExist) {
		t.Fatal(err)
	}

	if _, err := mnist.Open(t.TempDir(), mnist.Split(9)); err == nil {
		t.Fatal("accepted invalid split")
	}

	directory := t.TempDir()
	images := fixture(t, directory, "images", []uint32{2051, 1, 28, 28}, make([]byte, 784), true)
	labels := fixture(t, directory, "labels", []uint32{2049, 1}, []byte{0}, true)
	bytes, _ := os.ReadFile(images)

	bytes[len(bytes)-8] ^= 1
	if err := os.WriteFile(images, bytes, 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := mnist.OpenFiles(context.Background(), images, labels); err == nil {
		t.Fatal("accepted corrupt gzip checksum")
	}
}

func BenchmarkMNIST(b *testing.B) {
	directory := b.TempDir()
	fixture(b, directory, "train-images-idx3-ubyte", []uint32{2051, 128, 28, 28}, make([]byte, 128*784), true)
	fixture(b, directory, "train-labels-idx1-ubyte", []uint32{2049, 128}, make([]byte, 128), true)

	ds, err := mnist.Open(directory, mnist.Train)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("get", func(b *testing.B) {
		b.ReportAllocs()

		for b.Loop() {
			if _, err := ds.Get(context.Background(), 0); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("batch64", func(b *testing.B) {
		loader, _ := data.NewLoader(ds, data.CollateClassification, data.LoaderOptions{BatchSize: 64})

		b.ReportAllocs()

		for b.Loop() {
			for batch, err := range loader.Batches(context.Background(), 0) {
				if err != nil {
					b.Fatal(err)
				}

				_ = batch.Close()

				break
			}
		}
	})
}
