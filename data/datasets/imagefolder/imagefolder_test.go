package imagefolder_test

import (
	"context"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/data/datasets/imagefolder"
	"github.com/bpradana/goml/data/transform"
)

func writeImage(t testing.TB, root, path string, width, height int) {
	t.Helper()

	full := filepath.Join(root, path)
	if err := os.MkdirAll(filepath.Dir(full), 0o700); err != nil {
		t.Fatal(err)
	}

	file, err := os.Create(full)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = file.Close() }()

	img := image.NewRGBA(image.Rect(0, 0, width, height))
	img.Set(0, 0, color.RGBA{R: 255, A: 255})

	if filepath.Ext(path) == ".png" {
		err = png.Encode(file, img)
	} else {
		err = jpeg.Encode(file, img, nil)
	}

	if err != nil {
		t.Fatal(err)
	}
}

func TestImageFolder_StableClassesLazyImagesAndTransforms(t *testing.T) {
	root := t.TempDir()
	writeImage(t, root, "dogs/2.png", 3, 2)
	writeImage(t, root, "cats/b.jpg", 2, 3)
	writeImage(t, root, "cats/a.png", 2, 2)

	ds, err := imagefolder.Open(root, imagefolder.Options{})
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = ds.Close() }()

	classes := ds.Classes()
	if !slices.Equal(classes, []string{"cats", "dogs"}) || ds.Len() != 3 {
		t.Fatal(classes, ds.Len())
	}

	classes[0] = "changed"
	a, _ := ds.Get(context.Background(), 0)

	b, _ := ds.Get(context.Background(), 1)
	if a.Label != 0 || a.Input[0] != 1 || b.Shape[1] != 3 || ds.Classes()[0] != "cats" {
		t.Fatal("unstable metadata or decoding")
	}

	if batch, err := data.CollateClassification([]data.ClassificationSample{a, b}); err == nil {
		_ = batch.Close()

		t.Fatal("silently resized variable images")
	}

	resize, _ := transform.Resize(4, 4)

	resized, err := imagefolder.Open(root, imagefolder.Options{Transform: resize})
	if err != nil {
		t.Fatal(err)
	}

	defer func() { _ = resized.Close() }()

	var workers sync.WaitGroup
	for range 4 {
		workers.Go(func() {
			sample, err := resized.Get(context.Background(), 1)
			if err != nil || !slices.Equal(sample.Shape, []int{3, 4, 4}) {
				t.Error("resize/decode", err)
			}
		})
	}

	workers.Wait()
}

func TestImageFolder_RejectsLimitsCorruptionSymlinksAndUnsupportedFiles(t *testing.T) {
	for _, kind := range []string{"bytes", "pixels", "corrupt", "unsupported", "symlink"} {
		t.Run(kind, func(t *testing.T) {
			root := t.TempDir()
			writeImage(t, root, "class/image.png", 3, 3)

			options := imagefolder.Options{}

			switch kind {
			case "bytes":
				options.MaxBytes = 1
			case "pixels":
				options.MaxPixels = 1
			case "corrupt":
				if err := os.WriteFile(filepath.Join(root, "class/image.png"), []byte("bad"), 0o600); err != nil {
					t.Fatal(err)
				}
			case "unsupported":
				if err := os.WriteFile(filepath.Join(root, "class/file.txt"), nil, 0o600); err != nil {
					t.Fatal(err)
				}
			case "symlink":
				if err := os.Symlink(t.TempDir(), filepath.Join(root, "class/link")); err != nil {
					t.Skipf("symlinks unavailable: %v", err)
				}
			}

			ds, err := imagefolder.Open(root, options)
			if kind == "unsupported" || kind == "symlink" {
				if err == nil {
					_ = ds.Close()

					t.Fatal("unsupported filesystem entry accepted")
				}

				return
			}

			if err != nil {
				t.Fatal(err)
			}

			defer func() { _ = ds.Close() }()

			if _, err := ds.Get(context.Background(), 0); err == nil {
				t.Fatal("invalid image accepted")
			}
		})
	}
}

func BenchmarkImageFolder_Get(b *testing.B) {
	root := b.TempDir()
	writeImage(b, root, "class/image.png", 64, 64)

	ds, err := imagefolder.Open(root, imagefolder.Options{})
	if err != nil {
		b.Fatal(err)
	}

	defer func() { _ = ds.Close() }()

	b.ReportAllocs()
	b.ReportMetric(1, "images/op")

	for b.Loop() {
		if _, err := ds.Get(context.Background(), 0); err != nil {
			b.Fatal(err)
		}
	}
}

func TestImageFolder_CancellationBoundsCloseAndMissingRoot(t *testing.T) {
	root := t.TempDir()
	writeImage(t, root, "class/image.png", 1, 1)

	ds, err := imagefolder.Open(root, imagefolder.Options{Channels: 1})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := ds.Get(ctx, 0); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}

	for _, index := range []int{-1, 1} {
		if _, err := ds.Get(context.Background(), index); err == nil {
			t.Fatal("invalid index accepted")
		}
	}

	for range 2 {
		if err := ds.Close(); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := ds.Get(context.Background(), 0); err == nil {
		t.Fatal("closed dataset remained readable")
	}

	if _, err := imagefolder.Open(filepath.Join(root, "missing"), imagefolder.Options{}); !errors.Is(err, os.ErrNotExist) {
		t.Fatal(err)
	}
}
