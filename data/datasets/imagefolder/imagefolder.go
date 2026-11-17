// Package imagefolder lazily decodes local JPEG/PNG classification images.
// Class directories and filenames are sorted. Symlinks and nested directories
// are rejected; rooted file access prevents paths from escaping the dataset.
package imagefolder

import (
	"context"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/data/transform"
)

// Options configures conversion and bounded decoding. Zero Channels means RGB
// (3); one selects grayscale. Zero limits mean 16 million pixels and 32 MiB per
// encoded file. Transform runs after CHW conversion and must be concurrency-safe
// if Get is used concurrently. Images are never resized implicitly.
type Options struct {
	Channels  int
	MaxPixels int
	MaxBytes  int64
	Transform transform.Func[data.ClassificationSample]
}

type entry struct {
	path  string
	label int
}

// Dataset owns a rooted directory handle and immutable file metadata. Close it
// after all Get calls complete. Returned samples and batches outlive Close.
type Dataset struct {
	root      *os.Root
	classes   []string
	entries   []entry
	options   Options
	closeOnce sync.Once
	closeErr  error
}

// Open discovers one level of class directories without decoding or caching
// images. Every class entry must be a regular JPEG or PNG file. Missing roots,
// symlinks, unsupported formats, and invalid limits return descriptive errors.
func Open(directory string, options Options) (_ *Dataset, err error) {
	if options.Channels == 0 {
		options.Channels = 3
	}

	if options.MaxPixels == 0 {
		options.MaxPixels = 16_000_000
	}

	if options.MaxBytes == 0 {
		options.MaxBytes = 32 << 20
	}

	if (options.Channels != 1 && options.Channels != 3) || options.MaxPixels < 0 || options.MaxBytes < 0 {
		return nil, fmt.Errorf("imagefolder: invalid channel count or decoding limits")
	}

	info, err := os.Lstat(directory)
	if err != nil {
		return nil, err
	}

	if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
		return nil, fmt.Errorf("imagefolder: root must be a directory, not a symlink")
	}

	root, err := os.OpenRoot(directory)
	if err != nil {
		return nil, err
	}

	d := &Dataset{root: root, options: options}

	defer func() {
		if err != nil {
			_ = d.Close()
		}
	}()

	classes, err := readDirectory(root, ".")
	if err != nil {
		return nil, err
	}

	for _, class := range classes {
		if class.Type()&os.ModeSymlink != 0 || !class.IsDir() {
			return nil, fmt.Errorf("imagefolder: root entry %s must be a class directory, not a symlink or file", class.Name())
		}

		label := len(d.classes)
		d.classes = append(d.classes, class.Name())

		files, err := readDirectory(root, class.Name())
		if err != nil {
			return nil, err
		}

		for _, file := range files {
			path := filepath.Join(class.Name(), file.Name())

			info, err := root.Lstat(path)
			if err != nil {
				return nil, err
			}

			extension := strings.ToLower(filepath.Ext(file.Name()))
			if !info.Mode().IsRegular() || (extension != ".jpg" && extension != ".jpeg" && extension != ".png") {
				return nil, fmt.Errorf("imagefolder: unsupported file, nested directory, or symlink %s", path)
			}

			d.entries = append(d.entries, entry{path: path, label: label})
		}
	}

	return d, nil
}

func readDirectory(root *os.Root, name string) ([]os.DirEntry, error) {
	file, err := root.Open(name)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()

	entries, err := file.ReadDir(-1)
	slices.SortFunc(entries, func(a, b os.DirEntry) int { return strings.Compare(a.Name(), b.Name()) })

	return entries, err
}

// Len returns the stable discovered file count. File changes surface as Get errors.
func (d *Dataset) Len() int {
	return len(d.entries)
}

// Classes returns an independent, label-ordered class-name slice.
func (d *Dataset) Classes() []string {
	return slices.Clone(d.classes)
}

// Close releases the directory handle once. Samples and batches remain valid.
func (d *Dataset) Close() error {
	d.closeOnce.Do(func() { d.closeErr = d.root.Close() })
	return d.closeErr
}

type contextReader struct {
	ctx    context.Context
	source io.Reader
}

func (r contextReader) Read(dst []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}

	return r.source.Read(dst)
}

// Get decodes one image on demand and returns independently owned CHW values.
// Encoded bytes, header dimensions, and actual decoded dimensions are bounded.
// No decoded image is cached. Concurrent calls use independent file handles.
func (d *Dataset) Get(ctx context.Context, index int) (data.ClassificationSample, error) {
	if err := ctx.Err(); err != nil {
		return data.ClassificationSample{}, err
	}

	if index < 0 || index >= d.Len() {
		return data.ClassificationSample{}, fmt.Errorf("imagefolder: index %d out of range", index)
	}

	entry := d.entries[index]

	info, err := d.root.Lstat(entry.path)
	if err != nil {
		return data.ClassificationSample{}, err
	}

	if !info.Mode().IsRegular() || info.Size() > d.options.MaxBytes {
		return data.ClassificationSample{}, fmt.Errorf("imagefolder: %s is a symlink, nonregular, or oversized file", entry.path)
	}

	file, err := d.root.Open(entry.path)
	if err != nil {
		return data.ClassificationSample{}, err
	}
	defer func() { _ = file.Close() }()

	reader := func() io.Reader {
		return contextReader{ctx: ctx, source: io.LimitReader(file, d.options.MaxBytes)}
	}

	config, format, err := image.DecodeConfig(reader())
	if err != nil {
		return data.ClassificationSample{}, fmt.Errorf("imagefolder: header %s: %w", entry.path, err)
	}

	if (format != "png" && format != "jpeg") || config.Width <= 0 || config.Height <= 0 || config.Height > d.options.MaxPixels/config.Width {
		return data.ClassificationSample{}, fmt.Errorf("imagefolder: %s exceeds pixel limits or has unsupported format", entry.path)
	}

	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return data.ClassificationSample{}, err
	}

	decoded, _, err := image.Decode(reader())
	if err != nil {
		return data.ClassificationSample{}, fmt.Errorf("imagefolder: decode %s: %w", entry.path, err)
	}

	if decoded.Bounds().Dx() != config.Width || decoded.Bounds().Dy() != config.Height {
		return data.ClassificationSample{}, fmt.Errorf("imagefolder: decoded size differs from header in %s", entry.path)
	}

	sample, err := transform.ImageToCHW(ctx, decoded, d.options.Channels)
	if err != nil {
		return data.ClassificationSample{}, err
	}

	sample.Label = entry.label
	if d.options.Transform != nil {
		return d.options.Transform(ctx, sample)
	}

	return sample, nil
}
