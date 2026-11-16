package yolo

import (
	"archive/zip"
	"bytes"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strconv"

	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
)

const (
	maxArchiveBytes = 64 << 20
	maxNetworkBytes = 16 << 20
)

type manifest struct {
	Version      int      `json:"version"`
	Architecture string   `json:"architecture"`
	Classes      []string `json:"classes"`
	SourceSHA256 string   `json:"source_sha256"`
}

type convSpec struct {
	in, out, kernel, stride, padding int
	linear                           bool
}

type networkLayout struct {
	Root   int            `json:"root"`
	Layers []networkLayer `json:"layers"`
}

type networkLayer struct {
	Type       string    `json:"type"`
	Config     *geometry `json:"config,omitempty"`
	Children   []int     `json:"children,omitempty"`
	Parameters []int     `json:"parameters,omitempty"`
}

type geometry struct {
	Kernel  int `json:"kernel"`
	Stride  int `json:"stride"`
	Padding int `json:"padding"`
}

// Load reads a converted goml YOLO ZIP archive. It validates the format version,
// YOLOv8n topology, every convolution's geometry and parameter shapes, finite
// weights, and checkpoint checksums before returning a usable model. Archives
// are limited to 128 entries and 64 MiB compressed and uncompressed; individual
// network entries are limited to 16 MiB. No files are extracted or code executed.
func Load(path string) (*Model, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("yolo: load: %w", err)
	}
	defer func() { _ = file.Close() }()

	info, err := file.Stat()
	if err != nil {
		return nil, err
	}

	if info.Size() > maxArchiveBytes {
		return nil, fmt.Errorf("yolo: load: archive exceeds 64 MiB")
	}

	archive, err := zip.NewReader(file, info.Size())
	if err != nil {
		return nil, fmt.Errorf("yolo: load: %w", err)
	}

	return loadArchive(archive)
}

func readEntry(file *zip.File, limit int) ([]byte, error) {
	if file.UncompressedSize64 > uint64(limit) {
		return nil, fmt.Errorf("yolo: load: entry %q exceeds size limit", file.Name)
	}

	reader, err := file.Open()
	if err != nil {
		return nil, err
	}
	defer func() { _ = reader.Close() }()

	data, err := io.ReadAll(io.LimitReader(reader, int64(limit)+1))
	if err != nil {
		return nil, fmt.Errorf("yolo: load: entry %q: %w", file.Name, err)
	}

	if len(data) > limit {
		return nil, fmt.Errorf("yolo: load: oversized entry %q", file.Name)
	}

	return data, nil
}

func loadArchive(archive *zip.Reader) (*Model, error) {
	if len(archive.File) == 0 || len(archive.File) > 128 {
		return nil, fmt.Errorf("yolo: load: invalid archive entry count")
	}

	entries := make(map[string]*zip.File, len(archive.File))
	total := uint64(0)

	for _, file := range archive.File {
		if _, exists := entries[file.Name]; exists || file.UncompressedSize64 > maxNetworkBytes {
			return nil, fmt.Errorf("yolo: load: duplicate or oversized entry %q", file.Name)
		}

		total += file.UncompressedSize64
		if total > maxArchiveBytes {
			return nil, fmt.Errorf("yolo: load: uncompressed archive exceeds 64 MiB")
		}

		entries[file.Name] = file
	}

	metadata, ok := entries["manifest.json"]
	if !ok {
		return nil, fmt.Errorf("yolo: load: missing manifest.json")
	}

	delete(entries, "manifest.json")

	data, err := readEntry(metadata, 64<<10)
	if err != nil {
		return nil, err
	}

	var config manifest

	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(&config); err != nil {
		return nil, fmt.Errorf("yolo: load: manifest: %w", err)
	}

	if err := decoder.Decode(new(json.RawMessage)); err != io.EOF {
		return nil, fmt.Errorf("yolo: load: trailing manifest data")
	}

	hash, err := hex.DecodeString(config.SourceSHA256)
	if err != nil || len(hash) != 32 || config.Version != 1 || config.Architecture != "yolov8n-detect" || len(config.Classes) == 0 || len(config.Classes) > 1000 {
		return nil, fmt.Errorf("yolo: load: unsupported or invalid manifest")
	}

	for _, name := range config.Classes {
		if name == "" || len(name) > 256 {
			return nil, fmt.Errorf("yolo: load: invalid class name")
		}
	}

	load := func(name string, specs ...convSpec) (*layer.Sequential, error) {
		path := "networks/" + name + ".goml"

		file, ok := entries[path]
		if !ok {
			return nil, fmt.Errorf("yolo: load: missing %s", path)
		}

		delete(entries, path)

		data, err := readEntry(file, maxNetworkBytes)
		if err != nil {
			return nil, err
		}

		network, err := checkpoint.Unmarshal(data)
		if err != nil {
			return nil, fmt.Errorf("yolo: load: %s: %w", path, err)
		}

		if err := validateNetwork(network, specs); err != nil {
			return nil, fmt.Errorf("yolo: load: %s: %w", path, err)
		}

		return network, nil
	}

	model := &Model{classes: slices.Clone(config.Classes)}

	for _, stage := range []struct{ index, in, out int }{{0, 3, 16}, {1, 16, 32}, {3, 32, 64}, {5, 64, 128}, {7, 128, 256}, {16, 64, 64}, {19, 128, 128}} {
		network, err := load(strconv.Itoa(stage.index), convSpec{in: stage.in, out: stage.out, kernel: 3, stride: 2, padding: 1})
		if err != nil {
			return nil, err
		}

		model.stages[stage.index] = network
	}

	for _, stage := range []struct{ index, in, out, repeats int }{{2, 32, 32, 1}, {4, 64, 64, 2}, {6, 128, 128, 2}, {8, 256, 256, 1}, {12, 384, 128, 1}, {15, 192, 64, 1}, {18, 192, 128, 1}, {21, 384, 256, 1}} {
		prefix := strconv.Itoa(stage.index)
		hidden := stage.out / 2

		first, err := load(prefix+".cv1", convSpec{in: stage.in, out: 2 * hidden, kernel: 1, stride: 1})
		if err != nil {
			return nil, err
		}

		last, err := load(prefix+".cv2", convSpec{in: (2 + stage.repeats) * hidden, out: stage.out, kernel: 1, stride: 1})
		if err != nil {
			return nil, err
		}

		block := &c2f{first: first, last: last, hidden: hidden}

		for repeat := range stage.repeats {
			prefix := prefix + ".m." + strconv.Itoa(repeat)

			first, err := load(prefix+".cv1", convSpec{in: hidden, out: hidden, kernel: 3, stride: 1, padding: 1})
			if err != nil {
				return nil, err
			}

			second, err := load(prefix+".cv2", convSpec{in: hidden, out: hidden, kernel: 3, stride: 1, padding: 1})
			if err != nil {
				return nil, err
			}

			block.blocks = append(block.blocks, bottleneck{first: first, second: second, shortcut: stage.index < 9})
		}

		model.stages[stage.index] = block
	}

	first, err := load("9.cv1", convSpec{in: 256, out: 128, kernel: 1, stride: 1})
	if err != nil {
		return nil, err
	}

	last, err := load("9.cv2", convSpec{in: 512, out: 256, kernel: 1, stride: 1})
	if err != nil {
		return nil, err
	}

	pool, err := layer.NewMaxPool2D(5, 1, 2)
	if err != nil {
		return nil, err
	}

	model.stages[9] = &sppf{first: first, last: last, pool: pool}
	for _, index := range []int{10, 13} {
		model.stages[index], err = layer.NewUpsample2D(2, layer.Nearest)
		if err != nil {
			return nil, err
		}
	}

	classes := len(config.Classes)
	classWidth := max(64, min(classes, 100))

	for scale, channels := range [3]int{64, 128, 256} {
		model.regression[scale], err = load("22.reg."+strconv.Itoa(scale),
			convSpec{in: channels, out: 64, kernel: 3, stride: 1, padding: 1},
			convSpec{in: 64, out: 64, kernel: 3, stride: 1, padding: 1},
			convSpec{in: 64, out: 64, kernel: 1, stride: 1, linear: true})
		if err != nil {
			return nil, err
		}

		model.classification[scale], err = load("22.cls."+strconv.Itoa(scale),
			convSpec{in: channels, out: classWidth, kernel: 3, stride: 1, padding: 1},
			convSpec{in: classWidth, out: classWidth, kernel: 3, stride: 1, padding: 1},
			convSpec{in: classWidth, out: classes, kernel: 1, stride: 1, linear: true})
		if err != nil {
			return nil, err
		}
	}

	if len(entries) != 0 {
		return nil, fmt.Errorf("yolo: load: unexpected archive entries")
	}

	return model, nil
}

func validateNetwork(network *layer.Sequential, specs []convSpec) error {
	parameters := network.Parameters()
	if len(parameters) != len(specs)*2 {
		return fmt.Errorf("incorrect parameter count")
	}

	data, err := layer.MarshalConfig(network)
	if err != nil {
		return err
	}

	var layout networkLayout
	if err := json.Unmarshal(data, &layout); err != nil {
		return err
	}

	root := layout.Layers[layout.Root]
	if root.Type != "sequential" {
		return fmt.Errorf("expected a sequential convolution block")
	}

	cursor := 0

	for index, spec := range specs {
		weight, bias := parameters[index*2].Value().Data(), parameters[index*2+1].Value().Data()
		if !slices.Equal(weight.Shape(), []int{spec.in * spec.kernel * spec.kernel, spec.out}) || !slices.Equal(bias.Shape(), []int{spec.out}) {
			return fmt.Errorf("convolution %d has incompatible weights", index)
		}

		for _, parameter := range parameters[index*2 : index*2+2] {
			for _, value := range parameter.Value().Data().Data() {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return fmt.Errorf("nonfinite weights")
				}
			}
		}

		if cursor >= len(root.Children) {
			return fmt.Errorf("missing convolution %d", index)
		}

		conv := layout.Layers[root.Children[cursor]]
		cursor++

		if conv.Type != "conv2d" || conv.Config == nil || *conv.Config != (geometry{Kernel: spec.kernel, Stride: spec.stride, Padding: spec.padding}) || len(conv.Children) != 1 {
			return fmt.Errorf("convolution %d geometry does not match YOLOv8n", index)
		}

		dense := layout.Layers[conv.Children[0]]
		if dense.Type != "dense" || !slices.Equal(dense.Parameters, []int{2 * index, 2*index + 1}) {
			return fmt.Errorf("convolution %d parameter order does not match YOLOv8n", index)
		}

		if !spec.linear {
			if cursor >= len(root.Children) || layout.Layers[root.Children[cursor]].Type != "silu" {
				return fmt.Errorf("convolution %d requires SiLU", index)
			}

			cursor++
		}
	}

	if cursor != len(root.Children) {
		return fmt.Errorf("unexpected layers in convolution block")
	}

	return nil
}
