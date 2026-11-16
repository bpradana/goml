package yolo

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bpradana/goml/nn/activation"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/tensor"
)

func TestValidateNetwork_ConvolutionsAndSharedStatelessActivations(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	first, _ := layer.NewConv2D(2, 3, 3, 1, 1, true, rng)
	second, _ := layer.NewConv2D(3, 4, 1, 1, 0, true, rng)
	model, _ := layer.NewSequential(first, activation.NewSiLU(), second, activation.NewSiLU())

	encoded, err := checkpoint.Marshal(model)
	if err != nil {
		t.Fatal(err)
	}

	restored, err := checkpoint.Unmarshal(encoded)
	if err != nil {
		t.Fatal(err)
	}

	specs := []convSpec{{in: 2, out: 3, kernel: 3, stride: 1, padding: 1}, {in: 3, out: 4, kernel: 1, stride: 1}}
	if err := validateNetwork(restored, specs); err != nil {
		t.Fatal(err)
	}

	specs[1].linear = true
	if err := validateNetwork(restored, specs); err == nil {
		t.Fatal("extra activation accepted")
	}

	specs[1].linear = false

	specs[0].stride = 2
	if err := validateNetwork(restored, specs); err == nil {
		t.Fatal("wrong stride accepted")
	}

	specs[0].stride = 1
	parameter := restored.Parameters()[0]
	values := parameter.Value().Data().Data()
	values[0] = float32(math.NaN())

	data, _ := tensor.From(values, parameter.Value().Data().Shape()...)
	if err := parameter.Update(data); err != nil {
		t.Fatal(err)
	}

	if err := validateNetwork(restored, specs); err == nil {
		t.Fatal("nonfinite weights accepted")
	}
}

func TestValidateNetwork_RejectsMissingBiasWrongActivationAndShape(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))

	specs := []convSpec{{in: 1, out: 2, kernel: 1, stride: 1}}
	for _, test := range []struct {
		bias bool
		out  int
		silu bool
	}{{false, 2, true}, {true, 3, true}, {true, 2, false}} {
		conv, _ := layer.NewConv2D(1, test.out, 1, 1, 0, test.bias, rng)

		var model *layer.Sequential
		if test.silu {
			model, _ = layer.NewSequential(conv, activation.NewSiLU())
		} else {
			model, _ = layer.NewSequential(conv, activation.NewReLU())
		}

		if err := validateNetwork(model, specs); err == nil {
			t.Fatal(test)
		}
	}
}

func testZIP(t *testing.T, names, contents []string) []byte {
	t.Helper()

	var buffer bytes.Buffer

	archive := zip.NewWriter(&buffer)
	for index, name := range names {
		writer, err := archive.Create(name)
		if err != nil {
			t.Fatal(err)
		}

		if _, err := writer.Write([]byte(contents[index])); err != nil {
			t.Fatal(err)
		}
	}

	if err := archive.Close(); err != nil {
		t.Fatal(err)
	}

	return buffer.Bytes()
}

func TestLoadArchive_RejectsInvalidManifestMissingNetworksAndDuplicates(t *testing.T) {
	valid := manifest{Version: 1, Architecture: "yolov8n-detect", Classes: []string{"object"}, SourceSHA256: strings.Repeat("0", 64)}

	metadata, _ := json.Marshal(valid)
	for _, data := range [][]byte{
		testZIP(t, []string{"other"}, []string{"data"}),
		testZIP(t, []string{"manifest.json", "manifest.json"}, []string{string(metadata), string(metadata)}),
		testZIP(t, []string{"manifest.json"}, []string{`{"version":2}`}),
		testZIP(t, []string{"manifest.json"}, []string{`null`}),
		testZIP(t, []string{"manifest.json"}, []string{string(metadata) + ` {}`}),
		testZIP(t, []string{"manifest.json"}, []string{strings.TrimSuffix(string(metadata), "}") + `,"unknown":1}`}),
		testZIP(t, []string{"manifest.json"}, []string{string(metadata)}),
		testZIP(t, []string{"manifest.json", "networks/0.goml"}, []string{string(metadata), "corrupt checkpoint"}),
	} {
		archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
		if err != nil {
			t.Fatal(err)
		}

		if _, err := loadArchive(archive); err == nil {
			t.Fatal("invalid archive accepted")
		}
	}
}

func TestReadEntry_RejectsOversizedPayload(t *testing.T) {
	data := testZIP(t, []string{"entry"}, []string{"0123456789"})

	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}

	if _, err := readEntry(archive.File[0], 5); err == nil {
		t.Fatal("oversized entry accepted")
	}
}

func TestLoad_ReportsFilesystemAndZIPFailures(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "missing")); err == nil {
		t.Fatal("missing file accepted")
	}

	path := filepath.Join(t.TempDir(), "bad.goml")
	if err := os.WriteFile(path, []byte("not a zip"), 0o600); err != nil {
		t.Fatal(err)
	}

	if _, err := Load(path); err == nil {
		t.Fatal("bad zip accepted")
	}
}

func FuzzLoadArchive(f *testing.F) {
	f.Add([]byte("invalid zip"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			t.Skip()
		}

		archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
		if err == nil {
			_, _ = loadArchive(archive)
		}
	})
}
