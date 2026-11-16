package yolo

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/layer"

	"github.com/bpradana/goml/tensor"
)

func TestModel_RejectsUninitializedModelAndInvalidInputShapes(t *testing.T) {
	var missing *Model
	if _, err := missing.Forward(nil); err == nil {
		t.Fatal("nil model accepted")
	}

	model := &Model{classes: []string{"object"}}

	for _, shape := range [][]int{{3, 32, 32}, {2, 3, 32, 32}, {1, 1, 32, 32}, {1, 3, 32, 64}, {1, 3, 33, 33}} {
		input, _ := tensor.Zeros(shape...)
		if _, err := model.Forward(input); err == nil {
			t.Fatal(shape)
		}
	}

	classes := model.Classes()
	classes[0] = "changed"

	if model.Classes()[0] != "object" {
		t.Fatal("Classes exposed mutable storage")
	}
}

func writeFloats(t *testing.T, path string, values []float32) {
	t.Helper()

	data := make([]byte, len(values)*4)
	for index, value := range values {
		binary.LittleEndian.PutUint32(data[index*4:], math.Float32bits(value))
	}

	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

// Explicit local artifacts keep pretrained validation out of ordinary unit tests.
// The first run writes input.f32; reference.py produces the expected heads, and
// the next run compares every logit and the class-aware NMS output.
func TestModel_PretrainedReference(t *testing.T) {
	modelPath := os.Getenv("GOML_YOLO_MODEL")
	if modelPath == "" {
		t.Skip("set GOML_YOLO_MODEL to an explicitly converted local model")
	}

	size := 320

	if value := os.Getenv("GOML_YOLO_SIZE"); value != "" {
		parsed, err := strconv.Atoi(value)
		if err != nil || !validSize(parsed) {
			t.Fatal("invalid GOML_YOLO_SIZE")
		}

		size = parsed
	}

	model, err := Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}

	var input *tensor.Tensor

	if path := os.Getenv("GOML_YOLO_IMAGE"); path != "" {
		file, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}

		source, _, err := image.Decode(file)

		closeErr := file.Close()
		if closeErr != nil {
			t.Fatal(closeErr)
		}

		if err != nil {
			t.Fatal(err)
		}

		input, _, err = prepare(source, size)
		if err != nil {
			t.Fatal(err)
		}
	} else {
		values := make([]float32, 3*size*size)
		for index := range values {
			values[index] = float32(index%251) / 250
		}

		input, err = tensor.From(values, 1, 3, size, size)
		if err != nil {
			t.Fatal(err)
		}
	}

	started := time.Now()

	heads, err := model.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Forward: size=%d elapsed=%s", size, time.Since(started))

	options := DefaultOptions()
	options.Size = size

	candidates, err := decode(heads, model.classes, options)
	if err != nil {
		t.Fatal(err)
	}

	detections := suppress(candidates, options.IoU, options.MaxDetections)
	for _, detection := range detections {
		t.Logf("%s confidence=%.6f box=%+v", detection.Label, detection.Confidence, detection.Box)
	}

	directory := os.Getenv("GOML_YOLO_REFERENCE")
	if directory == "" {
		return
	}

	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}

	writeFloats(t, filepath.Join(directory, "input.f32"), input.Data())

	for scale, head := range heads {
		path := filepath.Join(directory, fmt.Sprintf("head-%d.f32", scale))

		expected, err := os.ReadFile(path)
		if os.IsNotExist(err) {
			t.Log("reference heads absent; input.f32 is ready for reference.py")
			return
		}

		if err != nil {
			t.Fatal(err)
		}

		values := head.Data()
		if len(expected) != len(values)*4 {
			t.Fatal("reference head length mismatch")
		}

		maximum := float64(0)

		for index, got := range values {
			want := math.Float32frombits(binary.LittleEndian.Uint32(expected[index*4:]))
			difference := math.Abs(float64(got) - float64(want))

			maximum = math.Max(maximum, difference)
			if math.IsNaN(difference) || difference > 1e-3+1e-4*math.Abs(float64(want)) {
				t.Fatalf("head %d logit %d: got %g want %g", scale, index, got, want)
			}
		}

		t.Logf("head %d: %d logits, maximum absolute error %.9g", scale, len(values), maximum)
	}

	data, err := os.ReadFile(filepath.Join(directory, "detections.json"))
	if err != nil {
		t.Fatal(err)
	}

	var expected [][]float32
	if err := json.Unmarshal(data, &expected); err != nil {
		t.Fatal(err)
	}

	if len(expected) != len(detections) {
		t.Fatalf("NMS count: got %d want %d", len(detections), len(expected))
	}

	for index, got := range detections {
		want := expected[index]
		if len(want) != 6 || got.ClassID != int(want[5]) || math.Abs(float64(got.Confidence-want[4])) > 1e-4 {
			t.Fatal("NMS class/confidence mismatch", got, want)
		}

		for axis, value := range []float32{got.Box.Left, got.Box.Top, got.Box.Right, got.Box.Bottom} {
			if math.Abs(float64(value-want[axis])) > 0.05 {
				t.Fatal("NMS box mismatch", got, want)
			}
		}
	}
}

type routeStage struct {
	tag  float32
	seen []float32
}

func (s *routeStage) Forward(input *nn.Value) (*nn.Value, error) {
	if !input.IsInference() || input.RequiresGrad() {
		return nil, fmt.Errorf("route lost inference mode")
	}

	s.seen = input.Data().Data()

	data, err := tensor.From([]float32{s.tag}, 1, 1, 1, 1)
	if err != nil {
		return nil, err
	}

	return nn.NewOperation(data, input, nil)
}

func (*routeStage) Parameters() []*nn.Parameter {
	return nil
}

func TestModel_ForwardUsesBackboneSkipsAndThreeHeadSources(t *testing.T) {
	model := &Model{classes: []string{"object"}}

	var stages [22]*routeStage
	for index := range stages {
		stages[index] = &routeStage{tag: float32(index)}
		model.stages[index] = stages[index]
	}

	identity, _ := layer.NewSequential()
	for scale := range 3 {
		model.regression[scale] = identity
		model.classification[scale] = identity
	}

	input, _ := tensor.Ones(1, 3, 32, 32)

	heads, err := model.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range []struct {
		index int
		want  []float32
	}{{12, []float32{10, 6}}, {15, []float32{13, 4}}, {18, []float32{16, 12}}, {21, []float32{19, 9}}} {
		if !slices.Equal(stages[test.index].seen, test.want) {
			t.Fatal(test.index, stages[test.index].seen, test.want)
		}
	}

	for scale, index := range [3]int{15, 18, 21} {
		if !slices.Equal(heads[scale].Data(), []float32{float32(index), float32(index)}) {
			t.Fatal(scale, heads[scale])
		}
	}
}
