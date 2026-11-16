package checkpoint

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func testParameter(t *testing.T, values []float32, shape ...int) *nn.Parameter {
	t.Helper()

	data, err := tensor.From(values, shape...)
	if err != nil {
		t.Fatal(err)
	}

	parameter, err := nn.NewParameter(data)
	if err != nil {
		t.Fatal(err)
	}

	return parameter
}

func seedGradients(t *testing.T, parameters []*nn.Parameter) {
	t.Helper()

	var objective *nn.Value

	for _, parameter := range parameters {
		term, err := nn.Sum(parameter.Value())
		if err != nil {
			t.Fatal(err)
		}

		if objective == nil {
			objective = term
		} else {
			objective, err = nn.Add(objective, term)
			if err != nil {
				t.Fatal(err)
			}
		}
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}
}

func TestSaveLoadWeights_PreservesFloat32BitsAndClearsGradients(t *testing.T) {
	bits := []uint32{0, 1 << 31, 0x7fc12345, 0x7fa12345, 0xffc12345, 0x7f800000, 0xff800000, 1, 0x7f7fffff}

	values := make([]float32, len(bits))
	for i, value := range bits {
		values[i] = math.Float32frombits(value)
	}

	source := []*nn.Parameter{testParameter(t, values, len(values)), testParameter(t, []float32{3})}

	path := filepath.Join(t.TempDir(), "nested", "weights.goml")
	if err := SaveWeights(path, source); err != nil {
		t.Fatal(err)
	}

	target := []*nn.Parameter{testParameter(t, make([]float32, len(values)), len(values)), testParameter(t, []float32{9})}
	seedGradients(t, target)
	identity := target[0].Value()

	original, err := identity.Data().Retain()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = original.Close() }()

	if err := LoadWeights(path, target); err != nil {
		t.Fatal(err)
	}

	for i, value := range backendtest.HostData(t, target[0].Value().Data()) {
		if math.Float32bits(value) != bits[i] {
			t.Fatalf("element %d: bits %08x, want %08x", i, math.Float32bits(value), bits[i])
		}
	}

	if target[0].Value() != identity || target[0].Value().Grad() != nil || target[1].Value().Grad() != nil || backendtest.HostData(t, target[1].Value().Data())[0] != 3 {
		t.Fatal("load changed identity or retained gradients")
	}

	if !slices.Equal(backendtest.HostData(t, original), make([]float32, len(values))) {
		t.Fatal("load mutated old immutable storage")
	}
}

func TestLoadWeights_LateShapeMismatchLeavesAllValuesAndGradientsUnchanged(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.goml")
	if err := SaveWeights(path, []*nn.Parameter{testParameter(t, []float32{1, 2}, 2), testParameter(t, []float32{3})}); err != nil {
		t.Fatal(err)
	}

	target := []*nn.Parameter{testParameter(t, []float32{7, 8}, 2), testParameter(t, []float32{9}, 1)}
	seedGradients(t, target)
	beforeData := []*tensor.Tensor{target[0].Value().Data(), target[1].Value().Data()}

	beforeGrad := []*tensor.Tensor{target[0].Value().Grad(), target[1].Value().Grad()}
	if err := LoadWeights(path, target); err == nil {
		t.Fatal("accepted scalar/vector mismatch")
	}

	for i, parameter := range target {
		if parameter.Value().Data() != beforeData[i] || parameter.Value().Grad() != beforeGrad[i] {
			t.Fatal("failed load changed an earlier parameter")
		}
	}
}

func TestSaveWeights_InvalidParametersDoNotReplaceExistingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "weights.goml")

	parameter := testParameter(t, []float32{1})
	if err := os.WriteFile(path, []byte("original"), 0o600); err != nil {
		t.Fatal(err)
	}

	for _, parameters := range [][]*nn.Parameter{{nil}, {{}}, {parameter, parameter}} {
		if err := SaveWeights(path, parameters); err == nil {
			t.Fatal("accepted invalid parameters")
		}

		data, err := os.ReadFile(path)
		if err != nil || string(data) != "original" {
			t.Fatalf("failed save changed destination: %q %v", data, err)
		}
	}
}

func TestLoadWeights_RejectsInvalidDTypeLengthsShapesAndCounts(t *testing.T) {
	for _, tt := range []struct {
		name   string
		stored []storedTensor
	}{
		{"DType", []storedTensor{{DType: "float64", Shape: []int{0}}}},
		{"Bytes", []storedTensor{{DType: "float32", Data: []byte{1}}}},
		{"ElementCount", []storedTensor{{DType: "float32", Shape: []int{2}, Data: make([]byte, 4)}}},
		{"NegativeShape", []storedTensor{{DType: "float32", Shape: []int{-1}}}},
		{"OverflowShape", []storedTensor{{DType: "float32", Shape: []int{int(^uint(0) >> 1), 2}}}},
		{"ParameterCount", nil},
	} {
		t.Run(tt.name, func(t *testing.T) {
			data, err := encodeDocument(document{Kind: "weights", Parameters: tt.stored})
			if err != nil {
				t.Fatal(err)
			}

			path := filepath.Join(t.TempDir(), "invalid.goml")
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}

			parameter := testParameter(t, []float32{7})
			if err := LoadWeights(path, []*nn.Parameter{parameter}); err == nil {
				t.Fatal("accepted invalid tensor metadata")
			}

			if backendtest.HostData(t, parameter.Value().Data())[0] != 7 {
				t.Fatal("failed load changed weights")
			}
		})
	}
}

func TestSaveLoadWeights_NoncontiguousAndEmptyParameters(t *testing.T) {
	data, err := tensor.From([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		t.Fatal(err)
	}

	data, err = data.Transpose()
	if err != nil {
		t.Fatal(err)
	}

	parameter, err := nn.NewParameter(data)
	if err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "weights.goml")
	if err := SaveWeights(path, []*nn.Parameter{parameter}); err != nil {
		t.Fatal(err)
	}

	target := testParameter(t, make([]float32, 6), 3, 2)
	if err := LoadWeights(path, []*nn.Parameter{target}); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, target.Value().Data()), []float32{1, 4, 2, 5, 3, 6}) {
		t.Fatal("noncontiguous logical order changed")
	}

	if err := SaveWeights(path, nil); err != nil {
		t.Fatal(err)
	}

	if err := LoadWeights(path, nil); err != nil {
		t.Fatal(err)
	}
}
