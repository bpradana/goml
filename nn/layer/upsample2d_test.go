package layer

import (
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func TestUpsample2D_BilinearAndBicubicMatchHalfPixelReferences(t *testing.T) {
	for _, test := range []struct {
		mode UpsampleMode
		want []float32
	}{
		{Bilinear, []float32{0, 0.25, 0.75, 1, 0, 0.25, 0.75, 1}},
		{Bicubic, []float32{-0.10546875, 0.2265625, 0.7734375, 1.10546875, -0.10546875, 0.2265625, 0.7734375, 1.10546875}},
	} {
		t.Run(string(test.mode), func(t *testing.T) {
			stage, _ := NewUpsample2D(2, test.mode)
			input := layerValue(t, []float32{0, 1}, []int{1, 1, 1, 2}, false)

			output, err := stage.Forward(input)
			if err != nil {
				t.Fatal(err)
			}

			for i, got := range output.Data().Data() {
				if math.Abs(float64(got-test.want[i])) > 1e-6 {
					t.Fatalf("pixel %d: got %g want %g", i, got, test.want[i])
				}
			}

			checkLayerGradients(t, stage, []float32{-1, 2, 3, 0.5, 4, -2}, []int{1, 1, 2, 3})
		})
	}

	if _, err := NewUpsample2D(2, "invalid"); err == nil {
		t.Fatal("unknown mode accepted")
	}
}

func TestUpsample2D_RepeatsPixelsAndAccumulatesGradients(t *testing.T) {
	stage, _ := NewUpsample2D(2, Nearest)
	input := layerValue(t, []float32{1, 2, 3, 4}, []int{1, 2, 1, 2}, true)

	output, err := stage.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(output.Data().Shape(), []int{1, 2, 2, 4}) || !slices.Equal(output.Data().Data(), []float32{1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4}) {
		t.Fatal(output.Data())
	}

	seed, _ := tensor.Ones(output.Data().Shape()...)
	if err := output.BackwardWith(seed); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(input.Grad().Data(), []float32{4, 4, 4, 4}) {
		t.Fatal(input.Grad())
	}

	checkLayerGradients(t, stage, []float32{1, 2, 3, 4}, []int{1, 1, 2, 2})
}

func TestUpsample2D_InferenceIdentityAndEmptyDimensions(t *testing.T) {
	stage, _ := NewUpsample2D(1, Nearest)

	for _, shape := range [][]int{{1, 1, 2, 2}, {0, 3, 2, 2}, {1, 2, 0, 3}} {
		data, _ := tensor.Ones(shape...)
		input, _ := nn.Inference(data)

		output, err := stage.Forward(input)
		if err != nil {
			t.Fatal(err)
		}

		if output.RequiresGrad() || !output.IsInference() || !slices.Equal(output.Data().Shape(), shape) || !slices.Equal(output.Data().Data(), data.Data()) || len(stage.Parameters()) != 0 {
			t.Fatal(output.Data())
		}
	}
}

func TestUpsample2D_RejectsInvalidScaleRankAndOverflow(t *testing.T) {
	if _, err := NewUpsample2D(0, Nearest); err == nil {
		t.Fatal("zero scale accepted")
	}

	stage, _ := NewUpsample2D(2, Nearest)
	if _, err := stage.Forward(nil); err == nil {
		t.Fatal("nil input accepted")
	}

	if _, err := stage.Forward(layerValue(t, []float32{1}, []int{1}, false)); err == nil {
		t.Fatal("rank one accepted")
	}

	data, _ := tensor.Zeros(0, 1, int(^uint(0)>>1), 1)

	input, _ := nn.Inference(data)
	if _, err := stage.Forward(input); err == nil {
		t.Fatal("overflow accepted")
	}
}

func TestUpsample2D_NearestPreservesSpecialValues(t *testing.T) {
	values := []float32{float32(math.Copysign(0, -1)), float32(math.NaN()), float32(math.Inf(-1)), float32(math.Inf(1))}
	input := layerValue(t, values, []int{1, 1, 1, 4}, false)
	stage, _ := NewUpsample2D(2, Nearest)

	output, err := stage.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	for index, got := range output.Data().Data() {
		want := values[(index%8)/2]
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatal(index, got, want)
		}
	}
}
