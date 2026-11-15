package layer

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/nn"
)

func TestConv2D_ChannelMixingAndNCHWOutputOrder(t *testing.T) {
	convolution, err := NewConv2D(2, 2, 1, 1, 0, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	setParameter(t, convolution.Parameters()[0], []float32{1, 2, -1, 0.5})
	setParameter(t, convolution.Parameters()[1], []float32{0.1, -0.2})
	input := layerValue(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, -1, 0, 1, 2, -3, -2, -1, 0}, []int{2, 2, 2, 2}, false)

	output, err := convolution.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	want := []float32{-3.9, -3.9, -3.9, -3.9, 4.3, 6.8, 9.3, 11.8, 2.1, 2.1, 2.1, 2.1, -3.7, -1.2, 1.3, 3.8}
	for i, value := range output.Data().Data() {
		if math.Abs(float64(value-want[i])) > 1e-5 {
			t.Fatalf("element %d: got %g, want %g", i, value, want[i])
		}
	}
}

func TestConv2D_ReferenceCrossCorrelationAndOverlapGradient(t *testing.T) {
	convolution, err := NewConv2D(1, 1, 2, 1, 0, true, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	setParameter(t, convolution.Parameters()[0], []float32{1, 1, 1, 1})
	setParameter(t, convolution.Parameters()[1], []float32{1})
	input := layerValue(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{1, 1, 3, 3}, true)

	output, err := convolution.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(output.Data().Shape(), []int{1, 1, 2, 2}) || !slices.Equal(output.Data().Data(), []float32{13, 17, 25, 29}) {
		t.Fatal(output.Data())
	}

	objective, err := nn.Sum(output)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(input.Grad().Data(), []float32{1, 2, 1, 2, 4, 2, 1, 2, 1}) {
		t.Fatal(input.Grad())
	}

	if !slices.Equal(convolution.Parameters()[0].Value().Grad().Data(), []float32{12, 16, 24, 28}) || convolution.Parameters()[1].Value().Grad().Data()[0] != 4 {
		t.Fatal("incorrect kernel or bias gradient")
	}
}

func TestConv2D_StridePaddingChannelsAndBatchesMatchFiniteDifferences(t *testing.T) {
	convolution, err := NewConv2D(2, 2, 2, 2, 1, true, rand.New(rand.NewPCG(3, 4)))
	if err != nil {
		t.Fatal(err)
	}

	values := make([]float32, 2*2*3*3)
	for i := range values {
		values[i] = float32(i%11-5) * 0.1
	}

	checkLayerGradients(t, convolution, values, []int{2, 2, 3, 3})
}

func TestNewConv2D_RejectsInvalidConfiguration(t *testing.T) {
	for _, configuration := range [][5]int{{0, 1, 1, 1, 0}, {1, 0, 1, 1, 0}, {1, 1, 0, 1, 0}, {1, 1, 1, 0, 0}, {1, 1, 1, 1, -1}, {2, 1, int(^uint(0) >> 1), 1, 0}} {
		if _, err := NewConv2D(configuration[0], configuration[1], configuration[2], configuration[3], configuration[4], false, rand.New(rand.NewPCG(1, 2))); err == nil {
			t.Fatalf("accepted %v", configuration)
		}
	}

	if _, err := NewConv2D(1, 1, 1, 1, 0, false, nil); err == nil {
		t.Fatal("accepted nil generator")
	}
}

func TestConv2D_RejectsInvalidInputsAndSupportsEmptyBatch(t *testing.T) {
	convolution, err := NewConv2D(1, 1, 2, 1, 0, false, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	for _, input := range []*nn.Value{nil, {}, layerValue(t, []float32{1}, []int{1, 1, 1, 1}, false), layerValue(t, []float32{1, 2}, []int{1, 2}, false)} {
		if _, err := convolution.Forward(input); err == nil {
			t.Fatal("accepted invalid input")
		}
	}

	input := layerValue(t, nil, []int{0, 1, 3, 3}, true)

	output, err := convolution.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(output.Data().Shape(), []int{0, 1, 2, 2}) {
		t.Fatal(output.Data())
	}

	objective, err := nn.Sum(output)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}
}
