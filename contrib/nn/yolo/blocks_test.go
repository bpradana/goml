package yolo

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/tensor"
)

func identityNetwork(t *testing.T, channels int) *layer.Sequential {
	t.Helper()

	conv, err := layer.NewConv2D(channels, channels, 1, 1, 0, false, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		t.Fatal(err)
	}

	values := make([]float32, channels*channels)
	for i := range channels {
		values[i*channels+i] = 1
	}

	data, _ := tensor.From(values, channels, channels)
	if err := conv.Parameters()[0].Update(data); err != nil {
		t.Fatal(err)
	}

	network, _ := layer.NewSequential(conv)

	return network
}

func TestC2f_SplitResidualAndConcatenationOrder(t *testing.T) {
	for _, shortcut := range []bool{false, true} {
		identity := identityNetwork(t, 1)
		passthrough, _ := layer.NewSequential()
		block := &c2f{first: passthrough, last: passthrough, hidden: 1, blocks: []bottleneck{{first: identity, second: identity, shortcut: shortcut}}}
		data, _ := tensor.From([]float32{1, 2, 3, 4}, 1, 2, 1, 2)
		input, _ := nn.Inference(data)

		output, err := block.Forward(input)
		if err != nil {
			t.Fatal(err)
		}

		want := []float32{1, 2, 3, 4, 3, 4}
		if shortcut {
			want = []float32{1, 2, 3, 4, 6, 8}
		}

		if !slices.Equal(backendtest.HostData(t, output.Data()), want) || output.RequiresGrad() || !output.IsInference() {
			t.Fatal(output.Data())
		}
	}
}

func TestSPPF_RepeatedPoolingPreservesBranchOrder(t *testing.T) {
	passthrough, _ := layer.NewSequential()
	pool, _ := layer.NewMaxPool2D(3, 1, 1)
	block := &sppf{first: passthrough, last: passthrough, pool: pool}
	data, _ := tensor.From([]float32{1, 2, 3, 4}, 1, 1, 2, 2)
	input, _ := nn.Inference(data)

	output, err := block.Forward(input)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(backendtest.HostData(t, output.Data()), []float32{1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}) || output.RequiresGrad() || !output.IsInference() {
		t.Fatal(output.Data())
	}
}
