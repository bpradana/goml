//go:build darwin && cgo

package examples

import (
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
)

func TestCheckpoint_MetalWeightsRestoreOnCPUAndGPUWithoutRetainingStagingBuffers(t *testing.T) {
	dense, err := layer.NewDense(2, 3, true, rand.New(rand.NewPCG(1, 3)))
	if err != nil {
		t.Fatal(err)
	}

	model, err := layer.NewSequential(dense)
	if err != nil {
		t.Fatal(err)
	}

	device := deviceForTest(t)
	if err := nn.ToDevice(model.Parameters(), device); err != nil {
		t.Fatal(err)
	}

	path := filepath.Join(t.TempDir(), "weights.goml")
	if err := checkpoint.SaveWeights(path, model.Parameters()); err != nil {
		t.Fatal(err)
	}

	encoded, err := checkpoint.Marshal(model)
	if err != nil {
		t.Fatal(err)
	}

	cpu, err := checkpoint.Unmarshal(encoded)
	if err != nil {
		t.Fatal(err)
	}

	baseline := device.Stats().LiveBytes

	identity := model.Parameters()[0].Value()
	for range 3 {
		if err := checkpoint.LoadWeights(path, model.Parameters()); err != nil {
			t.Fatal(err)
		}

		if err := device.Synchronize(); err != nil {
			t.Fatal(err)
		}

		if device.Stats().LiveBytes != baseline {
			t.Fatalf("checkpoint retained staging allocations: %+v", device.Stats())
		}
	}

	if model.Parameters()[0].Value() != identity {
		t.Fatal("restoration changed parameter identity")
	}

	for index, parameter := range cpu.Parameters() {
		if !parameter.Value().Data().IsCPU() {
			t.Fatal("portable checkpoint did not restore onto CPU")
		}

		compareTensor(t, model.Parameters()[index].Value().Data(), parameter.Value().Data(), 0)
	}
}
