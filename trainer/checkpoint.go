package trainer

import (
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
)

func saveModel(path string, model Model) error {
	if sequential, ok := model.(*layer.Sequential); ok {
		return checkpoint.Save(path, sequential)
	}

	// The existing codec supports arbitrary model weights without inventing
	// architecture registration or a second checkpoint format.
	return checkpoint.SaveWeights(path, model.Parameters())
}
