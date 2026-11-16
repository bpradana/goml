package main

import (
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/tensor"
)

func predict(model *layer.Sequential, data *tensor.Tensor, targets []int, batch int) (int, float32, error) {
	criterion := loss.NewCrossEntropy()
	correct := 0

	var total float32

	for start := 0; start < len(targets); start += batch {
		end := min(start+batch, len(targets))

		images, err := data.Slice(0, start, end)
		if err != nil {
			return 0, 0, err
		}

		logits, err := model.Predict(images)
		if err != nil {
			return 0, 0, err
		}

		value, err := nn.NewValue(logits, false)
		if err != nil {
			return 0, 0, err
		}

		objective, err := criterion.Forward(value, targets[start:end])
		if err != nil {
			return 0, 0, err
		}

		lossValue1, err := objective.Data().At()
		if err != nil {
			return 0, 0, err
		}

		total += lossValue1 * float32(end-start)
		classes := logits.Shape()[1]

		scores, err := logits.HostData()
		if err != nil {
			return 0, 0, err
		}

		for row := range end - start {
			best := 0
			for class := 1; class < classes; class++ {
				if scores[row*classes+class] > scores[row*classes+best] {
					best = class
				}
			}

			if best == targets[start+row] {
				correct++
			}
		}
	}

	return correct, total / float32(len(targets)), nil
}
