package loss

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

// CrossEntropy computes mean classification loss from logits.
// It has no mutable state or trainable parameters.
type CrossEntropy struct{}

// NewCrossEntropy constructs a classification cross-entropy loss.
func NewCrossEntropy() *CrossEntropy {
	return &CrossEntropy{}
}

// Forward returns a scalar using stable shifted log-sum-exp.
// Prediction must contain finite logits [batch,classes] with positive dimensions.
// Target contains one integer class index in [0,classes) per row.
// Only logits are differentiated. Targets are not retained after Forward.
// Invalid inputs or consumed graphs return errors. Inference disables recording.
func (*CrossEntropy) Forward(prediction *nn.Value, target []int) (*nn.Value, error) {
	if prediction == nil || prediction.Data() == nil {
		return nil, fmt.Errorf("nn/loss: cross-entropy: nil or uninitialized value")
	}

	data := prediction.Data()
	shape := data.Shape()

	if len(shape) != 2 || shape[0] == 0 || shape[1] == 0 || len(target) != shape[0] {
		return nil, fmt.Errorf("nn/loss: cross-entropy: expected nonempty [batch,classes] logits and batch targets, got %v and %d targets", shape, len(target))
	}

	input := data.Data()
	rows, columns := shape[0], shape[1]

	var (
		totalLoss float64
		gradient  []float32
	)

	if prediction.RequiresGrad() {
		gradient = make([]float32, len(input))
	}

	for row, label := range target {
		if label < 0 || label >= columns {
			return nil, fmt.Errorf("nn/loss: cross-entropy: target %d at row %d outside [0,%d)", label, row, columns)
		}

		maximum := math.Inf(-1)

		for _, x := range input[row*columns : (row+1)*columns] {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				return nil, fmt.Errorf("nn/loss: cross-entropy: nonfinite logit at row %d", row)
			}

			maximum = math.Max(maximum, float64(x))
		}

		var total float64

		for column := 0; column < columns; column++ {
			total += math.Exp(float64(input[row*columns+column]) - maximum)
		}

		totalLoss += math.Log(total) + (maximum - float64(input[row*columns+label]))

		if gradient != nil {
			for column := 0; column < columns; column++ {
				probability := math.Exp(float64(input[row*columns+column])-maximum) / total
				if column == label {
					probability--
				}

				gradient[row*columns+column] = float32(probability / float64(rows))
			}
		}
	}

	result := tensor.Scalar(float32(totalLoss / float64(rows)))
	if gradient == nil {
		return nn.NewOperation(result, prediction, nil)
	}

	saved, err := tensor.From(gradient, shape...)
	if err != nil {
		return nil, err
	}

	return nn.NewOperation(result, prediction, func(g *tensor.Tensor) (*tensor.Tensor, error) {
		return tensor.Scale(saved, g.Data()[0])
	})
}
