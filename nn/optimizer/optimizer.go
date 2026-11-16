package optimizer

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"
)

// Optimizer is the training-loop boundary shared by SGD and Adam.
type Optimizer interface {
	Step() error
	ZeroGrad()
}

func finite(x float32) bool {
	return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0)
}

func optimizerParameters(parameters []*nn.Parameter, learningRate float32) ([]*nn.Parameter, error) {
	if !finite(learningRate) || learningRate <= 0 {
		return nil, fmt.Errorf("nn: optimizer: learning rate must be finite and positive")
	}

	var unique []*nn.Parameter

	seen := make(map[*nn.Parameter]bool)

	for i, parameter := range parameters {
		if parameter == nil || parameter.Value() == nil {
			return nil, fmt.Errorf("nn: optimizer: nil or uninitialized parameter at index %d", i)
		}

		if parameter.Value().Data().IsClosed() {
			return nil, fmt.Errorf("nn: optimizer: parameter %d has closed device storage", i)
		}

		if len(unique) > 0 {
			first, current := unique[0].Value().Data().Device(), parameter.Value().Data().Device()
			if first.ID() != current.ID() {
				return nil, fmt.Errorf("nn: optimizer: parameters must use one device")
			}
		}

		if !seen[parameter] {
			seen[parameter] = true
			unique = append(unique, parameter)
		}
	}

	return unique, nil
}
