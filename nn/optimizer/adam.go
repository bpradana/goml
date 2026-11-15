package optimizer

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"

	"github.com/bpradana/goml/tensor"
)

type adamState struct {
	first  []float32
	second []float32
	steps  uint64
}

// Adam owns first and second moment estimates for its parameters.
type Adam struct {
	parameters   []*nn.Parameter
	learningRate float32
	beta1        float32
	beta2        float32
	epsilon      float32
	state        map[*nn.Parameter]*adamState
}

// NewAdam copies and deduplicates parameters, validating all hyperparameters.
// The rate and epsilon must be finite and positive; betas must lie in [0,1).
// Common settings are beta1=0.9, beta2=0.999, epsilon=1e-8.
// State is allocated lazily when a parameter first receives a gradient.
func NewAdam(parameters []*nn.Parameter, learningRate, beta1, beta2, epsilon float32) (*Adam, error) {
	unique, err := optimizerParameters(parameters, learningRate)
	if err != nil {
		return nil, err
	}

	if !finite(beta1) || !finite(beta2) || !finite(epsilon) ||
		beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || epsilon <= 0 {
		return nil, fmt.Errorf("nn: adam: betas must be finite in [0,1) and epsilon finite and positive")
	}

	return &Adam{
		parameters:   unique,
		learningRate: learningRate,
		beta1:        beta1,
		beta2:        beta2,
		epsilon:      epsilon,
		state:        make(map[*nn.Parameter]*adamState),
	}, nil
}

// Step applies bias-corrected Adam updates, preserving parameter identity.
// Missing gradients do not advance that parameter's moments or step counter.
// Gradients remain until ZeroGrad. Updates require external synchronization.
func (a *Adam) Step() error {
	if a.state == nil {
		return fmt.Errorf("nn: adam: use NewAdam to initialize the optimizer")
	}

	for _, parameter := range a.parameters {
		v := parameter.Value()
		if v.Grad() == nil {
			continue
		}

		state := a.state[parameter]
		if state == nil {
			state = &adamState{
				first:  make([]float32, v.Data().Numel()),
				second: make([]float32, v.Data().Numel()),
			}
			a.state[parameter] = state
		}

		state.steps++

		firstCorrection := 1 - math.Pow(float64(a.beta1), float64(state.steps))
		secondCorrection := 1 - math.Pow(float64(a.beta2), float64(state.steps))
		values, gradient := v.Data().Data(), v.Grad().Data()

		for i, g := range gradient {
			state.first[i] = a.beta1*state.first[i] + (1-a.beta1)*g
			state.second[i] = a.beta2*state.second[i] + (1-a.beta2)*g*g
			first := float64(state.first[i]) / firstCorrection
			second := float64(state.second[i]) / secondCorrection
			values[i] -= float32(float64(a.learningRate) * first / (math.Sqrt(second) + float64(a.epsilon)))
		}

		next, err := tensor.From(values, v.Data().Shape()...)
		if err != nil {
			return err
		}

		if err := parameter.Update(next); err != nil {
			return err
		}
	}

	return nil
}

// ZeroGrad clears all gradients owned by this optimizer's parameters.
func (a *Adam) ZeroGrad() {
	for _, parameter := range a.parameters {
		parameter.ZeroGrad()
	}
}
