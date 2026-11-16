package optimizer

import (
	"fmt"

	"github.com/bpradana/goml/nn"

	"github.com/bpradana/goml/tensor"
)

// SGD implements gradient descent without momentum.
type SGD struct {
	parameters   []*nn.Parameter
	learningRate float32
}

// NewSGD copies and deduplicates parameters. The rate must be finite and positive.
// Nil or uninitialized parameters return an error.
func NewSGD(parameters []*nn.Parameter, learningRate float32) (*SGD, error) {
	unique, err := optimizerParameters(parameters, learningRate)
	if err != nil {
		return nil, err
	}

	return &SGD{parameters: unique, learningRate: learningRate}, nil
}

// Step replaces parameter tensors with value-learningRate*gradient.
// Missing gradients are skipped; gradients are retained until ZeroGrad.
// It preserves parameter identity and requires external synchronization.
func (s *SGD) Step() error {
	if s.learningRate <= 0 {
		return fmt.Errorf("nn: sgd: use NewSGD to initialize the optimizer")
	}

	if len(s.parameters) > 0 && !s.parameters[0].Value().Data().HostAccessible() {
		return deviceSGD(s.parameters, s.learningRate)
	}

	for _, parameter := range s.parameters {
		v := parameter.Value()
		if v.Grad() == nil {
			continue
		}

		next, err := tensor.SubScaled(v.Data(), v.Grad(), s.learningRate)
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
func (s *SGD) ZeroGrad() {
	for _, parameter := range s.parameters {
		parameter.ZeroGrad()
	}
}
