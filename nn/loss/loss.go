package loss

import "github.com/bpradana/goml/nn"

// Loss computes a differentiable scalar from predictions and a typed target.
// Implementations document their shape, reduction, and target constraints.
// Losses with the same target type can be substituted within a training loop.
type Loss[T any] interface {
	Forward(prediction *nn.Value, target T) (*nn.Value, error)
}
