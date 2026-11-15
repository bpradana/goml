package layer

import "github.com/bpradana/goml/nn"

// Layer is the composition boundary for Sequential.
// Forward allocates a new value or view; Parameters returns the trainable leaves.
type Layer interface {
	Forward(*nn.Value) (*nn.Value, error)
	Parameters() []*nn.Parameter
}
