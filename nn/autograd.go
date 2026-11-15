package nn

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/tensor"
)

// Backward differentiates a scalar value using a seed of one.
// Gradients accumulate on tracked values until cleared. Non-scalar outputs,
// inference values, and consumed graphs return errors. Successful execution
// releases saved graph resources; create a new forward pass to differentiate again.
func (v *Value) Backward() error {
	if err := checkValues("backward", v); err != nil {
		return err
	}

	if v.data.Rank() != 0 {
		return fmt.Errorf("nn: backward: expected scalar, got shape %v", v.data.Shape())
	}

	return v.BackwardWith(tensor.Scalar(1))
}

// BackwardWith computes a vector-Jacobian product with an explicit seed.
// The seed shape must equal the output shape. It consumes the graph and
// accumulates gradients, following the same lifecycle as Backward.
func (v *Value) BackwardWith(seed *tensor.Tensor) error {
	if err := checkValues("backward", v); err != nil {
		return err
	}

	if !v.requires {
		return fmt.Errorf("nn: backward: value does not track gradients")
	}

	if seed == nil || !slices.Equal(seed.Shape(), v.data.Shape()) || seed.Numel() != v.data.Numel() {
		return fmt.Errorf("nn: backward: seed must match output shape %v", v.data.Shape())
	}

	var order []*Value

	seen := make(map[*Value]bool)

	var visit func(*Value) error

	visit = func(node *Value) error {
		if seen[node] {
			return nil
		}

		if node.consumed {
			return fmt.Errorf("nn: backward: dependency graph already consumed")
		}

		seen[node] = true

		for _, parent := range node.parents {
			if parent.requires {
				if err := visit(parent); err != nil {
					return err
				}
			}
		}

		order = append(order, node)

		return nil
	}

	if err := visit(v); err != nil {
		return err
	}

	// Per-pass adjoints prevent old accumulated intermediate gradients from
	// being propagated through a new branch a second time.
	adjoints := map[*Value]*tensor.Tensor{v: seed}

	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		g := adjoints[node]

		if node.backward == nil {
			continue
		}

		grads, err := node.backward(g)
		if err != nil {
			return fmt.Errorf("nn: backward: %w", err)
		}

		for j, parent := range node.parents {
			if !parent.requires {
				continue
			}

			if previous := adjoints[parent]; previous != nil {
				grads[j], err = tensor.Add(previous, grads[j])
				if err != nil {
					return err
				}
			}

			adjoints[parent] = grads[j]
		}
	}

	// Commit only after every backward operation succeeds.
	accumulated := make(map[*Value]*tensor.Tensor, len(order))

	for _, node := range order {
		g := adjoints[node]

		if node.grad != nil {
			var err error

			g, err = tensor.Add(node.grad, g)
			if err != nil {
				return err
			}
		}

		accumulated[node] = g
	}

	for _, node := range order {
		node.grad = accumulated[node]

		if node.backward != nil {
			node.parents = nil
			node.backward = nil
			node.consumed = true
		}
	}

	return nil
}

func reduceGradient(g *tensor.Tensor, shape []int) (*tensor.Tensor, error) {
	current := g.Shape()

	var axes []int

	leading := len(current) - len(shape)

	for i, d := range current {
		if i < leading || (shape[i-leading] == 1 && d != 1) {
			axes = append(axes, i)
		}
	}

	if len(axes) > 0 {
		var err error

		g, err = tensor.Sum(g, axes...)
		if err != nil {
			return nil, err
		}
	}

	return g.Contiguous().Reshape(shape...)
}
