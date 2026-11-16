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

	seed, err := tensor.FullLike(v.data, 1)
	if err != nil {
		return err
	}

	return v.BackwardWith(seed)
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

	if seed != nil && !sameDevice(seed, v.data) {
		return fmt.Errorf("nn: backward: seed device must match output")
	}

	if seed == nil || seed.IsClosed() || !slices.Equal(seed.Shape(), v.data.Shape()) || seed.Numel() != v.data.Numel() {
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
	for _, node := range order {
		g := adjoints[node]

		if node.grad != nil {
			var err error

			g, err = tensor.Add(node.grad, g)
			if err != nil {
				return err
			}
		}

		adjoints[node] = g
	}

	if device := v.data.Device(); device != nil {
		if err := device.Synchronize(); err != nil {
			return err
		}
	}

	// Leaves outlive individual forward scopes. Give them independent leases
	// so parameter gradients can accumulate across batches until ZeroGrad.
	owned := make(map[*Value]*tensor.Tensor)

	committed := false
	defer func() {
		if !committed {
			for _, gradient := range owned {
				_ = gradient.Close()
			}
		}
	}()

	for _, node := range order {
		if len(node.parents) == 0 && node.data.Device() != nil {
			gradient, err := adjoints[node].To(node.data.Device())
			if err != nil {
				return err
			}

			owned[node] = gradient
		}
	}

	for _, node := range order {
		if gradient := owned[node]; gradient != nil {
			node.ZeroGrad()
			node.grad, node.ownsGrad = gradient, true
		} else {
			node.grad = adjoints[node]
		}

		if node.backward != nil {
			node.parents = nil
			node.backward = nil
			node.consumed = true
		}
	}

	committed = true

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

	if !g.IsContiguous() {
		var err error

		g, err = g.MakeContiguous()
		if err != nil {
			return nil, err
		}
	}

	return g.Reshape(shape...)
}
