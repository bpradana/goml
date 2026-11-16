package tensor

import (
	"fmt"
	"math"
	"runtime"
	"slices"

	"github.com/bpradana/goml/tensor/backend"
)

// SubScaled allocates a-factor*b for equally shaped tensors, including views.
// The scaled product is rounded to Float32 before subtraction, as with Scale
// followed by Sub. Invalid inputs or unequal shapes return errors.
func SubScaled(a, b *Tensor, factor float32) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{a, b})

	if err := valid("subscaled", a, b); err != nil {
		return nil, err
	}

	if !slices.Equal(a.shape, b.shape) {
		return nil, fmt.Errorf("tensor: subscaled: shapes %v and %v differ", a.shape, b.shape)
	}

	if err := compatible("subscaled", a, b); err != nil {
		return nil, err
	}

	compute, ok := a.device.(backend.SubScaledBackend)
	if !ok {
		scaled, err := Scale(b, factor)
		if err != nil {
			return nil, err
		}
		defer func() { _ = scaled.Close() }()

		return Sub(a, scaled)
	}

	out, err := allocatedOn(a.shape, a, b)
	if err != nil {
		return nil, err
	}

	if err := compute.SubScaled(out.deviceView(), a.deviceView(), b.deviceView(), factor); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}

// AdamUpdate allocates a bias-corrected Adam parameter update, without changing
// values or gradient. First and second are caller-owned, independent moment
// buffers in logical element order; only those buffers are updated in place.
// Shapes and moment lengths must match. Step must be positive, rate and epsilon
// finite and positive, and betas in [0,1). Validation errors leave moments intact.
func AdamUpdate(values, gradient *Tensor, first, second []float32, rate, beta1, beta2, epsilon float32, step uint64) (*Tensor, error) {
	defer runtime.KeepAlive([]*Tensor{values, gradient})

	if err := valid("adam", values, gradient); err != nil {
		return nil, err
	}

	if !slices.Equal(values.shape, gradient.shape) || len(first) != values.size || len(second) != values.size {
		return nil, fmt.Errorf("tensor: adam: parameter, gradient, and moment shapes must match")
	}

	if step == 0 || !(rate > 0) || math.IsInf(float64(rate), 0) || !(epsilon > 0) || math.IsInf(float64(epsilon), 0) ||
		!(beta1 >= 0 && beta1 < 1 && beta2 >= 0 && beta2 < 1) {
		return nil, fmt.Errorf("tensor: adam: invalid step or hyperparameters")
	}

	if err := compatible("adam", values, gradient); err != nil {
		return nil, err
	}

	compute, ok := values.device.(backend.HostAdamBackend)
	if !ok {
		return nil, unsupported("host Adam", values.device)
	}

	out, err := allocatedOn(values.shape, values, gradient)
	if err != nil {
		return nil, err
	}

	options := backend.AdamOptions{Rate: rate, Beta1: beta1, Beta2: beta2, Epsilon: epsilon, Step: step}
	if err := compute.Adam(out.deviceView(), values.deviceView(), gradient.deviceView(), first, second, options); err != nil {
		_ = out.Close()
		return nil, err
	}

	return out, nil
}
