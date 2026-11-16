package nn

import "github.com/bpradana/goml/tensor"

// Add computes a+b using tensor broadcasting, allocating a new result.
// It records gradients when required; invalid shapes or values return errors.
func Add(a, b *Value) (*Value, error) {
	return binary("add", a, b)
}

// Sub computes a-b with broadcasting and records gradients when required.
func Sub(a, b *Value) (*Value, error) {
	return binary("sub", a, b)
}

// Mul computes a*b with broadcasting and records gradients when required.
func Mul(a, b *Value) (*Value, error) {
	return binary("mul", a, b)
}

// Div computes a/b with broadcasting. IEEE 754 division semantics apply.
func Div(a, b *Value) (*Value, error) {
	return binary("div", a, b)
}

func binary(op string, a, b *Value) (*Value, error) {
	if err := checkValues(op, a, b); err != nil {
		return nil, err
	}

	var (
		data *tensor.Tensor
		err  error
	)

	switch op {
	case "add":
		data, err = tensor.Add(a.data, b.data)
	case "sub":
		data, err = tensor.Sub(a.data, b.data)
	case "mul":
		data, err = tensor.Mul(a.data, b.data)
	case "div":
		data, err = tensor.Div(a.data, b.data)
	}

	if err != nil {
		return nil, err
	}

	out := record(data, a, b)
	if !out.requires {
		return out, nil
	}

	// Save immutable forward values, independent of later parameter updates.
	av, bv := a.data, b.data

	out.backward = func(g *tensor.Tensor) ([]*tensor.Tensor, error) {
		ga, gb := g, g

		var err error

		switch op {
		case "sub":
			gb, err = tensor.Neg(g)
		case "mul":
			ga, err = tensor.Mul(g, bv)
			if err == nil {
				gb, err = tensor.Mul(g, av)
			}
		case "div":
			ga, err = tensor.Div(g, bv)
			if err != nil {
				return nil, err
			}

			var quotient *tensor.Tensor

			quotient, err = tensor.Div(av, bv)
			if err == nil {
				gb, err = tensor.Mul(ga, quotient)
			}

			if err == nil {
				gb, err = tensor.Neg(gb)
			}
		}

		if err != nil {
			return nil, err
		}

		ga, err = reduceGradient(ga, av.Shape())
		if err != nil {
			return nil, err
		}

		gb, err = reduceGradient(gb, bv.Shape())
		if err != nil {
			return nil, err
		}

		return []*tensor.Tensor{ga, gb}, nil
	}

	return out, nil
}

// MatMul computes a new 2D matrix product and records its derivatives.
// Inputs must have compatible inner dimensions.
func MatMul(a, b *Value) (*Value, error) {
	if err := checkValues("matmul", a, b); err != nil {
		return nil, err
	}

	data, err := tensor.MatMul(a.data, b.data)
	if err != nil {
		return nil, err
	}

	out := record(data, a, b)
	if !out.requires {
		return out, nil
	}

	av, bv := a.data, b.data

	out.backward = func(g *tensor.Tensor) ([]*tensor.Tensor, error) {
		var ga, gb *tensor.Tensor

		if a.requires {
			bt, err := bv.Transpose()
			if err != nil {
				return nil, err
			}
			defer func() { _ = bt.Close() }()

			ga, err = tensor.MatMul(g, bt)
			if err != nil {
				return nil, err
			}
		}

		if b.requires {
			at, err := av.Transpose()
			if err != nil {
				return nil, err
			}
			defer func() { _ = at.Close() }()

			gb, err = tensor.MatMul(at, g)
			if err != nil {
				return nil, err
			}
		}

		return []*tensor.Tensor{ga, gb}, nil
	}

	return out, nil
}
