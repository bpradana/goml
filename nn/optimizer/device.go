package optimizer

import (
	"fmt"
	"math"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

type parameterUpdate struct {
	parameter *nn.Parameter
	next      *tensor.Tensor
}

func deviceSGD(parameters []*nn.Parameter, rate float32) error {
	scope, err := tensor.NewScope(parameters[0].Value().Data().Device())
	if err != nil {
		return err
	}
	defer func() { _ = scope.Close() }()

	updates := make([]parameterUpdate, 0, len(parameters))

	committed := false
	defer func() {
		if !committed {
			for _, update := range updates {
				_ = update.next.Close()
			}
		}
	}()

	for _, parameter := range parameters {
		value := parameter.Value()
		if value.Grad() == nil {
			continue
		}

		values, err := scope.To(value.Data())
		if err != nil {
			return err
		}

		gradient, err := scope.To(value.Grad())
		if err != nil {
			return err
		}

		next, err := tensor.SubScaled(values, gradient, rate)
		if err != nil {
			return err
		}

		updates = append(updates, parameterUpdate{parameter: parameter, next: next})
	}

	if err := parameters[0].Value().Data().Device().Synchronize(); err != nil {
		return err
	}

	for _, update := range updates {
		if err := update.parameter.Update(update.next); err != nil {
			return err
		}
	}

	committed = true

	return nil
}

type deviceAdamUpdate struct {
	parameter           *nn.Parameter
	state               *adamState
	next, first, second *tensor.Tensor
	initial             bool
}

func (a *Adam) deviceStep() error {
	scope, err := tensor.NewScope(a.parameters[0].Value().Data().Device())
	if err != nil {
		return err
	}
	defer func() { _ = scope.Close() }()

	updates := make([]deviceAdamUpdate, 0, len(a.parameters))

	committed := false
	defer func() {
		if !committed {
			for _, update := range updates {
				_ = update.next.Close()
				_ = update.first.Close()

				_ = update.second.Close()
				if update.initial {
					_ = update.state.gpuFirst.Close()
					_ = update.state.gpuSecond.Close()
				}
			}
		}
	}()

	for _, parameter := range a.parameters {
		value := parameter.Value()
		if value.Grad() == nil {
			continue
		}

		values, err := scope.To(value.Data())
		if err != nil {
			return err
		}

		state := a.state[parameter]

		initial := state == nil
		if initial {
			first, err := tensor.FullLike(values, 0, values.Shape()...)
			if err != nil {
				return err
			}

			second, err := tensor.FullLike(values, 0, values.Shape()...)
			if err != nil {
				_ = first.Close()
				return err
			}

			state = &adamState{gpuFirst: first, gpuSecond: second}
		}

		if state.gpuFirst == nil || state.gpuSecond == nil {
			return fmt.Errorf("nn: adam: recreate the optimizer after changing parameter devices")
		}

		gradient, err := scope.To(value.Grad())
		if err != nil {
			return err
		}

		firstView, err := scope.To(state.gpuFirst)
		if err != nil {
			return err
		}

		secondView, err := scope.To(state.gpuSecond)
		if err != nil {
			return err
		}

		scopedState := &adamState{gpuFirst: firstView, gpuSecond: secondView, steps: state.steps}

		next, first, second, err := a.computeDeviceUpdate(values, gradient, scopedState)
		if err != nil {
			if initial {
				_ = state.gpuFirst.Close()
				_ = state.gpuSecond.Close()
			}

			return err
		}

		updates = append(updates, deviceAdamUpdate{
			parameter: parameter, state: state, next: next,
			first: first, second: second, initial: initial,
		})
	}

	if err := a.parameters[0].Value().Data().Device().Synchronize(); err != nil {
		return err
	}

	for _, update := range updates {
		if err := update.parameter.Update(update.next); err != nil {
			return err
		}

		update.first.Keep()
		update.second.Keep()

		oldFirst, oldSecond := update.state.gpuFirst, update.state.gpuSecond
		update.state.gpuFirst, update.state.gpuSecond = update.first, update.second
		update.state.steps++
		a.state[update.parameter] = update.state

		_ = oldFirst.Close()
		_ = oldSecond.Close()
	}

	committed = true

	return nil
}

func (a *Adam) computeDeviceUpdate(values, gradient *tensor.Tensor, state *adamState) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	firstDecay, err := tensor.Scale(state.gpuFirst, a.beta1)
	if err != nil {
		return nil, nil, nil, err
	}

	firstGradient, err := tensor.Scale(gradient, 1-a.beta1)
	if err != nil {
		return nil, nil, nil, err
	}

	first, err := tensor.Add(firstDecay, firstGradient)
	if err != nil {
		return nil, nil, nil, err
	}

	squares, err := tensor.Mul(gradient, gradient)
	if err != nil {
		return nil, nil, nil, err
	}

	secondGradient, err := tensor.Scale(squares, 1-a.beta2)
	if err != nil {
		return nil, nil, nil, err
	}

	secondDecay, err := tensor.Scale(state.gpuSecond, a.beta2)
	if err != nil {
		return nil, nil, nil, err
	}

	second, err := tensor.Add(secondDecay, secondGradient)
	if err != nil {
		return nil, nil, nil, err
	}

	step := state.steps + 1
	correction1 := float32(1 - math.Pow(float64(a.beta1), float64(step)))
	correction2 := float32(1 - math.Pow(float64(a.beta2), float64(step)))

	correctedFirst, err := tensor.Scale(first, 1/correction1)
	if err != nil {
		return nil, nil, nil, err
	}

	correctedSecond, err := tensor.Scale(second, 1/correction2)
	if err != nil {
		return nil, nil, nil, err
	}

	deviation, err := tensor.Sqrt(correctedSecond)
	if err != nil {
		return nil, nil, nil, err
	}

	denominator, err := tensor.AddScalar(deviation, a.epsilon)
	if err != nil {
		return nil, nil, nil, err
	}

	direction, err := tensor.Div(correctedFirst, denominator)
	if err != nil {
		return nil, nil, nil, err
	}

	next, err := tensor.SubScaled(values, direction, a.learningRate)

	return next, first, second, err
}

// Close releases Adam's device moments and invalidates this optimizer. It does
// not close model parameters. CPU moment storage is released to Go's collector.
func (a *Adam) Close() error {
	for _, state := range a.state {
		if state.gpuFirst != nil {
			_ = state.gpuFirst.Close()
			_ = state.gpuSecond.Close()
		}
	}

	a.state = nil

	return nil
}
