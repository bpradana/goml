package nn

import (
	"fmt"

	"github.com/bpradana/goml/tensor"
)

func sameDevice(a, b *tensor.Tensor) bool {
	first, second := a.Device(), b.Device()
	return (first == nil && second == nil) || (first != nil && second != nil && first.ID() == second.ID())
}

// ToDevice transfers a parameter set atomically, preserving parameter identity
// and deduplicating shared parameters. Gradients are cleared after success.
// Transfer and synchronization failures leave the original parameter set intact.
// Create optimizers after moving parameters; synchronize against training.
func ToDevice(parameters []*Parameter, device tensor.Device) error {
	seen := make(map[*Parameter]bool)
	unique := make([]*Parameter, 0, len(parameters))
	prepared := make([]*tensor.Tensor, 0, len(parameters))

	committed := false
	defer func() {
		if !committed {
			for _, value := range prepared {
				_ = value.Close()
			}
		}
	}()

	for _, parameter := range parameters {
		if parameter == nil || parameter.value == nil {
			return fmt.Errorf("nn: transfer: invalid parameter")
		}

		if seen[parameter] {
			continue
		}

		seen[parameter] = true

		next, err := parameter.value.data.To(device)
		if err != nil {
			return err
		}

		unique = append(unique, parameter)
		prepared = append(prepared, next)
	}

	if device != nil {
		if err := device.Synchronize(); err != nil {
			return err
		}
	}

	for i, parameter := range unique {
		previous := parameter.value.data
		parameter.value.data = prepared[i]
		_ = previous.Close()

		parameter.ZeroGrad()
	}

	committed = true

	return nil
}
