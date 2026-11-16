package nn

import (
	"fmt"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

func sameDevice(a, b *tensor.Tensor) bool {
	first, second := a.Device(), b.Device()
	return first != nil && second != nil && first.ID() == second.ID()
}

// ToDevice transfers a parameter set atomically, preserving parameter identity
// and deduplicating shared parameters. Gradients are cleared after success.
// Transfer and synchronization failures leave the original parameter set intact.
// Create optimizers after moving parameters; synchronize against training.
// Cross-device transfers reject active parameter graphs; consume them first.
func ToDevice(parameters []*Parameter, device backend.Device) error {
	if device == nil {
		return fmt.Errorf("nn: transfer: device required; use tensor.CPU()")
	}

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

		if device.ID() != parameter.value.data.Device().ID() && parameter.value.parameter.inUse() {
			return fmt.Errorf("nn: transfer: consume active parameter graphs before changing devices")
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

	if err := device.Synchronize(); err != nil {
		return err
	}

	for i, parameter := range unique {
		parameter.replace(prepared[i])

		parameter.ZeroGrad()
	}

	committed = true

	return nil
}
