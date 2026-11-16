package yolo

import (
	"fmt"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func (m *Model) parameters() []*nn.Parameter {
	var parameters []*nn.Parameter

	for _, stage := range m.stages {
		if stage != nil {
			parameters = append(parameters, stage.Parameters()...)
		}
	}

	for scale := range m.regression {
		parameters = append(parameters, m.regression[scale].Parameters()...)
		parameters = append(parameters, m.classification[scale].Parameters()...)
	}

	return parameters
}

// ToDevice explicitly transfers all model weights, preserving shared parameter
// identity. A nil device selects CPU. Transfers are atomic and must be externally
// synchronized against inference. Load always constructs a CPU model first.
func (m *Model) ToDevice(device tensor.Device) error {
	if m == nil || len(m.classes) == 0 {
		return fmt.Errorf("yolo: transfer: uninitialized model")
	}

	return nn.ToDevice(m.parameters(), device)
}

// Device returns the current weight device, or nil for CPU or an empty model.
func (m *Model) Device() tensor.Device {
	if m == nil || len(m.classes) == 0 || m.stages[0] == nil {
		return nil
	}

	parameters := m.stages[0].Parameters()
	if len(parameters) == 0 {
		return nil
	}

	return parameters[0].Value().Data().Device()
}

// Close releases the model's GPU tensor leases. CPU tensors are unaffected.
// Views with independent leases remain valid; synchronize against model use.
func (m *Model) Close() error {
	if m == nil || len(m.classes) == 0 {
		return nil
	}

	for _, parameter := range m.parameters() {
		if err := parameter.Value().Data().Close(); err != nil {
			return err
		}
	}

	return nil
}
