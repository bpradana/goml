package checkpoint

import (
	"encoding/binary"
	"fmt"
	"math"
	"slices"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

// SaveWeights writes an ordered, unique parameter list to path. It preserves all
// float32 bit patterns and does not change parameters or gradients. Nil or
// uninitialized parameters, duplicates, oversized payloads, and I/O failures
// return errors. Parent directories are created; an existing file is replaced
// only after encoding and writing succeed.
func SaveWeights(path string, parameters []*nn.Parameter) error {
	stored, err := encodeParameters(parameters)
	if err != nil {
		return err
	}

	data, err := encodeDocument(document{Kind: "weights", Parameters: stored})
	if err != nil {
		return err
	}

	return writeFile(path, data)
}

// LoadWeights restores a weights-only file into an existing ordered, unique
// parameter list. Count, dtype, shapes, lengths, checksum, and file version must
// match. All tensors are validated before any updates. Success preserves parameter
// identity, replaces immutable tensor values, and clears gradients. Errors leave
// parameters and gradients unchanged. Optimizer state is not restored.
func LoadWeights(path string, parameters []*nn.Parameter) error {
	if err := validateParameters(parameters); err != nil {
		return err
	}

	state, err := readFile(path)
	if err != nil {
		return err
	}

	if state.Kind != "weights" {
		return fmt.Errorf("checkpoint: expected weights file, got %s", state.Kind)
	}

	if len(state.Parameters) != len(parameters) {
		return fmt.Errorf("checkpoint: expected %d parameters, got %d", len(parameters), len(state.Parameters))
	}

	values, err := decodeParameters(state.Parameters)
	if err != nil {
		return err
	}

	for i, value := range values {
		if !slices.Equal(value.Shape(), parameters[i].Value().Data().Shape()) {
			return fmt.Errorf("checkpoint: parameter %d shape %v does not match %v", i, value.Shape(), parameters[i].Value().Data().Shape())
		}
	}

	prepared := make([]*tensor.Tensor, len(values))

	defer func() {
		for _, value := range prepared {
			_ = value.Close()
		}
	}()

	devices := make(map[string]backend.Device)

	for i, value := range values {
		device := parameters[i].Value().Data().Device()

		prepared[i], err = value.To(device)
		if err != nil {
			return err
		}

		devices[device.ID()] = device
	}

	for _, device := range devices {
		if err := device.Synchronize(); err != nil {
			return err
		}
	}

	values = prepared

	for i, parameter := range parameters {
		if err := parameter.Update(values[i]); err != nil {
			return fmt.Errorf("checkpoint: update parameter %d: %w", i, err)
		}

		parameter.ZeroGrad()
	}

	return nil
}

func validateParameters(parameters []*nn.Parameter) error {
	seen := make(map[*nn.Parameter]bool, len(parameters))
	for i, parameter := range parameters {
		if parameter == nil || parameter.Value() == nil || parameter.Value().Data() == nil || parameter.Value().Data().IsClosed() || seen[parameter] {
			return fmt.Errorf("checkpoint: parameter %d is nil, uninitialized, or repeated", i)
		}

		seen[parameter] = true
	}

	return nil
}

func encodeParameters(parameters []*nn.Parameter) ([]storedTensor, error) {
	if err := validateParameters(parameters); err != nil {
		return nil, err
	}

	stored := make([]storedTensor, len(parameters))
	total := 0

	for i, parameter := range parameters {
		data := parameter.Value().Data()
		if data.Numel() > (maxPayloadSize-total)/4 {
			return nil, fmt.Errorf("checkpoint: parameter data exceeds payload limit")
		}

		total += data.Numel() * 4

		values, err := data.HostData()
		if err != nil {
			return nil, fmt.Errorf("checkpoint: parameter transfer: %w", err)
		}

		encoded := make([]byte, len(values)*4)
		for j, value := range values {
			binary.LittleEndian.PutUint32(encoded[j*4:], math.Float32bits(value))
		}

		stored[i] = storedTensor{DType: "float32", Shape: data.Shape(), Data: encoded}
	}

	return stored, nil
}

func decodeParameters(stored []storedTensor) ([]*tensor.Tensor, error) {
	values := make([]*tensor.Tensor, len(stored))
	for i, parameter := range stored {
		if parameter.DType != "float32" || len(parameter.Data)%4 != 0 {
			return nil, fmt.Errorf("checkpoint: parameter %d has invalid dtype or byte count", i)
		}

		data := make([]float32, len(parameter.Data)/4)
		for j := range data {
			data[j] = math.Float32frombits(binary.LittleEndian.Uint32(parameter.Data[j*4:]))
		}

		value, err := tensor.From(data, parameter.Shape...)
		if err != nil {
			return nil, fmt.Errorf("checkpoint: parameter %d: %w", i, err)
		}

		values[i] = value
	}

	return values, nil
}
