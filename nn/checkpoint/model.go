package checkpoint

import (
	"bytes"
	"fmt"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/layer"
)

// Save writes a built-in Sequential network's architecture and weights to path.
// Nested and shared layers are preserved. Graphs, gradients, optimizer state, and
// runtime recurrent states are excluded. Encoding or I/O failures return errors
// without replacing an existing checkpoint. See Marshal for supported layers.
func Save(path string, model *layer.Sequential) error {
	data, err := Marshal(model)
	if err != nil {
		return err
	}

	return writeFile(path, data)
}

// Load reads a complete network checkpoint and returns a trainable Sequential
// model with independent parameter storage and no saved gradients or graphs.
// Version, checksum, dtype, shape, topology, and I/O errors are reported before a
// model is returned. No random parameter arrays are allocated during restoration.
func Load(path string) (*layer.Sequential, error) {
	state, err := readFile(path)
	if err != nil {
		return nil, err
	}

	return restoreModel(state)
}

// Marshal serializes architecture and weights to a versioned binary checkpoint.
// All built-in nn/layer and nn/activation types are supported. Custom layers and
// invalid architectures return errors; use SaveWeights for custom models.
// Float32 bits and shared identities within the model are preserved. Payloads
// above 64 MiB fail. The source model and gradients remain unchanged.
func Marshal(model *layer.Sequential) ([]byte, error) {
	config, err := layer.MarshalConfig(model)
	if err != nil {
		return nil, err
	}

	parameters, err := encodeParameters(model.Parameters())
	if err != nil {
		return nil, err
	}

	return encodeDocument(document{Kind: "network", Architecture: config, Parameters: parameters})
}

// Unmarshal restores a complete model from Marshal bytes. It validates the entire
// checkpoint, rejects trailing bytes and unsupported types, and allocates new
// trainable parameters. The returned model does not share storage with data.
func Unmarshal(data []byte) (*layer.Sequential, error) {
	state, err := decodeDocument(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}

	return restoreModel(state)
}

func restoreModel(state document) (*layer.Sequential, error) {
	if state.Kind != "network" {
		return nil, fmt.Errorf("checkpoint: expected network file, got %s", state.Kind)
	}

	values, err := decodeParameters(state.Parameters)
	if err != nil {
		return nil, err
	}

	parameters := make([]*nn.Parameter, len(values))
	for i, value := range values {
		parameter, err := nn.NewParameter(value)
		if err != nil {
			return nil, err
		}

		parameters[i] = parameter
	}

	return layer.Restore(state.Architecture, parameters)
}
