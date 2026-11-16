package layer

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"slices"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
)

// Restore rebuilds a Sequential model from MarshalConfig JSON and an ordered,
// unique parameter list. The returned model references those parameters directly.
// It preserves shared layers and parameters without random weight initialization.
//
// Unknown layer types or fields, invalid topology, cycles, mismatched parameter
// shapes, unused nodes or parameters, and format limits return errors. Validation
// does not update the supplied parameters or their gradients. Architecture limits
// match MarshalConfig. No user-defined constructors or executable code are loaded.
func Restore(data []byte, parameters []*nn.Parameter) (*Sequential, error) {
	var config architecture
	if err := decodeConfig(data, &config); err != nil {
		return nil, fmt.Errorf("nn: restore: %w", err)
	}

	if len(config.Layers) == 0 || len(config.Layers) > maxConfigLayers {
		return nil, fmt.Errorf("nn: restore: invalid layer count %d", len(config.Layers))
	}

	unique := make(map[*nn.Parameter]bool, len(parameters))
	for i, parameter := range parameters {
		if parameter == nil || parameter.Value() == nil || parameter.Value().Data() == nil || unique[parameter] {
			return nil, fmt.Errorf("nn: restore: invalid or repeated parameter %d", i)
		}

		unique[parameter] = true
	}

	stages := make([]Layer, len(config.Layers))
	state := make([]uint8, len(config.Layers))
	used := make([]bool, len(parameters))

	var build func(int, int) (Layer, error)

	build = func(index, depth int) (Layer, error) {
		if index < 0 || index >= len(stages) || depth > maxConfigDepth {
			return nil, fmt.Errorf("nn: restore: invalid layer reference or depth")
		}

		if state[index] == 1 {
			return nil, fmt.Errorf("nn: restore: cyclic layer graph")
		}

		if state[index] == 2 {
			return stages[index], nil
		}

		state[index] = 1
		definition := config.Layers[index]

		children := make([]Layer, len(definition.Children))
		for i, child := range definition.Children {
			stage, err := build(child, depth+1)
			if err != nil {
				return nil, err
			}

			children[i] = stage
		}

		leaves := make([]*nn.Parameter, len(definition.Parameters))
		for i, parameter := range definition.Parameters {
			if parameter < 0 || parameter >= len(parameters) {
				return nil, fmt.Errorf("nn: restore: invalid parameter reference %d", parameter)
			}

			leaves[i] = parameters[parameter]
			used[parameter] = true
		}

		stage, err := restoreLayer(definition, children, leaves)
		if err != nil {
			return nil, fmt.Errorf("nn: restore: layer %d (%s): %w", index, definition.Type, err)
		}

		stages[index] = stage
		state[index] = 2

		return stage, nil
	}

	root, err := build(config.Root, 0)
	if err != nil {
		return nil, err
	}

	model, ok := root.(*Sequential)
	if !ok {
		return nil, fmt.Errorf("nn: restore: root must be sequential")
	}

	for _, status := range state {
		if status != 2 {
			return nil, fmt.Errorf("nn: restore: unreachable layer")
		}
	}

	for _, referenced := range used {
		if !referenced {
			return nil, fmt.Errorf("nn: restore: unreferenced parameter")
		}
	}

	if !slices.Equal(model.Parameters(), parameters) {
		return nil, fmt.Errorf("nn: restore: parameter order does not match architecture")
	}

	return model, nil
}

func decodeConfig(data []byte, target any) error {
	data = bytes.TrimSpace(data)
	if len(data) == 0 || data[0] != '{' {
		return fmt.Errorf("expected a JSON object")
	}

	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(target); err != nil {
		return err
	}

	if err := decoder.Decode(new(json.RawMessage)); err != io.EOF {
		return fmt.Errorf("unexpected data after JSON object")
	}

	return nil
}

func denseProjections(children []Layer) ([]*Dense, error) {
	projections := make([]*Dense, len(children))
	for i, child := range children {
		projection, ok := child.(*Dense)
		if !ok {
			return nil, fmt.Errorf("expected dense projection at child %d", i)
		}

		projections[i] = projection
	}

	return projections, nil
}

func restoreLayer(definition layerDefinition, children []Layer, parameters []*nn.Parameter) (Layer, error) {
	counts := map[string][2]int{
		"conv2d": {1, 0}, "rnn": {2, 0}, "lstm": {2, 0}, "attention": {4, 0},
		"layer_norm": {0, 2}, "transformer_encoder": {5, 0},
		"relu": {0, 0}, "sigmoid": {0, 0}, "tanh": {0, 0}, "softmax": {0, 0},
		"flatten": {0, 0}, "last_step": {0, 0}, "positional_encoding": {0, 0},
	}
	if expected, ok := counts[definition.Type]; ok {
		if len(children) != expected[0] || len(parameters) != expected[1] {
			return nil, fmt.Errorf("invalid child or parameter count")
		}
	} else if definition.Type != "sequential" && definition.Type != "dense" {
		return nil, fmt.Errorf("unsupported layer type %q", definition.Type)
	}

	if definition.Type != "conv2d" && definition.Type != "attention" && definition.Type != "layer_norm" && definition.Type != "softmax" && len(definition.Config) != 0 {
		if err := decodeConfig(definition.Config, &struct{}{}); err != nil {
			return nil, err
		}
	}

	switch definition.Type {
	case "sequential":
		if len(parameters) != 0 {
			return nil, fmt.Errorf("sequential cannot own direct parameters")
		}

		return NewSequential(children...)
	case "dense":
		if len(children) != 0 || len(parameters) < 1 || len(parameters) > 2 {
			return nil, fmt.Errorf("dense requires weight and optional bias")
		}

		shape := parameters[0].Value().Data().Shape()
		if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 {
			return nil, fmt.Errorf("dense weight must have positive shape [in,out]")
		}

		dense := &Dense{weight: parameters[0]}
		if len(parameters) == 2 {
			if !slices.Equal(parameters[1].Value().Data().Shape(), []int{shape[1]}) {
				return nil, fmt.Errorf("dense bias must match output features")
			}

			dense.bias = parameters[1]
		}

		return dense, nil
	case "conv2d":
		var config convolutionConfig
		if err := decodeConfig(definition.Config, &config); err != nil {
			return nil, err
		}

		if config.Kernel <= 0 || config.Stride <= 0 || config.Padding < 0 || config.Kernel > int(^uint(0)>>1)/config.Kernel {
			return nil, fmt.Errorf("invalid convolution geometry")
		}

		projections, err := denseProjections(children)
		if err != nil {
			return nil, err
		}

		shape := projections[0].weight.Value().Data().Shape()

		area := config.Kernel * config.Kernel
		if shape[0]%area != 0 {
			return nil, fmt.Errorf("kernel shape does not match projection")
		}

		return &Conv2D{
			inChannels:  shape[0] / area,
			outChannels: shape[1],
			kernel:      config.Kernel,
			stride:      config.Stride,
			padding:     config.Padding,
			projection:  projections[0],
		}, nil
	case "rnn", "lstm":
		projections, err := denseProjections(children)
		if err != nil {
			return nil, err
		}

		input, recurrent := projections[0], projections[1]
		shape := input.weight.Value().Data().Shape()

		hidden := shape[1]
		if definition.Type == "lstm" {
			if hidden%4 != 0 {
				return nil, fmt.Errorf("lstm gate dimension must be divisible by four")
			}

			hidden /= 4
		}

		if input.bias == nil || recurrent.bias != nil || !slices.Equal(recurrent.weight.Value().Data().Shape(), []int{hidden, shape[1]}) {
			return nil, fmt.Errorf("incompatible recurrent projections")
		}

		if definition.Type == "rnn" {
			return &RNN{
				inputProjection:  input,
				hiddenProjection: recurrent,
				in:               shape[0],
				hidden:           hidden,
			}, nil
		}

		return &LSTM{
			inputProjection:  input,
			hiddenProjection: recurrent,
			in:               shape[0],
			hidden:           hidden,
		}, nil
	case "layer_norm":
		var config normalizationConfig
		if err := decodeConfig(definition.Config, &config); err != nil {
			return nil, err
		}

		shape := parameters[0].Value().Data().Shape()
		if len(shape) != 1 || shape[0] <= 0 || !slices.Equal(parameters[1].Value().Data().Shape(), shape) || config.Epsilon <= 0 || math.IsNaN(float64(config.Epsilon)) || math.IsInf(float64(config.Epsilon), 0) {
			return nil, fmt.Errorf("invalid normalization shape or epsilon")
		}

		return &LayerNorm{
			scale:    parameters[0],
			bias:     parameters[1],
			features: shape[0],
			epsilon:  float64(config.Epsilon),
		}, nil
	case "attention":
		var config attentionConfig
		if err := decodeConfig(definition.Config, &config); err != nil {
			return nil, err
		}

		projections, err := denseProjections(children)
		if err != nil {
			return nil, err
		}

		shape := projections[0].weight.Value().Data().Shape()
		if config.Heads <= 0 || shape[0] != shape[1] || shape[0]%config.Heads != 0 {
			return nil, fmt.Errorf("invalid attention head geometry")
		}

		for _, projection := range projections {
			if projection.bias == nil || !slices.Equal(projection.weight.Value().Data().Shape(), shape) {
				return nil, fmt.Errorf("incompatible attention projections")
			}
		}

		return &MultiHeadAttention{
			query:    projections[0],
			key:      projections[1],
			value:    projections[2],
			output:   projections[3],
			features: shape[0],
			heads:    config.Heads,
			causal:   config.Causal,
		}, nil
	case "transformer_encoder":
		first, firstOK := children[0].(*LayerNorm)
		attention, attentionOK := children[1].(*MultiHeadAttention)
		second, secondOK := children[2].(*LayerNorm)

		projections, err := denseProjections(children[3:])
		if !firstOK || !attentionOK || !secondOK || err != nil {
			return nil, fmt.Errorf("invalid transformer sublayers")
		}

		inShape := projections[0].weight.Value().Data().Shape()

		outShape := projections[1].weight.Value().Data().Shape()
		if first.features != attention.features || second.features != attention.features || attention.causal ||
			projections[0].bias == nil || projections[1].bias == nil || inShape[0] != attention.features ||
			!slices.Equal(outShape, []int{inShape[1], attention.features}) {
			return nil, fmt.Errorf("incompatible transformer sublayers")
		}

		return &TransformerEncoder{
			normAttention:   first,
			attention:       attention,
			normFeedForward: second,
			feedForwardIn:   projections[0],
			feedForwardOut:  projections[1],
		}, nil
	case "relu":
		return activation.NewReLU(), nil
	case "sigmoid":
		return activation.NewSigmoid(), nil
	case "tanh":
		return activation.NewTanh(), nil
	case "softmax":
		var config softmaxConfig
		if err := decodeConfig(definition.Config, &config); err != nil {
			return nil, err
		}

		if config.Axis < 0 {
			return nil, fmt.Errorf("invalid softmax axis")
		}

		return activation.NewSoftmax(config.Axis), nil
	case "flatten":
		return NewFlatten(), nil
	case "last_step":
		return NewLastStep(), nil
	case "positional_encoding":
		return NewPositionalEncoding(), nil
	}

	return nil, fmt.Errorf("unsupported layer type %q", definition.Type)
}
