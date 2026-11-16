package layer

import (
	"encoding/json"
	"fmt"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/activation"
)

const maxConfigLayers = 10000

const maxConfigDepth = 64

type architecture struct {
	Root   int               `json:"root"`
	Layers []layerDefinition `json:"layers"`
}

type layerDefinition struct {
	Type       string          `json:"type"`
	Config     json.RawMessage `json:"config,omitempty"`
	Children   []int           `json:"children,omitempty"`
	Parameters []int           `json:"parameters,omitempty"`
}

type convolutionConfig struct {
	Kernel  int `json:"kernel"`
	Stride  int `json:"stride"`
	Padding int `json:"padding"`
}

type attentionConfig struct {
	Heads  int  `json:"heads"`
	Causal bool `json:"causal"`
}

type normalizationConfig struct {
	Epsilon float32 `json:"epsilon"`
}

type softmaxConfig struct {
	Axis int `json:"axis"`
}

// MarshalConfig describes a Sequential architecture as JSON without its weights.
// Parameter references follow model.Parameters order. Shared layers and parameters
// retain their identity through references. Restore rebuilds this description with
// supplied parameters, without allocating randomly initialized weight arrays.
//
// All built-in layers and activations are supported, including nested Sequential
// models. Custom, nil, uninitialized, cyclic, and excessively deep models fail.
// The limit is 10,000 distinct layers and 64 nested layer levels. Callers must
// synchronize serialization with parameter updates and model use.
func MarshalConfig(model *Sequential) ([]byte, error) {
	if model == nil {
		return nil, fmt.Errorf("nn: config: nil model")
	}

	var result architecture

	seen := make(map[Layer]int)
	visiting := make(map[Layer]bool)
	parameters := make(map[*nn.Parameter]int)

	var encode func(Layer, int) (int, error)

	encode = func(stage Layer, depth int) (int, error) {
		if depth > maxConfigDepth {
			return 0, fmt.Errorf("nn: config: layer depth exceeds %d", maxConfigDepth)
		}

		definition, children, leaves, err := describeLayer(stage)
		if err != nil {
			return 0, err
		}

		if visiting[stage] {
			return 0, fmt.Errorf("nn: config: cyclic layer graph")
		}

		if index, ok := seen[stage]; ok {
			return index, nil
		}

		if len(result.Layers) >= maxConfigLayers {
			return 0, fmt.Errorf("nn: config: too many layers")
		}

		index := len(result.Layers)
		seen[stage] = index
		visiting[stage] = true

		result.Layers = append(result.Layers, definition)

		for _, parameter := range leaves {
			if parameter == nil || parameter.Value() == nil || parameter.Value().Data() == nil {
				return 0, fmt.Errorf("nn: config: uninitialized %s parameter", definition.Type)
			}

			id, ok := parameters[parameter]
			if !ok {
				id = len(parameters)
				parameters[parameter] = id
			}

			result.Layers[index].Parameters = append(result.Layers[index].Parameters, id)
		}

		for _, child := range children {
			id, err := encode(child, depth+1)
			if err != nil {
				return 0, err
			}

			result.Layers[index].Children = append(result.Layers[index].Children, id)
		}

		visiting[stage] = false

		return index, nil
	}

	root, err := encode(model, 0)
	if err != nil {
		return nil, err
	}

	result.Root = root

	return json.Marshal(result)
}

func describeLayer(stage Layer) (layerDefinition, []Layer, []*nn.Parameter, error) {
	var (
		definition layerDefinition
		children   []Layer
		parameters []*nn.Parameter
		config     any
	)

	switch value := stage.(type) {
	case *Sequential:
		if value == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: nil sequential layer")
		}

		definition.Type = "sequential"
		children = value.layers
	case *Dense:
		if value == nil || value.weight == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: uninitialized dense layer")
		}

		definition.Type = "dense"
		parameters = value.Parameters()
	case *Conv2D:
		if value == nil || value.projection == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: uninitialized convolution")
		}

		definition.Type = "conv2d"
		config = convolutionConfig{Kernel: value.kernel, Stride: value.stride, Padding: value.padding}
		children = []Layer{value.projection}
	case *RNN:
		if value == nil || value.inputProjection == nil || value.hiddenProjection == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: uninitialized rnn")
		}

		definition.Type = "rnn"
		children = []Layer{value.inputProjection, value.hiddenProjection}
	case *LSTM:
		if value == nil || value.inputProjection == nil || value.hiddenProjection == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: uninitialized lstm")
		}

		definition.Type = "lstm"
		children = []Layer{value.inputProjection, value.hiddenProjection}
	case *LayerNorm:
		if value == nil || value.scale == nil || value.bias == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: uninitialized normalization")
		}

		definition.Type = "layer_norm"
		config = normalizationConfig{Epsilon: float32(value.epsilon)}
		parameters = value.Parameters()
	case *MultiHeadAttention:
		if value == nil || value.query == nil || value.key == nil || value.value == nil || value.output == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: uninitialized attention")
		}

		definition.Type = "attention"
		config = attentionConfig{Heads: value.heads, Causal: value.causal}
		children = []Layer{value.query, value.key, value.value, value.output}
	case *TransformerEncoder:
		if value == nil || value.normAttention == nil || value.attention == nil || value.normFeedForward == nil || value.feedForwardIn == nil || value.feedForwardOut == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: uninitialized transformer")
		}

		definition.Type = "transformer_encoder"
		children = []Layer{value.normAttention, value.attention, value.normFeedForward, value.feedForwardIn, value.feedForwardOut}
	case *activation.ReLU:
		if value == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: nil relu")
		}

		definition.Type = "relu"
	case *activation.Sigmoid:
		if value == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: nil sigmoid")
		}

		definition.Type = "sigmoid"
	case *activation.Tanh:
		if value == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: nil tanh")
		}

		definition.Type = "tanh"
	case *activation.Softmax:
		if value == nil || value.Axis() < 0 {
			return definition, nil, nil, fmt.Errorf("nn: config: invalid softmax axis")
		}

		definition.Type = "softmax"
		config = softmaxConfig{Axis: value.Axis()}
	case *Flatten:
		if value == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: nil flatten")
		}

		definition.Type = "flatten"
	case *LastStep:
		if value == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: nil last step")
		}

		definition.Type = "last_step"
	case *PositionalEncoding:
		if value == nil {
			return definition, nil, nil, fmt.Errorf("nn: config: nil positional encoding")
		}

		definition.Type = "positional_encoding"
	default:
		return definition, nil, nil, fmt.Errorf("nn: config: unsupported layer %T", stage)
	}

	if config != nil {
		data, err := json.Marshal(config)
		if err != nil {
			return definition, nil, nil, err
		}

		definition.Config = data
	}

	return definition, children, parameters, nil
}
