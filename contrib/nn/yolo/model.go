package yolo

import (
	"fmt"
	"slices"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/tensor"
)

// Model owns a validated, immutable YOLOv8n detection network. Use Load to create
// one. Inference calls allocate independent activations and do not record graphs.
type Model struct {
	classes        []string
	stages         [22]layer.Layer
	regression     [3]*layer.Sequential
	classification [3]*layer.Sequential
}

// Classes returns a copy of the class names, ordered by class ID.
func (m *Model) Classes() []string {
	return slices.Clone(m.classes)
}

// Forward runs one square NCHW RGB tensor [1,3,size,size] and returns raw heads
// at strides 8, 16, and 32. Each head is [1,64+classes,size/stride,size/stride]:
// 64 regression logits (four groups of sixteen bins), then class logits. Pixels
// normally lie in [0,1]. Size must be a multiple of 32 in [32,1280]. Invalid
// shapes and uninitialized models return errors. Outputs own their storage.
func (m *Model) Forward(input *tensor.Tensor) ([3]*tensor.Tensor, error) {
	var result [3]*tensor.Tensor
	if m == nil || len(m.classes) == 0 || input == nil {
		return result, fmt.Errorf("yolo: forward: uninitialized model or input")
	}

	shape := input.Shape()
	if len(shape) != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != shape[3] || !validSize(shape[2]) {
		return result, fmt.Errorf("yolo: forward: expected [1,3,size,size], size a multiple of 32 in [32,1280], got %v", shape)
	}

	previous, err := nn.Inference(input)
	if err != nil {
		return result, err
	}

	var outputs [22]*nn.Value
	for index := range outputs {
		switch index {
		case 11:
			previous, err = nn.Concat(1, outputs[10], outputs[6])
		case 14:
			previous, err = nn.Concat(1, outputs[13], outputs[4])
		case 17:
			previous, err = nn.Concat(1, outputs[16], outputs[12])
		case 20:
			previous, err = nn.Concat(1, outputs[19], outputs[9])
		default:
			previous, err = m.stages[index].Forward(previous)
		}

		if err != nil {
			return result, fmt.Errorf("yolo: forward: stage %d: %w", index, err)
		}

		outputs[index] = previous
	}

	for scale, index := range [3]int{15, 18, 21} {
		regression, err := m.regression[scale].Forward(outputs[index])
		if err != nil {
			return result, fmt.Errorf("yolo: forward: regression head %d: %w", scale, err)
		}

		classification, err := m.classification[scale].Forward(outputs[index])
		if err != nil {
			return result, fmt.Errorf("yolo: forward: classification head %d: %w", scale, err)
		}

		head, err := nn.Concat(1, regression, classification)
		if err != nil {
			return result, err
		}

		result[scale] = head.Data()
	}

	return result, nil
}

func validSize(size int) bool {
	return size >= 32 && size <= 1280 && size%32 == 0
}
