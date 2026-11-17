package transform

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/bpradana/goml/data"
)

// Normalize validates copied channel statistics and returns a CHW transform
// computing (x-mean[channel])/std[channel]. Standard deviations must be positive
// and statistics finite. It copies input/shape and preserves the label.
func Normalize(mean, std []float32) (Func[data.ClassificationSample], error) {
	if len(mean) == 0 || len(mean) != len(std) {
		return nil, fmt.Errorf("transform: normalize: matching channel statistics required")
	}

	for i, value := range mean {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) ||
			math.IsNaN(float64(std[i])) || math.IsInf(float64(std[i]), 0) || std[i] <= 0 {
			return nil, fmt.Errorf("transform: normalize: invalid statistics at channel %d", i)
		}
	}

	mean, std = slices.Clone(mean), slices.Clone(std)

	return func(ctx context.Context, sample data.ClassificationSample) (data.ClassificationSample, error) {
		if err := ctx.Err(); err != nil {
			return data.ClassificationSample{}, err
		}

		if err := validateImage(sample); err != nil {
			return data.ClassificationSample{}, err
		}

		if sample.Shape[0] != len(mean) {
			return data.ClassificationSample{}, fmt.Errorf("transform: normalize: channel count differs from statistics")
		}

		result := data.ClassificationSample{Input: make([]float32, len(sample.Input)), Shape: slices.Clone(sample.Shape), Label: sample.Label}
		area := sample.Shape[1] * sample.Shape[2]

		for channel := range mean {
			if err := ctx.Err(); err != nil {
				return data.ClassificationSample{}, err
			}

			for i := range area {
				result.Input[channel*area+i] = (sample.Input[channel*area+i] - mean[channel]) / std[channel]
			}
		}

		return result, nil
	}, nil
}
