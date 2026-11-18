package main

import (
	"context"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"

	"github.com/bpradana/goml/data/transform"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

func predictImage(ctx context.Context, directory, imagePath string, device backend.Device, output io.Writer) error {
	metadata, err := readMetadata(directory)
	if err != nil {
		return err
	}

	file, err := os.Open(imagePath)
	if err != nil {
		return err
	}

	defer func() { _ = file.Close() }()

	info, err := file.Stat()
	if err != nil {
		return err
	}

	if info.Size() > 32<<20 {
		return fmt.Errorf("image exceeds 32 MiB")
	}

	config, _, err := image.DecodeConfig(file)
	if err != nil {
		return err
	}

	if config.Width <= 0 || config.Height <= 0 || config.Width > 16000000/config.Height {
		return fmt.Errorf("image must have at most 16000000 pixels")
	}

	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return err
	}

	source, _, err := image.Decode(file)
	if err != nil {
		return err
	}

	sample, err := transform.ImageToCHW(ctx, source, 3)
	if err != nil {
		return err
	}

	resize, err := transform.Resize(32, 32)
	if err != nil {
		return err
	}

	sample, err = resize(ctx, sample)
	if err != nil {
		return err
	}

	for channel := range 3 {
		for pixel := range 1024 {
			index := channel*1024 + pixel
			sample.Input[index] = float32((float64(sample.Input[index]) - metadata.Normalization.Mean[channel]) / metadata.Normalization.Std[channel])
		}
	}

	host, err := tensor.From(sample.Input, 1, 3, 32, 32)
	if err != nil {
		return err
	}

	defer func() { _ = host.Close() }()

	model, err := checkpoint.Load(filepath.Join(directory, "best.goml"))
	if err != nil {
		return err
	}
	defer closeModel(model)

	if err := nn.ToDevice(model.Parameters(), device); err != nil {
		return err
	}

	scope, err := tensor.NewScope(device)
	if err != nil {
		return err
	}

	defer func() { _ = scope.Close() }()

	input, err := scope.To(host)
	if err != nil {
		return err
	}

	prediction, err := model.Predict(input)
	if err != nil {
		return err
	}

	scores, err := prediction.HostData()
	if err != nil {
		return err
	}

	if len(scores) != 10 {
		return fmt.Errorf("checkpoint must produce ten class scores")
	}

	for _, value := range scores {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("model produced nonfinite scores")
		}
	}

	maximum := slices.Max(scores)

	var (
		total         float64
		probabilities [10]float64
	)
	for class, score := range scores {
		probabilities[class] = math.Exp(float64(score) - float64(maximum))
		total += probabilities[class]
	}

	indices := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	slices.SortStableFunc(indices, func(a, b int) int {
		if probabilities[a] > probabilities[b] {
			return -1
		}

		if probabilities[a] < probabilities[b] {
			return 1
		}

		return 0
	})

	for _, class := range indices {
		if _, err := fmt.Fprintf(output, "class=%s probability=%.6f\n", classNames[class], probabilities[class]/total); err != nil {
			return err
		}
	}

	return nil
}
