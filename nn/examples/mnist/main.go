// Command mnist trains a small classifier from local gzip-compressed MNIST IDX files.
package main

import (
	"compress/gzip"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand/v2"
	"os"
	"path/filepath"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/layer"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
)

type runOptions struct {
	imagesPath      string
	labelsPath      string
	modelName       string
	limit           int
	epochs          int
	batch           int
	saveModelPath   string
	saveWeightsPath string
	loadModelPath   string
	loadWeightsPath string
	predict         bool
}

func main() {
	var options runOptions
	flag.StringVar(&options.imagesPath, "images", "data/train-images-idx3-ubyte.gz", "local MNIST image file")
	flag.StringVar(&options.labelsPath, "labels", "data/train-labels-idx1-ubyte.gz", "local MNIST label file")
	flag.StringVar(&options.modelName, "model", "dense", "classifier architecture: dense or cnn; ignored with -load-model")
	flag.IntVar(&options.limit, "limit", 1000, "number of examples")
	flag.IntVar(&options.epochs, "epochs", 3, "training epochs")
	flag.IntVar(&options.batch, "batch", 32, "batch size")
	flag.StringVar(&options.saveModelPath, "save-model", "data/mnist-model.goml", "network output path; empty disables saving")
	flag.StringVar(&options.saveWeightsPath, "save-weights", "data/mnist-weights.goml", "weights output path; empty disables saving")
	flag.StringVar(&options.loadModelPath, "load-model", "", "restore architecture and weights from a network checkpoint")
	flag.StringVar(&options.loadWeightsPath, "load-weights", "", "restore weights into the selected -model architecture")
	flag.BoolVar(&options.predict, "predict", false, "evaluate a loaded checkpoint without training or saving")

	flag.Parse()

	if err := run(options); err != nil {
		log.Fatal(err)
	}
}

func load(path string, headerWords int) ([]uint32, []byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = file.Close() }()

	reader, err := gzip.NewReader(file)
	if err != nil {
		return nil, nil, err
	}
	defer func() { _ = reader.Close() }()

	header := make([]uint32, headerWords)
	if err := binary.Read(reader, binary.BigEndian, header); err != nil {
		return nil, nil, err
	}

	if header[1] == 0 || header[1] > 60000 {
		return nil, nil, fmt.Errorf("mnist: invalid item count %d", header[1])
	}

	size := int(header[1])

	if headerWords == 4 {
		if header[0] != 2051 || header[2] != 28 || header[3] != 28 {
			return nil, nil, fmt.Errorf("mnist: expected image magic 2051 and 28x28 dimensions, got %v", header)
		}

		size *= 784
	} else if header[0] != 2049 {
		return nil, nil, fmt.Errorf("mnist: expected label magic 2049, got %d", header[0])
	}

	data, err := io.ReadAll(io.LimitReader(reader, int64(size)+1))
	if err != nil {
		return nil, nil, err
	}

	if len(data) != size {
		return nil, nil, fmt.Errorf("mnist: expected %d bytes, got %d", size, len(data))
	}

	return header, data, nil
}

func run(options runOptions) error {
	if options.limit <= 0 || options.batch <= 0 || (!options.predict && options.epochs <= 0) {
		return fmt.Errorf("mnist: limit, batch, and training epochs must be positive")
	}

	if options.loadModelPath != "" && options.loadWeightsPath != "" {
		return fmt.Errorf("mnist: choose either load-model or load-weights")
	}

	if options.predict && options.loadModelPath == "" && options.loadWeightsPath == "" {
		return fmt.Errorf("mnist: predict requires load-model or load-weights")
	}

	if !options.predict && options.saveModelPath != "" && options.saveWeightsPath != "" {
		modelPath, err := filepath.Abs(options.saveModelPath)
		if err != nil {
			return err
		}

		weightsPath, err := filepath.Abs(options.saveWeightsPath)
		if err != nil {
			return err
		}

		if modelPath == weightsPath {
			return fmt.Errorf("mnist: model and weights need different output paths")
		}
	}

	var (
		model *layer.Sequential
		err   error
	)
	if options.loadModelPath != "" {
		model, err = checkpoint.Load(options.loadModelPath)
	} else {
		model, err = newModel(options.modelName, rand.New(rand.NewPCG(1, 2)))
	}

	if err != nil {
		return err
	}

	if options.loadWeightsPath != "" {
		if err := checkpoint.LoadWeights(options.loadWeightsPath, model.Parameters()); err != nil {
			return err
		}
	}

	header, pixels, err := load(options.imagesPath, 4)
	if err != nil {
		return err
	}

	labelHeader, labels, err := load(options.labelsPath, 2)
	if err != nil {
		return err
	}

	if header[1] != labelHeader[1] || options.limit > int(header[1]) {
		return fmt.Errorf("mnist: image/label counts must match and contain at least %d examples", options.limit)
	}

	values := make([]float32, options.limit*784)
	targets := make([]int, options.limit)

	for i := range values {
		values[i] = float32(pixels[i]) / 255
	}

	for i := range targets {
		if labels[i] >= 10 {
			return fmt.Errorf("mnist: invalid class %d", labels[i])
		}

		targets[i] = int(labels[i])
	}

	data, err := tensor.From(values, options.limit, 1, 28, 28)
	if err != nil {
		return err
	}

	if options.predict {
		correct, meanLoss, err := predict(model, data, targets, options.batch)
		if err != nil {
			return err
		}

		fmt.Printf("examples=%d accuracy=%.2f%% mean_cross_entropy=%.4f\n", options.limit, 100*float64(correct)/float64(options.limit), meanLoss)

		return nil
	}

	opt, err := optimizer.NewAdam(model.Parameters(), 0.001, 0.9, 0.999, 1e-8)
	if err != nil {
		return err
	}

	var criterion loss.Loss[[]int] = loss.NewCrossEntropy()

	for epoch := 0; epoch < options.epochs; epoch++ {
		var total float32

		for start := 0; start < options.limit; start += options.batch {
			end := min(start+options.batch, options.limit)

			slice, err := data.Slice(0, start, end)
			if err != nil {
				return err
			}

			input, err := nn.NewValue(slice, false)
			if err != nil {
				return err
			}

			prediction, err := model.Forward(input)
			if err != nil {
				return err
			}

			objective, err := criterion.Forward(prediction, targets[start:end])
			if err != nil {
				return err
			}

			total += objective.Data().Data()[0] * float32(end-start)

			if err := objective.Backward(); err != nil {
				return err
			}

			if err := opt.Step(); err != nil {
				return err
			}

			opt.ZeroGrad()
		}

		fmt.Printf("epoch=%d mean_cross_entropy=%.4f\n", epoch+1, total/float32(options.limit))
	}

	if options.saveModelPath != "" {
		if err := checkpoint.Save(options.saveModelPath, model); err != nil {
			return err
		}

		fmt.Printf("saved_model=%s\n", options.saveModelPath)
	}

	if options.saveWeightsPath != "" {
		if err := checkpoint.SaveWeights(options.saveWeightsPath, model.Parameters()); err != nil {
			return err
		}

		fmt.Printf("saved_weights=%s\n", options.saveWeightsPath)
	}

	return nil
}
