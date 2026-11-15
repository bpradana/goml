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

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/loss"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor"
)

func main() {
	images := flag.String("images", "data/train-images-idx3-ubyte.gz", "local MNIST image file")
	labels := flag.String("labels", "data/train-labels-idx1-ubyte.gz", "local MNIST label file")
	model := flag.String("model", "dense", "classifier architecture: dense or cnn")
	limit := flag.Int("limit", 1000, "number of training examples")
	epochs := flag.Int("epochs", 3, "training epochs")
	batch := flag.Int("batch", 32, "batch size")

	flag.Parse()

	if err := run(*images, *labels, *model, *limit, *epochs, *batch); err != nil {
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

func run(imagesPath, labelsPath, modelName string, limit, epochs, batch int) error {
	if limit <= 0 || epochs <= 0 || batch <= 0 {
		return fmt.Errorf("mnist: limit, epochs, and batch must be positive")
	}

	model, err := newModel(modelName, rand.New(rand.NewPCG(1, 2)))
	if err != nil {
		return err
	}

	header, pixels, err := load(imagesPath, 4)
	if err != nil {
		return err
	}

	labelHeader, labels, err := load(labelsPath, 2)
	if err != nil {
		return err
	}

	if header[1] != labelHeader[1] || limit > int(header[1]) {
		return fmt.Errorf("mnist: image/label counts must match and contain at least %d examples", limit)
	}

	values := make([]float32, limit*784)
	targets := make([]int, limit)

	for i := range values {
		values[i] = float32(pixels[i]) / 255
	}

	for i := range targets {
		if labels[i] >= 10 {
			return fmt.Errorf("mnist: invalid class %d", labels[i])
		}

		targets[i] = int(labels[i])
	}

	data, err := tensor.From(values, limit, 1, 28, 28)
	if err != nil {
		return err
	}

	opt, err := optimizer.NewAdam(model.Parameters(), 0.001, 0.9, 0.999, 1e-8)
	if err != nil {
		return err
	}

	var criterion loss.Loss[[]int] = loss.NewCrossEntropy()

	for epoch := 0; epoch < epochs; epoch++ {
		var total float32

		for start := 0; start < limit; start += batch {
			end := min(start+batch, limit)

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

		fmt.Printf("epoch=%d mean_cross_entropy=%.4f\n", epoch+1, total/float32(limit))
	}

	return nil
}
