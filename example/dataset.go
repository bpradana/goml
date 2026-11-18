package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"

	"github.com/bpradana/goml/data"
)

const (
	imageSize  = 3 * 32 * 32
	recordSize = 1 + imageSize
)

var classNames = [10]string{"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"}

type normalization struct {
	Mean [3]float64 `json:"mean"`
	Std  [3]float64 `json:"std"`
}

type imageDataset struct {
	records       []byte
	indices       []int
	normalization normalization
}

func readRecords(ctx context.Context, directory string, training bool) ([]byte, error) {
	names := []string{"test_batch.bin"}
	if training {
		names = []string{"data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin", "data_batch_4.bin", "data_batch_5.bin"}
	}

	records := make([]byte, 0, len(names)*10000*recordSize)
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		file, err := os.Open(filepath.Join(directory, name))
		if err != nil {
			return nil, fmt.Errorf("read CIFAR-10 (run download first): %w", err)
		}

		contents, readErr := io.ReadAll(io.LimitReader(file, 10000*recordSize+1))
		if err := errors.Join(readErr, file.Close()); err != nil {
			return nil, fmt.Errorf("read CIFAR-10 %s: %w", name, err)
		}

		if len(contents) != 10000*recordSize {
			return nil, fmt.Errorf("CIFAR-10 %s: expected 10000 records, got %d bytes", name, len(contents))
		}

		if err := validateRecords(contents); err != nil {
			return nil, fmt.Errorf("CIFAR-10 %s: %w", name, err)
		}

		records = append(records, contents...)
	}

	return records, nil
}

func validateRecords(records []byte) error {
	if len(records) == 0 || len(records)%recordSize != 0 {
		return fmt.Errorf("expected nonempty %d-byte CIFAR records", recordSize)
	}

	for index := 0; index < len(records); index += recordSize {
		if records[index] >= 10 {
			return fmt.Errorf("record %d has invalid class %d", index/recordSize, records[index])
		}
	}

	return nil
}

func splitIndices(records []byte, trainPerClass, validationPerClass int, seed uint64) ([]int, []int, error) {
	if trainPerClass <= 0 || validationPerClass <= 0 || trainPerClass > len(records)/recordSize-validationPerClass {
		return nil, nil, fmt.Errorf("positive training/validation counts must fit the dataset")
	}

	var byClass [10][]int

	for index := 0; index < len(records)/recordSize; index++ {
		label := records[index*recordSize]
		byClass[label] = append(byClass[label], index)
	}

	var train, validation []int

	for class, indices := range byClass {
		if len(indices) < trainPerClass+validationPerClass {
			return nil, nil, fmt.Errorf("class %s has only %d images", classNames[class], len(indices))
		}

		rng := rand.New(rand.NewPCG(seed, uint64(class)))
		rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
		train = append(train, indices[:trainPerClass]...)
		validation = append(validation, indices[trainPerClass:trainPerClass+validationPerClass]...)
	}

	return train, validation, nil
}

func fitNormalization(records []byte, indices []int) (normalization, error) {
	var result normalization
	if len(indices) == 0 {
		return result, fmt.Errorf("normalization needs training images")
	}

	var sum, squares [3]float64

	for _, index := range indices {
		image := records[index*recordSize+1:][:imageSize]
		for channel := range 3 {
			for _, pixel := range image[channel*1024:][:1024] {
				value := float64(pixel) / 255
				sum[channel] += value
				squares[channel] += value * value
			}
		}
	}

	count := float64(len(indices) * 1024)
	for channel := range 3 {
		result.Mean[channel] = sum[channel] / count

		result.Std[channel] = math.Sqrt(math.Max(0, squares[channel]/count-result.Mean[channel]*result.Mean[channel]))
		if result.Std[channel] == 0 {
			return result, fmt.Errorf("training channel %d has zero variance", channel)
		}
	}

	return result, nil
}

func (d *imageDataset) Len() int { return len(d.indices) }

func (d *imageDataset) Get(ctx context.Context, index int) (data.ClassificationSample, error) {
	if err := ctx.Err(); err != nil {
		return data.ClassificationSample{}, err
	}

	if index < 0 || index >= len(d.indices) {
		return data.ClassificationSample{}, fmt.Errorf("image index %d out of range", index)
	}

	record := d.records[d.indices[index]*recordSize:][:recordSize]
	values := make([]float32, imageSize)

	for channel := range 3 {
		for pixel := range 1024 {
			values[channel*1024+pixel] = float32((float64(record[1+channel*1024+pixel])/255 - d.normalization.Mean[channel]) / d.normalization.Std[channel])
		}
	}

	return data.ClassificationSample{Input: values, Shape: []int{3, 32, 32}, Label: int(record[0])}, nil
}
