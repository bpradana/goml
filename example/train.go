package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/metrics"
	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/nn/checkpoint"
	"github.com/bpradana/goml/nn/optimizer"
	"github.com/bpradana/goml/tensor/backend"
	"github.com/bpradana/goml/trainer"
)

type trainConfig struct {
	Epochs             int    `json:"epochs"`
	BatchSize          int    `json:"batch_size"`
	Seed               uint64 `json:"seed"`
	TrainPerClass      int    `json:"train_per_class"`
	ValidationPerClass int    `json:"validation_per_class"`
	Patience           int    `json:"patience"`
}

func (c trainConfig) validate() error {
	if c.Epochs <= 0 || c.BatchSize <= 0 || c.TrainPerClass <= 0 || c.ValidationPerClass <= 0 || c.TrainPerClass > 5000-c.ValidationPerClass || c.Patience < 0 {
		return fmt.Errorf("epochs/batch/split counts must be positive; training plus validation must fit 5000 images per class; patience must be nonnegative")
	}

	return nil
}

type trainingSettings struct {
	LearningRate float32 `json:"learning_rate"`
	Beta1        float32 `json:"beta1"`
	Beta2        float32 `json:"beta2"`
	Epsilon      float32 `json:"epsilon"`
	Patience     int     `json:"patience"`
	MinDelta     float64 `json:"min_delta"`
	Loss         string  `json:"loss"`
	Selection    string  `json:"selection"`
}

type runMetadata struct {
	Version           int              `json:"version"`
	Model             string           `json:"model"`
	DatasetURL        string           `json:"dataset_url"`
	DatasetMD5        string           `json:"dataset_md5"`
	Device            string           `json:"device"`
	CreatedAt         time.Time        `json:"created_at"`
	Config            trainConfig      `json:"config"`
	Settings          trainingSettings `json:"settings"`
	Classes           [10]string       `json:"classes"`
	Normalization     normalization    `json:"normalization"`
	TrainIndices      []int            `json:"train_indices"`
	ValidationIndices []int            `json:"validation_indices"`
}

type epochRecord struct {
	trainer.EpochReport
	ElapsedSeconds float64 `json:"elapsed_seconds"`
}

type trainingHistory struct {
	trainer.History
	ElapsedSeconds float64 `json:"elapsed_seconds"`
	Error          string  `json:"error,omitempty"`
}

type evaluationReport struct {
	Split          string                       `json:"split"`
	Device         string                       `json:"device"`
	Samples        uint64                       `json:"samples"`
	MeanLoss       float64                      `json:"cross_entropy"`
	ElapsedSeconds float64                      `json:"elapsed_seconds"`
	Classes        [10]string                   `json:"classes"`
	Metrics        metrics.ClassificationResult `json:"metrics"`
}

func classificationLoader(dataset *imageDataset, batch int, shuffle bool, seed uint64) (*data.Loader[data.ClassificationSample, *data.ClassificationBatch], error) {
	return data.NewLoader(dataset, data.CollateClassification, data.LoaderOptions{BatchSize: batch, Shuffle: shuffle, Seed: seed})
}

func fitModel(ctx context.Context, records []byte, directory string, config trainConfig, device backend.Device, output io.Writer) error {
	if err := config.validate(); err != nil {
		return err
	}

	trainIndices, validationIndices, err := splitIndices(records, config.TrainPerClass, config.ValidationPerClass, config.Seed)
	if err != nil {
		return err
	}

	norm, err := fitNormalization(records, trainIndices)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Dir(directory), 0o755); err != nil {
		return err
	}

	if err := os.Mkdir(directory, 0o755); err != nil {
		return fmt.Errorf("create new run directory (choose an unused -run path): %w", err)
	}

	deviceName := device.ID()
	if named, ok := device.(interface{ Name() string }); ok {
		deviceName = named.Name()
	}

	metadata := runMetadata{
		Version: 1, Model: "cifar10-cnn-16-32-64", DatasetURL: datasetURL, DatasetMD5: datasetMD5,
		Device: deviceName, CreatedAt: time.Now().UTC(), Config: config, Classes: classNames,
		Normalization: norm, TrainIndices: trainIndices, ValidationIndices: validationIndices,
		Settings: trainingSettings{
			LearningRate: 0.001, Beta1: 0.9, Beta2: 0.999, Epsilon: 1e-8,
			Patience: config.Patience, MinDelta: 0.001, Loss: "cross_entropy", Selection: "minimum_validation_loss",
		},
	}

	encoded, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(filepath.Join(directory, "metadata.json"), encoded, 0o644); err != nil {
		return err
	}

	train, err := classificationLoader(&imageDataset{records: records, indices: trainIndices, normalization: norm}, config.BatchSize, true, config.Seed)
	if err != nil {
		return err
	}

	validation, err := classificationLoader(&imageDataset{records: records, indices: validationIndices, normalization: norm}, config.BatchSize, false, config.Seed)
	if err != nil {
		return err
	}

	model, err := newModel(config.Seed)
	if err != nil {
		return err
	}
	defer closeModel(model)

	if err := nn.ToDevice(model.Parameters(), device); err != nil {
		return err
	}

	settings := metadata.Settings

	opt, err := optimizer.NewAdam(model.Parameters(), settings.LearningRate, settings.Beta1, settings.Beta2, settings.Epsilon)
	if err != nil {
		return err
	}

	defer func() { _ = opt.Close() }()

	runner, err := trainer.NewClassifier(model, opt, device, len(classNames))
	if err != nil {
		return err
	}

	epochs, err := os.Create(filepath.Join(directory, "epochs.jsonl"))
	if err != nil {
		return err
	}

	defer func() { _ = epochs.Close() }()

	encoder := json.NewEncoder(epochs)
	started, previous := time.Now(), time.Now()

	if _, err := fmt.Fprintf(output, "device=%s train=%d validation=%d batch=%d seed=%d\n", deviceName, len(trainIndices), len(validationIndices), config.BatchSize, config.Seed); err != nil {
		return err
	}

	options := trainer.FitOptions{
		Epochs: config.Epochs, BestCheckpointPath: filepath.Join(directory, "best.goml"),
		OnEpochEnd: func(report trainer.EpochReport) error {
			now := time.Now()
			record := epochRecord{EpochReport: report, ElapsedSeconds: now.Sub(previous).Seconds()}
			previous = now

			if err := encoder.Encode(record); err != nil {
				return err
			}

			if err := epochs.Sync(); err != nil {
				return err
			}

			_, err := fmt.Fprintf(output, "epoch=%d train_loss=%.4f validation_loss=%.4f validation_accuracy=%.2f%% seconds=%.1f\n", report.Epoch, report.Train.MeanLoss, report.Validation.MeanLoss, 100*report.Classification.Accuracy, record.ElapsedSeconds)

			return err
		},
	}
	if settings.Patience > 0 {
		options.EarlyStopping = &trainer.EarlyStopping{Patience: settings.Patience, MinDelta: settings.MinDelta}
	}

	history, fitErr := runner.Fit(ctx, train, validation, options)

	summary := trainingHistory{History: history, ElapsedSeconds: time.Since(started).Seconds()}
	if fitErr != nil {
		summary.Error = fitErr.Error()

		summary.StopReason = "error"
		if errors.Is(fitErr, context.Canceled) || errors.Is(fitErr, context.DeadlineExceeded) {
			summary.StopReason = "canceled"
		}
	}

	encoded, err = json.MarshalIndent(summary, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(filepath.Join(directory, "history.json"), encoded, 0o644); err != nil {
		return errors.Join(fitErr, err)
	}

	if fitErr != nil {
		return fitErr
	}

	return checkpoint.Save(filepath.Join(directory, "final.goml"), model)
}

func readMetadata(directory string) (runMetadata, error) {
	var metadata runMetadata

	file, err := os.Open(filepath.Join(directory, "metadata.json"))
	if err != nil {
		return metadata, err
	}

	defer func() { _ = file.Close() }()

	decoder := json.NewDecoder(io.LimitReader(file, 4<<20))
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(&metadata); err != nil {
		return metadata, err
	}

	if err := decoder.Decode(new(json.RawMessage)); err != io.EOF {
		return metadata, fmt.Errorf("unexpected trailing metadata")
	}

	if metadata.Version != 1 || metadata.Model != "cifar10-cnn-16-32-64" || metadata.Classes != classNames {
		return metadata, fmt.Errorf("unsupported run metadata")
	}

	if err := metadata.Config.validate(); err != nil {
		return metadata, err
	}

	for channel := range 3 {
		mean, std := metadata.Normalization.Mean[channel], metadata.Normalization.Std[channel]
		if math.IsNaN(mean) || math.IsInf(mean, 0) || mean < 0 || mean > 1 || math.IsNaN(std) || math.IsInf(std, 0) || std <= 0 {
			return metadata, fmt.Errorf("invalid normalization for channel %d", channel)
		}
	}

	if len(metadata.TrainIndices) != 10*metadata.Config.TrainPerClass || len(metadata.ValidationIndices) != 10*metadata.Config.ValidationPerClass {
		return metadata, fmt.Errorf("saved split counts do not match configuration")
	}

	seen := make(map[int]bool)

	for _, indices := range [][]int{metadata.TrainIndices, metadata.ValidationIndices} {
		for _, index := range indices {
			if index < 0 || index >= 50000 || seen[index] {
				return metadata, fmt.Errorf("invalid or overlapping saved split index %d", index)
			}

			seen[index] = true
		}
	}

	return metadata, nil
}

func evaluateRun(ctx context.Context, dataDirectory, directory, split string, device backend.Device, output io.Writer) error {
	metadata, err := readMetadata(directory)
	if err != nil {
		return err
	}

	records, err := readRecords(ctx, dataDirectory, split == "validation")
	if err != nil {
		return err
	}

	indices := metadata.ValidationIndices
	if split == "test" {
		indices = make([]int, len(records)/recordSize)
		for index := range indices {
			indices[index] = index
		}
	}

	loader, err := classificationLoader(&imageDataset{records: records, indices: indices, normalization: metadata.Normalization}, metadata.Config.BatchSize, false, metadata.Config.Seed)
	if err != nil {
		return err
	}

	model, err := checkpoint.Load(filepath.Join(directory, "best.goml"))
	if err != nil {
		return err
	}
	defer closeModel(model)

	if err := nn.ToDevice(model.Parameters(), device); err != nil {
		return err
	}

	started := time.Now()

	evaluation, err := trainer.EvaluateClassification(ctx, model, device, 10, loader)
	if err != nil {
		return err
	}

	report := evaluationReport{
		Split: split, Device: device.ID(), Samples: evaluation.Samples, MeanLoss: evaluation.MeanLoss,
		ElapsedSeconds: time.Since(started).Seconds(), Classes: classNames, Metrics: *evaluation.Classification,
	}

	encoded, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return err
	}

	if err := os.WriteFile(filepath.Join(directory, split+".json"), encoded, 0o644); err != nil {
		return err
	}

	if _, err := fmt.Fprintf(output, "%s samples=%d accuracy=%.2f%% cross_entropy=%.4f macro_f1=%.4f seconds=%.1f\n", split, report.Samples, report.Metrics.Accuracy*100, report.MeanLoss, report.Metrics.F1Macro, report.ElapsedSeconds); err != nil {
		return err
	}

	for _, class := range report.Metrics.PerClass {
		if _, err := fmt.Fprintf(output, "class=%s support=%d precision=%.4f recall=%.4f f1=%.4f\n", classNames[class.Class], class.Support, class.Precision, class.Recall, class.F1); err != nil {
			return err
		}
	}

	return nil
}
