package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	"github.com/bpradana/goml/contrib/backend/metal"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if err := run(ctx, os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "cifar10:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, args []string, output io.Writer) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: go run . <download|train|evaluate|predict> [flags]")
	}

	command := args[0]
	switch command {
	case "download", "train", "evaluate", "predict":
	default:
		return fmt.Errorf("unknown command %q", command)
	}

	flags := flag.NewFlagSet(command, flag.ContinueOnError)
	flags.SetOutput(output)
	dataDirectory := flags.String("data", "archive/cifar-10-batches-bin", "local CIFAR-10 binary directory")
	directory := flags.String("run", "archive/run", "run directory containing checkpoint and preprocessing metadata")
	config := trainConfig{}

	var imagePath, split string

	downloadURL := datasetURL
	if command == "download" {
		flags.StringVar(&downloadURL, "url", datasetURL, "binary archive URL; the official checksum is always required")
	}

	if command == "train" {
		flags.IntVar(&config.Epochs, "epochs", 30, "maximum training epochs")
		flags.IntVar(&config.BatchSize, "batch", 32, "training and evaluation batch size")
		flags.Uint64Var(&config.Seed, "seed", 42, "model initialization, split, and epoch shuffle seed")
		flags.IntVar(&config.TrainPerClass, "train-per-class", 1000, "training images selected per class")
		flags.IntVar(&config.ValidationPerClass, "validation-per-class", 200, "disjoint validation images per class")
		flags.IntVar(&config.Patience, "patience", 5, "validation epochs without improvement before stopping; 0 disables early stopping")
	}

	if command == "predict" {
		flags.StringVar(&imagePath, "image", "", "PNG or JPEG image to classify")
	}

	if command == "evaluate" {
		flags.StringVar(&split, "split", "test", "test or validation")
	}

	if err := flags.Parse(args[1:]); err != nil {
		return err
	}

	if flags.NArg() != 0 {
		return fmt.Errorf("unexpected positional arguments: %v", flags.Args())
	}

	if command == "train" {
		if err := config.validate(); err != nil {
			return err
		}
	}

	if command == "evaluate" && split != "test" && split != "validation" {
		return fmt.Errorf("evaluate -split must be test or validation")
	}

	if command == "predict" && imagePath == "" {
		return fmt.Errorf("predict requires -image")
	}

	if command == "download" {
		if _, err := fmt.Fprintf(output, "downloading %s\n", downloadURL); err != nil {
			return err
		}

		if err := downloadDataset(ctx, *dataDirectory, downloadURL, datasetMD5); err != nil {
			return err
		}

		_, err := fmt.Fprintf(output, "dataset=%s checksum=%s\n", *dataDirectory, datasetMD5)

		return err
	}

	device, err := metal.Open()
	if err != nil {
		return fmt.Errorf("open Metal (requires macOS, CGo, and a supported GPU): %w", err)
	}
	defer func() { _ = device.Close() }()

	switch command {
	case "train":
		records, err := readRecords(ctx, *dataDirectory, true)
		if err != nil {
			return err
		}

		if err := fitModel(ctx, records, *directory, config, device, output); err != nil {
			return err
		}
		// The official test split is opened only after model selection is complete.
		return evaluateRun(ctx, *dataDirectory, *directory, "test", device, output)
	case "evaluate":
		return evaluateRun(ctx, *dataDirectory, *directory, split, device, output)
	case "predict":
		return predictImage(ctx, *directory, imagePath, device, output)
	}

	return nil
}
