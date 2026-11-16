package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"os"

	"github.com/bpradana/goml/contrib/nn/yolo"
)

func run(args []string, output io.Writer) error {
	options := yolo.DefaultOptions()
	flags := flag.NewFlagSet("yolo", flag.ContinueOnError)
	modelPath := flags.String("model", "", "converted YOLOv8n .goml archive")
	imagePath := flags.String("image", "", "input JPEG or PNG")
	flags.IntVar(&options.Size, "size", options.Size, "square input size, a multiple of 32")
	confidence := flags.Float64("confidence", float64(options.Confidence), "minimum class probability")
	iou := flags.Float64("iou", float64(options.IoU), "class-aware NMS IoU threshold")
	flags.IntVar(&options.MaxDetections, "max-detections", options.MaxDetections, "maximum retained detections")

	if err := flags.Parse(args); err != nil {
		return err
	}

	if *modelPath == "" || *imagePath == "" || flags.NArg() != 0 {
		return fmt.Errorf("usage: yolo -model yolov8n.goml -image image.jpg [-size 640]")
	}

	options.Confidence, options.IoU = float32(*confidence), float32(*iou)

	model, err := yolo.Load(*modelPath)
	if err != nil {
		return err
	}

	file, err := os.Open(*imagePath)
	if err != nil {
		return err
	}
	defer func() { _ = file.Close() }()

	config, _, err := image.DecodeConfig(file)
	if err != nil {
		return err
	}

	if config.Width <= 0 || config.Height <= 0 || config.Width > 65536 || config.Height > 65536 || int64(config.Width)*int64(config.Height) > 64<<20 {
		return fmt.Errorf("input image exceeds supported dimensions or 64 million pixels")
	}

	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return err
	}

	source, _, err := image.Decode(file)
	if err != nil {
		return err
	}

	detections, err := model.Detect(source, options)
	if err != nil {
		return err
	}

	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")

	return encoder.Encode(detections)
}

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
