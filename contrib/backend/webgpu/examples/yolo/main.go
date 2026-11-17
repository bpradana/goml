package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"time"

	"github.com/bpradana/goml/contrib/backend/webgpu"
	"github.com/bpradana/goml/contrib/nn/yolo"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "converted local YOLOv8n .goml archive")
	imagePath := flag.String("image", "", "local JPEG or PNG image")
	size := flag.Int("size", 640, "square inference size, for example 320 or 640")

	flag.Parse()

	if *modelPath == "" || *imagePath == "" {
		return fmt.Errorf("provide -model and -image")
	}

	device, err := webgpu.Open()
	if err != nil {
		return err
	}
	defer func() { _ = device.Close() }()

	model, err := yolo.Load(*modelPath)
	if err != nil {
		return err
	}
	defer func() { _ = model.Close() }()

	if err := model.ToDevice(device); err != nil {
		return err
	}

	file, err := os.Open(*imagePath)
	if err != nil {
		return err
	}

	source, _, err := image.Decode(file)
	_ = file.Close()

	if err != nil {
		return err
	}

	options := yolo.DefaultOptions()
	options.Size = *size
	started := time.Now()

	detections, err := model.Detect(source, options)
	if err != nil {
		return err
	}

	fmt.Fprintf(os.Stderr, "%s: detection=%s tracked_peak_bytes=%d\n", device.Name(), time.Since(started), device.Stats().PeakBytes)

	return json.NewEncoder(os.Stdout).Encode(detections)
}
