//go:build darwin && cgo

package examples

import (
	"encoding/binary"
	"fmt"
	"image"
	_ "image/jpeg"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/bpradana/goml/contrib/nn/yolo"
	"github.com/bpradana/goml/tensor"
)

func referenceInput(t testing.TB, path string, size int) *tensor.Tensor {
	t.Helper()

	file := filepath.Join(filepath.Dir(path), fmt.Sprintf("reference-%d", size), "input.f32")

	bytes, err := os.ReadFile(file)
	if err != nil {
		t.Fatal(err)
	}

	if len(bytes) != 3*size*size*4 {
		t.Fatal("reference input size mismatch")
	}

	values := make([]float32, len(bytes)/4)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(bytes[index*4:]))
	}

	input, err := tensor.From(values, 1, 3, size, size)
	if err != nil {
		t.Fatal(err)
	}

	return input
}

func TestYOLO_MetalRawHeadsAndDetectionsMatchCPU(t *testing.T) {
	path := os.Getenv("GOML_YOLO_MODEL")
	if path == "" {
		t.Skip("set GOML_YOLO_MODEL to an explicitly converted local YOLOv8n archive")
	}

	started := time.Now()

	cpu, err := yolo.Load(path)
	if err != nil {
		t.Fatal(err)
	}

	gpu, err := yolo.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = gpu.Close() }()

	t.Logf("Loaded two models in %s", time.Since(started))

	device := deviceForTest(t)
	started = time.Now()

	if err := gpu.ToDevice(device); err != nil {
		t.Fatal(err)
	}

	t.Logf("Weight upload: %s, allocations: %+v", time.Since(started), device.Stats())

	imageFile, err := os.Open(filepath.Join(filepath.Dir(path), "bus.jpg"))
	if err != nil {
		t.Fatal(err)
	}

	source, _, err := image.Decode(imageFile)
	_ = imageFile.Close()

	if err != nil {
		t.Fatal(err)
	}

	for _, size := range []int{320, 640} {
		t.Run(fmt.Sprint(size), func(t *testing.T) {
			input := referenceInput(t, path, size)

			started := time.Now()

			expected, err := cpu.Forward(input)
			if err != nil {
				t.Fatal(err)
			}

			cpuDuration := time.Since(started)
			started = time.Now()

			transferred, err := input.To(device)
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = transferred.Close() }()

			t.Logf("CPU forward: %s; input upload: %s", cpuDuration, time.Since(started))

			for iteration := range 2 {
				started = time.Now()

				heads, err := gpu.Forward(transferred)
				if err != nil {
					t.Fatal(err)
				}

				encoded := time.Since(started)
				started = time.Now()

				if err := device.Synchronize(); err != nil {
					t.Fatal(err)
				}

				synchronized := time.Since(started)
				started = time.Now()

				for scale, head := range heads {
					if head.Device() != device {
						t.Fatal("raw head left GPU storage")
					}

					actual, err := head.HostData()
					if err != nil {
						t.Fatal(err)
					}

					wanted := expected[scale].Data()
					maximum := float64(0)

					for index, value := range actual {
						delta := math.Abs(float64(value) - float64(wanted[index]))

						maximum = math.Max(maximum, delta)
						if math.IsNaN(delta) || delta > 0.001+0.0001*math.Abs(float64(wanted[index])) {
							t.Fatalf("head %d element %d: got %g, expected %g", scale, index, value, wanted[index])
						}
					}

					t.Logf("iteration %d head %d maximum absolute error %.9g", iteration, scale, maximum)

					if err := head.Close(); err != nil {
						t.Fatal(err)
					}
				}

				t.Logf("iteration %d: encode=%s synchronize=%s download/compare=%s memory=%+v", iteration, encoded, synchronized, time.Since(started), device.Stats())
			}

			options := yolo.DefaultOptions()
			options.Size = size

			cpuDetections, err := cpu.Detect(source, options)
			if err != nil {
				t.Fatal(err)
			}

			started = time.Now()

			gpuDetections, err := gpu.Detect(source, options)
			if err != nil {
				t.Fatal(err)
			}

			t.Logf("complete GPU detection: %s, objects=%d", time.Since(started), len(gpuDetections))

			if len(gpuDetections) != len(cpuDetections) {
				t.Fatalf("detection counts: GPU=%d CPU=%d", len(gpuDetections), len(cpuDetections))
			}

			for index, got := range gpuDetections {
				expected := cpuDetections[index]
				if got.ClassID != expected.ClassID || math.Abs(float64(got.Confidence-expected.Confidence)) > 0.0001 {
					t.Fatal("class or confidence mismatch", got, expected)
				}

				for axis, value := range []float32{got.Box.Left, got.Box.Top, got.Box.Right, got.Box.Bottom} {
					wanted := []float32{expected.Box.Left, expected.Box.Top, expected.Box.Right, expected.Box.Bottom}[axis]
					if math.Abs(float64(value-wanted)) > 0.1 {
						t.Fatal("box mismatch", got, expected)
					}
				}
			}
		})
	}

	if err := gpu.Close(); err != nil {
		t.Fatal(err)
	}

	if err := device.Synchronize(); err != nil {
		t.Fatal(err)
	}

	if stats := device.Stats(); stats.LiveBytes != 0 || stats.LiveBuffers != 0 {
		t.Fatalf("model shutdown retained native tensors: %+v", stats)
	}
}
