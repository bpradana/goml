# Metal YOLOv8n detection

[Examples](../README.md) · [YOLO](../../../../nn/yolo/README.md) · [Metal](../../README.md)

This program loads an explicitly converted local YOLOv8n archive and uploads weights once.
Detect uploads the prepared image, executes the network on Metal, then downloads final heads
for CPU decoding and NMS. Detection JSON goes to stdout. Timing and tracked peak bytes go to stderr.
The first invocation includes native graph compilation and warm-up.

```sh
mise run example:gpu:yolo -- -model data/yolo/yolov8n.goml -image data/yolo/bus.jpg -size 640
```

Use `-size 320` for the smaller validated input. The program does not download weights or images.
Follow the [conversion guide](../../../../nn/yolo/README.md#weight-conversion) for trusted checkpoint setup.
macOS, CGo, and Metal hardware are required.
