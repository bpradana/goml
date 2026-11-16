// Package yolo runs converted YOLOv8n object detection models on the CPU.
//
// Load validates a goml YOLO archive produced by the accompanying offline
// converter. Model.Detect resizes and letterboxes an image, executes an inference
// graph, decodes distributional box coordinates, and performs class-aware NMS.
// Model.Forward exposes the three raw detection heads for numerical validation.
// No Python, CGo, GPU, or external inference runtime is needed for inference.
//
// Loaded weights remain private and immutable. A Model may be shared by concurrent
// inference calls, but each call allocates its own activations. Only YOLOv8n
// detection is supported; other YOLO families, training, segmentation, pose, and
// end-to-end detection heads are outside this module's format.
package yolo
