// Package activation provides ReLU, Sigmoid, and Tanh activation layers.
//
// Construct layers with NewReLU, NewSigmoid, or NewTanh. The concrete types
// implement layer.Layer through Forward and Parameters, so they compose directly
// with layer.NewSequential. No separate activation interface is required.
//
// Forward preserves the input shape and registers derivatives through
// nn.NewOperation. The implementations own their numerical and derivative rules.
// Activations have no trainable parameters or mutable state.
//
// Inference inputs suppress gradient recording. Backward follows nn's consuming
// graph lifecycle and requires external synchronization when values are shared.
package activation
