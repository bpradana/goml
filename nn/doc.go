// Package nn provides eager reverse-mode differentiation and trainable values.
//
// Tensor values are immutable. A forward pass owns its graph; Backward consumes
// it and accumulates gradients until ZeroGrad is called. Parameters keep their
// identity while optimizers replace their immutable tensor values.
// Inference inputs suppress graph recording throughout a forward pass.
//
// Training, Backward, and optimizer updates require external synchronization.
// Independent inference passes may read the same model concurrently.
package nn
