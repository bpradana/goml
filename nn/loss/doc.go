// Package loss provides loss functions for training loops.
//
// Construct objectives with NewMSE or NewCrossEntropy. Each implements Loss,
// parameterized by its target type: *nn.Value for MSE and []int class labels
// for CrossEntropy. Both return a differentiable scalar.
//
// Loss implementations own their numerical calculation and derivative rules.
// Package nn owns graph recording, traversal, and gradient accumulation.
package loss
