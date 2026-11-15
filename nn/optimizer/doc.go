// Package optimizer provides SGD and Adam for updating trainable nn parameters.
//
// Construct an optimizer with NewSGD or NewAdam. Constructors validate
// hyperparameters and copy and deduplicate the supplied parameter list.
// The Optimizer interface exposes Step and ZeroGrad for training loops.
//
// Call Step after a successful backward pass. Step replaces immutable parameter
// tensors while preserving parameter identity and accumulated gradients.
// Parameters without gradients are skipped. Call ZeroGrad before accumulating
// gradients for the next training iteration.
//
// Adam owns its moment estimates and advances each parameter's step counter only
// when that parameter has a gradient. Optimizer state is separate from tensor
// storage. Updates and gradient clearing require external synchronization.
package optimizer
