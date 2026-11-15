// Package layer provides dense, convolutional, recurrent, and attention layers
// with sequential model composition.
//
// The Layer interface defines Forward and Parameters. NewSequential composes
// layers in order and returns each shared parameter only once from Parameters.
// NewDense uses Xavier normal initialization with a caller-supplied random
// generator. Layers from package activation also implement Layer and have no
// trainable parameters.
//
// Conv2D accepts NCHW images. RNN and LSTM accept batch-first [batch,time,features]
// sequences and expose explicit recurrent state through ForwardState. Attention
// and TransformerEncoder use the same sequence layout. PositionalEncoding adds
// fixed position information. LastStep and Flatten adapt outputs for Dense.
//
// Forward records gradients when inputs or parameters require them. A constant
// input does not disable parameter tracking. Use nn.Inference or
// Sequential.Predict to run without constructing a gradient graph.
//
// Layers own their parameters, while optimizers replace immutable parameter
// tensors through package nn. Training and parameter updates require external
// synchronization. Concurrent inference may read a model while its parameters
// remain unchanged.
package layer
