package backend

// Conv2DOptions describes square kernels, equal strides, and symmetric zero
// padding for NCHW cross-correlation. Weight shape is [in*kernel*kernel,out].
type Conv2DOptions struct {
	Kernel, Stride, Padding int
}

// ConvolutionBackend writes native convolution. A nil bias.Buffer means no bias;
// otherwise bias has shape [out]. All non-nil buffers belong to the same device.
type ConvolutionBackend interface {
	Conv2D(dst, input, weight, bias View, options Conv2DOptions) error
}

// ConvolutionBackwardBackend writes input and flattened weight derivatives.
// Inference-only backends do not need to implement this capability.
type ConvolutionBackwardBackend interface {
	Conv2DBackward(dx, dw, input, weight, gradient View, options Conv2DOptions) error
}

// PatchBackend supplies the existing CPU training patch extraction and scatter.
// Im2Col writes [batch*outH*outW,in*kernel*kernel]; Col2Im sums overlapping patches.
// A native convolution backend does not need to expose patches.
type PatchBackend interface {
	Im2Col(dst, input View, options Conv2DOptions) error
	Col2Im(dst, patches View, options Conv2DOptions) error
}
