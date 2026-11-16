package shape

import "fmt"

// Convolution validates NCHW geometry and returns output dimensions and patch width.
func Convolution(shape []int, kernel, stride, padding int) (int, int, int, error) {
	if len(shape) != 4 || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 || kernel <= 0 || stride <= 0 || padding < 0 {
		return 0, 0, 0, fmt.Errorf("tensor: convolution: expected NCHW input, positive channels, spatial dimensions, kernel and stride, and nonnegative padding")
	}

	maxInt := int(^uint(0) >> 1)
	if padding > (maxInt-shape[2])/2 || padding > (maxInt-shape[3])/2 || kernel > maxInt/shape[1] || kernel > maxInt/(shape[1]*kernel) {
		return 0, 0, 0, fmt.Errorf("tensor: convolution: padded shape or kernel size overflow")
	}

	height, width := shape[2]+2*padding, shape[3]+2*padding
	if kernel > height || kernel > width {
		return 0, 0, 0, fmt.Errorf("tensor: convolution: kernel exceeds padded spatial dimensions")
	}

	outH, outW, features := (height-kernel)/stride+1, (width-kernel)/stride+1, shape[1]*kernel*kernel
	if _, _, err := Layout([]int{shape[0], outH, outW, features}); err != nil {
		return 0, 0, 0, err
	}

	return outH, outW, features, nil
}

// Im2Col extracts zero-padded square patches from NCHW input into a row-major
// [batch*outH*outW,channels*kernel*kernel] matrix. Strided inputs are read directly.
// Invalid shapes, configuration, and size overflow return errors.
