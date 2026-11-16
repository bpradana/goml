package backendtest

import (
	"testing"

	"github.com/bpradana/goml/tensor/backend"
)

// RunConvolution checks strided, padded, multichannel NCHW convolution with
// optional bias against an independent direct cross-correlation reference.
func RunConvolution(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.ConvolutionBackend)
	if !ok {
		t.Fatal("convolution suite requires ConvolutionBackend")
	}

	for _, biased := range []bool{false, true} {
		name := "Convolution/NoBias"
		if biased {
			name = "Convolution/Bias"
		}

		t.Run(name, func(t *testing.T) {
			input, weights, bias, want, _, _ := convolutionReference()
			x, w, out := view(t, device, input, 2, 3, 5, 7), view(t, device, weights, 27, 4), view(t, device, nil, 2, 4, 3, 4)

			var b backend.View
			if biased {
				b = view(t, device, bias, 4)
				for i := range want {
					want[i] += bias[(i/12)%4]
				}
			}

			if err := compute.Conv2D(out, x, w, b, backend.Conv2DOptions{Kernel: 3, Stride: 2, Padding: 1}); err != nil {
				t.Fatal(err)
			}

			compare(t, device, out, want, 3e-4)
		})
	}
}

// RunConvolutionBackward checks both derivatives independently of the forward
// backend. Inference-only backends need not run this suite.
func RunConvolutionBackward(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.ConvolutionBackwardBackend)
	if !ok {
		t.Fatal("convolution gradient suite requires ConvolutionBackwardBackend")
	}

	t.Run("ConvolutionBackward/InputAndWeight", func(t *testing.T) {
		input, weights, _, _, wantX, wantW := convolutionReference()
		x, w := view(t, device, input, 2, 3, 5, 7), view(t, device, weights, 27, 4)

		upstream := make([]float32, 2*4*3*4)
		for i := range upstream {
			upstream[i] = float32(i%7-3) / 8
		}

		g := view(t, device, upstream, 2, 4, 3, 4)

		dx, dw := view(t, device, nil, 2, 3, 5, 7), view(t, device, nil, 27, 4)
		if err := compute.Conv2DBackward(dx, dw, x, w, g, backend.Conv2DOptions{Kernel: 3, Stride: 2, Padding: 1}); err != nil {
			t.Fatal(err)
		}

		compare(t, device, dx, wantX, 4e-4)
		compare(t, device, dw, wantW, 4e-4)
	})
}

func convolutionReference() (input, weight, bias, output, dx, dw []float32) {
	input, weight, bias = make([]float32, 2*3*5*7), make([]float32, 27*4), []float32{-1, 0, 0.5, 1}

	output, dx, dw = make([]float32, 2*4*3*4), make([]float32, len(input)), make([]float32, len(weight))
	for i := range input {
		input[i] = float32(i%13-6) / 8
	}

	for i := range weight {
		weight[i] = float32(i%11-5) / 8
	}

	for batch := range 2 {
		for channel := range 4 {
			for y := range 3 {
				for x := range 4 {
					target := ((batch*4+channel)*3+y)*4 + x
					gradient := float32(target%7-3) / 8

					for in := range 3 {
						for ky := range 3 {
							for kx := range 3 {
								iy, ix := y*2-1+ky, x*2-1+kx
								if iy < 0 || iy >= 5 || ix < 0 || ix >= 7 {
									continue
								}

								source := ((batch*3+in)*5+iy)*7 + ix
								feature := ((in*3+ky)*3+kx)*4 + channel
								output[target] += input[source] * weight[feature]
								dx[source] += gradient * weight[feature]
								dw[feature] += input[source] * gradient
							}
						}
					}
				}
			}
		}
	}

	return input, weight, bias, output, dx, dw
}
