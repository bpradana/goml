package tensor

import "fmt"

// MatMul allocates the 2D product [m,k] x [k,n] -> [m,n].
// Strided inputs and zero dimensions are supported. Invalid ranks or inner
// dimensions return an error. CPU accumulation is float32 in increasing k order.
// GPU accumulation follows the backend's Float32 ordering and may round differently.
func MatMul(a, b *Tensor) (*Tensor, error) {
	if err := valid("matmul", a, b); err != nil {
		return nil, err
	}

	if a.Rank() != 2 || b.Rank() != 2 || a.shape[1] != b.shape[0] {
		return nil, fmt.Errorf("tensor: matmul: incompatible shapes %v and %v", a.shape, b.shape)
	}

	if err := compatible("matmul", a, b); err != nil {
		return nil, err
	}

	if a.gpu != nil {
		return gpuMatMul(a, b)
	}

	m, k, n := a.shape[0], a.shape[1], b.shape[1]

	out, err := allocated([]int{m, n})
	if err != nil {
		return nil, err
	}

	matmulAdd(out.data, a, b, m, k, n)

	return out, nil
}

// matmulAdd accumulates into independently owned row-major storage. Blocking
// reuses right-hand rows across four output rows while keeping k in order.
func matmulAdd(dst []float32, a, b *Tensor, m, k, n int) {
	if m == 0 || k == 0 || n == 0 {
		return
	}

	// Packing is amortized across output rows. Small transposed products avoid
	// the buffer and use the strided dot-product path below.
	if b.strides[1] != 1 && m >= 8 && k >= 16 && n >= 8 {
		packed := make([]float32, k*n)
		for p := range k {
			bp := b.offset + p*b.strides[0]
			for j := range n {
				packed[p*n+j] = b.data[bp+j*b.strides[1]]
			}
		}

		b = &Tensor{data: packed, strides: []int{n, 1}}
	}

	if b.strides[1] != 1 {
		for i := range m {
			row := dst[i*n : (i+1)*n]
			for j := range n {
				ap, bp := a.offset+i*a.strides[0], b.offset+j*b.strides[1]
				sum := row[j]

				for range k {
					sum += a.data[ap] * b.data[bp]
					ap += a.strides[1]
					bp += b.strides[0]
				}

				row[j] = sum
			}
		}

		return
	}

	const blockM, blockN, blockK = 32, 128, 64

	for ii := 0; ii < m; ii += blockM {
		endM := min(ii+blockM, m)
		for jj := 0; jj < n; jj += blockN {
			endN := min(jj+blockN, n)
			for pp := 0; pp < k; pp += blockK {
				endK := min(pp+blockK, k)
				i := ii

				for ; i+3 < endM; i += 4 {
					r0 := dst[i*n+jj : i*n+endN]
					r1 := dst[(i+1)*n+jj : (i+1)*n+endN]
					r2 := dst[(i+2)*n+jj : (i+2)*n+endN]
					r3 := dst[(i+3)*n+jj : (i+3)*n+endN]
					ap := a.offset + i*a.strides[0] + pp*a.strides[1]

					for p := pp; p < endK; p++ {
						x0, x1 := a.data[ap], a.data[ap+a.strides[0]]
						x2, x3 := a.data[ap+2*a.strides[0]], a.data[ap+3*a.strides[0]]
						bp := b.offset + p*b.strides[0]
						values := b.data[bp+jj : bp+endN]

						for j, y := range values {
							r0[j] += x0 * y
							r1[j] += x1 * y
							r2[j] += x2 * y
							r3[j] += x3 * y
						}

						ap += a.strides[1]
					}
				}

				for ; i < endM; i++ {
					row := dst[i*n+jj : i*n+endN]
					ap := a.offset + i*a.strides[0] + pp*a.strides[1]

					for p := pp; p < endK; p++ {
						x := a.data[ap]
						bp := b.offset + p*b.strides[0]
						values := b.data[bp+jj : bp+endN]

						for j, y := range values {
							row[j] += x * y
						}

						ap += a.strides[1]
					}
				}
			}
		}
	}
}
