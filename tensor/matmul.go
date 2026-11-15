package tensor

import "fmt"

// MatMul allocates the 2D product [m,k] x [k,n] -> [m,n].
// Strided inputs and zero dimensions are supported. Invalid ranks or inner
// dimensions return an error. Accumulation is float32, in increasing k order.
func MatMul(a, b *Tensor) (*Tensor, error) {
	if err := valid("matmul", a, b); err != nil {
		return nil, err
	}

	if a.Rank() != 2 || b.Rank() != 2 || a.shape[1] != b.shape[0] {
		return nil, fmt.Errorf("tensor: matmul: incompatible shapes %v and %v", a.shape, b.shape)
	}

	m, k, n := a.shape[0], a.shape[1], b.shape[1]

	out, err := allocated([]int{m, n})
	if err != nil {
		return nil, err
	}

	for i := 0; i < m; i++ {
		for p := 0; p < k; p++ {
			x := a.data[a.offset+i*a.strides[0]+p*a.strides[1]]
			for j := 0; j < n; j++ {
				out.data[i*n+j] += x * b.data[b.offset+p*b.strides[0]+j*b.strides[1]]
			}
		}
	}

	return out, nil
}
