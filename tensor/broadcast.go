package tensor

// broadcastStrides aligns input strides with output axes. Broadcast dimensions
// have a zero stride so the arithmetic loop can advance without division.
func broadcastStrides(t *Tensor, shape []int, strides []int) {
	leading := len(shape) - t.Rank()
	for axis, size := range t.shape {
		if size != 1 {
			strides[leading+axis] = t.strides[axis]
		}
	}
}

func broadcastIndex(t *Tensor, outShape []int, i int) int {
	p := t.offset

	for axis := len(outShape) - 1; axis >= 0; axis-- {
		coordinate := i % outShape[axis]
		i /= outShape[axis]

		ta := axis - (len(outShape) - t.Rank())
		if ta >= 0 && t.shape[ta] != 1 {
			p += coordinate * t.strides[ta]
		}
	}

	return p
}
