package tensor

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
