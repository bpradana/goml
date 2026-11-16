package backend

import "fmt"

// View borrows immutable shape and stride metadata over a buffer. Offsets and
// strides are measured in elements, not bytes. It does not acquire ownership.
// Backends must copy metadata before retaining asynchronous work. Negative
// strides are unsupported. Empty views perform no physical memory accesses.
type View struct {
	Buffer  Buffer
	Shape   []int
	Strides []int
	Offset  int
}

// Numel returns the logical count, checking negative dimensions and overflow.
// A rank-zero view has one element. A zero dimension makes the count zero.
func (v View) Numel() (int, error) {
	empty := false

	for _, size := range v.Shape {
		if size < 0 {
			return 0, fmt.Errorf("%w: negative dimension", ErrInvalidView)
		}

		empty = empty || size == 0
	}

	if empty {
		return 0, nil
	}

	count, maximum := 1, int(^uint(0)>>1)
	for _, size := range v.Shape {
		if count > maximum/size {
			return 0, fmt.Errorf("%w: element count overflow", ErrInvalidView)
		}

		count *= size
	}

	return count, nil
}

// IsContiguous reports row-major logical storage. Singleton strides are ignored.
func (v View) IsContiguous() bool {
	count, err := v.Numel()
	if err != nil || len(v.Shape) != len(v.Strides) {
		return false
	}

	if count == 0 {
		return true
	}

	stride := 1
	for axis := len(v.Shape) - 1; axis >= 0; axis-- {
		if v.Shape[axis] > 1 && v.Strides[axis] != stride {
			return false
		}

		stride *= v.Shape[axis]
	}

	return true
}

// Validate checks physical bounds without imposing a backend's hardware limits.
// It is shared by implementations and does not infer mathematical output shapes.
func (v View) Validate() error {
	if v.Buffer == nil || len(v.Shape) != len(v.Strides) || v.Offset < 0 {
		return fmt.Errorf("%w: missing buffer or invalid metadata", ErrInvalidView)
	}

	if err := v.Buffer.Valid(); err != nil {
		return err
	}

	return v.ValidateLayout(v.Buffer.Len(), false)
}

// ValidateLayout checks bounds and optional write nonoverlap against elements.
// Backends may call it while holding their device lock after checking buffer
// ownership and liveness, avoiding reentrant calls to Buffer.Valid.
func (v View) ValidateLayout(elements int, writable bool) error {
	if elements < 0 || len(v.Shape) != len(v.Strides) || v.Offset < 0 {
		return ErrInvalidView
	}

	count, err := v.Numel()
	if err != nil {
		return err
	}

	maximum := v.Offset
	for axis, size := range v.Shape {
		stride := v.Strides[axis]
		if stride < 0 {
			return fmt.Errorf("%w: negative stride", ErrInvalidView)
		}

		if count == 0 || size <= 1 {
			continue
		}

		available := elements - 1 - maximum
		if available < 0 || stride > available/(size-1) {
			return fmt.Errorf("%w: storage bounds exceeded", ErrInvalidView)
		}

		maximum += (size - 1) * stride
	}

	if count != 0 && maximum >= elements {
		return fmt.Errorf("%w: storage bounds exceeded", ErrInvalidView)
	}

	if writable {
		return v.validateWrites()
	}

	return nil
}

// ValidateOutput also rejects overlapping writes. Standard row-major, sliced,
// and transposed layouts are supported. Exotic layouts that cannot be proven
// nonoverlapping are rejected conservatively.
func (v View) ValidateOutput() error {
	if err := v.Validate(); err != nil {
		return err
	}

	return v.validateWrites()
}

func (v View) validateWrites() error {
	count, _ := v.Numel()
	if count == 0 || v.IsContiguous() {
		return nil
	}

	remaining := 0

	for _, size := range v.Shape {
		if size > 1 {
			remaining++
		}
	}

	span, previous := 1, -1

	for remaining > 0 {
		next := -1
		for axis, size := range v.Shape {
			if size > 1 && v.Strides[axis] > previous && (next < 0 || v.Strides[axis] < v.Strides[next]) {
				next = axis
			}
		}

		if next < 0 || v.Strides[next] < span {
			return fmt.Errorf("%w: overlapping destination", ErrInvalidView)
		}

		span += (v.Shape[next] - 1) * v.Strides[next]
		previous = v.Strides[next]
		remaining--
	}

	return nil
}
