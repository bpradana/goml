package backend

import (
	"errors"
	"testing"
)

type viewBuffer struct {
	length int
	closed bool
}

func (b *viewBuffer) Len() int { return b.length }

func (*viewBuffer) DeviceID() string { return "test" }

func (b *viewBuffer) Valid() error {
	if b.closed {
		return ErrClosed
	}

	return nil
}

func (b *viewBuffer) Close() error {
	b.closed = true
	return nil
}

func TestView_ValidateBoundsAndOverlappingOutputs(t *testing.T) {
	for _, test := range []struct {
		name            string
		shape, strides  []int
		offset, length  int
		valid, writable bool
	}{
		{"scalar", nil, nil, 0, 1, true, true},
		{"contiguous", []int{2, 3}, []int{3, 1}, 0, 6, true, true},
		{"transpose", []int{3, 2}, []int{1, 3}, 0, 6, true, true},
		{"slice", []int{2, 2}, []int{3, 1}, 1, 6, true, true},
		{"broadcast", []int{2, 3}, []int{0, 1}, 0, 3, true, false},
		{"overlap", []int{2, 2}, []int{1, 1}, 0, 3, true, false},
		{"bounds", []int{2, 3}, []int{3, 1}, 1, 6, false, false},
		{"negative", []int{2}, []int{-1}, 1, 2, false, false},
		{"empty", []int{0, 3}, []int{3, 1}, 7, 0, true, true},
		{"rank", []int{2}, nil, 0, 2, false, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			view := View{Buffer: &viewBuffer{length: test.length}, Shape: test.shape, Strides: test.strides, Offset: test.offset}
			if err := view.Validate(); (err == nil) != test.valid || (err != nil && !errors.Is(err, ErrInvalidView)) {
				t.Fatalf("Validate = %v, valid=%v", err, test.valid)
			}

			if err := view.ValidateOutput(); (err == nil) != test.writable {
				t.Fatalf("ValidateOutput = %v, writable=%v", err, test.writable)
			}
		})
	}
}

func TestView_ClosedBuffersAndCountOverflow(t *testing.T) {
	buffer := &viewBuffer{length: 1}
	if err := buffer.Close(); err != nil {
		t.Fatal(err)
	}

	if err := (View{Buffer: buffer}).Validate(); !errors.Is(err, ErrClosed) {
		t.Fatal("closed buffer accepted", err)
	}

	if _, err := (View{Shape: []int{int(^uint(0) >> 1), 2}}).Numel(); !errors.Is(err, ErrInvalidView) {
		t.Fatal("count overflow accepted", err)
	}
}
