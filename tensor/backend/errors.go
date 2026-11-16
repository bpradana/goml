package backend

import "errors"

var (
	// ErrUnsupported identifies an unavailable operation or configuration.
	ErrUnsupported = errors.New("backend: unsupported operation or configuration")
	// ErrDeviceMismatch identifies buffers or operands from incompatible contexts.
	ErrDeviceMismatch = errors.New("backend: device mismatch")
	// ErrClosed identifies a released buffer, tensor handle, or device.
	ErrClosed = errors.New("backend: closed resource")
	// ErrInvalidView identifies invalid bounds or an overlapping output layout.
	ErrInvalidView = errors.New("backend: invalid storage view")
	// ErrAllocation identifies an invalid or unsuccessful allocation request.
	ErrAllocation = errors.New("backend: allocation failed")
)
