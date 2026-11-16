package backend

// Buffer is one device-owned allocation, measured in Float32 elements. Valid
// reports ErrClosed after release or owning-device shutdown. Close is idempotent.
// Queued operations must retain physical memory after the caller closes a buffer.
type Buffer interface {
	Len() int
	DeviceID() string
	Valid() error
	Close() error
}

// GCManagedBuffer is an optional lifetime optimization for allocations whose
// memory and bookkeeping are completely reclaimed by Go's garbage collector.
// Such buffers need no additional allocation cleanup when unreachable. Explicit
// Close still invalidates the buffer and releases its owned references. Native
// resources and device registries retaining allocations must not claim this.
// Tensor uses the capability to avoid cleanup-queue overhead for CPU arrays.
type GCManagedBuffer interface {
	Buffer
	GCManaged() bool
}
