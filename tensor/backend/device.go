package backend

// Device is one execution context with a stable, unique identity. It rejects
// foreign buffers and reports allocation, transfer, and execution failures.
// Upload and Download copy exactly one physical buffer. Download synchronizes
// earlier work; Upload must order writes safely against queued operations.
// Close waits for pending work and invalidates allocations on owned contexts.
// A documented shared stateless host context may implement Close as a no-op.
// Independent buffers may be used concurrently. Callers synchronize overlapping
// writes and do not mutate borrowed view metadata during execution.
type Device interface {
	ID() string
	Name() string
	Allocate(elements int) (Buffer, error)
	Upload(dst Buffer, values []float32) error
	Download(src Buffer, values []float32) error
	Synchronize() error
	Close() error
}
