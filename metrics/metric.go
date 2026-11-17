package metrics

// Accumulator is the update/reset boundary consumed by evaluation loops such as
// Trainer. Implementations validate a whole batch before mutation, borrow slices
// only during Update, and require caller synchronization for concurrent mutation.
// P and T are element types, for example int class IDs or Float32 predictions.
type Accumulator[P, T any] interface {
	Update(predictions []P, targets []T) error
	Reset()
}

// Metric adds a strongly typed report to an Accumulator. Classification satisfies
// Metric[int,int,ClassificationResult]; Regression satisfies
// Metric[float32,float32,RegressionResult]. External packages can implement their
// own report type without registration, reflection, or type-erased result maps.
type Metric[P, T, R any] interface {
	Accumulator[P, T]
	Result() (R, error)
}
