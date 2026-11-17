// Package transform provides typed, copying sample transforms. Image conversion
// uses normalized RGB values and explicit CHW layout. No transform selects a
// device, caches samples, or uses global random state.
package transform

import (
	"context"
	"fmt"
	"slices"
)

// Func transforms a typed value. Callers coordinate any mutable custom state.
type Func[T any] func(context.Context, T) (T, error)

// Compose copies the transform list and runs steps in order, stopping at the
// first error or canceled context. An empty composition is the identity.
func Compose[T any](steps ...Func[T]) Func[T] {
	steps = slices.Clone(steps)

	return func(ctx context.Context, value T) (T, error) {
		for index, step := range steps {
			if err := ctx.Err(); err != nil {
				return value, err
			}

			if step == nil {
				return value, fmt.Errorf("transform: nil step %d", index)
			}

			result, err := step(ctx, value)
			if err != nil {
				return value, err
			}

			value = result
		}

		return value, ctx.Err()
	}
}
