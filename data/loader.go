package data

import (
	"context"
	"fmt"
	"iter"
	"math/rand/v2"
)

// CollateFunc builds one caller-owned batch. On failure it must release any
// partial resources itself. The sample slice is borrowed only during the call.
type CollateFunc[S, B any] func(samples []S) (B, error)

// LoaderOptions controls synchronous batching. Seed and epoch determine a local
// Fisher–Yates permutation; no global random state or background workers are used.
type LoaderOptions struct {
	BatchSize int
	Shuffle   bool
	DropLast  bool
	Seed      uint64
}

// Loader describes a reusable iteration. Independent iterations have independent
// buffers and random generators. Concurrent use requires a safe dataset/collator.
type Loader[S, B any] struct {
	dataset Dataset[S]
	collate CollateFunc[S, B]
	options LoaderOptions
}

// NewLoader validates configuration. Dataset must be an initialized, nonnil
// implementation. It is borrowed, not closed or modified by the loader.
func NewLoader[S, B any](dataset Dataset[S], collate CollateFunc[S, B], options LoaderOptions) (*Loader[S, B], error) {
	if dataset == nil || collate == nil || options.BatchSize <= 0 {
		return nil, fmt.Errorf("data: loader: dataset, collator, and positive batch size required")
	}

	if dataset.Len() < 0 {
		return nil, fmt.Errorf("data: loader: negative dataset length")
	}

	return &Loader[S, B]{dataset: dataset, collate: collate, options: options}, nil
}

// Batches iterates one nonnegative epoch. The first fetch/collation/cancellation
// error is yielded once and ends iteration. Successful batches belong to the
// caller, including the last batch before an early break. No goroutines survive
// iteration. Length is snapshotted at the start of each iteration.
func (l *Loader[S, B]) Batches(ctx context.Context, epoch int) iter.Seq2[B, error] {
	return func(yield func(B, error) bool) {
		var zero B
		if epoch < 0 {
			yield(zero, fmt.Errorf("data: loader: epoch must be nonnegative"))
			return
		}

		if err := ctx.Err(); err != nil {
			yield(zero, err)
			return
		}

		count := l.dataset.Len()
		if count < 0 {
			yield(zero, fmt.Errorf("data: loader: negative dataset length"))
			return
		}

		var indices []int
		if l.options.Shuffle {
			indices = make([]int, count)
			for i := range indices {
				indices[i] = i
			}

			rng := rand.New(rand.NewPCG(l.options.Seed, uint64(epoch)))

			for i := count - 1; i > 0; i-- {
				if err := ctx.Err(); err != nil {
					yield(zero, err)
					return
				}

				j := rng.IntN(i + 1)
				indices[i], indices[j] = indices[j], indices[i]
			}
		}

		for start := 0; start < count; {
			size := min(l.options.BatchSize, count-start)
			if size < l.options.BatchSize && l.options.DropLast {
				return
			}

			samples := make([]S, size)
			for i := range samples {
				if err := ctx.Err(); err != nil {
					yield(zero, err)
					return
				}

				index := start + i
				if indices != nil {
					index = indices[index]
				}

				sample, err := l.dataset.Get(ctx, index)
				if err != nil {
					yield(zero, fmt.Errorf("data: sample %d: %w", index, err))
					return
				}

				samples[i] = sample
			}

			if err := ctx.Err(); err != nil {
				yield(zero, err)
				return
			}

			batch, err := l.collate(samples)
			if err != nil {
				yield(zero, fmt.Errorf("data: collate at sample %d: %w", start, err))
				return
			}

			if !yield(batch, nil) {
				return
			}

			start += size
		}
	}
}
