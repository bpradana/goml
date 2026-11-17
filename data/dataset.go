// Package data provides indexed datasets, synchronous epoch-based loaders, and
// CPU Tensor batches. Applications own yielded batches and select devices
// explicitly. Loading does not depend on NeuralNet or a concrete GPU backend.
package data

import (
	"context"
	"fmt"
	"slices"
)

// Dataset provides indexed samples. Len must remain stable during an iteration.
// Get must return errors for invalid indices and respect cancellation. Built-in
// datasets support concurrent Get calls and return independently mutable samples.
type Dataset[T any] interface {
	Len() int
	Get(context.Context, int) (T, error)
}

type mapped[S, T any] struct {
	source Dataset[S]
	apply  func(context.Context, S) (T, error)
}

// Map applies a typed transform after each successful fetch. It does not cache
// samples. Concurrent use requires the source and transform to be safe together.
func Map[S, T any](source Dataset[S], apply func(context.Context, S) (T, error)) (Dataset[T], error) {
	if source == nil || apply == nil {
		return nil, fmt.Errorf("data: map: dataset and transform required")
	}

	return &mapped[S, T]{source: source, apply: apply}, nil
}

func (d *mapped[S, T]) Len() int {
	return d.source.Len()
}

func (d *mapped[S, T]) Get(ctx context.Context, index int) (T, error) {
	var zero T
	if err := ctx.Err(); err != nil {
		return zero, err
	}

	if index < 0 || index >= d.Len() {
		return zero, fmt.Errorf("data: sample index %d out of range", index)
	}

	sample, err := d.source.Get(ctx, index)
	if err != nil {
		return zero, err
	}

	if err := ctx.Err(); err != nil {
		return zero, err
	}

	return d.apply(ctx, sample)
}

type subset[T any] struct {
	source  Dataset[T]
	indices []int
}

// Subset copies an ordered index selection. Repeated indices are allowed. The
// source remains responsible for its own lifetime and stable sample count.
func Subset[T any](source Dataset[T], indices []int) (Dataset[T], error) {
	if source == nil {
		return nil, fmt.Errorf("data: subset: dataset required")
	}

	for _, index := range indices {
		if index < 0 || index >= source.Len() {
			return nil, fmt.Errorf("data: subset: index %d out of range", index)
		}
	}

	return &subset[T]{source: source, indices: slices.Clone(indices)}, nil
}

func (d *subset[T]) Len() int {
	return len(d.indices)
}

func (d *subset[T]) Get(ctx context.Context, index int) (T, error) {
	var zero T
	if err := ctx.Err(); err != nil {
		return zero, err
	}

	if index < 0 || index >= d.Len() {
		return zero, fmt.Errorf("data: subset: index %d out of range", index)
	}

	return d.source.Get(ctx, d.indices[index])
}
