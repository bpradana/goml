package data_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"testing"

	"github.com/bpradana/goml/data"
)

type integerDataset struct {
	count int
	fail  int
	reads int
}

func (d *integerDataset) Len() int {
	return d.count
}

func (d *integerDataset) Get(ctx context.Context, index int) (int, error) {
	d.reads++

	if err := ctx.Err(); err != nil {
		return 0, err
	}

	if index < 0 || index >= d.count || index == d.fail {
		return 0, fmt.Errorf("bad index %d", index)
	}

	return index, nil
}

func collectIntegers(samples []int) ([]int, error) {
	return slices.Clone(samples), nil
}

func TestLoader_BatchSizesOrderAndDropLast(t *testing.T) {
	for _, tc := range []struct {
		count, size, batches, last int
		drop                       bool
	}{
		{5, 1, 5, 1, false},
		{6, 2, 3, 2, false},
		{5, 2, 3, 1, false},
		{5, 2, 2, 2, true},
		{0, 2, 0, 0, false},
		{1, 2, 0, 0, true},
	} {
		t.Run(fmt.Sprint(tc), func(t *testing.T) {
			ds := &integerDataset{count: tc.count, fail: -1}

			loader, err := data.NewLoader(ds, collectIntegers, data.LoaderOptions{BatchSize: tc.size, DropLast: tc.drop})
			if err != nil {
				t.Fatal(err)
			}

			var result []int

			batches, last := 0, 0

			for batch, err := range loader.Batches(context.Background(), 0) {
				if err != nil {
					t.Fatal(err)
				}

				result = append(result, batch...)
				batches++
				last = len(batch)
			}

			if batches != tc.batches || last != tc.last {
				t.Fatalf("got %d batches, final %d", batches, last)
			}

			for i, value := range result {
				if value != i {
					t.Fatalf("order %v", result)
				}
			}
		})
	}
}

func TestLoader_ShuffleIsEpochDeterministicAndAPermutation(t *testing.T) {
	ds := &integerDataset{count: 101, fail: -1}

	loader, err := data.NewLoader(ds, collectIntegers, data.LoaderOptions{BatchSize: 13, Shuffle: true, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}

	orders := make([][]int, 3)

	for i, epoch := range []int{7, 7, 8} {
		for batch, err := range loader.Batches(context.Background(), epoch) {
			if err != nil {
				t.Fatal(err)
			}

			orders[i] = append(orders[i], batch...)
		}
	}

	if !slices.Equal(orders[0], orders[1]) || slices.Equal(orders[0], orders[2]) {
		t.Fatal("shuffle is not seed/epoch deterministic")
	}

	for _, order := range orders {
		slices.Sort(order)

		for i, value := range order {
			if value != i {
				t.Fatal("missing or duplicate sample")
			}
		}
	}
}

func TestLoader_ErrorsCancellationAndEarlyBreakStopReads(t *testing.T) {
	for _, failure := range []string{"fetch", "collate", "cancel", "break"} {
		t.Run(failure, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			ds := &integerDataset{count: 9, fail: -1}
			collate := collectIntegers

			if failure == "fetch" {
				ds.fail = 0
			}

			if failure == "collate" {
				collate = func([]int) ([]int, error) { return nil, errors.New("collate failure") }
			}

			if failure == "cancel" {
				cancel()
			}

			loader, err := data.NewLoader(ds, collate, data.LoaderOptions{BatchSize: 2})
			if err != nil {
				t.Fatal(err)
			}

			yields := 0
			for _, err := range loader.Batches(ctx, 0) {
				yields++

				if failure == "break" {
					if err != nil {
						t.Fatal(err)
					}

					break
				}

				if err == nil {
					t.Fatal("expected failure")
				}

				if failure == "cancel" && !errors.Is(err, context.Canceled) {
					t.Fatal(err)
				}
			}

			if yields != 1 || ds.reads > 2 {
				t.Fatalf("yields=%d reads=%d", yields, ds.reads)
			}
		})
	}
}

func TestNewLoader_RejectsInvalidConfiguration(t *testing.T) {
	ds := &integerDataset{count: 1, fail: -1}
	for _, size := range []int{0, -1} {
		if _, err := data.NewLoader(ds, collectIntegers, data.LoaderOptions{BatchSize: size}); err == nil {
			t.Fatal("accepted batch size", size)
		}
	}

	if _, err := data.NewLoader[int, []int](nil, collectIntegers, data.LoaderOptions{BatchSize: 1}); err == nil {
		t.Fatal("accepted nil dataset")
	}

	if _, err := data.NewLoader[int, []int](ds, nil, data.LoaderOptions{BatchSize: 1}); err == nil {
		t.Fatal("accepted nil collator")
	}

	ds.count = -1
	if _, err := data.NewLoader(ds, collectIntegers, data.LoaderOptions{BatchSize: 1}); err == nil {
		t.Fatal("accepted negative length")
	}
}

func TestLoader_NegativeEpochAndCancellationBeforeCollation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	dataset, err := data.Map[int, int](&integerDataset{count: 4, fail: -1}, func(_ context.Context, value int) (int, error) {
		cancel()
		return value, nil
	})
	if err != nil {
		t.Fatal(err)
	}

	collated := false

	loader, err := data.NewLoader(dataset, func(samples []int) ([]int, error) {
		collated = true
		return samples, nil
	}, data.LoaderOptions{BatchSize: 2})
	if err != nil {
		t.Fatal(err)
	}

	for _, err := range loader.Batches(context.Background(), -1) {
		if err == nil {
			t.Fatal("accepted negative epoch")
		}
	}

	for _, err := range loader.Batches(ctx, 0) {
		if !errors.Is(err, context.Canceled) {
			t.Fatal(err)
		}
	}

	if collated {
		t.Fatal("collated after cancellation")
	}
}
