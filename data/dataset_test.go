package data_test

import (
	"context"
	"errors"
	"testing"

	"github.com/bpradana/goml/data"
)

func TestDataset_MapAndSubsetUseOnlyPublicContracts(t *testing.T) {
	indices := []int{3, 1}

	subset, err := data.Subset[int](&integerDataset{count: 4, fail: -1}, indices)
	if err != nil {
		t.Fatal(err)
	}

	indices[0] = 0

	mapped, err := data.Map(subset, func(_ context.Context, value int) (string, error) {
		return string(rune('a' + value)), nil
	})
	if err != nil {
		t.Fatal(err)
	}

	got, err := mapped.Get(context.Background(), 0)
	if err != nil || got != "d" || mapped.Len() != 2 {
		t.Fatalf("got %q %v", got, err)
	}

	for _, index := range []int{-1, 2} {
		if _, err := mapped.Get(context.Background(), index); err == nil {
			t.Fatal("accepted invalid index")
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := mapped.Get(ctx, 0); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}

	if _, err := data.Subset[int](subset, []int{2}); err == nil {
		t.Fatal("accepted invalid selection")
	}
}
