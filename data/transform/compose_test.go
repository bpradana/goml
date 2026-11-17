package transform_test

import (
	"context"
	"errors"
	"testing"

	"github.com/bpradana/goml/data/transform"
)

func TestCompose_TypedOrderFailureAndCancellation(t *testing.T) {
	first := transform.Func[int](func(_ context.Context, x int) (int, error) { return x + 1, nil })
	second := transform.Func[int](func(_ context.Context, x int) (int, error) { return x * 2, nil })

	result, err := transform.Compose(first, second)(context.Background(), 3)
	if err != nil || result != 8 {
		t.Fatal(result, err)
	}

	wanted := errors.New("stop")

	fail := transform.Func[int](func(context.Context, int) (int, error) { return 0, wanted })
	if _, err := transform.Compose(first, fail, second)(context.Background(), 3); !errors.Is(err, wanted) {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := transform.Compose[int]()(ctx, 3); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}

	if _, err := transform.Compose[int](nil)(context.Background(), 3); err == nil {
		t.Fatal("accepted nil transform")
	}
}
