package backendtest

import (
	"errors"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backend"

	"github.com/bpradana/goml/tensor"
)

// HostData reads logical values for a test and reports transfer errors at its
// call site. It is an explicit host read, including synchronization on GPUs.
func HostData(t testing.TB, value *tensor.Tensor) []float32 {
	t.Helper()

	values, err := value.HostData()
	if err != nil {
		t.Fatal(err)
	}

	return values
}

// RunTensor checks public Tensor ownership, scopes, explicit transfers, and
// unsupported capability errors. Copy, Fill, and Binary capabilities are needed.
func RunTensor(t *testing.T, device backend.Device) {
	t.Helper()
	t.Run("Tensor/IndependentViewLeases", func(t *testing.T) {
		input, err := tensor.FromOn(device, []float32{1, 2, 3, 4}, 2, 2)
		if err != nil {
			t.Fatal(err)
		}

		transposed, err := input.Transpose()
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = transposed.Close() }()

		if err := input.Close(); err != nil {
			t.Fatal(err)
		}

		got := HostData(t, transposed)
		if !slices.Equal(got, []float32{1, 3, 2, 4}) {
			t.Fatalf("view values %v", got)
		}

		if _, err := input.HostData(); !errors.Is(err, backend.ErrClosed) {
			t.Fatalf("closed tensor: %v", err)
		}
	})
	t.Run("Tensor/ScopeAndRetain", func(t *testing.T) {
		input, err := tensor.FromOn(device, []float32{1, 2}, 2)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = input.Close() }()

		scope, err := tensor.NewScope(device)
		if err != nil {
			t.Fatal(err)
		}

		scoped, err := scope.To(input)
		if err != nil {
			t.Fatal(err)
		}

		retained, err := scoped.Retain()
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = retained.Close() }()

		output, err := tensor.Add(scoped, scoped)
		if err != nil {
			t.Fatal(err)
		}

		output.Keep()
		defer func() { _ = output.Close() }()

		if err := scope.Close(); err != nil {
			t.Fatal(err)
		}

		if !scoped.IsClosed() || retained.IsClosed() || output.IsClosed() || input.IsClosed() {
			t.Fatal("scope released independent handles")
		}

		if !slices.Equal(HostData(t, output), []float32{2, 4}) {
			t.Fatal("queued result invalid after scope close")
		}
	})
	t.Run("Tensor/MissingCapability", func(t *testing.T) {
		limited := baseOnly{Device: device}

		input, err := tensor.FromOn(limited, []float32{1, 2}, 2)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = input.Close() }()

		if _, err := tensor.Add(input, input); !errors.Is(err, backend.ErrUnsupported) {
			t.Fatalf("missing binary capability: %v", err)
		}

		if _, err := input.MakeContiguous(); !errors.Is(err, backend.ErrUnsupported) {
			t.Fatalf("missing copy capability: %v", err)
		}
	})
}

type baseOnly struct{ backend.Device }

// RunHostAccess checks host-native logical reads and bounds-checked indexing.
func RunHostAccess(t *testing.T, device backend.Device) {
	t.Helper()

	compute, ok := device.(backend.HostAccess)
	if !ok {
		t.Fatal("host access suite requires HostAccess")
	}

	t.Run("HostAccess/StridedValuesAndBounds", func(t *testing.T) {
		input := view(t, device, []float32{1, 2, 3, 4, 5, 6}, 2, 3)
		input.Shape, input.Strides = []int{3, 2}, []int{1, 3}

		values := make([]float32, 6)
		if err := compute.Read(input, values); err != nil {
			t.Fatal(err)
		}

		if !slices.Equal(values, []float32{1, 4, 2, 5, 3, 6}) {
			t.Fatalf("logical values %v", values)
		}

		value, err := compute.At(input, []int{1, 1})
		if err != nil || value != 5 {
			t.Fatalf("At: %g, %v", value, err)
		}

		if _, err := compute.At(input, []int{3, 1}); !errors.Is(err, backend.ErrInvalidView) {
			t.Fatalf("index bounds: %v", err)
		}

		if err := compute.Read(input, make([]float32, 1)); !errors.Is(err, backend.ErrInvalidView) {
			t.Fatalf("read length: %v", err)
		}
	})
}
