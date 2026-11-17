package nn

import (
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/bpradana/goml/tensor/backendtest"

	"github.com/bpradana/goml/tensor"
	"github.com/bpradana/goml/tensor/backend"
)

func testValue(t *testing.T, data []float32, shape []int, tracked bool) *Value {
	t.Helper()

	array, err := tensor.From(data, shape...)
	if err != nil {
		t.Fatal(err)
	}

	value, err := NewValue(array, tracked)
	if err != nil {
		t.Fatal(err)
	}

	return value
}

func checkGradient(t *testing.T, data []float32, shape []int, forward func(*Value) (*Value, error)) {
	t.Helper()

	input := testValue(t, data, shape, true)

	output, err := forward(input)
	if err != nil {
		t.Fatal(err)
	}

	objective, err := Sum(output)
	if err != nil {
		t.Fatal(err)
	}

	if err := objective.Backward(); err != nil {
		t.Fatal(err)
	}

	got := backendtest.HostData(t, input.Grad())

	const epsilon float32 = 1e-3

	for i := range data {
		objectives := [2]float32{}

		for j, delta := range []float32{-epsilon, epsilon} {
			perturbed := slices.Clone(data)
			perturbed[i] += delta

			result, err := forward(testValue(t, perturbed, shape, false))
			if err != nil {
				t.Fatal(err)
			}

			total, err := tensor.Sum(result.Data())
			if err != nil {
				t.Fatal(err)
			}

			objectives[j] = backendtest.HostData(t, total)[0]
		}

		want := float64((objectives[1] - objectives[0]) / (2 * epsilon))
		if math.Abs(float64(got[i])-want) > 0.003+0.015*math.Abs(want) {
			t.Fatalf("gradient at %d: got %g, finite difference %g", i, got[i], want)
		}
	}
}

func TestNewValue_RejectsInvalidTensors(t *testing.T) {
	for _, data := range []*tensor.Tensor{nil, {}} {
		if _, err := NewValue(data, true); err == nil {
			t.Fatal("expected invalid tensor error")
		}
	}
}

func TestValue_IsInferenceDistinguishesConstantAndPredictionInputs(t *testing.T) {
	constant := testValue(t, []float32{1}, nil, false)
	if constant.IsInference() {
		t.Fatal("ordinary constant suppressed future parameter tracking")
	}

	prediction, err := Inference(constant.Data())
	if err != nil {
		t.Fatal(err)
	}

	if !prediction.IsInference() || prediction.RequiresGrad() {
		t.Fatal("inference mode was not exposed correctly")
	}
}

func TestValue_DetachOwnsIndependentHandleWithoutGraph(t *testing.T) {
	input := testValue(t, []float32{2}, nil, true)

	output, err := Mul(input, input)
	if err != nil {
		t.Fatal(err)
	}

	detached, err := output.Detach()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = detached.Data().Close() }()

	if detached.RequiresGrad() || detached.parents != nil || detached.backward != nil ||
		detached.parameter != nil || detached.savedParameters != nil || detached.IsInference() {
		t.Fatal("detach retained graph, parameter, or inference state")
	}

	if detached.Data() == output.Data() {
		t.Fatal("detach borrowed the source handle")
	}

	if err := output.Data().Close(); err != nil {
		t.Fatal(err)
	}

	if got := backendtest.HostData(t, detached.Data())[0]; got != 4 {
		t.Fatalf("detached value %g, want 4 after closing source", got)
	}
}

func TestValue_DetachSurvivesParameterUpdateBeforeFirstUse(t *testing.T) {
	original := tensor.Scalar(2)
	defer func() { _ = original.Close() }()

	parameter, err := NewParameter(original)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = parameter.Value().Data().Close() }()

	detached, err := parameter.Value().Detach()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = detached.Data().Close() }()

	replacement := tensor.Scalar(20)
	defer func() { _ = replacement.Close() }()

	if err := parameter.Update(replacement); err != nil {
		t.Fatal(err)
	}

	if got := backendtest.HostData(t, detached.Data())[0]; got != 2 {
		t.Fatalf("detached value %g, want original value 2", got)
	}

	input := testValue(t, []float32{3}, nil, true)

	product, err := Mul(input, detached)
	if err != nil {
		t.Fatal(err)
	}

	if err := product.Backward(); err != nil {
		t.Fatal(err)
	}

	if got := backendtest.HostData(t, input.Grad())[0]; got != 2 {
		t.Fatalf("gradient %g, want detached multiplier 2", got)
	}

	if parameter.Value().Grad() != nil {
		t.Fatal("detached parameter received a gradient")
	}

	// Consuming one graph must not release a constant used by another pass.
	product, err = Mul(input, detached)
	if err != nil {
		t.Fatal(err)
	}

	if err := product.Backward(); err != nil {
		t.Fatal(err)
	}

	if got := backendtest.HostData(t, input.Grad())[0]; got != 4 {
		t.Fatalf("accumulated gradient %g, want 4", got)
	}

	if err := detached.Data().Close(); err != nil {
		t.Fatal(err)
	}

	if got := backendtest.HostData(t, parameter.Value().Data())[0]; got != 20 {
		t.Fatalf("closing detached handle changed parameter to %g", got)
	}
}

func TestValue_DetachSurvivesSourceScopeClosure(t *testing.T) {
	scope, err := tensor.NewScope(tensor.CPU())
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = scope.Close() }()

	original := tensor.Scalar(7)
	defer func() { _ = original.Close() }()

	scoped, err := scope.To(original)
	if err != nil {
		t.Fatal(err)
	}

	value, err := NewValue(scoped, false)
	if err != nil {
		t.Fatal(err)
	}

	detached, err := value.Detach()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = detached.Data().Close() }()

	if err := scope.Close(); err != nil {
		t.Fatal(err)
	}

	if !scoped.IsClosed() {
		t.Fatal("source scope did not close its handle")
	}

	if got := backendtest.HostData(t, detached.Data())[0]; got != 7 {
		t.Fatalf("detached value %g, want 7 after scope closure", got)
	}
}

func TestValue_DetachAllowsConsumedGraph(t *testing.T) {
	input := testValue(t, []float32{3}, nil, true)

	output, err := Mul(input, input)
	if err != nil {
		t.Fatal(err)
	}

	if err := output.Backward(); err != nil {
		t.Fatal(err)
	}

	detached, err := output.Detach()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = detached.Data().Close() }()

	result, err := Add(detached, detached)
	if err != nil {
		t.Fatal(err)
	}

	if result.RequiresGrad() || backendtest.HostData(t, result.Data())[0] != 18 {
		t.Fatal("detached consumed result was not an ordinary constant")
	}
}

func TestValue_DetachRejectsInvalidSources(t *testing.T) {
	closed := testValue(t, []float32{1}, nil, false)
	if err := closed.Data().Close(); err != nil {
		t.Fatal(err)
	}

	for _, tt := range []struct {
		name  string
		value *Value
	}{
		{name: "nil"},
		{name: "uninitialized", value: &Value{}},
		{name: "uninitialized_tensor", value: &Value{data: &tensor.Tensor{}}},
		{name: "closed", value: closed},
	} {
		t.Run(tt.name, func(t *testing.T) {
			detached, err := tt.value.Detach()
			if err == nil || detached != nil {
				t.Fatalf("Detach() = %v, %v; want nil and error", detached, err)
			}

			if tt.value == closed && !errors.Is(err, backend.ErrClosed) {
				t.Fatalf("closed source error %v, want ErrClosed", err)
			}
		})
	}
}

func TestInference_SuppressesParameterTracking(t *testing.T) {
	input, err := Inference(tensor.Scalar(2))
	if err != nil {
		t.Fatal(err)
	}

	weight := testValue(t, []float32{3}, nil, true)

	output, err := Mul(input, weight)
	if err != nil {
		t.Fatal(err)
	}

	if output.RequiresGrad() || output.parents != nil || output.backward != nil {
		t.Fatal("inference constructed a graph")
	}

	if err := output.Backward(); err == nil {
		t.Fatal("inference backward must fail")
	}
}
