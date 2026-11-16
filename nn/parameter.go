package nn

import (
	"fmt"
	"runtime"
	"slices"
	"sync"

	"github.com/bpradana/goml/tensor"
)

// Parameter owns a trainable leaf and its gradient. Optimizer updates preserve
// the leaf's identity and replace its immutable tensor data.
type Parameter struct {
	value *Value
}

// NewParameter wraps immutable data in a trainable leaf.
// Nil or uninitialized tensors return an error.
func NewParameter(data *tensor.Tensor) (*Parameter, error) {
	if data != nil {
		owned, err := data.Retain()
		if err != nil {
			return nil, err
		}

		data = owned
	}

	v, err := NewValue(data, true)
	if err != nil {
		return nil, err
	}

	v.parameter = &parameterData{data: data}

	return &Parameter{value: v}, nil
}

// Value returns the trainable leaf used in differentiable computations.
func (p *Parameter) Value() *Value {
	return p.value
}

// ZeroGrad clears this parameter's accumulated gradient.
func (p *Parameter) ZeroGrad() {
	p.value.ZeroGrad()
}

// Update replaces the immutable tensor data while preserving the parameter's
// identity and accumulated gradient. The new shape must match the old shape.
// Optimizers use this boundary; callers must synchronize updates with training.
// Parameters own an independent handle. Replacement retains forward generations
// still used by live graphs; external callers retain their own views explicitly.
func (p *Parameter) Update(data *tensor.Tensor) error {
	if p == nil || p.value == nil {
		return fmt.Errorf("nn: parameter: uninitialized parameter")
	}

	if data == nil || data.IsClosed() || !slices.Equal(data.Shape(), p.value.data.Shape()) ||
		data.Numel() != p.value.data.Numel() {
		return fmt.Errorf("nn: parameter: update must preserve shape %v", p.value.data.Shape())
	}

	if !sameDevice(data, p.value.data) {
		return fmt.Errorf("nn: parameter: update must preserve device; use nn.ToDevice")
	}

	owned, err := data.Retain()
	if err != nil {
		return err
	}

	p.replace(owned)

	return nil
}

// replace accepts an already-owned handle. Readers of older forward passes
// keep that generation alive until their graph releases it.
func (p *Parameter) replace(data *tensor.Tensor) {
	previous := p.value.parameter
	p.value.data = data
	p.value.parameter = &parameterData{data: data}

	previous.retire()
}

type parameterData struct {
	mu      sync.Mutex
	data    *tensor.Tensor
	readers int
	retired bool
}

func (p *parameterData) acquire() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.readers++
}

func (p *parameterData) release() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.readers--
	if p.retired && p.readers == 0 {
		_ = p.data.Close()
	}
}

func (p *parameterData) retire() {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.retired = true
	if p.readers == 0 {
		_ = p.data.Close()
	}
}

// parameterReads is held by the graph cleanup argument, never by a Parameter.
// Its once guard coordinates explicit consumption with abandoned-graph cleanup.
type parameterReads struct {
	cleanup runtime.Cleanup
	once    sync.Once
	values  []*parameterData
}

func (r *parameterReads) release() {
	r.once.Do(func() {
		for _, value := range r.values {
			value.release()
		}

		r.values = nil
	})
}

func (p *parameterData) inUse() bool {
	p.mu.Lock()
	defer p.mu.Unlock()

	return p.readers > 0
}
