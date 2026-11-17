package trainer

import (
	"fmt"
	"math"
)

// EarlyStopping monitors validation mean loss. An improvement must be strictly
// greater than MinDelta from the last sufficient improvement. The first epoch
// establishes a baseline. Patience consecutive misses stop before the next
// epoch; Patience must be positive and MinDelta finite/nonnegative.
type EarlyStopping struct {
	Patience int
	MinDelta float64
}

type stoppingState struct {
	best   float64
	misses int
}

func (s *stoppingState) update(value float64, options EarlyStopping) bool {
	if math.IsInf(s.best, 1) || value < s.best-options.MinDelta {
		s.best = value
		s.misses = 0
	} else {
		s.misses++
	}

	return s.misses >= options.Patience
}

func (e EarlyStopping) validate() error {
	if e.Patience <= 0 || e.MinDelta < 0 || math.IsNaN(e.MinDelta) || math.IsInf(e.MinDelta, 0) {
		return fmt.Errorf("trainer: early stopping requires positive patience and finite nonnegative min delta")
	}

	return nil
}
