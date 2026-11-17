// Package metrics provides streaming Float64 evaluation statistics without
// dependencies outside the standard library. Accumulators require external
// synchronization for concurrent mutation and own no external resources.
package metrics

import (
	"fmt"
	"math"
	"slices"
)

// Classification accumulates single-label confusion counts. Construct it with
// NewClassification. Rows represent actual classes; columns predicted classes.
type Classification struct {
	classes   int
	count     uint64
	confusion []uint64
}

type logitsMetric struct {
	*Classification
}

// Logits adapts this accumulator to the generic raw-score metric interface used
// by Trainer classification evaluation. It shares the same state: Result and
// Reset apply to the original accumulator. Scores have shape [batch,classes].
func (m *Classification) Logits() Metric[float32, int, ClassificationResult] {
	return logitsMetric{Classification: m}
}

func (m logitsMetric) Update(scores []float32, targets []int) error {
	return m.UpdateLogits(scores, targets)
}

// ClassResult reports one class. Undefined precision, recall, and F1 are zero.
type ClassResult struct {
	Class                 int
	Support               uint64
	Precision, Recall, F1 float64
}

// ClassificationResult is an independent snapshot. Macro means include every
// configured class; weighted means use actual support. Confusion[actual][predicted]
// contains integer observations. Single-label micro means equal Accuracy.
type ClassificationResult struct {
	Count                                         uint64
	Accuracy                                      float64
	PrecisionMicro, RecallMicro, F1Micro          float64
	PrecisionMacro, RecallMacro, F1Macro          float64
	PrecisionWeighted, RecallWeighted, F1Weighted float64
	PerClass                                      []ClassResult
	Confusion                                     [][]uint64
}

// NewClassification allocates an O(classes²) confusion matrix. Classes must be
// positive and its byte count must fit the address space. No class is inferred.
func NewClassification(classes int) (*Classification, error) {
	if classes <= 0 || classes > int(^uint(0)>>1)/8/classes {
		return nil, fmt.Errorf("metrics: classification: invalid class count %d", classes)
	}

	return &Classification{classes: classes, confusion: make([]uint64, classes*classes)}, nil
}

func (m *Classification) validate(targets []int, count int) error {
	if m == nil || m.classes <= 0 {
		return fmt.Errorf("metrics: classification: use NewClassification")
	}

	if count != len(targets) || uint64(count) > ^uint64(0)-m.count {
		return fmt.Errorf("metrics: classification: count mismatch or counter overflow")
	}

	for index, target := range targets {
		if target < 0 || target >= m.classes {
			return fmt.Errorf("metrics: classification: target %d at row %d outside [0,%d)", target, index, m.classes)
		}
	}

	return nil
}

// Update validates the entire batch, then accumulates predicted/actual class
// IDs without retaining or mutating caller slices. Invalid updates are atomic.
// An empty update is valid but does not make an empty Result defined.
func (m *Classification) Update(predictions, targets []int) error {
	if err := m.validate(targets, len(predictions)); err != nil {
		return err
	}

	for index, prediction := range predictions {
		if prediction < 0 || prediction >= m.classes {
			return fmt.Errorf("metrics: classification: prediction %d at row %d outside [0,%d)", prediction, index, m.classes)
		}
	}

	for index, prediction := range predictions {
		m.confusion[targets[index]*m.classes+prediction]++
	}

	m.count += uint64(len(targets))

	return nil
}

// UpdateLogits accumulates flattened [batch,classes] finite scores. ArgMax uses
// the lowest class index on ties; softmax is unnecessary. It allocates no
// prediction array and leaves state unchanged on any invalid score/target.
func (m *Classification) UpdateLogits(logits []float32, targets []int) error {
	if err := m.validate(targets, len(targets)); err != nil {
		return err
	}

	if len(targets) > int(^uint(0)>>1)/m.classes || len(logits) != len(targets)*m.classes {
		return fmt.Errorf("metrics: classification: expected [%d,%d] logits, got %d values", len(targets), m.classes, len(logits))
	}

	for index, value := range logits {
		if !finite(value) {
			return fmt.Errorf("metrics: classification: nonfinite score at value %d", index)
		}
	}

	for row, target := range targets {
		best := 0
		for column := 1; column < m.classes; column++ {
			if logits[row*m.classes+column] > logits[row*m.classes+best] {
				best = column
			}
		}

		m.confusion[target*m.classes+best]++
	}

	m.count += uint64(len(targets))

	return nil
}

// Result computes an independently owned snapshot, or errors before any
// observations. Counts are converted to Float64 for ratios; integers beyond
// Float64's exact range can round in reported ratios, never in stored counters.
func (m *Classification) Result() (ClassificationResult, error) {
	if m == nil || m.count == 0 {
		return ClassificationResult{}, fmt.Errorf("metrics: classification: no observations")
	}

	result := ClassificationResult{Count: m.count, PerClass: make([]ClassResult, m.classes), Confusion: make([][]uint64, m.classes)}

	var correct uint64

	for class := range m.classes {
		result.Confusion[class] = slices.Clone(m.confusion[class*m.classes : (class+1)*m.classes])

		var support, predicted uint64
		for other := range m.classes {
			support += m.confusion[class*m.classes+other]
			predicted += m.confusion[other*m.classes+class]
		}

		tp := m.confusion[class*m.classes+class]

		item := ClassResult{Class: class, Support: support}
		if predicted != 0 {
			item.Precision = float64(tp) / float64(predicted)
		}

		if support != 0 {
			item.Recall = float64(tp) / float64(support)
		}

		if support != 0 || predicted != 0 {
			item.F1 = 2 * float64(tp) / (float64(support) + float64(predicted))
		}

		weight := float64(support) / float64(m.count)
		result.PerClass[class] = item
		result.PrecisionMacro += item.Precision / float64(m.classes)
		result.RecallMacro += item.Recall / float64(m.classes)
		result.F1Macro += item.F1 / float64(m.classes)
		result.PrecisionWeighted += weight * item.Precision
		result.RecallWeighted += weight * item.Recall
		result.F1Weighted += weight * item.F1
		correct += tp
	}

	result.Accuracy = float64(correct) / float64(m.count)
	result.PrecisionMicro, result.RecallMicro, result.F1Micro = result.Accuracy, result.Accuracy, result.Accuracy

	return result, nil
}

// Reset clears all counters while preserving configuration/allocation. Returned
// result snapshots remain unchanged. Reset does not allocate.
func (m *Classification) Reset() {
	clear(m.confusion)
	m.count = 0
}

func finite(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
