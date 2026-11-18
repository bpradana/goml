package metrics_test

import (
	"fmt"
	"testing"

	"github.com/bpradana/goml/metrics"
)

type (
	biasResult struct {
		Count int
		Mean  float64
	}
	meanBias struct {
		count int
		sum   float64
	}
)

func (m *meanBias) Update(predictions, targets []float32) error {
	if len(predictions) != len(targets) {
		return fmt.Errorf("length mismatch")
	}

	for i, prediction := range predictions {
		m.sum += float64(prediction) - float64(targets[i])
		m.count++
	}

	return nil
}

func (m *meanBias) Result() (biasResult, error) {
	if m.count == 0 {
		return biasResult{}, fmt.Errorf("empty metric")
	}

	return biasResult{Count: m.count, Mean: m.sum / float64(m.count)}, nil
}

func (m *meanBias) Reset() { *m = meanBias{} }

func TestMetric_ExternalTypedImplementationAndBuiltins(t *testing.T) {
	var custom metrics.Metric[float32, float32, biasResult] = &meanBias{}
	if err := custom.Update([]float32{1, 3}, []float32{0, 1}); err != nil {
		t.Fatal(err)
	}

	result, err := custom.Result()
	if err != nil || result.Count != 2 || result.Mean != 1.5 {
		t.Fatal(result, err)
	}

	custom.Reset()

	if _, err := custom.Result(); err == nil {
		t.Fatal("reset failed")
	}
}

var (
	_ metrics.Metric[int, int, metrics.ClassificationResult]     = (*metrics.Classification)(nil)
	_ metrics.Metric[float32, float32, metrics.RegressionResult] = (*metrics.Regression)(nil)
)

func TestMetric_ClassificationLogitsAdapterSharesTypedState(t *testing.T) {
	classification, err := metrics.NewClassification(2)
	if err != nil {
		t.Fatal(err)
	}

	metric := classification.Logits()
	if err := metric.Update([]float32{0, 1, 2, 0}, []int{1, 1}); err != nil {
		t.Fatal(err)
	}

	result, err := metric.Result()
	if err != nil || result.Accuracy != 0.5 {
		t.Fatal(result, err)
	}

	classification.Reset()

	if _, err := metric.Result(); err == nil {
		t.Fatal("adapter did not share reset")
	}
}

type rankedPrediction struct{ Labels []string }

type expectedLabel string

type rankingResult struct{ Count, Hits int }

type rankingMetric struct{ result rankingResult }

func (m *rankingMetric) Update(predictions []rankedPrediction, targets []expectedLabel) error {
	if len(predictions) != len(targets) {
		return fmt.Errorf("count mismatch")
	}

	for index, prediction := range predictions {
		m.result.Count++

		for _, label := range prediction.Labels {
			if label == string(targets[index]) {
				m.result.Hits++
				break
			}
		}
	}

	return nil
}

func (m *rankingMetric) Result() (rankingResult, error) { return m.result, nil }

func (m *rankingMetric) Reset() { m.result = rankingResult{} }

func TestMetric_ExternalPredictionTargetAndReportTypes(t *testing.T) {
	var metric metrics.Metric[rankedPrediction, expectedLabel, rankingResult] = &rankingMetric{}
	for range 2 {
		if err := metric.Update([]rankedPrediction{{Labels: []string{"cat", "dog"}}}, []expectedLabel{"dog"}); err != nil {
			t.Fatal(err)
		}
	}

	result, err := metric.Result()
	if err != nil || result.Count != 2 || result.Hits != 2 {
		t.Fatal(result, err)
	}

	metric.Reset()

	cleared, _ := metric.Result()
	if cleared.Count != 0 || result.Count != 2 {
		t.Fatal(result, cleared)
	}
}
