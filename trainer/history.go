package trainer

import (
	"fmt"
	"math"
	"slices"

	"github.com/bpradana/goml/metrics"
)

// EpochStats reports sample-weighted mean loss; Batches counts successful steps.
type EpochStats struct {
	Samples, Batches uint64
	MeanLoss         float64
}

// Evaluation contains task-specific snapshots. Custom Fit steps report Stats
// only and can accumulate additional typed statistics in application-owned state.
type Evaluation struct {
	EpochStats
	Classification *metrics.ClassificationResult
	Regression     *metrics.RegressionResult
}

// EpochReport numbers epochs from one; loaders receive the corresponding
// zero-based epoch. Validation is nil when disabled. Reports own no tensors.
type EpochReport struct {
	Epoch          int
	Train          EpochStats
	Validation     *EpochStats
	Classification *metrics.ClassificationResult
	Regression     *metrics.RegressionResult
}

// History contains completed epochs, including those completed before an error.
// A failed/canceled run returns this history alongside its error. StopReason is
// empty on error, "completed" normally, or "early_stopping" on a patience stop.
// Best checkpoints are not restored automatically.
type History struct {
	Epochs       []EpochReport
	StoppedEarly bool
	StopReason   string
}

func (s *EpochStats) add(step StepResult) error {
	if step.Samples == 0 || step.Samples > ^uint64(0)-s.Samples || s.Batches == ^uint64(0) || math.IsNaN(step.MeanLoss) || math.IsInf(step.MeanLoss, 0) {
		return fmt.Errorf("trainer: step requires a finite mean loss, positive sample count, and nonoverflowing totals")
	}

	count := s.Samples + step.Samples
	// Convex weighting avoids overflow from summing large weighted losses.
	s.MeanLoss = s.MeanLoss*(float64(s.Samples)/float64(count)) + step.MeanLoss*(float64(step.Samples)/float64(count))
	s.Samples = count
	s.Batches++

	return nil
}

func cloneReport(report EpochReport) EpochReport {
	if report.Validation != nil {
		stats := *report.Validation
		report.Validation = &stats
	}

	if report.Classification != nil {
		result := *report.Classification
		result.PerClass = slices.Clone(result.PerClass)

		result.Confusion = make([][]uint64, len(result.Confusion))
		for row, values := range report.Classification.Confusion {
			result.Confusion[row] = slices.Clone(values)
		}

		report.Classification = &result
	}

	if report.Regression != nil {
		result := *report.Regression
		report.Regression = &result
	}

	return report
}
