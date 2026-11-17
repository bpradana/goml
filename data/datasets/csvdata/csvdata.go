// Package csvdata loads finite numerical CSV classification/regression data into
// host memory. Columns are selected explicitly; no types or labels are inferred.
package csvdata

import (
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strconv"

	"github.com/bpradana/goml/data"
)

// Options selects columns by exact header name when Header is true, or by
// zero-based decimal index strings ("0", "2") for headerless files. Features
// and Targets must both be nonempty and contain no duplicates or overlap.
type Options struct {
	Header   bool
	Features []string
	Targets  []string
}

type table struct {
	inputs                       []float32
	targets                      []float32
	labels                       []int
	count, features, targetCount int
}

// Classification owns finite features and nonnegative integer labels. It has no
// open files after construction and supports concurrent Get calls.
type Classification struct{ table table }

// Regression owns finite numerical inputs and targets. Get copies sample data.
type Regression struct{ table table }

// OpenClassification reads a local CSV with exactly one selected target column.
// Class labels use integer syntax: "1" is valid, "1.0" is rejected explicitly.
func OpenClassification(ctx context.Context, path string, options Options) (*Classification, error) {
	if len(options.Targets) != 1 {
		return nil, fmt.Errorf("csvdata: classification requires one integer target column")
	}

	table, err := read(ctx, path, options, true)
	if err != nil {
		return nil, err
	}

	return &Classification{table: table}, nil
}

// OpenRegression reads finite Float32 feature and target columns in selection
// order. Sample shapes are [features] and [targets], even for a single column.
func OpenRegression(ctx context.Context, path string, options Options) (*Regression, error) {
	table, err := read(ctx, path, options, false)
	if err != nil {
		return nil, err
	}

	return &Regression{table: table}, nil
}

type contextReader struct {
	ctx    context.Context
	source io.Reader
}

func (r contextReader) Read(dst []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}

	return r.source.Read(dst)
}

func read(ctx context.Context, path string, options Options, classification bool) (table, error) {
	result := table{features: len(options.Features), targetCount: len(options.Targets)}
	if len(options.Features) == 0 || len(options.Targets) == 0 {
		return table{}, fmt.Errorf("csvdata: explicit feature and target columns required")
	}

	if err := ctx.Err(); err != nil {
		return table{}, err
	}

	file, err := os.Open(path)
	if err != nil {
		return table{}, err
	}
	defer func() { _ = file.Close() }()

	reader := csv.NewReader(contextReader{ctx: ctx, source: file})

	first, err := reader.Read()
	if err != nil {
		return table{}, fmt.Errorf("csvdata: first row: %w", err)
	}

	header := make(map[string]int)

	if options.Header {
		for index, name := range first {
			if _, exists := header[name]; exists {
				return table{}, fmt.Errorf("csvdata: duplicate header column %q", name)
			}

			header[name] = index
		}
	}

	selected := append(slices.Clone(options.Features), options.Targets...)
	columns := make([]int, len(selected))
	seen := make(map[int]bool)

	for i, name := range selected {
		var index int

		if options.Header {
			var exists bool

			index, exists = header[name]
			if !exists {
				return table{}, fmt.Errorf("csvdata: missing column %q", name)
			}
		} else {
			index, err = strconv.Atoi(name)
			if err != nil || index < 0 || index >= len(first) {
				return table{}, fmt.Errorf("csvdata: invalid zero-based column %q", name)
			}
		}

		if seen[index] {
			return table{}, fmt.Errorf("csvdata: duplicate selected column %q", name)
		}

		seen[index] = true
		columns[i] = index
	}

	record := first
	if options.Header {
		record, err = reader.Read()
	}

	for err != io.EOF {
		if err != nil {
			return table{}, fmt.Errorf("csvdata: record %d: %w", result.count+1, err)
		}

		if err := ctx.Err(); err != nil {
			return table{}, err
		}

		for i, column := range columns {
			line, position := reader.FieldPos(column)
			if classification && i == len(options.Features) {
				label, err := strconv.Atoi(record[column])
				if err != nil || label < 0 {
					return table{}, fmt.Errorf("csvdata: row %d column %q (byte %d): expected nonnegative integer class", line, selected[i], position)
				}

				result.labels = append(result.labels, label)

				continue
			}

			value, err := strconv.ParseFloat(record[column], 32)
			if err != nil || math.IsNaN(value) || math.IsInf(value, 0) {
				return table{}, fmt.Errorf("csvdata: row %d column %q (byte %d): invalid finite Float32 %q", line, selected[i], position, record[column])
			}

			if i < len(options.Features) {
				result.inputs = append(result.inputs, float32(value))
			} else {
				result.targets = append(result.targets, float32(value))
			}
		}

		result.count++
		record, err = reader.Read()
	}

	return result, nil
}

// Len returns the stable parsed sample count.
func (d *Classification) Len() int {
	return d.table.count
}

// Get copies a classification sample, checking context and index bounds.
func (d *Classification) Get(ctx context.Context, index int) (data.ClassificationSample, error) {
	if err := ctx.Err(); err != nil {
		return data.ClassificationSample{}, err
	}

	if index < 0 || index >= d.Len() {
		return data.ClassificationSample{}, fmt.Errorf("csvdata: index %d out of range", index)
	}

	width := d.table.features

	return data.ClassificationSample{Input: slices.Clone(d.table.inputs[index*width : (index+1)*width]), Shape: []int{width}, Label: d.table.labels[index]}, nil
}

// Len returns the stable parsed sample count.
func (d *Regression) Len() int {
	return d.table.count
}

// Get copies a regression sample, checking context and index bounds.
func (d *Regression) Get(ctx context.Context, index int) (data.RegressionSample, error) {
	if err := ctx.Err(); err != nil {
		return data.RegressionSample{}, err
	}

	if index < 0 || index >= d.Len() {
		return data.RegressionSample{}, fmt.Errorf("csvdata: index %d out of range", index)
	}

	w, tw := d.table.features, d.table.targetCount

	return data.RegressionSample{Input: slices.Clone(d.table.inputs[index*w : (index+1)*w]), Shape: []int{w}, Target: slices.Clone(d.table.targets[index*tw : (index+1)*tw]), TargetShape: []int{tw}}, nil
}
