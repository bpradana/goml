package main

import (
	"context"
	"math"
	"reflect"
	"testing"
)

func fixtureRecords(perClass int) []byte {
	records := make([]byte, 10*perClass*3073)
	for label := range 10 {
		for sample := range perClass {
			record := records[(label*perClass+sample)*3073:][:3073]
			record[0] = byte(label)

			for channel := range 3 {
				for pixel := range 1024 {
					record[1+channel*1024+pixel] = byte((label*19 + sample*7 + channel*31 + pixel) % 256)
				}
			}
		}
	}

	return records
}

func TestRecordsRejectTruncationAndInvalidLabels(t *testing.T) {
	valid := fixtureRecords(1)
	if err := validateRecords(valid); err != nil {
		t.Fatal(err)
	}

	for _, input := range [][]byte{nil, valid[:len(valid)-1], append([]byte{10}, valid[1:]...)} {
		if err := validateRecords(input); err == nil {
			t.Fatal("accepted malformed CIFAR records")
		}
	}
}

func TestSplitIsBalancedDisjointAndReproducible(t *testing.T) {
	records := fixtureRecords(5)

	train, validation, err := splitIndices(records, 2, 1, 42)
	if err != nil {
		t.Fatal(err)
	}

	again, againValidation, err := splitIndices(records, 2, 1, 42)
	if err != nil || !reflect.DeepEqual(train, again) || !reflect.DeepEqual(validation, againValidation) {
		t.Fatal("split is not reproducible", err)
	}

	seen := make(map[int]bool)

	for split, indices := range [][]int{train, validation} {
		var counts [10]int

		for _, index := range indices {
			if seen[index] {
				t.Fatal("duplicate or overlapping split index", index)
			}

			seen[index] = true
			counts[records[index*3073]]++
		}

		for _, count := range counts {
			if count != 2-split {
				t.Fatal("unbalanced split", counts)
			}
		}
	}

	if _, _, err := splitIndices(records, 5, 1, 42); err == nil {
		t.Fatal("accepted an oversized split")
	}
}

func TestNormalizationUsesOnlyTrainingPixelsAndPreservesCHW(t *testing.T) {
	records := make([]byte, 3*3073)

	for sample := range 3 {
		for channel := range 3 {
			for pixel := range 1024 {
				records[sample*3073+1+channel*1024+pixel] = byte([]int{0, 255, 64}[sample])
			}
		}
	}

	norm, err := fitNormalization(records, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}

	for channel := range 3 {
		if math.Abs(norm.Mean[channel]-0.5) > 1e-12 || math.Abs(norm.Std[channel]-0.5) > 1e-12 {
			t.Fatal("normalization leaked non-training pixels", norm)
		}
	}

	dataset := imageDataset{records: records, indices: []int{2}, normalization: norm}

	sample, err := dataset.Get(context.Background(), 0)
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(sample.Shape, []int{3, 32, 32}) || math.Abs(float64(sample.Input[0])-(-0.4980392156862745)) > 1e-6 {
		t.Fatal("wrong normalized CHW sample", sample.Shape, sample.Input[0])
	}

	sample.Input[0] = 999

	if records[2*3073+1] != 64 {
		t.Fatal("sample mutated source records")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := dataset.Get(ctx, 0); err == nil {
		t.Fatal("ignored cancellation")
	}

	if _, err := dataset.Get(context.Background(), 1); err == nil {
		t.Fatal("accepted out-of-range sample")
	}
}
