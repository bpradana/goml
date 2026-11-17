package main

import "testing"

func TestRun_ClassificationDatasetAndPartialBatch(t *testing.T) {
	if err := run(); err != nil {
		t.Fatal(err)
	}
}
