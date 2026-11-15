package main

import "testing"

func TestRun_LinearRegressionTraining(t *testing.T) {
	if err := run(); err != nil {
		t.Fatal(err)
	}
}
