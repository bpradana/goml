package main

import "testing"

func TestRun_LSTMLearnsFirstBitRecall(t *testing.T) {
	if err := run(); err != nil {
		t.Fatal(err)
	}
}
