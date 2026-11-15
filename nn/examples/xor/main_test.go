package main

import "testing"

func TestRun_XORTraining(t *testing.T) {
	if err := run(); err != nil {
		t.Fatal(err)
	}
}
