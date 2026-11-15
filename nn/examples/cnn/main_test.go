package main

import "testing"

func TestRun_CNNLearnsLineOrientation(t *testing.T) {
	if err := run(); err != nil {
		t.Fatal(err)
	}
}
