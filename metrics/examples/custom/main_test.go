package main

import "testing"

func TestRun_CustomTypedMetric(t *testing.T) {
	if err := run(); err != nil {
		t.Fatal(err)
	}
}
