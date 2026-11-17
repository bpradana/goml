package main

import "testing"

func TestRun_StreamingReport(t *testing.T) {
	if err := run(); err != nil {
		t.Fatal(err)
	}
}
