package data_test

import (
	"testing"

	"github.com/bpradana/goml/data"
	"github.com/bpradana/goml/tensor"
)

func TestBatch_CloseIsIndependentAndIdempotent(t *testing.T) {
	var empty *data.ClassificationBatch
	if err := empty.Close(); err != nil {
		t.Fatal(err)
	}

	x, _ := tensor.Ones(2)
	y, _ := tensor.Ones(1)

	b := &data.RegressionBatch{Inputs: x, Targets: y}
	for range 2 {
		if err := b.Close(); err != nil {
			t.Fatal(err)
		}
	}

	if !x.IsClosed() || !y.IsClosed() {
		t.Fatal("regression tensors remain open")
	}
}
