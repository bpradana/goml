package main

import (
	"context"
	"testing"
)

func TestCorpus_ArbitrarySamplesAndCustomCollation(t *testing.T) {
	if err := run(); err != nil {
		t.Fatal(err)
	}

	if _, err := (corpus{"one"}).Get(context.Background(), -1); err == nil {
		t.Fatal("accepted negative index")
	}
}
