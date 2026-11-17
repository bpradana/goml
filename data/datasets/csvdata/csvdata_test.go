package csvdata_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/bpradana/goml/data/datasets/csvdata"
)

func csvFile(t *testing.T, content string) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "values.csv")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	return path
}

func TestCSV_ExplicitColumnOrderAndIntegerLabels(t *testing.T) {
	path := csvFile(t, "age,income,target\n25,5000,1\n40,9000,0\n")
	options := csvdata.Options{Header: true, Features: []string{"income", "age"}, Targets: []string{"target"}}

	ds, err := csvdata.OpenClassification(context.Background(), path, options)
	if err != nil {
		t.Fatal(err)
	}

	sample, err := ds.Get(context.Background(), 0)
	if err != nil || !slices.Equal(sample.Input, []float32{5000, 25}) || sample.Label != 1 || ds.Len() != 2 {
		t.Fatal(sample, err)
	}

	var workers sync.WaitGroup
	for range 4 {
		workers.Go(func() {
			sample, err := ds.Get(context.Background(), 0)
			if err != nil || sample.Input[0] != 5000 {
				t.Error(err)
				return
			}

			sample.Input[0] = 99
		})
	}

	workers.Wait()

	regression, err := csvdata.OpenRegression(context.Background(), csvFile(t, "2,3,4\n5,6,7\n"), csvdata.Options{Features: []string{"1"}, Targets: []string{"0", "2"}})
	if err != nil {
		t.Fatal(err)
	}

	target, err := regression.Get(context.Background(), 1)
	if err != nil || !slices.Equal(target.Input, []float32{6}) || !slices.Equal(target.Target, []float32{5, 7}) {
		t.Fatal(target, err)
	}
}

func TestCSV_RejectsMalformedRowsMissingColumnsAndNonfiniteNumbers(t *testing.T) {
	base := csvdata.Options{Header: true, Features: []string{"x"}, Targets: []string{"y"}}
	for _, content := range []string{"", "x,x,y\n1,2,3\n", "z,y\n1,2\n", "x,y\n1\n", "x,y\n\"bad\n", "x,y\nNaN,1\n", "x,y\nInf,1\n", "x,y\n1e100,1\n", "x,y\n1,1.0\n", "x,y\n1,-1\n"} {
		if _, err := csvdata.OpenClassification(context.Background(), csvFile(t, content), base); err == nil {
			t.Fatalf("accepted %q", content)
		}
	}

	for _, options := range []csvdata.Options{{}, {Header: true, Features: []string{"x", "x"}, Targets: []string{"y"}}, {Header: true, Features: []string{"x"}, Targets: []string{"x"}}} {
		if _, err := csvdata.OpenClassification(context.Background(), csvFile(t, "x,y\n1,2\n"), options); err == nil {
			t.Fatal("accepted invalid columns")
		}
	}

	_, err := csvdata.OpenClassification(context.Background(), csvFile(t, "x,y\n1,0\nwrong,1\n"), base)
	if err == nil || !strings.Contains(err.Error(), "row 3 column \"x\"") {
		t.Fatal("missing row/column context", err)
	}
}

func TestCSV_IndexBoundsEmptyHeaderAndCancellation(t *testing.T) {
	options := csvdata.Options{Header: true, Features: []string{"x"}, Targets: []string{"y"}}
	path := csvFile(t, "x,y\n")

	ds, err := csvdata.OpenRegression(context.Background(), path, options)
	if err != nil || ds.Len() != 0 {
		t.Fatal(ds, err)
	}

	if _, err := ds.Get(context.Background(), -1); err == nil {
		t.Fatal("accepted invalid index")
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, err := csvdata.OpenRegression(ctx, path, options); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
