package optimizer

import (
	"fmt"
	"testing"

	"github.com/bpradana/goml/nn"
	"github.com/bpradana/goml/tensor"
)

func BenchmarkOptimizerStep(b *testing.B) {
	for _, configuration := range []struct {
		name string
		size int
	}{{"SGD", 128}, {"Adam", 128}, {"SGD", 65536}, {"Adam", 65536}} {
		name := configuration.name
		b.Run(fmt.Sprintf("%s/n=%d", name, configuration.size), func(b *testing.B) {
			data, err := tensor.Ones(configuration.size)
			if err != nil {
				b.Fatal(err)
			}

			parameter, err := nn.NewParameter(data)
			if err != nil {
				b.Fatal(err)
			}

			loss, err := nn.Sum(parameter.Value())
			if err != nil {
				b.Fatal(err)
			}

			if err := loss.Backward(); err != nil {
				b.Fatal(err)
			}

			var opt Optimizer
			if name == "SGD" {
				opt, err = NewSGD([]*nn.Parameter{parameter}, 0.001)
			} else {
				opt, err = NewAdam([]*nn.Parameter{parameter}, 0.001, 0.9, 0.999, 1e-8)
			}

			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()

			for b.Loop() {
				if err := opt.Step(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
