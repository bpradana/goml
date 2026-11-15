package tensor

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

func BenchmarkMatMul(b *testing.B) {
	for _, dimensions := range [][3]int{{32, 32, 32}, {128, 16, 32}, {16, 128, 32}, {31, 29, 37}} {
		b.Run(fmt.Sprint(dimensions), func(b *testing.B) {
			rng := rand.New(rand.NewPCG(1, 2))

			a, err := Rand(rng, dimensions[0], dimensions[1])
			if err != nil {
				b.Fatal(err)
			}

			c, err := Rand(rng, dimensions[1], dimensions[2])
			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()

			for b.Loop() {
				if _, err := MatMul(a, c); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkAdd(b *testing.B) {
	for _, layout := range []string{"Contiguous", "Transposed", "Broadcast"} {
		b.Run(layout, func(b *testing.B) {
			a, err := Ones(64, 64)
			if err != nil {
				b.Fatal(err)
			}

			c := a

			switch layout {
			case "Transposed":
				a, err = a.Transpose()
			case "Broadcast":
				c, err = Ones(64)
			}

			if err != nil {
				b.Fatal(err)
			}

			b.ReportAllocs()

			for b.Loop() {
				if _, err := Add(a, c); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkSum(b *testing.B) {
	a, err := Ones(64, 64)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()

	for b.Loop() {
		if _, err := Sum(a, 0); err != nil {
			b.Fatal(err)
		}
	}
}
