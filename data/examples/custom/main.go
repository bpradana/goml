package main

import (
	"context"
	"fmt"
	"os"

	"github.com/bpradana/goml/data"
)

type sentence struct {
	text   string
	tokens []int
}

type corpus []string

func (c corpus) Len() int {
	return len(c)
}

func (c corpus) Get(ctx context.Context, index int) (sentence, error) {
	if err := ctx.Err(); err != nil {
		return sentence{}, err
	}

	if index < 0 || index >= len(c) {
		return sentence{}, fmt.Errorf("corpus: index %d out of range", index)
	}

	tokens := make([]int, 0, len(c[index]))
	for _, character := range c[index] {
		tokens = append(tokens, int(character))
	}

	return sentence{text: c[index], tokens: tokens}, nil
}

type batch struct {
	texts  []string
	tokens [][]int
}

func collate(samples []sentence) (batch, error) {
	result := batch{texts: make([]string, len(samples)), tokens: make([][]int, len(samples))}
	for index, sample := range samples {
		result.texts[index] = sample.text
		result.tokens[index] = append([]int(nil), sample.tokens...)
	}

	return result, nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	loader, err := data.NewLoader(corpus{"Go", "numerical computing", "typed samples"}, collate, data.LoaderOptions{BatchSize: 2})
	if err != nil {
		return err
	}

	for batch, err := range loader.Batches(context.Background(), 0) {
		if err != nil {
			return err
		}

		fmt.Printf("texts=%v variable_length_tokens=%v\n", batch.texts, batch.tokens)
	}

	return nil
}
