# MNIST

[Data module](../../README.md) · [Training example](../../../nn/examples/mnist/README.md)

Open standard local MNIST train or held-out test files. No dataset download occurs.

```go
train, err := mnist.Open("archive", mnist.Train)
if err != nil {
	return err
}

loader, err := data.NewLoader(train, data.CollateClassification, data.LoaderOptions{
	BatchSize: 64,
	Shuffle:   true,
	Seed:      42,
})
if err != nil {
	return err
}
```

Train uses `train-images-idx3-ubyte` and `train-labels-idx1-ubyte`; Test uses
`t10k-images-idx3-ubyte` and `t10k-labels-idx1-ubyte`. Plain filenames take precedence
when both plain and `.gz` versions exist. `OpenContext` supports cancellation
while reading/decompressing. `OpenFiles` accepts explicit plain/gzip file paths.

Construction validates magic numbers, nonzero bounded counts, 28×28 dimensions,
matching label count, complete payloads, trailing data, gzip checksums, and all
labels in `[0,10)`. Train/explicit pairs permit at most 60,000 samples; Test at
most 10,000. Compressed data is bounded by the validated payload size.

The dataset owns compact image bytes and labels and leaves no files open.
Concurrent Get calls allocate independent Float32 samples in `[0,1]`, shaped
`[1,28,28]`. Labels remain integers. A classification batch has shape
`[batch,1,28,28]`. Get checks cancellation and index bounds.

Use `data.Subset` to select a limit without expanding the complete dataset into
Float32. Files are validated fully even when training uses a smaller subset.
Tests generate IDX/gzip fixtures locally; no real MNIST files are required.

Licensed under [MIT](../../../LICENSE).
