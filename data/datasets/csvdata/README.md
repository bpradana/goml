# Numerical CSV datasets

[Data module](../../README.md) · [Dataset guide](../../../docs/datasets.md)

CSV data is parsed once with encoding/csv and retained as numerical host arrays.
No DataFrame, schema inference, categorical encoding, or global registry is used.

```go
dataset, err := csvdata.OpenClassification(ctx, "archive/samples.csv", csvdata.Options{
	Header:   true,
	Features: []string{"age", "income", "score"},
	Targets:  []string{"target"},
})
if err != nil {
	return err
}
```

Select exact header names when Header is true. Without a header, supply
zero-based decimal column strings such as `Features: []string{"0", "2"}` and
`Targets: []string{"3"}`. Both selections are mandatory. Duplicate/missing header
names, duplicate selected columns, feature/target overlap, malformed CSV records,
and inconsistent field counts return errors.

Classification requires one nonnegative integer target column. `1` is accepted;
`1.0`, negative labels, and implicit numeric/categorical conversions are rejected.
Regression accepts one or more finite Float32 targets using `OpenRegression`.
Feature and regression-target values must parse as finite Float32; NaN, infinities,
and overflow fail with row/column context. Selected column order determines vector order.

Samples are independent copies shaped `[features]` and, for regression,
`[targets]`. A single numerical target has shape `[1]`. Record order is stable,
concurrent Get calls are safe, and canceled contexts/invalid indices return errors.
An empty file fails; a valid header-only file creates an empty dataset.

Use `data.CollateClassification` or `data.CollateRegression`. Parsing leaves no
open files, so the dataset needs no Close method. Tests use small local CSV files.

Licensed under [MIT](../../../LICENSE).
