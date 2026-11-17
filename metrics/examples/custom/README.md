# Custom metric and Trainer plugin

[Metrics](../../README.md) · [Mean-bias implementation](meanbias/meanbias.go)

```sh
mise run example:metrics:custom
```

The program implements and uses a custom streaming mean signed error through
`metrics.Metric[float32,float32,meanbias.Result]`. Its report is a concrete struct,
not a dynamic result map. The [meanbias package](meanbias/README.md) depends only
on metrics and the standard library and can be imported by another application.

The [Trainer regression program](../../../trainer/examples/regression/main.go)
uses this exact implementation as a pluggable validation metric:

```go
bias := meanbias.New()
runner, err := trainer.NewRegressor(model, opt, device, bias)
if err != nil {
	return err
}
```

Read bias.Result() in OnEpochEnd or after Evaluate/Fit. Trainer resets the metric
at each evaluation and passes each prediction/target batch once. The application
owns the metric and its typed results. No registration or Trainer dependency is
needed in the custom metric implementation itself.

The Trainer program is a separate application; it is not imported into metrics.
Metrics remains standard-library-only. Negative bias means underprediction and
positive bias overprediction. Empty results return an error.

Licensed under [MIT](../../../LICENSE).
