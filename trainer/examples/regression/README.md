# Regression and a custom metric

[Trainer](../../README.md) · [Source](main.go) · [Custom metric](../../../metrics/examples/custom/README.md)

```sh
mise run example:trainer:regression
```

A seeded Dense model learns y = 2x + 1 with Adam, MSE, and data.Loader. The
example passes the independently defined meanbias.Metric directly to NewRegressor.
Trainer resets it at each validation and updates it with every prediction batch.
OnEpochEnd reads each concrete Result and appends an application-owned typed\nhistory. The application also reads Result after Fit and prints MeanBias alongside
built-in MSE and R². It needs no registry or adapter in the metric implementation.

The four generated samples serve both training and validation to demonstrate the
API and deterministic convergence. This is not a held-out accuracy measurement.

Licensed under [MIT](../../../LICENSE).
