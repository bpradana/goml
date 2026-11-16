# Metal linear regression

[Examples](../README.md) · [Metal](../../README.md)

A seeded Dense layer learns `y = 2x + 1` over four examples in 100 SGD steps.
Parameters, gradients, and updates remain on Metal. Each step closes a temporary scope.
Only the first and last losses and final parameters are downloaded for reporting.

```sh
mise run example:gpu:training
mise run example:gpu:training -- -checkpoint /tmp/linear.goml
```

The optional checkpoint preserves the existing portable network format.
It can be loaded on CPU without Metal. The measured run ended below `1e-12` MSE.
macOS, CGo, and Metal hardware are required.
