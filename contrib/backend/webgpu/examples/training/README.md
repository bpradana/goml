# WebGPU training and checkpoints

[Examples](../README.md) · [Backend](../../README.md)

The program learns `y=2x+1` with a seeded Dense layer and 100 SGD updates. Inputs,
targets, parameters, and gradients stay on the selected WebGPU context. Each
iteration uses a temporary scope through Backward, Step, and ZeroGrad.

```sh
mise run example:gpu:webgpu:training
mise run example:gpu:webgpu:training -- -checkpoint /tmp/webgpu-linear.goml
```

Only the first and last loss are explicitly downloaded for reporting. The
convergence check requires final MSE below `0.0001`. Optional checkpoint saving
downloads portable parameter values, preserving the existing nn/checkpoint format;
the saved network loads on CPU without WebGPU.

The matching tests also validate CPU/WebGPU model gradients and SGD/Adam updates.
Adam moments remain GPU-resident, and repeated steps must return to a stable
tracked allocation baseline. Run them with:

```sh
mise run test:gpu:webgpu:parity
```

Licensed under [MIT](../../../../../LICENSE).
