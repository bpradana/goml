# Trainer validation and performance

[Trainer](../../trainer/README.md) · [Training contracts](../training.md) · [Metrics measurements](metrics.md)

Validated locally on 2026-11-17 with Go 1.27.3, Apple M1 Pro, and eight Go
processors. CPU tests use generated data and local IDX fixtures. Raw logs,
benchmark binaries, profiler output, datasets, and checkpoints are not committed.

## Correctness and ownership

The complete CPU CI task passed formatting, build, vet, lint, ordinary tests,
race detection, and vulnerability analysis. Govulncheck found no reachable
vulnerabilities; it reported one advisory in an uncalled required module.
Metrics and Trainer also passed GOWORK=off tests against published dependencies.
Linux amd64/arm64 and Windows amd64 cross-builds passed with CGo disabled; Linux
scalar/SIMD cross-builds passed through the repository portable task.

| Area | Exercised behavior |
| --- | --- |
| Classification | Real seeded Dense models converge at batch sizes one, three, and four; partial batches and optimizer step counts agree. |
| Regression | Adam learns a generated linear relation, with exact prediction/target shapes and independent numerical validation. |
| Evaluation | Predict path, no optimizer steps or parameter changes, sample-weighted loss, explicit empty errors, and streaming metrics. |
| Custom metrics | Built-in logits adapter and external mean-bias implementation; per-evaluation reset, typed results, and metric-error cleanup. |
| History | One-based completed reports, partial history on failure, and isolated callback snapshots. |
| Model selection | Strict MinDelta/patience behavior, lowest-loss checkpoint selection, and preservation of an existing file after codec errors. |
| Failures | Dataset, collation, forward, loss, backward, optimizer, callback, and checkpoint errors; owned temporary closure and gradient clearing. |
| Custom tasks | Typed regularized reconstruction through generic Fit with application-owned steps. |
| MNIST application | Generated local gzip fixtures, dense/CNN training, checkpoint save/load, prediction, held-out split selection, and conflicting-output validation. |

These tests establish API behavior on their workloads. They do not establish
real-MNIST generalization, optimizer-state resume, or support for arbitrary
application code that does not follow Tensor scope/ownership contracts.

## Native device integration

Both optional tasks executed with the race detector on actual M1 Pro hardware:

```sh
mise run test:trainer:metal -- -run TestTrainer_ -v
mise run test:trainer:webgpu -- -run TestTrainer_ -v
```

Each backend runs the same initialized models, loaders, optimizer settings, and
batch order as CPU. Classification uses SGD for twenty epochs; regression uses
Adam for one hundred. Each epoch validates all four generated samples with a
partial final batch. Validation loss must agree with CPU within an absolute
0.001. Classification reaches accuracy one and regression MSE is below 0.001.

| Backend | Classification final loss | Regression final MSE | Stable Float32 buffer bytes, classifier / regressor |
| --- | ---: | ---: | ---: |
| Metal | 0.0848295 | 4.12719 × 10⁻¹¹ | 16 / 24 |
| WebGPU | 0.0848295 | 4.12719 × 10⁻¹¹ | 16 / 24 |

Tracked live bytes remain constant from the first completed epoch. After model
and optimizer cleanup, they return to zero. The peak tracked buffer payload is
396 bytes in both tiny test sequences. These counters measure backend-owned
Float32 buffers, not shader libraries, driver allocations, command metadata, or
physical GPU resident memory. Test callbacks explicitly synchronize for accounting;
normal Trainer execution does not synchronize after every operation.

The Metal-tag suite additionally executes Dense and CNN training on MNIST-shaped
fixtures and checks that explicit GPU uploads survive CPU batch closure. These
are correctness/memory tests, not GPU throughput or speedup measurements.

## Toy orchestration workload

Medians of three 200 ms benchmark samples, four generated samples and batch size
three, with model/loader construction outside timing:

| Workload | Time per epoch | Go bytes/epoch | Go allocations/epoch |
| --- | ---: | ---: | ---: |
| Classification Fit, one epoch without validation | 12.981 µs | 11,624 | 301 |
| Classification Evaluate, two batches | 4.702 µs | 4,752 | 121 |

These include Tensor operations and ownership, not isolated dispatch overhead.
The training samples varied from 12.016 to 37.191 µs; that spread precludes a
precise latency guarantee. Run longer repeated samples on an otherwise idle host
for regression decisions. Metrics reports deliberately allocate independent
snapshots; training does not retain per-sample predictions.

## MNIST fixture comparison

The manual baseline is the application at commit cdf2063, built in a temporary
module against the same local nn/Tensor/CPU kernels. Both paths read generated
128-image/label gzip files, initialize the same Dense network, and run one epoch
of Adam at batch size 32. Shuffling, validation, and checkpoint writes are disabled
in both. Parsing, model construction, loss reporting, updates, and cleanup are
timed. Both print the same fixture cross-entropy, 2.2999.

Medians of five samples, thirty timed epochs per sample:

| Path | Epoch time | Go bytes/epoch | Go allocations/epoch |
| --- | ---: | ---: | ---: |
| Handwritten DataLoader training loop | 3.018 ms | 3,186,555 | 1,499 |
| Trainer Fit | 2.868 ms | 3,187,251 | 1,526 |

The additional sampled allocation cost is 696 bytes and 27 allocations per
epoch. Timing is noisy: baseline samples range from 2.945 to 3.897 ms, versus
2.797 to 2.939 ms for Trainer. The runs are sequential, not randomized paired
trials, so this does not establish a speedup or a statistically significant
regression. The comparison includes lifecycle/orchestration changes rather than
isolating interface dispatch. It is not a real-dataset convergence measurement.
Reproduce the Trainer side with:

```sh
mise run bench:mnist -- -benchtime=30x -count=5
```

## Limits and reproduction

```sh
mise run ci
mise run test:standalone:metrics
mise run test:standalone:trainer
mise run bench:metrics
mise run bench:trainer
mise run bench:mnist
mise run build:trainer:portable
```

Full real-MNIST epoch time, held-out accuracy, maximum Go heap/RSS, and accelerator
training throughput were not measured for this change. Native checks validate
small actual hardware workloads without claiming a GPU speedup. There is no new
pool, unsafe ownership, worker goroutine, or hidden fallback to improve numbers.
The separate MNIST application uses workspace Trainer until its first release;
Trainer's library module itself passes independent dependency checks.
