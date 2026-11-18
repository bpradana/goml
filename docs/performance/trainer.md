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

Full real-MNIST epoch time, held-out accuracy, and maximum Go heap/RSS were not
measured. The initial native checks above validate small actual hardware workloads;
the subsequent readback investigation measures throughput for one synthetic workload. There is no new
pool, unsafe ownership, worker goroutine, or hidden fallback to improve numbers.
The separate MNIST application uses workspace Trainer for development APIs;
Trainer's library module itself passes independent dependency checks.

## Loss readback investigation

Measured on 2026-11-18, Apple M1 Pro, Go 1.27.3, eight Go processors. The baseline
for source changes is 71dfb08. No native numerical kernels or optimizer algorithms
were changed. The benchmark builds an identical seeded Dense 64→10 classifier
for each timed epoch, SGD 0.01, 256 deterministic samples, and eight CPU-collated
batches of 32. Initialization/device opening, shader startup, model upload, weight
comparison, and final resource destruction are outside timing. Input transfers,
forward/backward, updates, scalar reporting, and a final queue barrier are inside.
No validation or shuffling is enabled in any variant.

Each epoch checks mean loss and final weights against an independently run
Trainer control to absolute tolerance 1e-6. That control does not depend on which
benchmark subcase is selected. Native live-buffer counters return to zero after
each epoch's model cleanup. Medians of three samples, five timed epochs each:

| Device | Variant | Epoch ms | Samples/s | Go bytes/epoch | Allocations/epoch |
| --- | --- | ---: | ---: | ---: | ---: |
| CPU | Trainer | 0.380 | 673,389 | 365,137 | 1,643 |
| CPU | Manual, read each batch | 0.440 | 581,444 | 364,547 | 1,616 |
| CPU | Manual, read at epoch end | 0.377 | 679,406 | 365,528 | 1,628 |
| Metal | Trainer | 17.353 | 14,753 | 417,840 | 3,525 |
| Metal | Manual, read each batch | 21.081 | 12,144 | 417,361 | 3,505 |
| Metal | Manual, read at epoch end | 14.330 | 17,864 | 418,336 | 3,517 |
| WebGPU | Trainer | 64.582 | 3,964 | 472,651 | 4,324 |
| WebGPU | Manual, read each batch | 63.400 | 4,038 | 472,060 | 4,297 |
| WebGPU | Manual, read at epoch end | 58.364 | 4,386 | 473,809 | 4,269 |

CPU timings are noisy (the per-batch manual samples span 0.348–0.615 ms).
Metal Trainer samples span 15.544–20.391 ms. These short sequential trials show a
candidate synchronization cost, not a general device speedup or a statistically
established orchestration regression. This small workload is much faster on CPU.

The manual controls time HostData calls directly. Summed readback medians per
epoch are CPU 0.001717/0.001099 ms, Metal 3.775700/0.006267 ms, and WebGPU
15.575109/11.638166 ms for per-batch/end-of-epoch reads respectively. Trainer's
individual reads are not instrumented. The final explicit barrier costs less than
0.002 ms in all samples; it excludes earlier synchronization. In particular,
existing deviceSGD synchronizes every Step before committing parameter updates.
Deferred scalar reads therefore do not make the whole epoch asynchronous.

Metal/WebGPU sampled live Float32 payload is 2,600 bytes after ordinary batch
cleanup and 2,632 with eight retained scalar losses. These samples exclude
in-operation temporary peaks, driver/library memory, and pending native retention.
CPU's zero native-byte counter is not a Go heap measurement. There are no raw
logs, profiler files, trained weights, or benchmark binaries in the repository.

The deferred control retains only eight independent scalar leases, closes them
after reading, and checks every loss for finiteness. However, those checks happen
**after parameter updates**. A nonfinite objective would violate Trainer's current
fail-before-Step contract. The experiment remains test-only: no reporting-frequency
option, delayed validation, training-logit download, worker, or buffering system
was added to production. EpochStats.MeanLoss and error timing remain unchanged.
Further on-device loss accumulation must preserve both safeguards before adoption.

```sh
mise run bench:trainer:readback:cpu
mise run bench:trainer:readback:metal
mise run bench:trainer:readback:webgpu
```

## Extensibility verification

The new configured-loss tests exercise typed targets, training/validation reuse,
scalar/device checks, nil/nonfinite outputs, backward dependencies, cleanup, and
no optimizer update after a loss failure. Custom checkpoint tests cover strict
selection, override of unsupported Sequential codecs, generic Fit, callback
ordering/errors, fresh-run retry, weights-only fallback, and invalid configuration.
Existing checkpoint and early-stopping tests remain in the suite.

The denoising example uses a configured two-term objective without a new lifecycle
helper. The generic reconstruction example still demonstrates arbitrary batches.
Custom mean-bias histories remain concrete application-owned snapshots. Metrics
checks include custom prediction/target/report types and JSON null for undefined
R² without changing numerical values or streaming interfaces.

The existing 200 ms toy benchmarks were repeated before/after this refactor
(three samples each). Median classification Fit changed from 15.303 to 15.461 µs,
and Evaluate from 5.738 to 5.757 µs. Fit still allocates 11,624 bytes/301 objects;
Evaluate still allocates 4,752 bytes/121 objects. The after-change Fit samples
span 15.125–21.281 µs, so these observations do not establish a timing regression.
No per-element dispatch, extra prediction readback, or runtime worker was added.
