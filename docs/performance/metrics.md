# Metrics validation and measurements

[Metrics](../../metrics/README.md) · [Contracts](../metrics.md)

Measured on 2026-11-17 with Go 1.27.3, Apple M1 Pro, and eight Go processors.
Benchmarks are single-threaded accumulation loops; these are not GPU or model
training throughput measurements. Raw logs are kept outside the repository.

## Numerical validation

Standalone tests passed without dependencies outside the standard library.
Independent multiclass references cover confusion orientation, all averaging
conventions, unsupported classes, imbalanced support, binary perfect/incorrect
predictions, partition invariance, snapshot ownership, Reset, ties, and atomic
invalid updates. Regression tests cover independently calculated MAE/MSE/RMSE/R²,
streaming partitions, constant targets, insufficient samples, finite validation,
large-offset target variance, extreme Float32 magnitudes, and tiny errors.

The generic Metric/Accumulator contracts are exercised by an external-package
custom implementation and compile-time built-in assertions. Classification's
raw-logit adapter shares typed state with its original accumulator.

## Update throughput

Medians of three 200 ms Go benchmark samples:

| Operation | Batch | Time | Go allocations/op |
| --- | --- | ---: | ---: |
| Classification UpdateLogits | 256 rows × 10 classes | 5.469 µs | 0 |
| Regression Update | 1024 scalar pairs | 10.980 µs | 0 |

This is about 46.8 million classification rows/s and 93.3 million scalar
regression observations/s for these inputs. Report construction is excluded;
classification reports deliberately allocate independently owned O(classes²)
confusion snapshots. Accumulation retains only counts/running statistics.

```sh
mise run test:standalone:metrics
mise run lint:metrics
mise run bench:metrics
```

Large class matrices, native amd64 execution, and statistically rigorous
cross-machine comparisons were not measured. Invalid-input checks and atomic
validation are included in update timing. No numerical shortcut, global pool,
unsafe ownership, or internal parallelism was added to improve these numbers.

## Extensibility and JSON follow-up

On 2026-11-18 the 71dfb08 baseline and updated source were measured with the same
three 200 ms sample tasks on M1 Pro/Go 1.27.3. Median UpdateLogits was 5.487 µs
before and 5.501 µs after; Regression.Update was 11.007 µs before and 10.989 µs
after. Both still allocate zero bytes and zero objects per update. MarshalJSON
runs only when serializing a report and does not change accumulator state or
its numerical update path. The small timing changes are within sampled noise.
The new serialization tests cover constant targets, a single observation, valid
negative R², unchanged fields, and rejection of unrelated nonfinite fields.
