# Performance baseline

[Documentation](README.md) · [Validation](testing.md)

Measured locally on 2026-11-16 using Go 1.27.3, macOS arm64, Apple M1 Pro.
Source commit: `71293b863b5c5427abffcb7911c7741eea70d956`. Recorded UTC timestamp: `2026-11-15T17:23:03Z`.

Each reported value is the median of three samples with a 100ms benchmark window.
Kernels and matrix multiplication are sequential. These measurements describe
one machine and do not establish statistical significance across hardware.

Reproduce the complete run with:

```sh
mise run bench:record
```

The task writes a UTC timestamp, source commit, hardware metadata, scalar-build results, and SIMD-build results
to the ignored `benchmark-results/` directory. GitHub's manual benchmark workflow
uploads those files separately for native amd64 and arm64 runners.
Run longer samples on an otherwise idle machine before performance decisions.

## Add kernels

The following scalar reference and experimental kernel measurements come from
the same SIMD-enabled binary. Speedup is scalar median divided by SIMD median.
All of these kernel cases allocate zero bytes and zero objects.

| Elements | Scalar ns/op | SIMD ns/op | Speedup |
| --- | --- | --- | --- |
| 8 | 5.768 | 8.655 | 0.67× |
| 64 | 40.66 | 37.73 | 1.08× |
| 1,024 | 666.4 | 568.3 | 1.17× |
| 1,027 | 651.5 | 579.8 | 1.12× |
| 65,536 | 41,088 | 36,079 | 1.14× |
| 1,048,576 | 683,361 | 589,599 | 1.16× |

The eight-element case is slower with SIMD. Larger cases show modest benefits
on this CPU. The experiment remains opt-in. These results do not justify a
universal speedup claim or an architecture-independent dispatch threshold.

## Other arithmetic kernels

The same benchmark suite covers sizes 8, 64, 1,024, 1,027, 65,536, and 1,048,576.
Representative 1,024-element results follow. All have zero allocations.

| Operation | Scalar ns/op | SIMD ns/op | Speedup |
| --- | --- | --- | --- |
| Sub | 645.1 | 568.1 | 1.14× |
| Mul | 641.9 | 561.8 | 1.14× |
| Div | 641.7 | 562.1 | 1.14× |
| Scale | 496.9 | 427.2 | 1.16× |
| Axpy | 647.3 | 589.7 | 1.10× |

Dot and Sum retain reference accumulation order. Max remains a scalar kernel.
No reduction SIMD optimization is claimed.

## Tensor and network baselines

These results use the default scalar build. Times include the allocations that
the public immutable APIs require. The training case uses a 16 → 8 Dense/Tanh
model, batch size 8, a mean loss, backward, SGD step, and gradient clearing.
Optimizer-only cases contain one vector parameter with 128 elements.

| Workload | ns/op | B/op | Allocs/op |
| --- | --- | --- | --- |
| MatMul [32,32] × [32,32] | 46,791 | 4,242 | 5 |
| MatMul [128,16] × [16,32] | 93,919 | 16,528 | 5 |
| MatMul [16,128] × [128,32] | 90,681 | 2,192 | 5 |
| MatMul [31,29] × [29,37] | 46,416 | 5,008 | 5 |
| Tensor Add, contiguous [64,64] | 4,075 | 16,528 | 5 |
| Tensor Add, transposed [64,64] | 32,266 | 16,528 | 5 |
| Tensor Add, broadcast [64,64] + [64] | 30,280 | 16,528 | 5 |
| Sequential forward [8,16] → [8,8] | 3,617 | 1,888 | 27 |
| Sequential inference [8,16] → [8,8] | 3,366 | 1,648 | 19 |
| Full SGD training iteration | 13,566 | 8,744 | 133 |
| SGD step, 128 parameters | 911 | 1,272 | 11 |
| Adam step, 128 parameters | 1,687 | 1,672 | 9 |

The 1,024-element sigmoid inference benchmark uses 7 allocations.
Parent-list allocation occurs only for tracked operations.
Inference retains no parent lists or backward closures.

Strided and broadcast arithmetic use general logical indexing and are slower
than contiguous kernels. Matrix multiplication is a numerical baseline with
improved traversal, without blocking, packing, or parallel workers.
Allocation reuse, specialized strided loops, and blocked multiplication remain
measurable future opportunities. There are no global pools or worker engines.

The local results do not measure native amd64 performance or actual MNIST
dataset accuracy. Native CI benchmark runs require pushing the repository first.
