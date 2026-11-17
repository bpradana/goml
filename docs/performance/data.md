# Data loading validation and performance

[Data module](../../data/README.md) · [Dataset contracts](../datasets.md) · [MNIST application](../../nn/examples/mnist/README.md)

Measured locally on 2026-11-17 using Go 1.27.3, Apple M1 Pro, and eight Go
processors. Tests use generated local files. No fixtures, downloaded datasets,
model weights, raw benchmark logs, or profiler output are committed.

## Implementation and verification

| Phase | Source | Verified behavior |
| --- | --- | --- |
| Core contracts | data/dataset.go, loader.go | External typed samples/collators, ordering, epoch determinism, DropLast, errors, cancellation, and early break. |
| Tensor batches | data/array.go, batch.go, collate.go | Copied snapshots, independent sample/batch ownership, shape checks, scalar targets, source closure, and concurrent Get. |
| MNIST | data/datasets/mnist | Plain/gzip files, train/test splits, bounds, malformed/truncated payloads, checksums, normalization, and local batch construction. |
| Application | nn/examples/mnist | Dense/CNN, seeded loading, held-out selection, Adam, checkpoint restoration, prediction, and batch cleanup. |
| File datasets/transforms | data/datasets/csvdata, imagefolder, transform | Explicit numerical columns, diagnostics, sorted classes/files, bounded lazy decode, symlink rejection, typed copying transforms, and 16-bit color precision. |
| Hardware integration | MNIST metal-tag tests | Explicit complete-batch transfers, CPU cleanup after upload, no implicit accelerator dataset reads, Dense/CNN training, and race checks on real Metal. |
| Contributor experience | READMEs, docs/datasets.md, examples | Public-only custom dataset/collator, runnable classification program, standalone module validation, and pure-Go cross-builds. |

The full CPU CI gate passed build, formatting, vet, lint, ordinary tests, races,
and govulncheck. No reachable vulnerabilities were found; the workspace scan
reported one uncalled module advisory. Linux amd64/arm64 and Windows amd64
cross-builds passed with CGo disabled. Shared data tests also pass independently
with GOWORK off against published Tensor dependencies.

## Synchronous preparation

Numbers below are medians of three 200 ms Go benchmark samples. File fixtures
are generated before timing. Image reads use a 64×64 PNG and a warm local
filesystem; these measurements do not represent remote storage or large JPEGs.

| Operation | Time | Go bytes/op | Go allocations/op |
| --- | ---: | ---: | ---: |
| Collate 64 existing `[1,28,28]` samples | 43.96 µs | 410,474 | 11 |
| MNIST Get, one image | 0.788 µs | 3,224 | 2 |
| MNIST loader batch, 64 images | 91.33 µs | 620,910 | 140 |
| ImageFolder Get, 64×64 PNG → RGB CHW | 244.96 µs | 122,088 | 38 |
| Collate 64 samples + explicit Metal upload/synchronization | 107.88 µs | 615,605 | 22 |

The sampled MNIST loader delivers about 10,949 batches/s, or 700,000 samples/s
for this preparation-only workload. The PNG case decodes about 4,082 images/s.
These are operation-throughput calculations, not model training throughput.

Before the image conversion change, the same PNG benchmark used 4,134 Go
allocations and a median 271.31 µs. Selecting the standard RGBA64At path avoids
boxing a color interface per pixel, reducing the measured count to 38. Tests
check both the standard-image path and custom image.Image fallback against
16-bit component values. No precision-reducing raster conversion was introduced.

Collation uses Tensor's copying public constructor. It needs a temporary packed
Go slice and final immutable Tensor storage. The MNIST batch additionally owns
64 decoded sample slices. There is no individual Tensor or GPU allocation per
sample. These bounded copies make ownership independent; they are included in
the measurements rather than hidden behind a global pool.

## End-to-end MNIST fixture comparison

The previous manual example was taken from commit 9e66110 and built in a temporary
module against the same local nn/Tensor/CPU kernels. Both paths use generated
128-image/label gzip files, the original seeded Dense network, one epoch,
batch size 32, no shuffle, and no checkpoint writes. File decompression, model
construction, four optimizer updates, and cleanup are inside timing.

The table uses medians of three benchmark samples, three epochs per sample:

| Path | Epoch time | Go bytes/epoch | Go allocations/epoch |
| --- | ---: | ---: | ---: |
| Previous manual loading/slicing | 3.382 ms | 2,601,930 | 1,137 |
| Dataset/DataLoader and explicit scopes | 3.510 ms | 3,186,640 | 1,500 |

Both report the same fixture cross-entropy, 2.2999. The approximately 3.8% timing
difference is a short fixture observation, not a statistically established
regression. Extra allocations come from independently owned per-sample values,
packed batches, loader metadata, and explicit scopes. The new implementation
also closes per-batch temporaries that the older application left to cleanup.
This comparison includes those lifecycle changes; it does not isolate dispatch
cost or establish real-MNIST accuracy/convergence.

A separate resource-counter observation ran each precompiled test binary for ten
timed epochs plus benchmark warm-up, excluding compilation. macOS time -l
reported the same maximum resident
set size, 13,172,736 bytes (12.56 MiB), in both processes. This is one small-fixture
process observation, not a peak-memory bound for full datasets or accelerator drivers.

MNIST retains 47,040,000 image bytes for all 60,000 training images. A fully
expanded Float32 representation would require 188,160,000 bytes before Tensor
copies or model memory. These are payload sizes, not measured resident memory.
Only requested samples are expanded and batches released. ImageFolder retains
paths and decodes on demand; it does not keep a full decoded-image cache.

## Limits and reproduction

Loading remains synchronous. No Workers/Prefetch fields, worker pools, channels,
or speculative caches were added. The measured small-image/preparation workloads
do not establish a benefit for a particular parallel scheduling design. Future
parallel loading should compare representative real files, preserve ordering,
and demonstrate cancellation/early-break cleanup before adding configuration.

```sh
mise run test:data:race
mise run test:standalone:data
mise run build:data:portable
mise run bench:data
mise run bench:mnist
mise run bench:data:metal
```

Metal measurements require supported Apple hardware and CGo. Transfer-inclusive
numbers exclude device startup, image decoding, and model execution; their input
sample slices already exist. Go allocation counts exclude Metal's native buffer.
No full real-dataset training study, held-out accuracy measurement, browser run,
or Windows/Linux native execution was performed here. Cross-builds establish
compilation, not execution on those platforms.
