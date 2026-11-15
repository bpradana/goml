<div align="center">

<img src="../docs/assets/simdops.svg" width="88" height="88" alt="simdops logo">

# simdops

**Allocation-free float32 slice operations, with scalar references and optional portable SIMD.**

[API](#api) · [Memory contract](#memory-contract) · [SIMD](#simd) · [Verification](#verification) · [Project](../README.md)

</div>

---

simdops operates on ordinary `[]float32` values.
It has no dependency on tensor or nn.
Length validation occurs before the destination buffer is written.

The module path is `github.com/bpradana/goml/simdops`.
During bootstrap, use it from the local workspace.
The v0.1.0 tag is not published yet.

## First program

```go
package main

import (
	"fmt"
	"log"

	"github.com/bpradana/goml/simdops"
)

func main() {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	dst := make([]float32, len(a))

	if err := simdops.Add(dst, a, b); err != nil {
		log.Fatal(err)
	}

	dot, err := simdops.Dot(a, b)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(dst)
	fmt.Println(dot)
}
```

```text
[5 7 9]
32
```

## API

| Function | Calculation | Constraint |
| --- | --- | --- |
| `Add(dst, a, b)` | `dst[i] = a[i] + b[i]` | All lengths equal. |
| `Sub(dst, a, b)` | `dst[i] = a[i] - b[i]` | All lengths equal. |
| `Mul(dst, a, b)` | `dst[i] = a[i] * b[i]` | All lengths equal. |
| `Div(dst, a, b)` | `dst[i] = a[i] / b[i]` | All lengths equal. |
| `Scale(dst, a, factor)` | `dst[i] = a[i] * factor` | Destination and input lengths equal. |
| `Axpy(dst, x, y, alpha)` | `dst[i] = alpha*x[i] + y[i]` | All lengths equal. |
| `Dot(a, b)` | Sum of pairwise products. | Input lengths equal. |
| `Sum(a)` | Sum of all elements. | Empty input returns zero. |
| `Max(a)` | Maximum element. | Empty input returns an error. |

Destination-writing functions return `error`.
Dot and Max return `(float32, error)`.
Sum returns `float32`.

Read the function comments in [arithmetic.go](arithmetic.go) and
[reduction.go](reduction.go) for the public contract.
Use `mise run docs:api` to read the API locally.

## Memory contract

The caller allocates destination buffers.
Successful operations allocate no memory.

Exact aliasing is supported. For example, `Add(a, a, b)` replaces every element
of `a` with its sum with `b`.
Partial overlap is unsupported and is not detected.
Do not pass shifted slices of the same backing array as overlapping inputs and output.

Invalid lengths return an error before any destination element changes.
The entire destination length must match the operation's required length.
A destination with excess capacity is valid only when its length is correct.

## Empty inputs and special values

| Case | Result |
| --- | --- |
| Empty Add, Sub, Mul, Div, Scale, Axpy | Success with an empty destination. |
| Dot of two empty slices | Zero. |
| Sum of an empty slice | Zero. |
| Max of an empty slice | Error. |
| Division by zero | IEEE 754 infinity or NaN. |
| NaN in a reduction | NaN propagates. |
| Max over positive and negative zero | Positive zero wins. |

Arithmetic uses float32 values.
Reduction order can affect floating-point results.
The current Dot and Sum kernels retain scalar accumulation order.
Do not assume future optimized reductions will be bit-identical.

## SIMD

The default build uses scalar kernels.
The experimental build selects portable SIMD arithmetic:

```sh
mise run test:simd
GOEXPERIMENT=simd mise run bench:simdops
```

The kernel files use complementary `goexperiment.simd` build constraints.
Vector operations process complete groups, then scalar kernels process the tail.
Public signatures expose no experimental types.

| Source | Responsibility |
| --- | --- |
| [internal/kernels/scalar.go](internal/kernels/scalar.go) | Scalar references and maximum reduction. |
| [internal/kernels/vector_fallback.go](internal/kernels/vector_fallback.go) | Default scalar dispatch. |
| [internal/kernels/vector_simd.go](internal/kernels/vector_simd.go) | Experimental arithmetic and scalar tails. |

Only arithmetic is vectorized.
Dot, Sum, and Max remain scalar.

## Performance

The [project performance report](../docs/performance.md) compares kernels in the
same SIMD-enabled binary on an Apple M1 Pro with Go 1.27.3.

At 1,024 elements, measured Add medians were 666.4 ns/op for scalar and
568.3 ns/op for SIMD. Both allocated zero bytes.
At eight elements, SIMD was slower.
Use measurements from your workload before choosing the experiment.

The suite includes regular and irregular lengths from 8 to 1,048,576 elements:

```sh
mise run bench:simdops
mise run bench:record
```

## Verification

Run these tasks from the repository root:

| Purpose | Command |
| --- | --- |
| Default correctness tests | `mise run test:simdops` |
| Module lint | `mise run lint:simdops` |
| Experimental parity across all modules | `mise run test:simd` |
| Race detection across all modules | `mise run test:race` |

Tests cover known scalar results, destination validation, exact aliasing,
irregular tails, NaN, infinity, signed zero, and allocation-free Add.
Implementation and test files are paired.
See the [validation guide](../docs/testing.md) for methods and remaining evidence.

## Limits

simdops supports float32 slices.
It provides no matrix API, tensor storage, concurrency engine, GPU runtime, or
manual assembly backend. Higher-level arrays belong to [tensor](../tensor/README.md).

Licensed under [MIT](../LICENSE).
