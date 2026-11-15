# tensor

**Immutable N-dimensional float32 arrays with broadcasting and strided views.**

[First program](#first-program) · [API](#api) · [Views](#views-and-ownership) · [Mathematics](#mathematics) · [Project](../README.md)

tensor provides numerical arrays independently of nn.
Contiguous arithmetic uses [simdops](../simdops/README.md).
Shape validation, indexing, and memory ownership remain part of tensor's contract.

The module path is `github.com/bpradana/goml/tensor`.
Use the local workspace while upstream module tags remain unpublished.

## First program

```go
package main

import (
	"fmt"
	"log"

	"github.com/bpradana/goml/tensor"
)

func main() {
	a, err := tensor.From([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	if err != nil {
		log.Fatal(err)
	}

	at, err := a.Transpose()
	if err != nil {
		log.Fatal(err)
	}

	product, err := tensor.MatMul(a, at)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(at)
	fmt.Println(product)
}
```

```text
tensor(shape=[3 2], data=[1 4 2 5 3 6])
tensor(shape=[2 2], data=[14 32 32 77])
```

The transposed array shares the original storage.
The matrix product owns a new contiguous buffer.

Run the repository example with `mise run example:tensor`.

## API

| Area | Operations |
| --- | --- |
| Construction | `From`, `Scalar`, `Zeros`, `Ones`, `Rand`, `Randn` |
| Metadata | `Shape`, `Strides`, `Rank`, `Numel`, `IsContiguous` |
| Reading | `At`, `Data`, bounded `String` |
| Views | `Reshape`, `Transpose`, `Slice` |
| Materialization | `Contiguous` |
| Arithmetic | `Add`, `Sub`, `Mul`, `Div`, `Scale`, `Neg` |
| Mathematics | `Exp`, `Log`, `Sqrt` |
| Reductions | `Sum`, `Mean`, `Max` |
| Linear algebra | `MatMul` |

Fallible constructors and operations return errors.
Use constructors rather than the zero Tensor value.
Random constructors require a caller-owned `*math/rand/v2.Rand`.

Function comments are the API reference.
Run `mise run docs:api` or read [tensor.go](tensor.go),
[view.go](view.go), [arithmetic.go](arithmetic.go), and [reduction.go](reduction.go).

## Shapes and indexing

Contiguous arrays use row-major order.
A scalar has rank zero, shape `[]`, and one element.
Dimensions can be zero but cannot be negative.
Element counts and row-major strides are checked for integer overflow.

At takes one index per dimension.
For a scalar, At takes no indices.
Negative indices and indices outside the shape return errors.

Shape and Strides return copies.
Data returns a copy in logical row-major order, including for transposed and sliced arrays.

## Views and ownership

| Operation | Allocates element storage | Shares original storage |
| --- | --- | --- |
| From | Yes, copies the supplied values. | No. |
| Transpose | No. | Yes. |
| Slice | No. | Yes. |
| Reshape of a contiguous array | No. | Yes. |
| Reshape of an unsupported noncontiguous array | Returns an error. | No result. |
| Contiguous | Yes, including for a contiguous input. | No. |
| Arithmetic and reductions | Yes. | No. |

Views allocate metadata while sharing element storage.
There is no public writable view or unrestricted mutation API.

Transpose with no axes reverses the dimension order.
Explicit axes must form a complete permutation.
Slice selects a half-open `[start,end)` interval with unit step.

Use Contiguous before a reshape that needs row-major materialization.
Reshape does not infer a `-1` dimension or copy implicitly.

## Mathematics

Binary arithmetic aligns trailing dimensions:

| Input shapes | Output shape |
| --- | --- |
| `[32,128]` and `[128]` | `[32,128]` |
| `[10,1,64]` and `[1,64]` | `[10,1,64]` |
| `[4,8]` and scalar `[]` | `[4,8]` |
| `[0,3]` and `[1,3]` | `[0,3]` |

Two aligned dimensions must be equal, or one must be 1.
Zero can broadcast with 1, but cannot broadcast with 2.

Reductions remove selected dimensions.
No axes means all dimensions.
For example, Sum over axis 0 of a `[2,3]` array produces `[3]`.
Negative, duplicate, and out-of-range axes return errors.

Empty sums are zero.
Mean and Max reject empty reduction groups.
NaNs propagate through Max and arithmetic.
Log of zero returns negative infinity, and Sqrt of a negative value returns NaN.

## Matrix multiplication

MatMul computes `[m,k] × [k,n] → [m,n]`.
Inputs can be contiguous or strided.
A zero inner dimension produces a zero-filled result.

Only rank-two inputs are accepted.
A shape error identifies the operation and both shapes:

```text
tensor: matmul: incompatible shapes [2 3] and [4 5]
```

The current implementation is sequential and accumulates in increasing inner-dimension order.
There is no batched multiplication, operand packing, or worker pool.

## Verification and performance

Run from the repository root:

| Purpose | Command |
| --- | --- |
| Tensor tests | `mise run test:tensor` |
| Tensor lint | `mise run lint:tensor` |
| Shape, broadcasting, and view fuzzing | `mise run test:fuzz` |
| Tensor benchmarks | `mise run bench:tensor` |

Reference tests cover ownership, indexing, broadcasting, strided multiplication,
empty dimensions, invalid shapes, and overflow.
The [performance report](../docs/performance.md) compares contiguous, transposed,
and broadcast operations.

The [tensor contract](../docs/tensor-semantics.md) provides the complete semantics.
See [architecture](../docs/architecture.md) for private storage and kernel boundaries.

## Limits

The current API supports float32, unit-step slicing, explicit axes, and 2D matrix products.
It has no arbitrary dtypes, negative strides, advanced indexing, or mutable broadcast views.
Gradient tracking belongs to [nn](../nn/README.md).

Licensed under [MIT](../LICENSE).
