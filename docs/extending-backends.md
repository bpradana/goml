# Extending backends

A goml backend is an ordinary Go package implementing
[`tensor/backend`](../tensor/backend). It does not register globally or import
NeuralNet. Implement one device, its buffers, and only the numerical capabilities
needed by your workload. Tensor and custom neural-network layers remain unchanged.

The complete tutorial backend is in [`tensor/examples/backend`](../tensor/examples/backend).
It uses Go memory and implements MatMul only. Its source compiles in the Tensor
module and passes the same Device and MatMul conformance suites as CPU and Metal.
It is a contributor example, not an additional optimized runtime.

## 1. Define the execution context

Read [`device.go`](../tensor/examples/backend/device.go). `Open` creates a unique
context identity and a private allocation registry. There is no package-level
mutable device state. Device methods serialize access to that context.

Implement the mandatory contract:

```go
var _ backend.Device = (*Device)(nil)
```

The registry lets an owned context invalidate outstanding buffers during Close.
A stateless shared context can instead document a harmless Close, as CPU does.
Do not invent successful device implementations for CUDA or ROCm reservations.

## 2. Define physical storage

Read [`buffer.go`](../tensor/examples/backend/buffer.go). Each allocation has an
owner, immutable length, private storage, and a closed state. `Valid` checks both
buffer and context liveness. Repeated release is harmless. The example releases Go
memory synchronously; an asynchronous backend must retain native resources until
queued operations finish.

Ownership checks use the actual buffer type and owner. They do not trust a string
ID supplied by another implementation. Closed buffers return `ErrClosed`; foreign
buffers return `ErrDeviceMismatch`.

## 3. Implement allocation and transfers

`Allocate` validates the element count and allocates private storage. It does not
promise zero initialization. `Upload` and `Download` require the exact physical
buffer length and copy caller data. Downloads synchronize queued execution on an
asynchronous device. Report allocation failures with `ErrAllocation` and transfer
length failures with `ErrInvalidView`, preserving native details through wrapping.

The example's `Synchronize` is a liveness check because its operations finish
before returning. Metal instead submits and waits for its ordered command queue.
Do not return an unowned native pointer as a long-lived Go slice.

## 4. Add one numerical capability

Read [`matmul.go`](../tensor/examples/backend/matmul.go). The example implements:

```go
var _ backend.MatMulBackend = (*Device)(nil)
```

Tensor supplies validated `[m,k]` and `[k,n]` inputs and an allocated `[m,n]`
destination. The implementation checks ownership and physical layout under its
context lock, then runs its concrete kernel. `View.ValidateLayout` checks bounds
without calling `Buffer.Valid` recursively under that lock. Inputs can be strided;
this example requires a complete contiguous output and reports `ErrUnsupported`
for unsupported destinations.

The tutorial scalar loop makes indexing easy to inspect. Production CPU and Metal
keep their existing optimized kernels. Dispatch happens once for the whole
operation. Do not call an interface for every numerical element.

## 5. Leave other capabilities absent

There is no catch-all `Execute` method. This backend implements neither arithmetic
nor Fill. `tensor.Add` and `tensor.ZerosOn` therefore return errors wrapping
`backend.ErrUnsupported`. That is a valid MatMul-only backend. Upload an explicit
zero slice with `FromOn` when initialization is needed without Fill.

If an implemented family supports only some operation identifiers or layouts,
return `ErrUnsupported` for the others. A native failure of a supported operation
must retain its actual failure details, not masquerade as missing support.

## 6. Run conformance tests

The example's tests create real contexts and execute real allocations and kernels:

```go
func TestMatMul_Conformance(t *testing.T) {
	device, err := examplebackend.Open()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = device.Close() }()

	backendtest.RunMatMul(t, device)
}
```

Run the base Device suite in `device_test.go`. Run destructive `RunShutdown` last
on an owned context. Run each claimed numerical suite beside its matching source
file. Keep tests named for the behavior they verify, such as
`TestMatMul_PublicTensorIntegrationAndMissingCapability`.

```sh
mise run test:backend
mise run test:backend:cpu
mise run test:backend:metal
mise run lint:backend
```

Metal suites require supported Apple hardware and CGo. Ordinary CPU tests do not.
The tests compare against independent Go references rather than relying solely on
agreement between two backend implementations. For a partial unary family, select
its claimed operations explicitly, for example `backendtest.RunUnary(t, device,
backend.Exp, backend.ReLU)`. Inference-only devices do not need activation derivatives.
MatMul suites also accept explicit `[3]int{m,k,n}` cases for hardware constraints;
unsupported strided inputs must return ErrUnsupported and are reported as skipped.

## 7. Use the device through Tensor

This is the same public API used by CPU, Metal, and the example:

```go
device, err := examplebackend.Open()
if err != nil {
	return err
}
defer func() { _ = device.Close() }()

a, err := tensor.FromOn(device, []float32{1, 2, 3, 4}, 2, 2)
if err != nil {
	return err
}
defer func() { _ = a.Close() }()

result, err := tensor.MatMul(a, a)
if err != nil {
	return err
}
defer func() { _ = result.Close() }()

values, err := result.HostData()
if err != nil {
	return err
}

fmt.Println(values) // [7 10 15 22]
```

The integration test in `matmul_test.go` compiles and verifies this path, including
missing-capability errors. No changes to Tensor or NeuralNet enable the example.

## 8. Extend support without changing existing interfaces

Implement another focused interface, such as `BinaryBackend`, in `arithmetic.go`.
Use the shared typed operations, preserve shape and Float32 semantics, and add
`arithmetic_test.go` invoking `backendtest.RunBinary`. Implement gradient interfaces
only when training is supported. Native convolution does not need PatchBackend.

A neural-network layer that composes Tensor operations works once its required
capabilities are present. Keep model weights, gradients, and optimizer moments on
the selected device. Do not create backend-specific neural-network model classes.

Before contributing an optimized implementation, record representative baseline
benchmarks and compare numerical correctness, latency, allocations, and native
memory. Separate cold-start, resident execution, transfer, and complete model
measurements. Keep raw outputs outside the repository and document reproducible
summary results. See [Performance](performance.md) and the
[backend contract](backend-contract.md) for ownership and numerical details.
