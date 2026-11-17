# Backend API migration

The development backend refactor makes CPU and Metal use the same storage and
capability contracts. Existing model checkpoint files keep their mathematical
values and architecture representation; device handles are never serialized.
CPU checkpoint loading does not require Metal.

| Previous development API | Current API |
| --- | --- |
| Nil device means CPU | `tensor.CPU()` is an explicit shared device. Nil device arguments fail. |
| `tensor.Device`, `tensor.Buffer`, `tensor.DeviceView` | `backend.Device`, `backend.Buffer`, `backend.View` from `tensor/backend`. |
| Combined `ArithmeticBackend` / `SpatialBackend` | Independent typed operation and forward/backward capabilities. |
| String operation identifiers in backend methods | `backend.UnaryOp`, `BinaryOp`, `ScalarOp`, `ReduceOp`, and `ResizeMode`. |
| `Tensor.Data()` | `Tensor.HostData() ([]float32, error)`. |
| `Tensor.Contiguous()` | `Tensor.MakeContiguous() (*Tensor, error)`. |
| `Value.Detach() *Value` borrows its source handle | `Value.Detach() (*Value, error)` owns an independent unscoped handle; close its Data after the last use. |
| Same-device `To` shares GPU storage | `To` copies on all devices; `Retain` explicitly shares immutable storage. |
| CPU Tensor Close is a no-op | Close releases that handle on every device. Other retained views remain valid. |
| String prints CPU values | String prints bounded metadata on every device. Read HostData explicitly for values. |

Default constructors still create CPU tensors. To construct directly on another
context, use `FromOn`, `ZerosOn`, `OnesOn`, or `ScalarOn`. All mathematical operations
allocate on the input device. Missing capabilities wrap `backend.ErrUnsupported`;
there is no hidden CPU fallback. See [error and ownership contracts](backend-contract.md).

Before a parameter update, `Value.Data()` is a borrowed tensor handle. Keep an
independent reference with `Retain` when caller code needs its values after the
update. Autograd retains parameter generations used by recorded forward passes
until successful graph consumption. Gradients, optimizers, and layer composition
continue using ordinary Tensor operations. Recreate optimizers after moving model
parameters between devices. Cross-device parameter transfers reject active
graphs; consume those graphs before moving parameters. Detached values retain
their original storage independently and stay on their original device.

A scope now works on CPU as well as Metal. `Scope.To` shares same-device storage
through a new handle, unlike `Tensor.To`, which copies. Close each scope after its
synchronous callers finish, and Keep outputs that must survive it. A kept tensor
still requires Close. Asynchronous commands retain physical resources until they
finish even when logical handles close.

Publish compatible Tensor changes before NeuralNet and downstream modules, then
validate each with `GOWORK=off`. The workspace validates local implementations but
does not prove published dependency resolution. Do not change existing release
tags, commit local replacements, or fabricate dependency checksums. CUDA and ROCm
remain unpublished, unimplemented reservations.
