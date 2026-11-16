// Package backendtest provides reusable conformance tests for goml backends.
// RunDevice is mandatory. Run each numerical suite for capabilities a device
// claims to implement. Suites synchronize before reading results and close their
// own allocations, but leave the caller's device open. RunShutdown is destructive
// and applies only to owned execution contexts, not the shared CPU context.
package backendtest
