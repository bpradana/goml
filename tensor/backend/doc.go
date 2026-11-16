// Package backend defines goml's Float32 execution and storage contracts.
// Implement Device and independently selected numerical capabilities. Views
// borrow metadata for the duration of a call; asynchronous implementations must
// copy metadata and retain physical buffers before returning. Mathematical shape
// validation belongs to tensor; implementations validate ownership, bounds, and
// hardware limitations. No implementation is registered globally.
package backend
