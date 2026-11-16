// Package checkpoint saves float32 weights and built-in Sequential networks.
//
// SaveWeights and LoadWeights operate on an existing ordered parameter list.
// LoadWeights validates every tensor before updating any parameter, preserves
// parameter identity, and clears gradients after a successful load.
//
// Save and Load include architecture and weights. Marshal and Unmarshal provide
// the same network format as bytes. Restored networks preserve shared layers and
// parameters and can run inference or build fresh training graphs.
//
// The versioned format stores JSON metadata, little-endian float32 bytes, and a
// SHA-256 payload checksum. Float32 bit patterns are preserved, including signed
// zero, NaN, and infinity. Payloads are limited to 64 MiB. The checksum detects
// corruption; it is not a signature authenticating the producer.
//
// Files are written through a temporary file in the destination directory and
// renamed after a complete write and sync. Files use owner-only permissions.
// Saving replaces an existing destination; parent directories are created when
// necessary. Atomic replacement assumes the supported POSIX filesystem behavior.
//
// Gradients, computation graphs, optimizer state, and runtime recurrent state are
// excluded. Recreate an optimizer when starting training from restored weights.
// Callers must synchronize saving and loading with training and parameter updates.
package checkpoint
