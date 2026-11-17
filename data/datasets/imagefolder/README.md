# ImageFolder

[Data module](../../README.md) · [Transforms](../../transform/README.md)

Each directory immediately under the root is a class. JPEG/PNG files immediately
inside each class are its samples. Classes and filenames are sorted lexically;
`Classes()` returns the stable label-ordered names as an independent slice.

```go
resize, err := transform.Resize(224, 224)
if err != nil {
	return err
}

dataset, err := imagefolder.Open("archive/images", imagefolder.Options{
	Channels:  3,
	Transform: resize,
})
if err != nil {
	return err
}
defer func() { _ = dataset.Close() }()
```

Open discovers paths without decoding images. Get opens, bounds-checks, decodes,
and converts one image into owned CHW Float32 values. RGB (3 channels) is the
default; Channels=1 selects BT.601 grayscale. Components lie in `[0,1]`; alpha
is composited onto black. A configured transform runs afterward and preserves
variable image dimensions unless the application explicitly resizes/crops them.

MaxPixels defaults to 16,000,000; MaxBytes defaults to 32 MiB per encoded file.
Headers are checked before raster allocation. Invalid/corrupt images, oversized
files/dimensions, unsupported formats, nested directories, and symlinks return
errors. Rooted file access prevents escape from the chosen directory. Symlink
roots, class directories, and file entries are rejected. Empty roots/classes
are allowed; there is no recursive directory traversal or decoded-image cache.

Concurrent Get calls use independent files and samples. Custom transforms must
also be concurrency-safe. Finish readers before Close. Close is idempotent and
releases only the rooted directory handle; returned samples/batches stay valid.
Context cancellation is checked during file reads and pixel conversion.

Images with different shapes cannot be collated until explicitly transformed.
Tests generate small JPEG/PNG files, corrupt fixtures, and symlink cases locally.

Licensed under [MIT](../../../LICENSE).
