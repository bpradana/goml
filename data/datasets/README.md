# Built-in datasets

[Data module](../README.md) · [Contributor guide](../../docs/datasets.md)

| Package | Local source | Returned sample |
| --- | --- | --- |
| [mnist](mnist/README.md) | Plain/gzip MNIST IDX files. | Normalized `[1,28,28]` image and integer label. |
| [csvdata](csvdata/README.md) | Explicit numerical CSV columns. | Classification or regression feature/target vectors. |
| [imagefolder](imagefolder/README.md) | Sorted JPEG/PNG class directories. | Lazy RGB/grayscale CHW image and stable label. |

No package downloads data. Use the root data package for Tensor-backed datasets,
generic loaders, custom collators, Map transforms, and ordered Subset selections.

Licensed under [MIT](../../LICENSE).
