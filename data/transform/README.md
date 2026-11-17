# Typed transforms

[Data module](../README.md) · [ImageFolder](../datasets/imagefolder/README.md)

Transforms use ordinary concrete values and `Func[T]`; no reflection or dynamic
argument maps are needed. `Compose` copies its step list, runs steps in order,
and stops at the first error or canceled context. Use `data.Map` to apply a
transform to a dataset, or pass a classification transform to ImageFolder.

```go
resize, err := transform.Resize(224, 224)
if err != nil {
	return err
}

normalize, err := transform.Normalize(
	[]float32{0.485, 0.456, 0.406},
	[]float32{0.229, 0.224, 0.225},
)
if err != nil {
	return err
}

pipeline := transform.Compose(resize, normalize)
```

Statistics above are explicit application configuration, not learned or inferred
values. Normalize applies `(x-mean)/std` per CHW channel. Means and standard
deviations are copied, finite, matching in length, and deviations must be positive.

| Operation | Input and output |
| --- | --- |
| ToFloat32 | image.Image → interleaved RGB HWC `[]float32` in `[0,1]`. |
| ImageToCHW | image.Image → ClassificationSample with `[channels,height,width]`, label zero. |
| Resize | CHW sample → bilinear half-pixel resize with clamped edges. |
| CenterCrop | CHW sample → centered crop, rejecting oversized requests. |
| Normalize | CHW sample → copied channel-normalized values. |
| Compose | Ordered same-type transforms with propagated errors and cancellation. |

Image conversion supports nonzero image origins, RGB or grayscale, and 16-bit
component precision. Alpha is composited onto black. Standard decoded images
use RGBA64At to avoid allocating a boxed color per pixel. Custom image types can
use the ordinary image.Image interface. Image conversion does not resize.

Resize, crop, and normalization return independent slices and preserve labels.
No input sample is mutated. Normalize before resizing only if that ordering is
intentional. No random augmentation, global random source, or automatic GPU
placement is implemented.

Licensed under [MIT](../../LICENSE).
