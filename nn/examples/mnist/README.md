# MNIST training example

[Neural network guide](../../README.md) · [Project overview](../../../README.md)

This example trains a seeded 784 → 32 → 10 classifier on local MNIST files.
It uses ReLU, stable cross-entropy, and Adam.
The program reports mean training cross-entropy after each epoch.

Dataset downloads are explicit.
Unit tests use generated local IDX fixtures and do not access the dataset host.

## Prepare the data

Run these commands from the repository root:

```sh
mkdir -p data
curl -fL https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz -o data/train-images-idx3-ubyte.gz
curl -fL https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz -o data/train-labels-idx1-ubyte.gz
mise run example:mnist
```

The data directory is ignored by Git.
Keep dataset files outside release commits.

## Configure the run

The default run uses 1,000 examples, three epochs, and batches of 32.
Pass flags after the mise task separator:

```sh
mise run example:mnist -- -limit 10000 -epochs 5 -batch 64
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-images` | `data/train-images-idx3-ubyte.gz` | Local gzip-compressed image IDX file. |
| `-labels` | `data/train-labels-idx1-ubyte.gz` | Local gzip-compressed label IDX file. |
| `-limit` | `1000` | Number of examples selected from the files. |
| `-epochs` | `3` | Number of training epochs. |
| `-batch` | `32` | Maximum batch size. The last batch can be smaller. |

Limit, epochs, and batch must be positive.
Limit cannot exceed the available image and label counts.

The loader reads complete IDX payloads before selecting the training limit.
A smaller limit reduces training arrays but does not reduce dataset-file reads.

## Model and training flow

```mermaid
flowchart LR
    Pixels["28 by 28 pixels<br/>normalized to float32"] --> Hidden["Dense 784 to 32"]
    Hidden --> ReLU["ReLU"]
    ReLU --> Output["Dense 32 to 10<br/>logits"]
    Output --> Loss["CrossEntropy"]
    Labels["Integer class labels"] --> Loss
    Loss --> Backward["Backward"]
    Backward --> Adam["Adam Step"]
    Adam --> Clear["ZeroGrad"]
```

The random generator uses a fixed seed.
Adam uses rate 0.001, beta1 0.9, beta2 0.999, and epsilon 1e-8.
Batches use the selected data in file order.
The example does not add data shuffling or an evaluation split.

## Input validation

The loader checks gzip reads, IDX magic numbers, image dimensions, counts, and exact payload lengths.
Images must be 28 by 28 pixels.
Files must contain 1 through 60,000 items, and image and label counts must match.

The training loop validates selected labels in `[0,10)`.
Pixels are normalized by dividing each byte value by 255.

| Failure | Resolution |
| --- | --- |
| File cannot be opened | Check the supplied local path and download completion. |
| Invalid magic or dimensions | Supply MNIST IDX image and label files, not another archive or format. |
| Truncated or extra payload | Replace the damaged or incorrect file. |
| Limit exceeds available examples | Choose a smaller positive limit. |
| Invalid label | Check that the label file corresponds to the image file. |

## What the example establishes

The source compiles as part of nn's tests.
Fixture tests verify valid and malformed IDX data and a complete training step.
The reported metric is training loss.

Real-dataset convergence and held-out accuracy were not measured for this checkout.
Do not treat fixture tests or lower training loss as an accuracy result.
