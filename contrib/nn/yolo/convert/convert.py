"""Export trusted YOLOv8n PyTorch checkpoints to goml's CPU inference archive."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import zipfile

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2f, SPPF, Conv, Detect


def network_checkpoint(block: torch.nn.Module) -> bytes:
    """Serialize standard convolutions using goml's existing network format."""
    layers = [{"type": "sequential", "children": []}]
    parameters = []

    def append_conv(conv: torch.nn.Conv2d, activation: torch.nn.Module) -> None:
        if (
            conv.groups != 1
            or conv.dilation != (1, 1)
            or conv.kernel_size[0] != conv.kernel_size[1]
            or conv.stride[0] != conv.stride[1]
            or conv.padding[0] != conv.padding[1]
            or conv.padding_mode != "zeros"
            or conv.bias is None
        ):
            raise ValueError("only fused, square, ungrouped convolutions are supported")
        if not isinstance(activation, (torch.nn.SiLU, torch.nn.Identity)):
            raise ValueError(
                f"unsupported convolution activation: {type(activation).__name__}"
            )

        weight = conv.weight.detach().cpu().permute(1, 2, 3, 0).contiguous()
        weight = weight.reshape(-1, conv.out_channels).numpy().astype("<f4")
        bias = conv.bias.detach().cpu().numpy().astype("<f4")
        first_parameter = len(parameters)
        for array in (weight, bias):
            if not np.isfinite(array).all():
                raise ValueError("checkpoint contains nonfinite weights")
            parameters.append(
                {
                    "dtype": "float32",
                    "shape": list(array.shape),
                    "data": base64.b64encode(array.tobytes()).decode("ascii"),
                }
            )

        index = len(layers)
        layers[0]["children"].append(index)
        layers.append(
            {
                "type": "conv2d",
                "config": {
                    "kernel": conv.kernel_size[0],
                    "stride": conv.stride[0],
                    "padding": conv.padding[0],
                },
                "children": [index + 1],
            }
        )
        layers.append(
            {"type": "dense", "parameters": [first_parameter, first_parameter + 1]}
        )
        if isinstance(activation, torch.nn.SiLU):
            layers[0]["children"].append(len(layers))
            layers.append({"type": "silu"})

    def visit(module: torch.nn.Module) -> None:
        if isinstance(module, Conv):
            if hasattr(module, "bn"):
                raise ValueError("BatchNorm must be fused before export")
            append_conv(module.conv, module.act)
        elif isinstance(module, torch.nn.Conv2d):
            append_conv(module, torch.nn.Identity())
        elif isinstance(module, torch.nn.Sequential):
            for child in module:
                visit(child)
        else:
            raise ValueError(f"unsupported network block: {type(module).__name__}")

    visit(block)
    payload = json.dumps(
        {
            "kind": "network",
            "architecture": {"root": 0, "layers": layers},
            "parameters": parameters,
        },
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return (
        b"GOMLCP01"
        + struct.pack("<Q", len(payload))
        + hashlib.sha256(payload).digest()
        + payload
    )


def export_networks(model: torch.nn.Module) -> dict[str, bytes]:
    """Require the YOLOv8n topology and export each learned block by stable name."""
    expected = [
        "Conv",
        "Conv",
        "C2f",
        "Conv",
        "C2f",
        "Conv",
        "C2f",
        "Conv",
        "C2f",
        "SPPF",
        "Upsample",
        "Concat",
        "C2f",
        "Upsample",
        "Concat",
        "C2f",
        "Conv",
        "Concat",
        "C2f",
        "Conv",
        "Concat",
        "C2f",
        "Detect",
    ]
    if [type(stage).__name__ for stage in model.model] != expected:
        raise ValueError("expected the 23-stage YOLOv8n detection topology")
    routes = {11: [-1, 6], 14: [-1, 4], 17: [-1, 12], 20: [-1, 9], 22: [15, 18, 21]}
    widths = {0: 16, 1: 32, 3: 64, 5: 128, 7: 256, 16: 64, 19: 128}
    repeats = {2: 1, 4: 2, 6: 2, 8: 1, 12: 1, 15: 1, 18: 1, 21: 1}
    networks = {}
    for index, stage in enumerate(model.model):
        if stage.f != routes.get(index, -1):
            raise ValueError(f"unsupported source routing at stage {index}")
        if isinstance(stage, Conv):
            if stage.conv.out_channels != widths[index]:
                raise ValueError("expected YOLOv8 nano channel widths")
            networks[str(index)] = network_checkpoint(stage)
        elif isinstance(stage, C2f):
            if len(stage.m) != repeats[index]:
                raise ValueError("expected YOLOv8 nano repeat counts")
            networks[f"{index}.cv1"] = network_checkpoint(stage.cv1)
            networks[f"{index}.cv2"] = network_checkpoint(stage.cv2)
            for number, bottleneck in enumerate(stage.m):
                if bottleneck.add != (index < 9):
                    raise ValueError("unsupported bottleneck residual configuration")
                networks[f"{index}.m.{number}.cv1"] = network_checkpoint(bottleneck.cv1)
                networks[f"{index}.m.{number}.cv2"] = network_checkpoint(bottleneck.cv2)
        elif isinstance(stage, SPPF):
            if (stage.m.kernel_size, stage.m.stride, stage.m.padding) != (5, 1, 2):
                raise ValueError("expected 5x5 SPPF pooling")
            networks["9.cv1"] = network_checkpoint(stage.cv1)
            networks["9.cv2"] = network_checkpoint(stage.cv2)
        elif isinstance(stage, Detect):
            if (
                stage.reg_max != 16
                or stage.stride.tolist() != [8, 16, 32]
                or stage.end2end
            ):
                raise ValueError("unsupported detection head geometry")
            projection = stage.dfl.conv.weight.detach().cpu().reshape(-1)
            if not torch.equal(projection, torch.arange(16, dtype=projection.dtype)):
                raise ValueError("expected the fixed sixteen-bin DFL projection")
            for scale in range(3):
                networks[f"22.reg.{scale}"] = network_checkpoint(stage.cv2[scale])
                networks[f"22.cls.{scale}"] = network_checkpoint(stage.cv3[scale])
        elif isinstance(stage, torch.nn.Upsample):
            if stage.mode != "nearest" or stage.scale_factor != 2:
                raise ValueError("expected nearest upsampling by two")
        elif stage.d != 1:
            raise ValueError("expected channel concatenation")
    return networks


def convert(source: Path, destination: Path) -> None:
    """Fold BatchNorm and write a deterministic ZIP without altering the source."""
    model = (
        YOLO(str(source), task="detect").model.cpu().float().eval().fuse(verbose=False)
    )
    networks = export_networks(model)
    names = [model.names[index] for index in range(model.model[-1].nc)]
    manifest = {
        "version": 1,
        "architecture": "yolov8n-detect",
        "classes": names,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            with zipfile.ZipFile(
                handle, "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                entries = {
                    "manifest.json": json.dumps(
                        manifest, separators=(",", ":")
                    ).encode()
                }
                entries.update(
                    {f"networks/{name}.goml": data for name, data in networks.items()}
                )
                for name, data in sorted(entries.items()):
                    info = zipfile.ZipInfo(name)
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Exported {len(networks)} networks, {len(names)} classes to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="trusted YOLOv8n .pt checkpoint")
    parser.add_argument("destination", type=Path, help="output .goml archive")
    args = parser.parse_args()
    if args.source.suffix != ".pt" or not args.source.is_file():
        parser.error("source must be an existing .pt checkpoint")
    if args.source.resolve() == args.destination.resolve():
        parser.error("destination must differ from source")
    convert(args.source, args.destination)


if __name__ == "__main__":
    main()
