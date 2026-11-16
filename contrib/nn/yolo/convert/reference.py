"""Produce reference raw heads and NMS results from the exact Go input tensor."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("directory", type=Path, help="directory containing input.f32")
    parser.add_argument("--size", type=int, default=320)
    args = parser.parse_args()
    values = np.fromfile(args.directory / "input.f32", dtype="<f4")
    if values.size != 3 * args.size * args.size:
        parser.error("input.f32 length does not match size")
    model = (
        YOLO(str(args.checkpoint), task="detect")
        .model.cpu()
        .float()
        .eval()
        .fuse(verbose=False)
    )
    with torch.inference_mode():
        decoded, heads = model(
            torch.from_numpy(values.reshape(1, 3, args.size, args.size))
        )
        detections = non_max_suppression(
            decoded, conf_thres=0.25, iou_thres=0.45, max_det=300
        )[0]
    for scale, head in enumerate(heads):
        head.cpu().numpy().astype("<f4").tofile(args.directory / f"head-{scale}.f32")
    (args.directory / "detections.json").write_text(
        json.dumps(detections.cpu().tolist())
    )
    print(f"Saved three reference heads and {len(detections)} detections")


if __name__ == "__main__":
    main()
