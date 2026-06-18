"""Evaluate a trained detector and export the deployment weights.

Runs Ultralytics validation on a held-out split (default: ``test``), prints the
headline mAP metrics and per-class AP, and copies the evaluated checkpoint to
``models/best.pt`` — the weights the Streamlit app loads. The confusion matrix
and PR curves are written by Ultralytics into the validation run directory.

Usage::

    uv run --extra gpu python scripts/evaluate.py \
        --weights runs/detect/ewaste-yolo11s/weights/best.pt --split test
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Trained .pt checkpoint.")
    parser.add_argument("--data", type=Path, default=Path("data.yaml"), help="Dataset YAML.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'.")
    parser.add_argument("--out", type=Path, default=Path("models/best.pt"), help="Export path.")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Only report metrics; do not copy weights to --out.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate the model, print metrics, export weights; return an exit code."""
    from ultralytics import YOLO

    args = parse_args(argv)
    if not args.weights.is_file():
        print(f"error: weights not found: {args.weights}")
        return 2

    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
    )

    print(f"\n=== {args.split} metrics ===")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP75    : {metrics.box.map75:.4f}")
    print("\nper-class AP50:")
    names = metrics.names
    for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
        print(f"  {names[int(i)]:<16} {ap:.4f}")
    print(f"\nplots/confusion matrix saved to: {metrics.save_dir}")

    if not args.no_export:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.weights, args.out)
        print(f"exported deployment weights -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
