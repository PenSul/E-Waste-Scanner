"""Train the E-Waste Scanner detector (YOLO11) on the prepared dataset.

Reads the top-level ``data.yaml`` produced by ``scripts/prepare_data.py`` and
fine-tunes a pretrained YOLO11 checkpoint. We deploy **YOLO11s** (the accuracy/
size balance that fits Streamlit Community Cloud's 1 GB, CPU-only runtime) and
keep **YOLO11n** as an out-of-memory fallback (see
``docs/adr/0001-model-architecture-and-size.md``).

The dataset is single-object Kaggle crops plus real multi-object scenes; mosaic
and a little mixup synthesise extra clutter so the model generalises to messy
upload photos. Run on the local CUDA GPU::

    uv run --extra gpu python scripts/train.py --model yolo11s.pt --epochs 100

Quick pipeline smoke test (a few minutes)::

    uv run --extra gpu python scripts/train.py --epochs 2 --fraction 0.04 --name smoke

Resume an interrupted run::

    uv run --extra gpu python scripts/train.py --name ewaste-yolo11s --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data.yaml"), help="Dataset YAML.")
    parser.add_argument("--model", default="yolo11s.pt", help="Pretrained weights to fine-tune.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=float,
        default=-1,
        help="Batch size; -1 auto-fits VRAM, a 0-1 float uses that fraction of VRAM.",
    )
    parser.add_argument("--device", default="0", help="CUDA device index, or 'cpu'.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cache",
        default="false",
        help="Image cache: 'ram', 'disk', or 'false' (default; 37k imgs won't fit in RAM).",
    )
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience (epochs).")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation prob.")
    parser.add_argument("--mixup", type=float, default=0.1, help="MixUp augmentation prob.")
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic for the final N epochs to sharpen box fit.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the train set to use (set <1 for a fast smoke test).",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="ewaste-yolo11s", help="Run name under --project.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume the run named by --name.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run training; return a process exit code."""
    from ultralytics import YOLO

    args = parse_args(argv)
    cache = args.cache if args.cache in {"ram", "disk"} else False
    # Ultralytics batch semantics: int = fixed size, -1 = auto-fit VRAM,
    # 0<f<1 = fraction of VRAM. torch's sampler rejects float sizes, so coerce
    # whole numbers to int and leave only the fractional auto modes as floats.
    batch: int | float = args.batch
    if batch == -1 or batch >= 1:
        batch = int(batch)
    # Resolve project to an absolute path: a relative value gets re-rooted under
    # Ultralytics' own runs dir, producing a doubled runs/detect/runs/detect path.
    project = str(args.project.resolve())

    if args.resume:
        ckpt = args.project.resolve() / args.name / "weights" / "last.pt"
        if not ckpt.is_file():
            print(f"error: cannot resume, no checkpoint at {ckpt}")
            return 2
        model = YOLO(str(ckpt))
        model.train(resume=True)
        return 0

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        cache=cache,
        patience=args.patience,
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
        fraction=args.fraction,
        project=project,
        name=args.name,
        seed=args.seed,
        deterministic=True,
        plots=True,
        exist_ok=True,
    )
    best = Path(project) / args.name / "weights" / "best.pt"
    print(f"\ntraining done. best weights: {best}")
    print("evaluate + export with: uv run --extra gpu python scripts/evaluate.py "
          f"--weights {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
