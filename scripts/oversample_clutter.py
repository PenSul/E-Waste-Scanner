"""Oversample cluttered (multi-object pile) images in a YOLO training split.

The detector's training data is overwhelmingly single, centred objects, with only
a handful of real cluttered pile scenes. That imbalance means the model learns
isolated objects well but generalises poorly to dense piles. This script raises
the cluttered share by duplicating each cluttered image/label pair ``multiplier``
times: the image is symlinked (no extra disk) and the label is copied. Because
Ultralytics applies fresh online augmentation (mosaic, scale, flip, HSV) to every
sample each epoch, the duplicates are not seen identically — this is the standard
oversampling trick for a rare subset.

Only the train split should be oversampled; leave val/test representative so the
reported metrics stay honest.

Usage::

    uv run --extra cpu python scripts/oversample_clutter.py \
        --train-dir datasets/train --multiplier 30
    # undo:
    uv run --extra cpu python scripts/oversample_clutter.py \
        --train-dir datasets/train --clean
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

#: Suffix marking a generated duplicate, e.g. ``cluttered_drawer1..._os7``.
_OS_RE = re.compile(r"_os\d+$")


def _originals(images_dir: Path, prefix: str) -> list[Path]:
    """Return the source cluttered images, excluding prior duplicates."""
    return sorted(
        p
        for p in images_dir.iterdir()
        if p.is_file()
        and p.stem.lower().startswith(prefix)
        and not _OS_RE.search(p.stem)
    )


def _generated(images_dir: Path, labels_dir: Path, prefix: str) -> Iterable[Path]:
    """Yield previously generated ``_os`` duplicate files (images and labels)."""
    for d in (images_dir, labels_dir):
        for p in d.iterdir():
            if p.is_file() and p.stem.lower().startswith(prefix) and _OS_RE.search(p.stem):
                yield p


def clean(images_dir: Path, labels_dir: Path, prefix: str) -> int:
    """Remove all generated duplicates; return how many files were deleted."""
    removed = 0
    for p in _generated(images_dir, labels_dir, prefix):
        p.unlink()
        removed += 1
    return removed


def oversample(train_dir: Path, multiplier: int, prefix: str) -> None:
    """Duplicate cluttered pairs ``multiplier``x (1x original + extra copies)."""
    images_dir = train_dir / "images"
    labels_dir = train_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"error: expected images/ and labels/ under {train_dir}")

    removed = clean(images_dir, labels_dir, prefix)
    if removed:
        print(f"removed {removed} stale duplicate files")

    originals = _originals(images_dir, prefix)
    if not originals:
        raise SystemExit(f"error: no images starting with '{prefix}' in {images_dir}")

    made = 0
    skipped = 0
    for img in originals:
        label = labels_dir / f"{img.stem}.txt"
        if not label.is_file():
            skipped += 1
            continue
        for i in range(1, multiplier):
            dup_img = images_dir / f"{img.stem}_os{i}{img.suffix}"
            dup_lbl = labels_dir / f"{img.stem}_os{i}.txt"
            if not dup_img.exists():
                dup_img.symlink_to(img.resolve())
            shutil.copy2(label, dup_lbl)
            made += 1

    total_clutter = len(originals) * multiplier
    total_imgs = sum(1 for p in images_dir.iterdir() if p.is_file())
    print(f"originals: {len(originals)} (skipped {skipped} without labels)")
    print(f"created {made} duplicate pairs -> cluttered total {total_clutter}")
    print(
        f"train images now {total_imgs}; "
        f"cluttered share ~{total_clutter / total_imgs:.1%}"
    )

    cache = labels_dir.with_suffix(".cache")
    if cache.exists():
        cache.unlink()
        print(f"removed label cache {cache.name} (Ultralytics will rebuild)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("datasets/train"))
    parser.add_argument("--multiplier", type=int, default=30, help="Total copies per scene.")
    parser.add_argument("--prefix", default="cluttered", help="Filename prefix to match.")
    parser.add_argument("--clean", action="store_true", help="Remove duplicates and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run oversampling or cleanup; return an exit code."""
    args = parse_args(argv)
    prefix = args.prefix.lower()
    if args.clean:
        images_dir, labels_dir = args.train_dir / "images", args.train_dir / "labels"
        removed = clean(images_dir, labels_dir, prefix)
        cache = labels_dir.with_suffix(".cache")
        if cache.exists():
            cache.unlink()
        print(f"removed {removed} duplicate files")
        return 0
    if args.multiplier < 1:
        raise SystemExit("error: --multiplier must be >= 1")
    oversample(args.train_dir, args.multiplier, prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
