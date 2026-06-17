"""Normalise YOLO labels to bounding boxes for detection training.

The Roboflow export mixes two label shapes: single near-full-frame *boxes*
(``class cx cy w h``, from the bootstrapped Kaggle images) and *polygons*
(``class x1 y1 x2 y2 ...``, from the multi-object segmentation annotations). A
handful of files even mix both shapes on different lines, which Ultralytics'
``detect`` loader mishandles (one polygon line flips the whole file onto the
segment path, corrupting the genuine box lines).

This script rewrites every label file so each annotation is a plain bounding
box. Polygon lines are reduced to their axis-aligned bounds; box lines pass
through unchanged. It is **idempotent** — re-running it leaves box-only files
untouched — and operates in place on the (git-ignored) export.

Usage::

    uv run --extra gpu python scripts/boxify_labels.py --export-dir datasets/ewaste-v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SPLIT_NAMES = ("train", "valid", "val", "test")


def polygon_to_box(coords: list[float]) -> tuple[float, float, float, float]:
    """Return ``(cx, cy, w, h)`` bounding a flat ``[x1, y1, x2, y2, ...]`` polygon."""
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return cx, cy, x_max - x_min, y_max - y_min


def clamp01(value: float) -> float:
    """Clamp a normalised coordinate into ``[0, 1]``."""
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def boxify_line(line: str) -> str | None:
    """Convert one label line to ``class cx cy w h``; return ``None`` to drop it."""
    parts = line.split()
    if not parts:
        return None
    cls = parts[0]
    nums = [float(x) for x in parts[1:]]
    if len(nums) == 4:  # already a box
        cx, cy, w, h = nums
    elif len(nums) >= 6 and len(nums) % 2 == 0:  # polygon (>= 3 points)
        cx, cy, w, h = polygon_to_box(nums)
    else:  # malformed (odd coord count, or a 1-2 point degenerate shape)
        return None
    cx, cy = clamp01(cx), clamp01(cy)
    w, h = clamp01(w), clamp01(h)
    if w <= 0.0 or h <= 0.0:
        return None
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def boxify_file(path: Path) -> tuple[int, int]:
    """Rewrite a label file in place. Return ``(kept_lines, dropped_lines)``."""
    out: list[str] = []
    dropped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        converted = boxify_line(line)
        if converted is None:
            dropped += 1
            continue
        out.append(converted)
    path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    return len(out), dropped


def iter_label_dirs(export_dir: Path) -> list[Path]:
    """Return existing ``<split>/labels`` directories under the export."""
    dirs = []
    for name in SPLIT_NAMES:
        labels = export_dir / name / "labels"
        if labels.is_dir():
            dirs.append(labels)
    return dirs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=Path("datasets/ewaste-v1"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Normalise all label files to boxes; return a process exit code."""
    args = parse_args(argv)
    export_dir = args.export_dir.resolve()
    label_dirs = iter_label_dirs(export_dir)
    if not label_dirs:
        print(f"error: no <split>/labels dirs under {export_dir}", file=sys.stderr)
        return 2

    for labels in label_dirs:
        files = kept = dropped = 0
        for txt in labels.glob("*.txt"):
            k, d = boxify_file(txt)
            files += 1
            kept += k
            dropped += d
        print(f"{labels.parent.name}: {files} files, {kept} boxes kept, {dropped} dropped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
