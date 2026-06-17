"""Bootstrap YOLO detection labels for the Kaggle classification dataset.

The Kaggle dataset stores one centered object per image, grouped into per-class
folders. This script writes a single near-full-frame YOLO bounding box next to
each image, using the folder name (resolved through ``reference/class_map.yaml``)
as the class. The result can be uploaded to Roboflow as pre-annotations and then
reviewed/tightened, which is far faster than drawing ~6,800 boxes by hand.

The generated box is intentionally large (a centered square/rectangle covering
``--coverage`` of the frame): it teaches the object's appearance, while YOLO's
mosaic augmentation synthesises the multi-object layouts at training time. Review
images where the object is small or off-centre.

Labels are written *in place* beside each image (e.g. ``image_001.txt`` next to
``image_001.jpg``); the source image files are never modified. A ``classes.txt``
and ``data.yaml`` are written at the images-dir root so Roboflow can map class
ids to names.

Usage::

    uv run --extra cpu python scripts/autobox_kaggle.py \
        --images-dir data/balanced_waste_images --coverage 0.92
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_map(path: Path) -> tuple[list[str], dict[str, str], set[str]]:
    """Return ``(classes, aliases, ignore)`` from a class-map YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes: list[str] = list(data["classes"])
    aliases: dict[str, str] = {k.lower(): v for k, v in (data.get("aliases") or {}).items()}
    ignore: set[str] = {str(x).lower() for x in (data.get("ignore") or [])}
    return classes, aliases, ignore


def resolve_class(raw: str, classes: list[str], aliases: dict[str, str]) -> str | None:
    """Resolve a folder name to a canonical class, or ``None`` if unmappable."""
    key = raw.strip().lower()
    by_canonical = {c.lower(): c for c in classes}
    if key in by_canonical:
        return by_canonical[key]
    return aliases.get(key)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/balanced_waste_images"),
        help="Root folder containing one sub-folder per class.",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=Path("reference/class_map.yaml"),
        help="Canonical taxonomy used to resolve folder names to class ids.",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.92,
        help="Fraction of width/height the centered box spans (0-1).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite label files that already exist (default: skip them).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Generate the bootstrap labels; return a process exit code."""
    args = parse_args(argv)
    if not 0.0 < args.coverage <= 1.0:
        print("error: --coverage must be in (0, 1]", file=sys.stderr)
        return 2
    if not args.images_dir.is_dir():
        print(f"error: images dir not found: {args.images_dir}", file=sys.stderr)
        return 2

    classes, aliases, _ignore = load_class_map(args.class_map)
    class_to_id = {name: i for i, name in enumerate(classes)}
    box = f"{0.5:.6f} {0.5:.6f} {args.coverage:.6f} {args.coverage:.6f}"

    written = skipped = unresolved = 0
    unresolved_folders: set[str] = set()

    for folder in sorted(p for p in args.images_dir.iterdir() if p.is_dir()):
        canonical = resolve_class(folder.name, classes, aliases)
        if canonical is None:
            unresolved_folders.add(folder.name)
            continue
        line = f"{class_to_id[canonical]} {box}\n"
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = image.with_suffix(".txt")
            if label.exists() and not args.overwrite:
                skipped += 1
                continue
            label.write_text(line, encoding="utf-8")
            written += 1

    if unresolved_folders:
        unresolved = len(unresolved_folders)
        print(
            "warning: skipped unmappable folders (add them to class_map.yaml): "
            + ", ".join(sorted(unresolved_folders)),
            file=sys.stderr,
        )

    # Class index files so Roboflow can map ids -> names on import.
    (args.images_dir / "classes.txt").write_text(
        "\n".join(classes) + "\n", encoding="utf-8"
    )
    (args.images_dir / "data.yaml").write_text(
        yaml.safe_dump({"nc": len(classes), "names": classes}, sort_keys=False),
        encoding="utf-8",
    )

    print(f"labels written: {written}, skipped (existing): {skipped}, unmapped folders: {unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
