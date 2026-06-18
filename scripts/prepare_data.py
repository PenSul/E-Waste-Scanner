"""Normalise a Roboflow YOLO export into the project's ``data.yaml``.

A Roboflow object-detection export contains its own ``data.yaml`` plus
``train``/``valid``/``test`` folders (each with ``images/`` and ``labels/``).
This script:

1. reads the export's class names,
2. resolves them to our canonical taxonomy (``reference/class_map.yaml``),
   failing loudly on any class that is neither a canonical class, a known alias,
   nor on the ignore list,
3. optionally (``--remap``) rewrites every label file so class ids follow the
   canonical order (Battery=0, ...), dropping boxes for ignored classes, and
4. writes a top-level ``data.yaml`` whose ``names`` is the full canonical list,
   so class ids stay aligned with the reference value/impact tables.

Usage::

    uv run --extra cpu python scripts/prepare_data.py \
        --export-dir datasets/ewaste-v1 --out data.yaml --remap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SPLIT_DIRS = {"train": ["train"], "val": ["valid", "val"], "test": ["test"]}


def load_class_map(path: Path) -> tuple[list[str], dict[str, str], set[str]]:
    """Return ``(classes, aliases, ignore)`` from a class-map YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes: list[str] = list(data["classes"])
    aliases: dict[str, str] = {k.lower(): v for k, v in (data.get("aliases") or {}).items()}
    ignore: set[str] = {str(x).lower() for x in (data.get("ignore") or [])}
    return classes, aliases, ignore


def export_class_names(export_dir: Path) -> list[str]:
    """Read the ordered class names from the Roboflow export's data.yaml."""
    yml = export_dir / "data.yaml"
    if not yml.is_file():
        raise FileNotFoundError(f"no data.yaml in export dir: {export_dir}")
    names = yaml.safe_load(yml.read_text(encoding="utf-8")).get("names")
    if isinstance(names, dict):  # ultralytics dict form {0: name, ...}
        names = [names[k] for k in sorted(names)]
    if not names:
        raise ValueError(f"no 'names' in {yml}")
    return list(names)


def build_id_map(
    names: list[str],
    classes: list[str],
    aliases: dict[str, str],
    ignore: set[str],
) -> dict[int, int | None]:
    """Map each export class id to a canonical id (or ``None`` to drop it)."""
    canonical_id = {c.lower(): i for i, c in enumerate(classes)}
    id_map: dict[int, int | None] = {}
    unresolved: list[str] = []
    for old_id, name in enumerate(names):
        key = name.strip().lower()
        if key in ignore:
            id_map[old_id] = None
        elif key in canonical_id:
            id_map[old_id] = canonical_id[key]
        elif key in aliases:
            id_map[old_id] = canonical_id[aliases[key].lower()]
        else:
            unresolved.append(name)
    if unresolved:
        raise SystemExit(
            "error: these export classes are not in class_map.yaml (remap them in "
            "Roboflow or add aliases/ignore entries): " + ", ".join(sorted(set(unresolved)))
        )
    return id_map


def find_split_dir(export_dir: Path, candidates: list[str]) -> Path | None:
    """Return the first existing split image dir among ``candidates``."""
    for name in candidates:
        images = export_dir / name / "images"
        if images.is_dir():
            return export_dir / name
    return None


def remap_labels(split_dir: Path, id_map: dict[int, int | None]) -> tuple[int, int]:
    """Rewrite label ids in a split's ``labels/`` dir. Return ``(files, dropped)``."""
    labels_dir = split_dir / "labels"
    files = dropped = 0
    if not labels_dir.is_dir():
        return (0, 0)
    for txt in labels_dir.glob("*.txt"):
        out_lines: list[str] = []
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            new_id = id_map.get(int(parts[0]))
            if new_id is None:
                dropped += 1
                continue
            out_lines.append(" ".join([str(new_id), *parts[1:]]))
        txt.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        files += 1
    return (files, dropped)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True, help="Roboflow YOLO export dir.")
    parser.add_argument("--out", type=Path, default=Path("data.yaml"), help="Output data.yaml path.")
    parser.add_argument("--class-map", type=Path, default=Path("reference/class_map.yaml"))
    parser.add_argument(
        "--remap",
        action="store_true",
        help="Rewrite label files into canonical class-id order (modifies the export in place).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate the export and write the project data.yaml; return an exit code."""
    args = parse_args(argv)
    export_dir = args.export_dir.resolve()
    if not export_dir.is_dir():
        print(f"error: export dir not found: {export_dir}", file=sys.stderr)
        return 2

    classes, aliases, ignore = load_class_map(args.class_map)
    names = export_class_names(export_dir)
    id_map = build_id_map(names, classes, aliases, ignore)

    splits: dict[str, Path] = {}
    for split, candidates in SPLIT_DIRS.items():
        found = find_split_dir(export_dir, candidates)
        if found is not None:
            splits[split] = found
    if "train" not in splits:
        print("error: no train split found in export", file=sys.stderr)
        return 2

    if args.remap:
        for split, split_dir in splits.items():
            files, dropped = remap_labels(split_dir, id_map)
            print(f"remapped {split}: {files} label files, {dropped} boxes dropped (ignored classes)")

    data = {
        "path": str(export_dir),
        **{split: f"{split_dir.name}/images" for split, split_dir in splits.items()},
        "nc": len(classes),
        "names": classes,
    }
    args.out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.out} (path={export_dir}, splits={list(splits)}, nc={len(classes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
