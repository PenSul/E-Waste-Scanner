"""Test configuration: make the repo root and app package importable."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "app"):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
