"""Application settings: paths, cache TTLs, and detection thresholds.

Settings are plain data with sensible defaults derived from the repository
layout. :func:`load_settings` overlays environment variables (and, when present,
Streamlit secrets) so the same code runs in tests, locally, and on Streamlit
Community Cloud without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Repository root, two packages up from this file (``src/ewaste/config.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the scanner.

    Paths default to the repository layout; thresholds and TTLs default to
    values suited to a 1 GB CPU-only deployment.
    """

    reference_dir: Path = REPO_ROOT / "reference"
    weights_path: Path = REPO_ROOT / "models" / "best.pt"
    min_confidence: float = 0.25
    imgsz: int = 640
    device: str = "cpu"
    price_ttl_seconds: int = 3600

    @property
    def composition_csv(self) -> Path:
        """Path to the per-class material-composition table."""
        return self.reference_dir / "materials_composition.csv"

    @property
    def impact_csv(self) -> Path:
        """Path to the per-class EPA WARM impact-factor table."""
        return self.reference_dir / "impact_factors.csv"

    @property
    def material_csv(self) -> Path:
        """Path to the per-material WEEE-LCA and price-fallback table."""
        return self.reference_dir / "material_factors.csv"


def _env(name: str) -> str | None:
    """Return an environment variable, treating blank as unset."""
    value = os.environ.get(name)
    return value if value else None


def load_settings() -> Settings:
    """Build :class:`Settings`, overlaying ``EWASTE_*`` environment variables."""
    defaults = Settings()
    weights = _env("EWASTE_WEIGHTS")
    reference = _env("EWASTE_REFERENCE_DIR")
    min_conf = _env("EWASTE_MIN_CONFIDENCE")
    device = _env("EWASTE_DEVICE")
    ttl = _env("EWASTE_PRICE_TTL")
    return Settings(
        reference_dir=Path(reference) if reference else defaults.reference_dir,
        weights_path=Path(weights) if weights else defaults.weights_path,
        min_confidence=float(min_conf) if min_conf else defaults.min_confidence,
        imgsz=defaults.imgsz,
        device=device or defaults.device,
        price_ttl_seconds=int(ttl) if ttl else defaults.price_ttl_seconds,
    )
