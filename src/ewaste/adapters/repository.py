"""CSV-backed reference-data repositories.

These adapters read the cited reference tables in ``reference/`` and return the
domain value objects the service layer expects. They do no network I/O and hold
no torch dependency, so they load fast and are trivially testable.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path

from ewaste.domain.model import (
    ImpactFactor,
    Material,
    MaterialComposition,
    MaterialFactor,
    MetalPrice,
)
from ewaste.ports import ImpactFactorRepository, MaterialsRepository

#: Material columns in ``materials_composition.csv``, matching the enum values.
_MATERIAL_COLUMNS: tuple[Material, ...] = tuple(Material)


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Return the rows of a CSV file as dictionaries."""
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class CsvMaterialsRepository(MaterialsRepository):
    """Loads per-class material compositions from ``materials_composition.csv``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, MaterialComposition] | None = None

    def compositions(self) -> Mapping[str, MaterialComposition]:
        """Return each class composition, keyed by canonical class name (cached)."""
        if self._cache is None:
            self._cache = self._load()
        return self._cache

    def _load(self) -> dict[str, MaterialComposition]:
        out: dict[str, MaterialComposition] = {}
        for row in _read_rows(self._path):
            fractions = {
                material: float(row[material.value])
                for material in _MATERIAL_COLUMNS
                if float(row[material.value]) > 0.0
            }
            name = row["waste_class"]
            out[name] = MaterialComposition(
                waste_class=name,
                device_mass_g=float(row["device_mass_g"]),
                fractions=fractions,
                source=row.get("source", ""),
            )
        return out


class CsvImpactFactorRepository(ImpactFactorRepository):
    """Loads WARM class factors and WEEE-LCA material factors from two CSVs."""

    def __init__(self, class_path: Path, material_path: Path) -> None:
        self._class_path = class_path
        self._material_path = material_path
        self._class_cache: dict[str, ImpactFactor] | None = None
        self._material_cache: dict[Material, MaterialFactor] | None = None

    def class_factors(self) -> Mapping[str, ImpactFactor]:
        """Return per-class EPA WARM factors, keyed by class name (cached)."""
        if self._class_cache is None:
            self._class_cache = self._load_class()
        return self._class_cache

    def material_factors(self) -> Mapping[Material, MaterialFactor]:
        """Return per-material WEEE-LCA factors and price fallbacks (cached)."""
        if self._material_cache is None:
            self._material_cache = self._load_material()
        return self._material_cache

    def _load_class(self) -> dict[str, ImpactFactor]:
        out: dict[str, ImpactFactor] = {}
        for row in _read_rows(self._class_path):
            name = row["waste_class"]
            out[name] = ImpactFactor(
                waste_class=name,
                warm_co2e_kg_per_kg=float(row["warm_co2e_kg_per_kg"]),
                recyclable=row["recyclable"].strip().lower() == "true",
                source=row.get("source", ""),
            )
        return out

    def _load_material(self) -> dict[Material, MaterialFactor]:
        out: dict[Material, MaterialFactor] = {}
        for row in _read_rows(self._material_path):
            material = Material(row["material"])
            out[material] = MaterialFactor(
                material=material,
                lca_energy_mj_per_kg=float(row["lca_energy_mj_per_kg"]),
                lca_co2e_kg_per_kg=float(row["lca_co2e_kg_per_kg"]),
                fallback_usd_per_gram=float(row["fallback_usd_per_gram"]),
                priced=row["priced"].strip().lower() == "true",
                source=row.get("source", ""),
            )
        return out


def fallback_prices(
    material_factors: Mapping[Material, MaterialFactor],
) -> dict[Material, MetalPrice]:
    """Build static fallback prices from the priced rows of the material table.

    Used by the market adapter when the live feed is unavailable, so the app
    still reports a defensible, cited value offline.
    """
    return {
        material: MetalPrice(
            material=material,
            usd_per_gram=factor.fallback_usd_per_gram,
            asof=None,
            source=f"fallback:{factor.source}" if factor.source else "fallback",
        )
        for material, factor in material_factors.items()
        if factor.priced
    }
