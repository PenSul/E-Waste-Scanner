"""Validate the shipped reference CSVs: coverage, ranges, and parseability.

These tests read the CSVs with the stdlib only (no adapter, no torch), so a
typo in the data files fails fast and independently of the rest of the stack.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

REFERENCE = Path(__file__).resolve().parents[2] / "reference"

MATERIAL_COLUMNS = [
    "gold", "silver", "palladium", "platinum", "copper",
    "aluminum", "ferrous", "plastic", "glass", "other",
]


def canonical_classes() -> list[str]:
    data = yaml.safe_load((REFERENCE / "class_map.yaml").read_text())
    return list(data["classes"])


def read_csv(name: str) -> list[dict]:
    with (REFERENCE / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_compositions_cover_every_canonical_class():
    rows = read_csv("materials_composition.csv")
    classes = {r["waste_class"] for r in rows}
    assert classes == set(canonical_classes())
    assert len(rows) == len(canonical_classes())  # no duplicates


def test_composition_fractions_are_valid():
    for row in read_csv("materials_composition.csv"):
        fracs = [float(row[c]) for c in MATERIAL_COLUMNS]
        assert all(0.0 <= f <= 1.0 for f in fracs), row["waste_class"]
        assert sum(fracs) <= 1.0 + 1e-6, f"{row['waste_class']} fractions exceed 1.0"
        assert float(row["device_mass_g"]) > 0.0


def test_impact_factors_cover_every_class():
    rows = read_csv("impact_factors.csv")
    classes = {r["waste_class"] for r in rows}
    assert classes == set(canonical_classes())
    for row in rows:
        assert float(row["warm_co2e_kg_per_kg"]) >= 0.0
        assert row["recyclable"] in {"true", "false"}


def test_material_factors_cover_every_material():
    rows = read_csv("material_factors.csv")
    materials = {r["material"] for r in rows}
    assert materials == set(MATERIAL_COLUMNS)
    priced = {r["material"] for r in rows if r["priced"] == "true"}
    # the seven commodity metals must carry a fallback price
    assert priced == {
        "gold", "silver", "palladium", "platinum", "copper", "aluminum", "ferrous",
    }
    for row in rows:
        assert float(row["lca_energy_mj_per_kg"]) >= 0.0
        assert float(row["lca_co2e_kg_per_kg"]) >= 0.0
        if row["priced"] == "true":
            assert float(row["fallback_usd_per_gram"]) > 0.0


@pytest.mark.parametrize("name", [
    "materials_composition.csv", "impact_factors.csv", "material_factors.csv",
])
def test_no_blank_source(name):
    for row in read_csv(name):
        assert row["source"].strip(), f"missing source in {name}: {row}"
