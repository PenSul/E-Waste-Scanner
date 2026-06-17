"""Integration tests: CSV repositories load the shipped reference data."""

from __future__ import annotations

from pathlib import Path

import pytest

from ewaste.adapters.repository import (
    CsvImpactFactorRepository,
    CsvMaterialsRepository,
    fallback_prices,
)
from ewaste.domain.model import Material

REFERENCE = Path(__file__).resolve().parents[2] / "reference"


def test_materials_repository_loads_all_classes():
    repo = CsvMaterialsRepository(REFERENCE / "materials_composition.csv")
    comps = repo.compositions()
    assert len(comps) == 17
    pcb = comps["PCB"]
    assert pcb.device_mass_g == pytest.approx(120.0)
    assert pcb.fractions[Material.GOLD] == pytest.approx(0.00025)
    # zero fractions are dropped, so a pure metal class has no plastic key
    assert Material.PLASTIC not in comps["Metal"].fractions


def test_materials_repository_is_cached():
    repo = CsvMaterialsRepository(REFERENCE / "materials_composition.csv")
    assert repo.compositions() is repo.compositions()


def test_impact_repository_loads_class_and_material_factors():
    repo = CsvImpactFactorRepository(
        REFERENCE / "impact_factors.csv", REFERENCE / "material_factors.csv"
    )
    classes = repo.class_factors()
    assert len(classes) == 17
    assert classes["Trash"].recyclable is False
    materials = repo.material_factors()
    assert len(materials) == 10
    assert materials[Material.GOLD].priced is True
    assert materials[Material.PLASTIC].priced is False


def test_fallback_prices_covers_only_priced_metals():
    repo = CsvImpactFactorRepository(
        REFERENCE / "impact_factors.csv", REFERENCE / "material_factors.csv"
    )
    prices = fallback_prices(repo.material_factors())
    assert set(prices) == {
        Material.GOLD,
        Material.SILVER,
        Material.PALLADIUM,
        Material.PLATINUM,
        Material.COPPER,
        Material.ALUMINUM,
        Material.FERROUS,
    }
    assert prices[Material.GOLD].usd_per_gram > 0
    assert prices[Material.GOLD].source.startswith("fallback")
