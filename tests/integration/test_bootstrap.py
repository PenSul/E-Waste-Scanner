"""Integration test: bootstrap wires real adapters from settings."""

from __future__ import annotations

from ewaste.adapters.detector import YoloDetector
from ewaste.adapters.market import YFinanceMarketProvider
from ewaste.adapters.repository import (
    CsvImpactFactorRepository,
    CsvMaterialsRepository,
)
from ewaste.bootstrap import build
from ewaste.config import Settings


def test_build_wires_concrete_adapters():
    container = build(Settings())
    assert isinstance(container.materials, CsvMaterialsRepository)
    assert isinstance(container.impacts, CsvImpactFactorRepository)
    assert isinstance(container.market, YFinanceMarketProvider)
    assert isinstance(container.detector, YoloDetector)
    # repositories read the shipped reference data
    assert len(container.materials.compositions()) == 17
    # market provider already holds the static fallbacks built from the table
    fallbacks = container.market._fallbacks  # noqa: SLF001 (white-box wiring check)
    assert len(fallbacks) == 7
