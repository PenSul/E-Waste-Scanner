"""Dependency-injection wiring: build the concrete adapters for the app.

The service layer and UI depend only on the ports; this module is the single
place that chooses concrete implementations (CSV repositories, the yfinance
market provider, the YOLO detector) and binds them together. Swapping an
implementation — a different price source, a mock detector — happens here and
nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from ewaste.adapters.detector import YoloDetector
from ewaste.adapters.market import YFinanceMarketProvider
from ewaste.adapters.repository import (
    CsvImpactFactorRepository,
    CsvMaterialsRepository,
    fallback_prices,
)
from ewaste.config import Settings, load_settings
from ewaste.ports import (
    ImpactFactorRepository,
    MarketPriceProvider,
    MaterialsRepository,
    ObjectDetector,
)


@dataclass(frozen=True)
class Container:
    """The wired application dependencies."""

    settings: Settings
    materials: MaterialsRepository
    impacts: ImpactFactorRepository
    market: MarketPriceProvider
    detector: ObjectDetector


def build(settings: Settings | None = None) -> Container:
    """Construct and wire all adapters from ``settings`` (defaults to env)."""
    settings = settings or load_settings()

    materials = CsvMaterialsRepository(settings.composition_csv)
    impacts = CsvImpactFactorRepository(settings.impact_csv, settings.material_csv)

    market = YFinanceMarketProvider(
        fallbacks=fallback_prices(impacts.material_factors()),
        ttl_seconds=settings.price_ttl_seconds,
    )

    detector = YoloDetector(
        settings.weights_path,
        min_confidence=settings.min_confidence,
        imgsz=settings.imgsz,
        device=settings.device,
        tiled=settings.tiled,
        tile_size=settings.tile_size,
        tile_overlap=settings.tile_overlap,
        tile_iou=settings.tile_iou,
    )

    return Container(
        settings=settings,
        materials=materials,
        impacts=impacts,
        market=market,
        detector=detector,
    )
