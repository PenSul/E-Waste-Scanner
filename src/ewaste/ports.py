"""Ports: the abstract boundaries between the domain and the outside world.

Following the ports-and-adapters pattern, the service layer depends only on
these interfaces. Concrete adapters in :mod:`ewaste.adapters` implement them
(Ultralytics for detection, yfinance for prices, CSV files for reference data),
and :mod:`ewaste.bootstrap` wires the chosen implementations together.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from ewaste.domain.model import (
    Detection,
    ImpactFactor,
    Material,
    MaterialComposition,
    MaterialFactor,
    MetalPrice,
)


class ObjectDetector(ABC):
    """Detects waste objects in an image, returning canonical-class detections."""

    @abstractmethod
    def detect(self, image: Any) -> list[Detection]:
        """Return the detections found in ``image`` (a path, array, or PIL image)."""
        raise NotImplementedError


class MarketPriceProvider(ABC):
    """Supplies current metals-market prices in USD per gram."""

    @abstractmethod
    def prices(self) -> dict[Material, MetalPrice]:
        """Return a price per priced material (live, cached, or fallback)."""
        raise NotImplementedError


class MaterialsRepository(ABC):
    """Loads per-class material compositions from reference data."""

    @abstractmethod
    def compositions(self) -> Mapping[str, MaterialComposition]:
        """Return the composition for each known waste class, keyed by class name."""
        raise NotImplementedError


class ImpactFactorRepository(ABC):
    """Loads the environmental-impact factors used by the WARM and LCA methods."""

    @abstractmethod
    def class_factors(self) -> Mapping[str, ImpactFactor]:
        """Return the per-class EPA WARM factors, keyed by class name."""
        raise NotImplementedError

    @abstractmethod
    def material_factors(self) -> Mapping[Material, MaterialFactor]:
        """Return the per-material WEEE-LCA factors and price fallbacks."""
        raise NotImplementedError
