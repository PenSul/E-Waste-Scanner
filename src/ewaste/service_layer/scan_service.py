"""Application service: turn an uploaded image into a costed, scored result.

This is the one place that orchestrates the pipeline described in
``CONTEXT.md``: detect objects, aggregate them into a haul, then run the pure
valuation and impact maths against the loaded reference data and live prices.
It depends only on the ports (via the :class:`~ewaste.bootstrap.Container`), so
the UI stays a thin caller and the pipeline is testable with a mock detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ewaste.bootstrap import Container
from ewaste.domain import services
from ewaste.domain.model import (
    Detection,
    DetectedItem,
    Haul,
    ImpactEstimate,
    Material,
    MetalPrice,
    Money,
    Valuation,
)


@dataclass(frozen=True)
class LineItem:
    """One row of the receipt: a class, its count, mass, and recoverable value."""

    waste_class: str
    count: int
    mass_kg: float
    value: Money


@dataclass(frozen=True)
class ScanResult:
    """Everything produced from one uploaded photo, ready for display.

    ``detections`` keep their pixel boxes for the overlay; ``lines`` are the
    per-class receipt rows; ``unknown_classes`` lists detected classes that have
    no reference composition (so the UI can disclose the gap rather than hide it).
    """

    detections: list[Detection]
    items: list[DetectedItem]
    lines: list[LineItem]
    valuation: Valuation
    impact: ImpactEstimate
    prices: dict[Material, MetalPrice]
    unknown_classes: list[str]

    @property
    def total_count(self) -> int:
        """Total number of detected objects."""
        return sum(item.count for item in self.items)

    @property
    def total_value(self) -> Money:
        """Recoverable value of the whole haul."""
        return self.valuation.total


class ScanService:
    """Runs the detect -> value -> impact pipeline for one image."""

    def __init__(self, container: Container) -> None:
        self._c = container

    def scan(self, image: Any, min_confidence: float | None = None) -> ScanResult:
        """Detect, value, and score the contents of ``image``.

        ``min_confidence`` overrides the configured detection floor for this
        call (used by the UI's confidence slider); it is applied to both the
        detector and the haul aggregation so the two never disagree.
        """
        settings = self._c.settings
        threshold = settings.min_confidence if min_confidence is None else min_confidence
        detector = self._c.detector
        if hasattr(detector, "min_confidence"):
            detector.min_confidence = threshold
        detections = detector.detect(image)
        haul = Haul(detections, min_confidence=threshold)
        items = haul.items()

        compositions = self._c.materials.compositions()
        class_factors = self._c.impacts.class_factors()
        material_factors = self._c.impacts.material_factors()
        prices = self._c.market.prices()

        valuation = services.valuation(items, compositions, prices)
        impact = services.impact(
            items, compositions, class_factors, material_factors, prices
        )

        lines: list[LineItem] = []
        unknown: list[str] = []
        for item in items:
            comp = compositions.get(item.waste_class)
            if comp is None:
                unknown.append(item.waste_class)
            mass_kg = comp.mass_kg * item.count if comp else 0.0
            value = valuation.per_class.get(item.waste_class, Money.zero())
            lines.append(
                LineItem(
                    waste_class=item.waste_class,
                    count=item.count,
                    mass_kg=mass_kg,
                    value=value,
                )
            )

        return ScanResult(
            detections=haul.kept(),
            items=items,
            lines=lines,
            valuation=valuation,
            impact=impact,
            prices=dict(prices),
            unknown_classes=unknown,
        )
