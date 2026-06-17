"""Domain model for the E-Waste Scanner.

Pure value objects and the ``Haul`` aggregate. This module performs **no I/O**
(no file, network, or model access): adapters load reference data and live
prices, and the service layer feeds them in. Keeping the core pure makes the
valuation and impact arithmetic trivially unit-testable.

Vocabulary follows ``CONTEXT.md``: a *Detection* is one predicted box, a
*Haul* is everything found in one uploaded photo, a *Valuation* is the
recoverable material value, and an *ImpactEstimate* is the avoided environmental
burden under three published methods.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum


class Material(str, Enum):
    """A recoverable material fraction tracked per device.

    The first seven are commodity-priced (precious metals, copper, and the
    base metals aluminium and ferrous/steel scrap); the last three carry mass
    and environmental weight but no metals-market price.
    """

    GOLD = "gold"
    SILVER = "silver"
    PALLADIUM = "palladium"
    PLATINUM = "platinum"
    COPPER = "copper"
    ALUMINUM = "aluminum"
    FERROUS = "ferrous"
    PLASTIC = "plastic"
    GLASS = "glass"
    OTHER = "other"


#: Materials with a metals-market quote; only these contribute to recoverable value.
PRICED_MATERIALS: tuple[Material, ...] = (
    Material.GOLD,
    Material.SILVER,
    Material.PALLADIUM,
    Material.PLATINUM,
    Material.COPPER,
    Material.ALUMINUM,
    Material.FERROUS,
)


@dataclass(frozen=True)
class Money:
    """A monetary amount in a single currency (USD by default).

    Amounts are floats: every figure in this app is an estimate, so exact
    decimal arithmetic would imply false precision. Round only for display.
    """

    amount: float
    currency: str = "USD"

    @classmethod
    def zero(cls, currency: str = "USD") -> Money:
        """Return a zero amount in ``currency``."""
        return cls(0.0, currency)

    def __add__(self, other: Money) -> Money:
        if self.currency != other.currency:
            raise ValueError(f"currency mismatch: {self.currency} vs {other.currency}")
        return Money(self.amount + other.amount, self.currency)

    def __mul__(self, factor: float) -> Money:
        return Money(self.amount * factor, self.currency)

    __rmul__ = __mul__

    def rounded(self, ndigits: int = 2) -> Money:
        """Return a copy rounded to ``ndigits`` decimal places."""
        return Money(round(self.amount, ndigits), self.currency)


@dataclass(frozen=True)
class MetalPrice:
    """A live (or fallback) market price for one material, in USD per gram."""

    material: Material
    usd_per_gram: float
    asof: str | None = None
    source: str = ""


@dataclass(frozen=True)
class MaterialComposition:
    """Typical material make-up of one device of a given waste class.

    ``fractions`` are mass fractions (0-1) of ``device_mass_g`` per material;
    they need not sum to exactly 1 (the remainder is implicitly unrecovered).
    """

    waste_class: str
    device_mass_g: float
    fractions: Mapping[Material, float]
    source: str = ""

    @property
    def mass_kg(self) -> float:
        """Device mass in kilograms."""
        return self.device_mass_g / 1000.0

    def grams(self) -> dict[Material, float]:
        """Grams of each material in one device."""
        return {m: self.device_mass_g * f for m, f in self.fractions.items()}


@dataclass(frozen=True)
class ImpactFactor:
    """Per-class EPA WARM factor: net CO2e avoided by recycling vs landfilling.

    ``warm_co2e_kg_per_kg`` is kilograms of CO2-equivalent avoided per kilogram
    of material diverted (positive = benefit). ``recyclable`` is False for
    streams with no recovery pathway (e.g. mixed trash).
    """

    waste_class: str
    warm_co2e_kg_per_kg: float
    recyclable: bool = True
    source: str = ""


@dataclass(frozen=True)
class MaterialFactor:
    """Per-material WEEE-LCA factors plus a static price fallback.

    The LCA factors are the primary-production burden avoided per kilogram of
    recovered material; ``fallback_usd_per_gram`` is used when the live market
    feed is unavailable.
    """

    material: Material
    lca_energy_mj_per_kg: float
    lca_co2e_kg_per_kg: float
    fallback_usd_per_gram: float
    priced: bool
    source: str = ""


@dataclass(frozen=True)
class Detection:
    """One predicted object: a canonical class, a confidence, and a box.

    ``bbox`` is ``(x1, y1, x2, y2)`` in pixels, or ``None`` for synthetic
    detections in tests.
    """

    waste_class: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class DetectedItem:
    """A waste class and how many of it were found in a haul."""

    waste_class: str
    count: int


@dataclass(frozen=True)
class MaterialValue:
    """Recovered grams of one material and the money it is worth."""

    material: Material
    grams: float
    value: Money


@dataclass(frozen=True)
class Valuation:
    """Recoverable value of a haul, with per-material and per-class breakdowns."""

    total: Money
    per_material: dict[Material, MaterialValue]
    per_class: dict[str, Money]


@dataclass(frozen=True)
class ImpactEstimate:
    """Environmental upside of recovering a haul, under three published methods.

    - ``recoverable_value``: UNEP/ITU raw-material value (equals the valuation
      total; restated here as the economic dimension of impact).
    - ``co2e_kg_avoided``: EPA WARM, recycling vs landfilling.
    - ``energy_mj_saved`` / ``lca_co2e_kg_avoided``: bottom-up WEEE-LCA from the
      recovered material mix.
    """

    mass_kg: float
    recoverable_value: Money
    co2e_kg_avoided: float
    energy_mj_saved: float
    lca_co2e_kg_avoided: float


@dataclass
class Haul:
    """Everything detected in one uploaded photo.

    Holds raw detections and applies a confidence floor when aggregating to
    per-class counts. The ``valuation`` and ``impact`` methods are thin wrappers
    over the pure functions in :mod:`ewaste.domain.services`.
    """

    detections: list[Detection]
    min_confidence: float = 0.0

    def kept(self) -> list[Detection]:
        """Detections at or above the confidence floor."""
        return [d for d in self.detections if d.confidence >= self.min_confidence]

    def items(self) -> list[DetectedItem]:
        """Per-class counts of the kept detections, in stable class order."""
        counts = Counter(d.waste_class for d in self.kept())
        return [DetectedItem(cls, n) for cls, n in sorted(counts.items())]

    def valuation(
        self,
        compositions: Mapping[str, MaterialComposition],
        prices: Mapping[Material, MetalPrice],
    ) -> Valuation:
        """Recoverable value of this haul (see :func:`services.valuation`)."""
        from ewaste.domain import services

        return services.valuation(self.items(), compositions, prices)

    def impact(
        self,
        compositions: Mapping[str, MaterialComposition],
        class_factors: Mapping[str, ImpactFactor],
        material_factors: Mapping[Material, MaterialFactor],
        prices: Mapping[Material, MetalPrice],
    ) -> ImpactEstimate:
        """Environmental impact of this haul (see :func:`services.impact`)."""
        from ewaste.domain import services

        return services.impact(
            self.items(), compositions, class_factors, material_factors, prices
        )


def total_count(items: Iterable[DetectedItem]) -> int:
    """Total number of detected objects across all classes."""
    return sum(item.count for item in items)
