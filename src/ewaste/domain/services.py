"""Pure valuation and impact arithmetic for the E-Waste Scanner.

These functions are the analytical heart of the app. They take already-loaded
domain data (per-class compositions, market prices, impact factors) and return
domain results. No I/O happens here, so the maths is deterministic and unit
tests need no mocks.

Three published methods are implemented (see ``docs/adr/0002-valuation-methodology.md``):

1. **UNEP/ITU recoverable value** — recovered metal grams times market price.
2. **EPA WARM** — net CO2e avoided by recycling vs landfilling, per class.
3. **WEEE-LCA** — bottom-up primary-production energy and CO2 avoided, from the
   recovered material mix.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

from ewaste.domain.model import (
    DetectedItem,
    ImpactEstimate,
    ImpactFactor,
    Material,
    MaterialComposition,
    MaterialFactor,
    MaterialValue,
    MetalPrice,
    Money,
    Valuation,
)


def aggregate(detections: Iterable, min_confidence: float = 0.0) -> list[DetectedItem]:
    """Group raw detections into per-class counts above a confidence floor."""
    counts = Counter(
        d.waste_class for d in detections if d.confidence >= min_confidence
    )
    return [DetectedItem(cls, n) for cls, n in sorted(counts.items())]


def recovered_grams(
    items: Iterable[DetectedItem],
    compositions: Mapping[str, MaterialComposition],
) -> dict[Material, float]:
    """Total grams of each material recoverable from the haul.

    Classes absent from ``compositions`` contribute nothing (and are surfaced by
    the service layer as a warning rather than silently miscounted).
    """
    grams: dict[Material, float] = defaultdict(float)
    for item in items:
        comp = compositions.get(item.waste_class)
        if comp is None:
            continue
        for material, g in comp.grams().items():
            grams[material] += g * item.count
    return dict(grams)


def valuation(
    items: Iterable[DetectedItem],
    compositions: Mapping[str, MaterialComposition],
    prices: Mapping[Material, MetalPrice],
) -> Valuation:
    """Recoverable value of a haul, with per-material and per-class breakdowns."""
    items = list(items)
    per_material: dict[Material, MaterialValue] = {}
    total = Money.zero()
    grams = recovered_grams(items, compositions)
    for material, g in grams.items():
        price = prices.get(material)
        value = Money(g * price.usd_per_gram) if price else Money.zero()
        per_material[material] = MaterialValue(material, g, value)
        total = total + value

    per_class: dict[str, Money] = {}
    for item in items:
        comp = compositions.get(item.waste_class)
        if comp is None:
            continue
        class_value = Money.zero()
        for material, g in comp.grams().items():
            price = prices.get(material)
            if price:
                class_value = class_value + Money(g * item.count * price.usd_per_gram)
        per_class[item.waste_class] = class_value
    return Valuation(total=total, per_material=per_material, per_class=per_class)


def impact(
    items: Iterable[DetectedItem],
    compositions: Mapping[str, MaterialComposition],
    class_factors: Mapping[str, ImpactFactor],
    material_factors: Mapping[Material, MaterialFactor],
    prices: Mapping[Material, MetalPrice],
) -> ImpactEstimate:
    """Environmental impact of a haul under the EPA WARM and WEEE-LCA methods."""
    items = list(items)

    mass_kg = 0.0
    co2e_kg_avoided = 0.0
    for item in items:
        comp = compositions.get(item.waste_class)
        if comp is None:
            continue
        item_mass_kg = comp.mass_kg * item.count
        mass_kg += item_mass_kg
        factor = class_factors.get(item.waste_class)
        if factor and factor.recyclable:
            co2e_kg_avoided += item_mass_kg * factor.warm_co2e_kg_per_kg

    grams = recovered_grams(items, compositions)
    energy_mj_saved = 0.0
    lca_co2e_kg_avoided = 0.0
    for material, g in grams.items():
        mf = material_factors.get(material)
        if mf is None:
            continue
        kg = g / 1000.0
        energy_mj_saved += kg * mf.lca_energy_mj_per_kg
        lca_co2e_kg_avoided += kg * mf.lca_co2e_kg_per_kg

    recoverable_value = valuation(items, compositions, prices).total
    return ImpactEstimate(
        mass_kg=mass_kg,
        recoverable_value=recoverable_value,
        co2e_kg_avoided=co2e_kg_avoided,
        energy_mj_saved=energy_mj_saved,
        lca_co2e_kg_avoided=lca_co2e_kg_avoided,
    )
