"""Unit tests for the pure valuation and impact arithmetic."""

from __future__ import annotations

import pytest

from ewaste.domain import services
from ewaste.domain.model import (
    Detection,
    Haul,
    ImpactFactor,
    Material,
    MaterialComposition,
    MaterialFactor,
    MetalPrice,
    Money,
)

# --- small, exact fixtures so the maths can be asserted to the cent ---------

COMPOSITIONS = {
    "PCB": MaterialComposition(
        waste_class="PCB",
        device_mass_g=100.0,
        fractions={Material.GOLD: 0.001, Material.COPPER: 0.20},
    ),
}

PRICES = {
    Material.GOLD: MetalPrice(Material.GOLD, usd_per_gram=100.0),
    Material.COPPER: MetalPrice(Material.COPPER, usd_per_gram=0.01),
}

CLASS_FACTORS = {
    "PCB": ImpactFactor("PCB", warm_co2e_kg_per_kg=1.5, recyclable=True),
}

MATERIAL_FACTORS = {
    Material.GOLD: MaterialFactor(Material.GOLD, 200000.0, 12000.0, 100.0, True),
    Material.COPPER: MaterialFactor(Material.COPPER, 45.0, 3.0, 0.01, True),
}


def test_money_arithmetic():
    assert (Money(1.5) + Money(2.0)).amount == pytest.approx(3.5)
    assert (Money(2.0) * 3).amount == pytest.approx(6.0)
    assert (3 * Money(2.0)).amount == pytest.approx(6.0)
    assert Money.zero().amount == 0.0
    with pytest.raises(ValueError):
        Money(1.0, "USD") + Money(1.0, "EUR")


def test_aggregate_groups_and_filters_by_confidence():
    dets = [
        Detection("PCB", 0.9),
        Detection("PCB", 0.8),
        Detection("Mobile", 0.2),  # below floor
    ]
    items = services.aggregate(dets, min_confidence=0.5)
    counts = {i.waste_class: i.count for i in items}
    assert counts == {"PCB": 2}


def test_recovered_grams_scales_with_count():
    items = services.aggregate([Detection("PCB", 1.0), Detection("PCB", 1.0)])
    grams = services.recovered_grams(items, COMPOSITIONS)
    assert grams[Material.GOLD] == pytest.approx(0.2)   # 0.1 g/device * 2
    assert grams[Material.COPPER] == pytest.approx(40.0)  # 20 g/device * 2


def test_valuation_total_and_breakdown():
    items = [services_item("PCB", 2)]
    val = services.valuation(items, COMPOSITIONS, PRICES)
    assert val.total.amount == pytest.approx(20.40)  # 0.2g*100 + 40g*0.01
    assert val.per_material[Material.GOLD].value.amount == pytest.approx(20.0)
    assert val.per_material[Material.COPPER].value.amount == pytest.approx(0.40)
    assert val.per_class["PCB"].amount == pytest.approx(20.40)


def test_unpriced_and_unknown_classes_contribute_zero():
    # Material with no price entry, and a class missing from compositions.
    items = [services_item("PCB", 1), services_item("Unobtainium", 5)]
    prices = {Material.GOLD: MetalPrice(Material.GOLD, 100.0)}  # no copper price
    val = services.valuation(items, COMPOSITIONS, prices)
    # only gold priced: 0.1g * 100 = 10.0; copper present but unpriced -> 0
    assert val.total.amount == pytest.approx(10.0)
    assert val.per_material[Material.COPPER].value.amount == 0.0


def test_impact_warm_and_weee_lca():
    items = [services_item("PCB", 2)]
    imp = services.impact(items, COMPOSITIONS, CLASS_FACTORS, MATERIAL_FACTORS, PRICES)
    assert imp.mass_kg == pytest.approx(0.2)  # 0.1 kg/device * 2
    # WARM: 0.2 kg * 1.5
    assert imp.co2e_kg_avoided == pytest.approx(0.30)
    # WEEE-LCA energy: gold 0.0002kg*200000 + copper 0.04kg*45 = 40 + 1.8
    assert imp.energy_mj_saved == pytest.approx(41.8)
    # WEEE-LCA co2: gold 0.0002*12000 + copper 0.04*3 = 2.4 + 0.12
    assert imp.lca_co2e_kg_avoided == pytest.approx(2.52)
    # UNEP/ITU value mirrors the valuation total
    assert imp.recoverable_value.amount == pytest.approx(20.40)


def test_non_recyclable_class_has_no_warm_benefit():
    comps = {
        "Trash": MaterialComposition("Trash", 50.0, {Material.OTHER: 1.0}),
    }
    factors = {"Trash": ImpactFactor("Trash", 0.0, recyclable=False)}
    items = [services_item("Trash", 4)]
    imp = services.impact(items, comps, factors, MATERIAL_FACTORS, PRICES)
    assert imp.co2e_kg_avoided == 0.0
    assert imp.mass_kg == pytest.approx(0.2)


def test_haul_methods_delegate_to_services():
    haul = Haul(
        detections=[Detection("PCB", 0.9), Detection("PCB", 0.4)],
        min_confidence=0.5,
    )
    assert [(i.waste_class, i.count) for i in haul.items()] == [("PCB", 1)]
    val = haul.valuation(COMPOSITIONS, PRICES)
    assert val.total.amount == pytest.approx(10.20)  # one PCB
    imp = haul.impact(COMPOSITIONS, CLASS_FACTORS, MATERIAL_FACTORS, PRICES)
    assert imp.recoverable_value.amount == pytest.approx(10.20)


# helper -------------------------------------------------------------------


def services_item(waste_class: str, count: int):
    """Build a DetectedItem without importing the class into every test."""
    from ewaste.domain.model import DetectedItem

    return DetectedItem(waste_class, count)
