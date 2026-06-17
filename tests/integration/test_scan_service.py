"""Integration tests: the full detect -> value -> impact -> receipt pipeline.

A fake detector and an offline market provider make the pipeline deterministic
while exercising the real CSV repositories and rendering.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from ewaste.adapters.market import YFinanceMarketProvider
from ewaste.adapters.repository import fallback_prices
from ewaste.bootstrap import build
from ewaste.config import Settings
from ewaste.domain.model import Detection
from ewaste.ports import ObjectDetector
from ewaste.service_layer import receipt
from ewaste.service_layer.scan_service import ScanService

FIXED_TIME = datetime(2026, 6, 18, 12, 0, 0, tzinfo=timezone.utc)


class FakeDetector(ObjectDetector):
    """Returns a canned set of detections regardless of the image."""

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def detect(self, image):  # noqa: ANN001
        return list(self._detections)


def _service(detections: list[Detection]) -> ScanService:
    container = build(Settings())
    offline_market = YFinanceMarketProvider(
        fallback_prices(container.impacts.material_factors()),
        fetcher=lambda _sym: None,  # force the static fallback, no network
    )
    container = dataclasses.replace(
        container, detector=FakeDetector(detections), market=offline_market
    )
    return ScanService(container)


DETECTIONS = [
    Detection("PCB", 0.91, (0, 0, 10, 10)),
    Detection("Mobile", 0.80, (10, 10, 20, 20)),
    Detection("PCB", 0.70, (20, 20, 30, 30)),
    Detection("Glubbo", 0.95, (30, 30, 40, 40)),  # not in reference data
]


def test_scan_aggregates_values_and_flags_unknowns():
    result = _service(DETECTIONS).scan("junk.jpg")
    counts = {line.waste_class: line.count for line in result.lines}
    assert counts == {"Glubbo": 1, "Mobile": 1, "PCB": 2}
    assert result.total_count == 4
    assert result.unknown_classes == ["Glubbo"]
    # PCB + Mobile carry priced metals -> positive recoverable value
    assert result.total_value.amount > 0
    # the unknown class is counted but contributes no mass or value
    glubbo = next(line for line in result.lines if line.waste_class == "Glubbo")
    assert glubbo.mass_kg == 0.0
    assert glubbo.value.amount == 0.0
    # impact populated under all three methods
    assert result.impact.mass_kg > 0
    assert result.impact.energy_mj_saved > 0


def test_per_class_value_sums_to_total():
    result = _service(DETECTIONS).scan("junk.jpg")
    summed = sum(line.value.amount for line in result.lines)
    assert summed == pytest.approx(result.total_value.amount)


def test_receipt_csv_is_parseable_and_complete():
    result = _service(DETECTIONS).scan("junk.jpg")
    text = receipt.to_csv(result, generated_at=FIXED_TIME)
    assert "E-Waste Scanner receipt" in text
    assert "2026-06-18T12:00:00" in text
    assert "recoverable_value_usd" in text
    assert "PCB" in text
    assert "Glubbo" in text  # unknown still listed


def test_receipt_html_has_disclaimer_and_no_emoji():
    result = _service(DETECTIONS).scan("junk.jpg")
    page = receipt.to_html(result, generated_at=FIXED_TIME)
    assert "<!doctype html>" in page
    assert "Estimates only" in page
    assert "Recoverable value:" in page
    assert "No reference composition for: Glubbo" in page
    assert page.isascii()  # emoji-free, deployment-safe


def test_summary_metrics_match_result():
    result = _service(DETECTIONS).scan("junk.jpg")
    metrics = receipt.summary_metrics(result)
    assert metrics["items"] == 4
    assert metrics["recoverable_value_usd"] == pytest.approx(
        round(result.total_value.amount, 2)
    )
    assert metrics["mass_kg"] == pytest.approx(round(result.impact.mass_kg, 3))
