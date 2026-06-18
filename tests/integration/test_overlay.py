"""Tests for the detection overlay drawing."""

from __future__ import annotations

from PIL import Image

from app.components import overlay  # noqa: F401  (path set in conftest)
from ewaste.domain.model import Detection


def test_draw_detections_returns_modified_copy():
    base = Image.new("RGB", (200, 150), "white")
    dets = [Detection("PCB", 0.9, (10, 10, 90, 90))]
    out = overlay.draw_detections(base, dets)
    assert out.size == base.size
    assert out.mode == "RGB"
    assert out is not base
    # something was drawn: at least one pixel is no longer white
    assert out.getcolors(maxcolors=10_000) != base.getcolors(maxcolors=10_000)


def test_detections_without_box_are_skipped():
    base = Image.new("RGB", (50, 50), "white")
    out = overlay.draw_detections(base, [Detection("Mobile", 0.8, None)])
    # nothing to draw -> identical pixels
    assert out.tobytes() == base.tobytes()
