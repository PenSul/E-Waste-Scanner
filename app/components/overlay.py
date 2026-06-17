"""Draw detection bounding boxes and labels onto an uploaded image."""

from __future__ import annotations

from collections.abc import Iterable

from PIL import Image, ImageDraw, ImageFont

from ewaste.domain.model import Detection

#: A fixed, high-contrast palette; classes map to a colour by stable hash.
_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9",
]


def _colour_for(waste_class: str) -> str:
    """Return a stable colour for a class name."""
    return _PALETTE[hash(waste_class) % len(_PALETTE)]


def _font() -> ImageFont.ImageFont:
    """Load a legible font, falling back to PIL's bitmap default."""
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: Iterable[Detection]) -> Image.Image:
    """Return a copy of ``image`` with each detection's box and label drawn.

    Detections without a box (e.g. synthetic ones) are skipped. Line and label
    sizes scale gently with the image so overlays stay readable on large photos.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()
    width = max(2, round(min(canvas.size) / 300))

    for det in detections:
        if det.bbox is None:
            continue
        x1, y1, x2, y2 = det.bbox
        colour = _colour_for(det.waste_class)
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=width)

        label = f"{det.waste_class} {det.confidence:.0%}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        tw, th = right - left, bottom - top
        ly = max(0, y1 - th - 4)
        draw.rectangle((x1, ly, x1 + tw + 6, ly + th + 4), fill=colour)
        draw.text((x1 + 3, ly + 2), label, fill="white", font=font)

    return canvas
