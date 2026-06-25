"""Ultralytics YOLO11 object-detector adapter.

Wraps a trained YOLO11 model behind the :class:`ObjectDetector` port. The heavy
torch/Ultralytics model is loaded lazily on first use so importing this module
(and constructing the adapter) stays cheap — important for Streamlit Community
Cloud, where the model is wrapped in ``@st.cache_resource`` and loaded once.

The model's class names are expected to already be the canonical taxonomy
(training used ``data.yaml`` built from ``reference/class_map.yaml``), so no
remapping happens here.

Optional **tiled (SAHI-style) inference** slices a large image into overlapping
windows, detects within each, offsets the boxes back to full-image coordinates,
adds a full-image pass, and merges everything with class-aware non-max
suppression. Because the detector is trained mostly on single, centred objects,
slicing turns a tiny object buried in a pile into a large, centred object inside
its tile — the regime the model handles best — which sharply raises recall on
small items at the cost of running inference once per tile. It is off by default.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ewaste.domain.model import Detection
from ewaste.ports import ObjectDetector


def _default_model_factory(weights: Path) -> Any:
    """Load an Ultralytics YOLO model from a weights file."""
    from ultralytics import YOLO

    return YOLO(str(weights))


def _slice_boxes(
    width: int, height: int, tile: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Return ``(x0, y0, x1, y1)`` windows tiling a ``width`` x ``height`` image.

    Windows are ``tile`` pixels square with ``overlap`` fractional overlap; the
    final row/column is shifted back to the edge so coverage is complete. An
    image already no larger than a tile yields a single full-frame window.
    """
    if width <= tile and height <= tile:
        return [(0, 0, width, height)]
    step = max(1, int(round(tile * (1.0 - overlap))))

    def starts(total: int) -> list[int]:
        if total <= tile:
            return [0]
        points = list(range(0, total - tile + 1, step))
        if points[-1] != total - tile:
            points.append(total - tile)
        return points

    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in starts(height)
        for x in starts(width)
    ]


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _merge(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Class-aware greedy NMS over detections gathered from many tiles.

    Boxes of *different* classes never suppress each other (a battery resting on
    a keyboard is two valid detections); within a class, a lower-confidence box
    that overlaps a kept box by more than ``iou_threshold`` is dropped.
    """
    kept: list[Detection] = []
    for cls in {d.waste_class for d in detections}:
        group = sorted(
            (d for d in detections if d.waste_class == cls),
            key=lambda d: d.confidence,
            reverse=True,
        )
        cls_kept: list[Detection] = []
        for det in group:
            if det.bbox is None or all(
                _iou(det.bbox, k.bbox) < iou_threshold
                for k in cls_kept
                if k.bbox is not None
            ):
                cls_kept.append(det)
        kept.extend(cls_kept)
    return kept


class YoloDetector(ObjectDetector):
    """Detects waste objects with a trained YOLO11 model.

    ``model_factory`` is injectable so tests can supply a stub model with
    ``predict`` and ``names`` instead of loading real weights.
    """

    def __init__(
        self,
        weights: Path,
        *,
        min_confidence: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
        tiled: bool = False,
        tile_size: int = 640,
        tile_overlap: float = 0.2,
        tile_iou: float = 0.45,
        model_factory: Callable[[Path], Any] = _default_model_factory,
    ) -> None:
        self._weights = weights
        # Public, runtime-tunable knobs: the UI mutates these per scan (the same
        # pattern the scan service uses for ``min_confidence``).
        self.min_confidence = min_confidence
        self.imgsz = imgsz
        self.tiled = tiled
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_iou = tile_iou
        self._device = device
        self._model_factory = model_factory
        self._model: Any | None = None

    def load(self) -> None:
        """Eagerly load the model (otherwise loaded on first :meth:`detect`)."""
        if self._model is None:
            self._model = self._model_factory(self._weights)

    def detect(self, image: Any) -> list[Detection]:
        """Run detection on ``image`` and return canonical-class detections.

        ``image`` may be anything Ultralytics accepts (path, PIL image, numpy
        array). Detections below ``min_confidence`` are filtered by the model.
        When tiled inference is enabled the image is sliced into overlapping
        windows and the merged detections are returned instead.
        """
        self.load()
        if self.tiled:
            return self._detect_tiled(image)
        return self._predict(image, self.imgsz)

    def _predict(
        self, image: Any, imgsz: int, offset: tuple[float, float] = (0.0, 0.0)
    ) -> list[Detection]:
        """Run one inference pass and map boxes to :class:`Detection` objects.

        ``offset`` is added to each box so detections from a cropped tile land in
        full-image coordinates.
        """
        results = self._model.predict(
            image,
            conf=self.min_confidence,
            imgsz=imgsz,
            device=self._device,
            verbose=False,
        )
        names = self._model.names
        ox, oy = offset
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        waste_class=names[cls_id],
                        confidence=confidence,
                        bbox=(x1 + ox, y1 + oy, x2 + ox, y2 + oy),
                    )
                )
        return detections

    def _detect_tiled(self, image: Any) -> list[Detection]:
        """Slice the image, detect per tile plus a full pass, and merge (NMS)."""
        pil = _to_pil(image)
        width, height = pil.size
        detections: list[Detection] = []
        for x0, y0, x1, y1 in _slice_boxes(
            width, height, self.tile_size, self.tile_overlap
        ):
            tile = pil.crop((x0, y0, x1, y1))
            detections.extend(
                self._predict(tile, self.tile_size, offset=(float(x0), float(y0)))
            )
        # A full-frame pass recovers large objects that span several tiles.
        detections.extend(self._predict(pil, self.imgsz))
        return _merge(detections, self.tile_iou)


def _to_pil(image: Any) -> Any:
    """Coerce ``image`` to an RGB PIL image for slicing.

    Accepts a PIL image, a path/string, or an RGB numpy array. Imported lazily
    so the module stays cheap to import where tiling is unused.
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    return Image.fromarray(image).convert("RGB")
