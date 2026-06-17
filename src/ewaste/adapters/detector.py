"""Ultralytics YOLO11 object-detector adapter.

Wraps a trained YOLO11 model behind the :class:`ObjectDetector` port. The heavy
torch/Ultralytics model is loaded lazily on first use so importing this module
(and constructing the adapter) stays cheap — important for Streamlit Community
Cloud, where the model is wrapped in ``@st.cache_resource`` and loaded once.

The model's class names are expected to already be the canonical taxonomy
(training used ``data.yaml`` built from ``reference/class_map.yaml``), so no
remapping happens here.
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
        model_factory: Callable[[Path], Any] = _default_model_factory,
    ) -> None:
        self._weights = weights
        self._min_confidence = min_confidence
        self._imgsz = imgsz
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
        """
        self.load()
        results = self._model.predict(
            image,
            conf=self._min_confidence,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )
        names = self._model.names
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        waste_class=names[cls_id],
                        confidence=confidence,
                        bbox=xyxy,  # type: ignore[arg-type]
                    )
                )
        return detections
