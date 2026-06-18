"""Integration tests: YOLO detector adapter, driven by a stub model.

A fake model factory stands in for Ultralytics so the adapter's result-mapping
and lazy-loading behaviour can be tested without torch or real weights.
"""

from __future__ import annotations

from pathlib import Path

from ewaste.adapters.detector import YoloDetector


class _FakeTensorCell:
    """Mimics ``box.cls[0]`` / ``box.conf[0]`` indexable single values."""

    def __init__(self, value: float) -> None:
        self._value = value

    def __getitem__(self, idx: int) -> float:
        return self._value


class _FakeXYXYRow:
    def __init__(self, coords: list[float]) -> None:
        self._coords = coords

    def __getitem__(self, idx: int) -> "_FakeXYXYRow":
        return self  # box.xyxy[0]

    def tolist(self) -> list[float]:
        return self._coords


class _FakeBox:
    def __init__(self, cls_id: int, conf: float, coords: list[float]) -> None:
        self.cls = _FakeTensorCell(cls_id)
        self.conf = _FakeTensorCell(conf)
        self.xyxy = _FakeXYXYRow(coords)


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class _FakeModel:
    names = {0: "PCB", 1: "Mobile"}

    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, image, **kwargs):  # noqa: ANN001, D401
        self.predict_calls += 1
        return [
            _FakeResult(
                [
                    _FakeBox(0, 0.91, [10.0, 20.0, 30.0, 40.0]),
                    _FakeBox(1, 0.55, [50.0, 60.0, 70.0, 80.0]),
                ]
            )
        ]


def _detector(model: _FakeModel) -> YoloDetector:
    return YoloDetector(Path("unused.pt"), model_factory=lambda _w: model)


def test_detect_maps_boxes_to_canonical_detections():
    model = _FakeModel()
    detections = _detector(model).detect("photo.jpg")
    assert [(d.waste_class, round(d.confidence, 2)) for d in detections] == [
        ("PCB", 0.91),
        ("Mobile", 0.55),
    ]
    assert detections[0].bbox == (10.0, 20.0, 30.0, 40.0)


def test_model_is_loaded_lazily_and_once():
    factory_calls: list[Path] = []

    def factory(weights: Path) -> _FakeModel:
        factory_calls.append(weights)
        return _FakeModel()

    detector = YoloDetector(Path("w.pt"), model_factory=factory)
    assert factory_calls == []  # not loaded on construction
    detector.detect("a.jpg")
    detector.detect("b.jpg")
    assert len(factory_calls) == 1  # loaded once, reused
