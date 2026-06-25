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


class _CountingModel:
    """Records how many windows it was asked to predict on."""

    names = {0: "PCB"}

    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, image, **kwargs):  # noqa: ANN001, D401
        self.predict_calls += 1
        # One detection near the centre of whatever window it is given.
        return [_FakeResult([_FakeBox(0, 0.80, [5.0, 5.0, 15.0, 15.0])])]


def test_tiled_inference_slices_and_offsets_boxes():
    from PIL import Image

    model = _CountingModel()
    detector = YoloDetector(
        Path("unused.pt"),
        tiled=True,
        tile_size=100,
        tile_overlap=0.0,
        model_factory=lambda _w: model,
    )
    # 200x100 image with 100px tiles, no overlap -> 2 tiles + 1 full pass = 3.
    image = Image.new("RGB", (200, 100))
    detections = detector.detect(image)

    assert model.predict_calls == 3  # 2 tiles + 1 full-frame pass
    # Tile 1 and the full-frame pass both yield a box at x=5 (identical, so NMS
    # merges them); tile 2's box is offset by +100. Two detections survive.
    xs = sorted(round(d.bbox[0]) for d in detections)
    assert xs == [5, 105]


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
