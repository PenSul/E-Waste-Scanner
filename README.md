# E-Waste Scanner

https://github.com/user-attachments/assets/25b3ffb6-f486-4960-af52-62080bb56956

Upload a photo of a junk pile, drawer, or messy desk. The application detects
electronic waste and mixed materials with a YOLO11 object detector, draws
bounding boxes, and estimates two things for the haul: the **recoverable
material value** (live metals pricing applied to per-device material
composition) and the **environmental impact** of recycling it rather than
landfilling. The result is shown as a dashboard and offered as a downloadable
digital receipt.

This is an awareness and triage tool. Every number is a coarse,
order-of-magnitude estimate derived from typical-device assumptions, not a
settlement-grade valuation or a certified life-cycle assessment. See
[Estimation methodology](#estimation-methodology) and `reference/SOURCES.md`.

## What it does

- Detects 17 classes of e-waste and mixed materials in a single uploaded image
  (no live camera; upload only).
- Estimates **Current Haul Value** from live commodity metals prices
  (`yfinance`) with a cited static fallback when the network is unavailable.
- Reports environmental impact through three published methods: UNEP/ITU
  recoverable value, EPA WARM avoided CO2e, and a WEEE-LCA energy/CO2 estimate.
- Generates a digital receipt (HTML and CSV) with the per-class breakdown, the
  totals, and the assumptions behind them.

## How it works

```
image
  -> YoloDetector.detect()        # YOLO11s, lazy-loaded
  -> [Detection]                  # class, confidence, box
  -> Haul (domain aggregate)
  -> MaterialsRepository          # typical mass + material fractions per class
  -> MarketPriceProvider          # live USD/gram, with fallback
  -> Valuation + ImpactEstimate   # pure domain calculation, no I/O
  -> ScanResult -> Receipt (HTML / CSV)
```

The codebase follows a ports-and-adapters (hexagonal) structure. The domain core
(`src/ewaste/domain/`) is pure and does no I/O; all I/O lives in adapters
(`src/ewaste/adapters/`) behind abstract ports (`src/ewaste/ports.py`), wired
together by `src/ewaste/bootstrap.py`. This keeps the valuation and impact maths
unit-testable without a model, a network, or any mocks.

## Model and accuracy

The deployed detector is **YOLO11s** (Ultralytics), fine-tuned for 50 epochs at
960px on a dataset whose cluttered, multi-object scenes are oversampled to close
the single-object-to-pile domain gap. On the held-out test split it reaches
**mAP50 0.947 / mAP50-95 0.913**.

Dense piles remain the hard case: most training images are single, centred
objects, so recall on small items buried in clutter is limited. Tiled inference
(see [Configuration](#configuration)) mitigates this, and the durable improvement
is more varied, fully-labelled pile photos.

Weights are committed to the repository and loaded lazily on first request:
`models/best.pt` is the deployed YOLO11s (about 19 MB). `models/best-nano.pt` is
the checkpoint the memory-constrained cloud demo loads; it is currently the same
YOLO11s weights rather than a separate smaller model.

## Estimation methodology

A 2-D photo yields a class and a count per object, never a mass or an assay. The
app bridges counts to dollars and emissions with typical-device profiles times
the detected count, fed by cited reference tables in `reference/`:

1. **Recoverable value (UNEP/ITU basis).** Per class, a typical device mass and
   the mass fraction of each recoverable material. Recovered grams of the priced
   metals (gold, silver, palladium, platinum, copper, aluminium, ferrous) are
   multiplied by live prices to give the Current Haul Value.
2. **EPA WARM CO2e avoided.** The net CO2e avoided by recycling versus
   landfilling, per kg, applied to device mass (WARM's comparative convention).
3. **WEEE-LCA energy and CO2 avoided.** A bottom-up estimate from recovered
   grams times the primary-production burden avoided per kg.

Every figure in the reference tables is cited in `reference/SOURCES.md`.

## Project structure

```
src/ewaste/
  domain/        # pure value objects, aggregate (Haul), and calculation services
  adapters/      # YoloDetector, YFinanceMarketProvider, CSV repositories
  service_layer/ # scan orchestration and receipt rendering
  ports.py       # abstract ports (ABCs)
  config.py      # settings, env-overridable
  bootstrap.py   # dependency-injection wiring
app/
  streamlit_app.py     # thin Streamlit entrypoint (upload -> scan -> render)
  components/           # dashboard, bounding-box overlay
  requirements.txt     # CPU-only dependency pin for Streamlit Community Cloud
reference/       # class_map.yaml + cited composition / impact CSVs + SOURCES.md
scripts/         # prepare_data.py, train.py, evaluate.py
models/          # best.pt (deployed weights, tracked in git)
tests/           # unit + integration tests
```

## Local development

The project is managed with [uv](https://docs.astral.sh/uv/) and pinned to
Python 3.12. PyTorch is split into mutually exclusive `cpu` and `gpu` extras;
always sync or run with exactly one of them.

```bash
# Tests, CI, and running the app locally (CPU PyTorch):
uv sync --extra cpu

# Local GPU training (CUDA 13.x PyTorch):
uv sync --extra gpu

# Run the app:
uv run --extra cpu streamlit run app/streamlit_app.py
```

## Training and evaluation

```bash
# (Optional) oversample cluttered pile scenes to weight them in training:
uv run --extra cpu python scripts/oversample_clutter.py --multiplier 30

# Fine-tune YOLO11s on the prepared dataset (expects data.yaml from prepare_data.py):
uv run --extra gpu python scripts/train.py --model yolo11s.pt --epochs 50 --imgsz 960

# Evaluate on the test split and export deployment weights to models/best.pt:
uv run --extra gpu python scripts/evaluate.py \
    --weights runs/detect/ewaste-yolo11s/weights/best.pt --split test --imgsz 960
```

## Testing

```bash
uv run --extra cpu python -m pytest
```

The suite covers the domain maths, the adapters (with the network and the model
faked), the full scan pipeline, and a headless render of the Streamlit page via
`streamlit.testing.v1.AppTest`.

## Try it online

A hosted demo runs on Streamlit Community Cloud's free tier:
**https://e-waste-scanner-aaf8pq3fnphvhbbhxjdydy.streamlit.app/**

The free tier is tightly memory-constrained, so the hosted demo can be slow or
briefly unavailable under load and may run at a reduced inference resolution.
Treat it as a rough preview — run it locally (see above) for the best results,
where you can raise the resolution and enable tiled inference.

## Configuration

Settings have sensible defaults and can be overridden by environment variables:
`EWASTE_WEIGHTS` (path to the model checkpoint), `EWASTE_REFERENCE_DIR`,
`EWASTE_MIN_CONFIDENCE`, `EWASTE_IMGSZ` (inference resolution; match it to the
training image size), `EWASTE_DEVICE`, and `EWASTE_PRICE_TTL`. See
`src/ewaste/config.py`.

For dense piles, tiled (SAHI-style) inference raises recall on small objects by
slicing the image into overlapping windows, detecting in each, and merging:
`EWASTE_TILED` (`1` to enable; off by default), `EWASTE_TILE_SIZE` (window size
in pixels, default 640), `EWASTE_TILE_OVERLAP` (fractional overlap, default 0.2),
and `EWASTE_TILE_IOU` (merge threshold, default 0.45). It runs inference once per
tile, so it is intended for local use, not the memory-constrained cloud demo. The
Streamlit sidebar also exposes the inference resolution and the tiling toggle, so
they can be tuned per session without setting environment variables.

## Further reading

- `reference/SOURCES.md` — citations for every composition and impact figure.
