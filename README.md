# E-Waste Scanner

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

The deployed detector is **YOLO11s** (Ultralytics), fine-tuned for 88 epochs
(early-stopped from a 100-epoch budget). On the held-out test split it reaches
**mAP50 0.738 / mAP50-95 0.694**. A smaller **YOLO11n** out-of-memory fallback is
also trained and committed (**mAP50 0.730 / mAP50-95 0.687** at a third the
parameters). The rationale and full per-class results are recorded in
[ADR-0001](docs/adr/0001-model-architecture-and-size.md).

Weights are committed to the repository and loaded lazily on first request:
`models/best.pt` (the deployed YOLO11s, about 19 MB) and `models/best-nano.pt`
(the YOLO11n fallback, about 5 MB).

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

The full reasoning is in [ADR-0002](docs/adr/0002-valuation-and-impact-methodology.md),
and every figure in the reference tables is cited in `reference/SOURCES.md`.

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
docs/adr/        # architecture decision records
tests/           # unit + integration tests
CONTEXT.md       # bounded-context and ubiquitous-language reference
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
# Fine-tune YOLO11s on the prepared dataset (expects data.yaml from prepare_data.py):
uv run --extra gpu python scripts/train.py --model yolo11s.pt --epochs 100

# Train the nano fallback:
uv run --extra gpu python scripts/train.py --model yolo11n.pt --name ewaste-yolo11n

# Evaluate on the test split and export deployment weights to models/best.pt:
uv run --extra gpu python scripts/evaluate.py \
    --weights runs/detect/ewaste-yolo11s/weights/best.pt --split test
```

Data preparation, labeling, and the Roboflow workflow are documented in
`docs/labeling-guide.md`.

## Testing

```bash
uv run --extra cpu python -m pytest
```

The suite covers the domain maths, the adapters (with the network and the model
faked), the full scan pipeline, and a headless render of the Streamlit page via
`streamlit.testing.v1.AppTest`.

## Deployment (Streamlit Community Cloud)

Community Cloud provides roughly 1 GB of RAM and is CPU-only, so the deployment
must use the CPU build of PyTorch. The CUDA build that PyPI ships for Linux would
risk running out of memory.

Two files configure the deployment:

- `app/requirements.txt` pins the Python dependencies, including the CPU-only
  torch and torchvision wheels referenced by direct URL. Community Cloud searches
  the entrypoint's directory before the repository root and uses the first
  dependency file it finds, so this file is used in preference to the root
  `uv.lock` (whose torch lives behind the `cpu`/`gpu` extras and has no default).
  It resolves identically under uv and pip.
- `packages.txt` (repo root) installs the system libraries that ultralytics'
  opencv dependency needs at import time and that the container does not ship by
  default: `libgl1` (for `libGL.so.1`) and `libglib2.0-0t64` (for
  `libgthread-2.0.so.0`). The `t64` suffix matches Community Cloud's Debian
  trixie base image, where the old `libglib2.0-0` name is not installable.

To deploy:

1. Push the repository to GitHub (it must be public for the free tier).
2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at
   `app/streamlit_app.py` and select **Python 3.12** (the pinned torch wheels are
   built for cp312 / 64-bit Linux).
3. Community Cloud installs from `app/requirements.txt` plus `packages.txt` and
   serves the app.

If the container runs short of memory under load, swap to the committed nano
fallback: set `EWASTE_WEIGHTS` to `models/best-nano.pt`, or replace
`models/best.pt` with it. No code change is required, because the detector reads
its class map and geometry from the checkpoint itself.

## Configuration

Settings have sensible defaults and can be overridden by environment variables:
`EWASTE_WEIGHTS` (path to the model checkpoint), `EWASTE_REFERENCE_DIR`,
`EWASTE_MIN_CONFIDENCE`, `EWASTE_DEVICE`, and `EWASTE_PRICE_TTL`. See
`src/ewaste/config.py`.

## Further reading

- `CONTEXT.md` — bounded context and ubiquitous language.
- `docs/adr/` — architecture decision records (model, methodology, environment).
- `reference/SOURCES.md` — citations for every composition and impact figure.
