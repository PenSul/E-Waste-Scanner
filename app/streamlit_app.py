"""E-Waste Scanner - Streamlit entrypoint.

Upload a photo of a junk pile or drawer; the app detects e-waste and mixed
materials, estimates recoverable value and environmental impact, and issues a
Digital Receipt. This module stays thin: it builds the wired service once,
collects the upload, and delegates rendering to ``app/components``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image

# Ensure both the app components and the src package are importable when run via
# `streamlit run app/streamlit_app.py`. Locally the project is pip-installed
# (editable, src layout) so `ewaste` resolves anyway, but on Streamlit Community
# Cloud only the third-party requirements are installed and the repo code is just
# cloned -- so `src/` must be put on the path explicitly here.
_APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_APP_DIR))
sys.path.insert(0, str(_APP_DIR.parent / "src"))

from components import dashboard, overlay  # noqa: E402

from ewaste.bootstrap import build  # noqa: E402
from ewaste.service_layer.receipt import DISCLAIMER  # noqa: E402
from ewaste.service_layer.scan_service import ScanService  # noqa: E402


@st.cache_resource
def get_service() -> ScanService:
    """Build and cache the wired service (and lazily-loaded model) once."""
    return ScanService(build())


def main() -> None:
    """Render the page."""
    st.set_page_config(page_title="E-Waste Scanner", layout="wide")
    st.title("E-Waste Scanner")
    st.caption(
        "Upload a photo of a junk pile or drawer to estimate its recoverable "
        "material value and environmental impact."
    )

    service = get_service()
    settings = service._c.settings  # noqa: SLF001 (entrypoint reads its own wiring)

    res_options = [640, 960, 1280, 1536, 2048]
    tile_options = [512, 640, 1024, 1536, 2048]
    with st.sidebar:
        st.header("Settings")
        min_conf = st.slider(
            "Detection confidence",
            min_value=0.05,
            max_value=0.90,
            value=float(settings.min_confidence),
            step=0.05,
            help="Minimum confidence for a detection to be counted. Lower it for "
            "dense piles to trade precision for recall.",
        )
        imgsz = st.select_slider(
            "Inference resolution",
            options=res_options,
            value=settings.imgsz if settings.imgsz in res_options else 1280,
            help="Higher resolution finds smaller objects but is slower and uses "
            "more memory.",
        )
        tiled = st.checkbox(
            "Tiled inference (dense piles)",
            value=settings.tiled,
            help="Slice the image into overlapping windows, detect in each, and "
            "merge. Greatly raises recall on small objects in clutter, at the "
            "cost of running inference once per tile.",
        )
        tile_size = settings.tile_size
        if tiled:
            tile_size = st.select_slider(
                "Tile size (px)",
                options=tile_options,
                value=settings.tile_size if settings.tile_size in tile_options else 1024,
                help="Smaller tiles magnify small objects more; larger tiles keep "
                "more context. A tile at or above the image size is one "
                "high-resolution pass.",
            )
        st.caption(f"Model weights: {settings.weights_path.name}")

    # Apply the UI inference choices to the cached detector before scanning. The
    # detector reads these attributes at detection time, so mutating them here is
    # the same per-call override pattern the scan service uses for confidence.
    detector = service._c.detector  # noqa: SLF001 (entrypoint reads its own wiring)
    for attr, value in (("imgsz", imgsz), ("tiled", tiled), ("tile_size", tile_size)):
        if hasattr(detector, attr):
            setattr(detector, attr, value)

    weights_ready = settings.weights_path.exists()
    if not weights_ready:
        st.warning(
            f"Model weights not found at `{settings.weights_path}`. Train and "
            "export a model (scripts/train.py then scripts/evaluate.py) before "
            "scanning."
        )

    upload = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "webp", "bmp"]
    )
    if upload is None:
        st.info("Upload a photo to begin.")
        return

    image = Image.open(upload)

    if not weights_ready:
        st.image(image, caption="Uploaded image", use_container_width=True)
        return

    try:
        with st.spinner("Detecting and valuing..."):
            result = service.scan(image, min_confidence=min_conf)
    except Exception as exc:  # surface model/runtime errors to the user
        st.error(f"Scan failed: {exc}")
        return

    left, right = st.columns([3, 2])
    with left:
        st.image(
            overlay.draw_detections(image, result.detections),
            caption=f"{result.total_count} item(s) detected",
            use_container_width=True,
        )
    with right:
        dashboard.render_summary(result)

    dashboard.render_line_items(result)
    dashboard.render_market(result.prices)
    dashboard.render_downloads(result)

    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
