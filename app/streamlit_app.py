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
# `streamlit run app/streamlit_app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

    with st.sidebar:
        st.header("Settings")
        min_conf = st.slider(
            "Detection confidence",
            min_value=0.05,
            max_value=0.90,
            value=float(settings.min_confidence),
            step=0.05,
            help="Minimum confidence for a detection to be counted.",
        )
        st.caption(f"Model weights: {settings.weights_path.name}")

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
