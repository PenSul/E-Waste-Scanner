"""Headless smoke test of the Streamlit entrypoint via AppTest.

Runs the page with no upload and (typically) no exported weights, asserting it
renders without error and degrades gracefully instead of crashing.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"


def test_app_renders_without_exception():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "E-Waste Scanner"
    # with no file uploaded, the page asks for one (info or weights warning)
    messages = [m.value for m in at.info] + [w.value for w in at.warning]
    assert any("Upload" in m or "weights" in m for m in messages)
