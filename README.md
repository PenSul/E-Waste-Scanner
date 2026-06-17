# E-Waste Scanner

Upload a photo of a junk pile, drawer, or messy desk. The application detects
electronic waste and mixed materials, draws bounding boxes, and estimates the
recoverable material value and the environmental impact of recycling the haul.
A digital receipt summarises the result.

> Status: under active development. This README is a placeholder and will be
> finalised once the application is feature-complete (Phase 7 of the plan).

## Quick start (development)

This project is managed with [uv](https://docs.astral.sh/uv/) and pinned to
Python 3.12.

```bash
# Local development / tests (CPU PyTorch):
uv sync --extra cpu

# Local GPU training (CUDA 13.x PyTorch):
uv sync --extra gpu

# Run the app:
uv run streamlit run app/streamlit_app.py
```

PyTorch is split into mutually exclusive `cpu` and `gpu` extras; always sync with
exactly one of them.
