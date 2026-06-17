# 0003 — Runtime environment and tooling

Status: accepted
Date: 2026-06-17

## Context

The project must train a YOLO model locally on an NVIDIA RTX 4070 (CUDA 13.2
toolkit installed) and also deploy a CPU-only inference app to Streamlit
Community Cloud (1 GB RAM, no GPU, public repository required).

The original stack target was Python 3.14.5. Research at project start found that
PyTorch does not yet publish reliable CUDA wheels for CPython 3.14 (cp314): users
report CPU-only fallback installs persisting into mid-2026. GPU training on 3.14
is therefore not dependable today.

PyTorch publishes per-accelerator builds on separate indexes
(`download.pytorch.org/whl/{cpu,cu130,...}`) using local version specifiers
(e.g. `2.x+cpu`, `2.x+cu130`). The default PyPI Linux wheel bundles a large CUDA
build that risks exceeding Community Cloud's 1 GB limit.

## Decision

* Pin the whole project to **Python 3.12**, managed by **uv**.
* Manage PyTorch through two mutually exclusive uv extras:
  * `cpu` -> `https://download.pytorch.org/whl/cpu` (tests, CI, deployment)
  * `gpu` -> `https://download.pytorch.org/whl/cu130` (local GPU training;
    CUDA 13.0 wheels are compatible with the installed 13.2 runtime)
* The extras are declared as a uv `conflicts` pair and routed via
  `[tool.uv.sources]` with `explicit = true` indexes, so neither index leaks into
  the resolution of other packages.
* The deployed app and CI always use `--extra cpu`. Local training uses
  `--extra gpu`.

## Consequences

* `uv sync` with no extra installs no PyTorch; one extra must always be chosen.
  This is documented in the README and CONTEXT.
* Reproducible, small CPU installs for Community Cloud; full GPU acceleration
  locally without a second project.
* If/when cp314 CUDA wheels stabilise, revisit the Python pin; the extras
  mechanism is independent of the interpreter version.
