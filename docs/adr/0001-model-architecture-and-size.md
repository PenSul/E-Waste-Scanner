# 0001 — YOLO model architecture and size

Status: accepted
Date: 2026-06-18

## Context

The detector must run on **Streamlit Community Cloud**: one container, ~1 GB RAM,
CPU-only, no GPU. The weights are committed to a public repo and loaded lazily at
first request. We therefore need an architecture that (a) is accurate enough to
find e-waste and mixed materials in cluttered upload photos across our 17-class
taxonomy, yet (b) loads and infers within the memory and latency budget of a
free CPU container.

Candidate family: **Ultralytics YOLO11**, which ships pretrained COCO
checkpoints at five sizes (n/s/m/l/x). Larger variants raise accuracy but also
raise the parameter count, the on-disk weight size, the resident memory after
load, and per-image CPU latency. The two smallest variants are the only ones
that comfortably fit the runtime:

| Variant | Params | `best.pt` on disk | Relative CPU latency |
| ------- | ------ | ----------------- | -------------------- |
| YOLO11n | ~2.6 M | ~5–6 MB           | fastest              |
| YOLO11s | ~9.4 M | ~19 MB            | ~2–3× nano           |
| YOLO11m+| 20 M+  | 40 MB+            | exceeds budget       |

## Decision

**Deploy YOLO11s as the primary model; keep YOLO11n as a documented
out-of-memory fallback.**

YOLO11s is the accuracy/size sweet spot: 9.4 M parameters and a 19 MB checkpoint
load well inside the 1 GB container, while inference stays interactive on CPU
(~3 ms/image measured on the training GPU; CPU is slower but acceptable for an
upload-and-wait flow). It is trained by `scripts/train.py` (default
`--model yolo11s.pt`), evaluated and exported to `models/best.pt` by
`scripts/evaluate.py`, and loaded by `YoloDetector`.

YOLO11n exists for the case where the container OOMs under load: the same
training and export scripts produce it via `--model yolo11n.pt`, and swapping the
deployed weights is a one-file change with no code change, because `YoloDetector`
reads class names and geometry from the checkpoint itself. It is trained,
evaluated, and committed to `models/best-nano.pt` (5.3 MB); deploy it by pointing
`EWASTE_WEIGHTS` at that file or replacing `models/best.pt` with it.

Training configuration (see `scripts/train.py` for the full set): fine-tune the
pretrained COCO checkpoint at `imgsz=640`, mosaic=1.0 and mixup=0.1 to synthesise
clutter from the largely single-object source crops, `close_mosaic=10` to sharpen
box fit near the end, and early stopping at `patience=20`. The production run
stopped at epoch 88 of a 100-epoch budget.

## Consequences

* **Measured accuracy.** On the held-out **test** split, YOLO11s reaches
  **mAP50 0.738 / mAP50-95 0.694** (validation split: mAP50 0.815 /
  mAP50-95 0.774). Strong classes include PCB, Washing Machine, Printer, and
  Mobile (AP50 > 0.91); the weakest are Metal (0.28), Plastic (0.42), and
  Paper (0.44), reflecting visual ambiguity and label overlap among the generic
  material classes rather than a capacity limit of the architecture.
* The 19 MB checkpoint is committed directly to the public repo (`models/best.pt`,
  intentionally git-tracked); no Git LFS or release-asset step is needed.
* Choosing `s` over `n` trades some headroom for accuracy. If the live container
  proves memory-tight, the nano fallback is the first lever, ahead of any code
  change — and it costs little accuracy: on the test split YOLO11n reaches
  **mAP50 0.730 / mAP50-95 0.687** (versus 0.738 / 0.694 for `s`) at roughly a
  third of the parameters and a 5.3 MB checkpoint. It was trained on the same
  data and early-stopped once it had converged.
* Accuracy on the generic material classes is data-bound, not model-bound:
  improving those rows (more cluttered real-scene labels, cleaner class
  boundaries) is higher-leverage than scaling the model up — which the runtime
  cannot afford anyway.
* The deployed weights carry their own class map, so the model's class order can
  change without touching the reference CSVs, which are keyed by canonical class
  name (see [0002](0002-valuation-and-impact-methodology.md)).
