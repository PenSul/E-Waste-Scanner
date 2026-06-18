# Labeling Guide (Roboflow) — E-Waste Scanner

This guide walks you through turning the **classification** dataset in
`data/balanced_waste_images/` into an **object-detection** dataset that YOLO11
can train on, and supplementing it with a real multi-object dataset so the model
works on cluttered photos.

Do this **before** training (Phase 2). The output is a Roboflow export that
`scripts/prepare_data.py` turns into the `data.yaml` the trainer reads.

---

## 0. Why this step exists

The Kaggle dataset has **one centered object per image, one class per folder** —
that is classification data, not detection data. Object detection needs a
**bounding box + class** on every object in every image.

Two gaps to close:

1. **No boxes yet.** Each Kaggle image needs at least one box.
2. **Single-object vs. clutter.** Real uploads are messy piles with many objects.
   We close this two ways:
   - **YOLO mosaic/mixup augmentation** (done automatically during training)
     stitches 4 images into one synthetic multi-object scene.
   - **A real multi-object e-waste dataset** from Roboflow Universe, merged in.

Our model only predicts the **17 canonical classes** in
`reference/class_map.yaml`. Keep every class name you create in Roboflow
identical to those 17 strings.

---

## 1. Create the Roboflow project

1. Sign in at <https://app.roboflow.com> and create a **New Project**.
2. Project Type: **Object Detection**.
3. Annotation Group: `waste`.
4. License: note the dataset licenses you combine (Kaggle terms + the CC-BY of
   any Universe dataset you add). Record them in the project description.

---

## 2. Get boxes onto the Kaggle images (fast)

You have ~6,785 images, so do **not** hand-draw every box. Use a bootstrap, then
review. Pick one path:

### Path A — Pre-annotate locally, then upload (recommended, fastest)

Because each Kaggle image is a single centered object, a near-full-frame box is a
good first label. Generate those boxes locally and upload images **with** their
labels, then just spot-check in Roboflow:

```bash
# Writes one YOLO box per image (a centered box covering ~92% of the frame),
# with the class taken from the folder name and mapped via class_map.yaml.
# Labels are written *in place* next to each image; images are not modified.
uv run --extra cpu python scripts/autobox_kaggle.py \
    --images-dir data/balanced_waste_images \
    --coverage 0.92
```

Then in Roboflow: **Upload** → drag the `data/balanced_waste_images` folder.
Roboflow detects the YOLO `.txt` annotations alongside the images (using the
generated `classes.txt`) and imports them as pre-labeled. Review a sample per
class and tighten any loose boxes.

> The near-full-frame box is intentional: combined with mosaic augmentation it
> teaches the model the object's appearance while synthetic scenes teach layout.
> Still, fix images where the object is small or off-center.

### Path B — Roboflow Auto Label / Label Assist

Upload the raw images (no labels), then use Roboflow's **Auto Label** (a
foundation model such as Grounding DINO) or **Label Assist** to propose boxes
from class prompts, and accept/correct them. Slower but higher quality on images
with background clutter. See <https://docs.roboflow.com/annotate/use-roboflow-annotate>.

### Class names

Whichever path, ensure the classes are exactly the 17 in
`reference/class_map.yaml` (e.g. `Mobile`, `PCB`, `Washing Machine`). Roboflow's
**Classes & Tags** page lets you rename/merge.

---

## 3. Add a real multi-object dataset (Roboflow Universe)

This is the supplement we agreed on, to handle clutter.

1. Browse Universe for an e-waste detection set, e.g.:
   - *Balanced E-Waste Dataset* (~7.2k images, 37 classes) —
     <https://universe.roboflow.com/electronic-waste-detection/balanced-e-waste-dataset>
   - *TRCProject E-waste detection model* (~1.7k images, 14 classes, CC-BY) —
     <https://universe.roboflow.com/trcproject/e-waste-detection-model>
2. Use **Clone / Add to Project** to bring its images+annotations into your
   project (or download in YOLO format and upload).
3. **Remap its classes to our taxonomy** on the Classes page, using the
   `aliases` table in `reference/class_map.yaml` (e.g. `Smart Phone` → `Mobile`,
   `Printed Circuit Board PCB` → `PCB`, `9V Battery` → `Battery`). **Delete or
   ignore** classes in the `ignore` list (Laptop, HDD, Router, …) — we have no
   value/impact data for them.

After merging, your project should contain **only the 17 canonical classes**.

---

## 4. Add a few real "junk pile" photos for validation

Take 20–50 photos of actual cluttered drawers/desks/piles with several items,
label them carefully, and make sure they land in **valid/test** (Step 5). These
are your honest measure of real-world performance — the metric that matters for
this app.

---

## 5. Generate a dataset version

On the **Generate** page:

- **Train/Valid/Test split**: 70 / 20 / 10. Put your real "junk pile" photos in
  valid/test.
- **Preprocessing**: Auto-Orient (on), Resize → **640×640** (Fit/Stretch).
- **Augmentation**: keep it light — Roboflow does *offline* augmentation, and
  YOLO already does mosaic/mixup/flip/HSV *online* during training. A little
  brightness/exposure and rotation is fine; **do not** double up on flips/mosaic.

Generate the version.

---

## 6. Export for training

- **Format**: `YOLOv11` (or `YOLOv8` — same layout: `data.yaml` + `train/valid/
  test` each with `images/` and `labels/`).
- Choose **"download zip to computer"**, unzip it into the (git-ignored)
  `datasets/` folder, e.g. `datasets/ewaste-v1/`.

> Tip: the "show download code" option gives a `roboflow` Python snippet you can
> paste into `scripts/prepare_data.py` to pull the export programmatically with
> your API key (store the key in `.env`, never commit it).

---

## 7. Normalise into the project's data.yaml

```bash
uv run --extra cpu python scripts/prepare_data.py \
    --export-dir datasets/ewaste-v1 \
    --out data.yaml --remap
```

This validates the export's classes against `reference/class_map.yaml`, rewrites
class ids into our canonical order when `--remap` is given, and writes a top-level
`data.yaml`. You are then ready for Phase 2 training.

---

## Checklist

- [ ] Roboflow Object-Detection project created
- [ ] Kaggle images boxed (Path A or B) with the 17 canonical class names
- [ ] A Roboflow Universe set merged and remapped to our taxonomy
- [ ] 20–50 real junk-pile photos labeled into valid/test
- [ ] Version generated (70/20/10, resize 640, light aug)
- [ ] Exported as YOLOv11 into `datasets/`
- [ ] `scripts/prepare_data.py` produced `data.yaml`
