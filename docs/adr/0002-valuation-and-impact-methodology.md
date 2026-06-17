# 0002 — Valuation and environmental-impact methodology

Status: accepted
Date: 2026-06-18

## Context

From a single 2-D photo the detector yields, per object, only a **class and a
confidence** — never a mass, an alloy, or an assay. Yet the app must report a
"Recoverable Value" and an "Environmental Impact". We therefore need a defensible
way to go from *counts of typical devices* to dollars and to avoided emissions,
and we must be explicit that the result is an estimate.

We also have three different audiences for the impact number (economic,
climate, and energy), and the published methods for each disagree on units and
conventions.

## Decision

Model the bridge as **typical-device profiles times detected count**, computed by
a pure domain core (`ewaste.domain.model` + `ewaste.domain.services`) fed by
cited reference tables in `reference/` (see `reference/SOURCES.md`).

Three methods, each with its own reference table:

1. **Recoverable value (UNEP/ITU basis).** Per class, a typical device mass and
   the mass fraction of each material (`materials_composition.csv`). Recovered
   grams are multiplied by live metals prices (USD/gram) to give the Current
   Haul Value. Only the seven commodity metals (Au, Ag, Pd, Pt, Cu, Al, ferrous)
   carry a price; plastics/glass/other carry mass but no market value.

2. **EPA WARM CO2e avoided.** Per class, the net CO2e avoided by recycling vs
   landfilling, in kg CO2e per kg (`impact_factors.csv`), applied to device mass.
   This is WARM's comparative convention (recycle minus landfill).

3. **WEEE-LCA energy and CO2 avoided.** Bottom-up: recovered grams of each
   material times the primary-production burden avoided per kg
   (`material_factors.csv`, from Nuss & Eckelman 2014 and industry LCI).

Supporting decisions:

* **Money is float, not Decimal.** Every figure is an estimate; exact decimal
  arithmetic would imply false precision. Round only at display time.
* **Reference data lives in CSVs, not code**, keyed by the canonical class names
  in `reference/class_map.yaml` (not by integer class id), so a future re-order
  of the model's classes cannot silently misalign value/impact rows.
* **The domain core does no I/O.** Adapters load prices and tables and pass plain
  domain objects in, keeping the maths unit-testable without mocks.

## Consequences

* Numbers are coarse, order-of-magnitude estimates. The UI, `SOURCES.md`, and
  this ADR must state that prominently; outputs are for awareness/triage, not a
  settlement-grade valuation or a certified LCA.
* Accuracy is bounded by the typical-device assumptions far more than by the
  detector, so improving the CSVs (measured masses, better WARM mappings) is the
  highest-leverage future work.
* Adding a class means adding one row to each of the three CSVs; the
  `test_reference_data.py` suite enforces full coverage and valid ranges.
* Live pricing is the only runtime input to value; its provider must degrade to
  the cited static fallbacks in `material_factors.csv` (handled in Phase 4).
