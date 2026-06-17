# CONTEXT — E-Waste Scanner

Single bounded context. This document is the source of truth for the ubiquitous
language and the domain model. Architectural decisions are recorded as ADRs under
`docs/adr/`.

## Purpose

Given a user-uploaded photograph of mixed waste, detect electronic-waste and
material items, then estimate two things:

1. **Recoverable value** — the market value of the metals recoverable from the
   detected items.
2. **Environmental impact** — the benefit of recycling the haul rather than
   landfilling it, expressed through three published methods.

The result is presented as a live dashboard and a downloadable digital receipt.

## Ubiquitous language

| Term | Meaning |
| --- | --- |
| **Detection** | One bounding box produced by the model: a class label, a confidence score, and pixel coordinates. |
| **Detected item** | A detection promoted to the domain: a waste class plus a count. |
| **Waste class** | A canonical category in our taxonomy (see `reference/class_map.yaml`), e.g. `Mobile`, `PCB`, `Battery`. Roboflow/Kaggle source labels are mapped onto these. |
| **Material content** | For one unit of a waste class, the typical device mass and the mass of each recoverable material (copper, gold, silver, palladium, iron, aluminium, plastic, glass). |
| **Haul** | The complete set of detected items from a single scan. The aggregate root. |
| **Valuation** | The monetary value of a haul, computed from its material content and current market prices. Drives "Current Haul Value". |
| **Market price** | The current price per gram of a metal, from the market-price provider. |
| **Impact estimate** | The environmental impact of recycling the haul, computed by the three methods below. |
| **Receipt** | The generated report summarising a scan: items, value, impact, and the assumptions/citations behind the numbers. |

## Domain model (aggregate: Haul)

```
Haul
 ├── items: list[DetectedItem]        # waste_class + count + representative confidence
 ├── valuation() -> Valuation         # via MaterialContent x MarketPrice
 └── impact() -> ImpactEstimate       # via MaterialContent x ImpactFactors
```

Value objects: `Metal`, `Money`, `MaterialContent`, `Detection`, `Valuation`,
`ImpactEstimate`. The domain is pure; all external data (model inference, prices,
reference tables) enters through ports.

## Estimation methods

Recoverable value and environmental impact are **estimates**. A 2-D photo yields
a class and a count, not a mass, so we assume a typical device mass per class
(cited in `reference/materials_composition.csv`). The UI states this clearly.

* **Current Haul Value / UNEP-ITU** — recoverable raw-material value, following
  the UN Global E-waste Monitor / UNEP material-recovery basis.
* **EPA WARM** — CO2e avoided by recycling vs. landfilling, using the WARM
  electronics emission factors (comparative MTCO2E convention).
* **WEEE-LCA** — a simplified life-cycle energy/CO2 saving per kilogram of
  material recovered.

## Pricing

Metal prices come from `yfinance` futures (`GC=F`, `SI=F`, `HG=F`, `ALI=F`,
`PL=F`, `PA=F`) converted to USD/gram, cached with a TTL, and backed by a static
cited fallback table for offline / cloud-blocked operation. Per-item scrap-board
pricing (e.g. ScrapMonster) is out of the core path: brittle and ToS-sensitive,
it lives behind a feature flag that is off by default.

## Boundaries / non-goals

* No live camera or WebRTC — upload only.
* Not a certified measurement tool. Figures are indicative estimates with cited
  sources, intended for awareness and comparison, not regulatory reporting.
