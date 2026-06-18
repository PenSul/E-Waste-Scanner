# Reference data sources and assumptions

The three CSVs in this folder drive every number the app reports. **They are
deliberately coarse, order-of-magnitude estimates.** A 2-D photo yields a class
and a count, never a mass or an assay, so the app multiplies a *typical device*
profile by the detected count. Treat all outputs as ballpark figures for
awareness and triage, not as a settlement-grade valuation or a certified LCA.

The `source` column in each row uses the short keys below.

## `materials_composition.csv`

Per *typical device* of each class: its mass in grams and the mass fraction
(0-1) of each material. Fractions need not sum to 1 — the remainder is treated
as unrecovered. Precious-metal fractions are tiny (ppm-scale) and dominate value
only for `PCB` and `Mobile`.

- **GEM2024** — Forti V., Baldé C.P., Kuehr R., Bel G. et al., *The Global
  E-waste Monitor* (UNITAR/ITU). Per-fraction metal content of mobile phones and
  e-waste streams.
- **UNEP2009** — Schluep M. et al., *Recycling — From E-waste to Resources*
  (UNEP/StEP, 2009). Printed-circuit-board and device metal grades.
- **Hagelueken** — Hagelüken C. / Umicore precious-metals-in-PCB figures
  (Au ~250 ppm, Ag ~1000 ppm, Pd ~110 ppm for medium-grade boards).
- **EST** — author estimate from product specifications and typical bill-of-
  materials for the device category. Lower confidence; refine with measured data.

## `impact_factors.csv`

Per class: net CO2e avoided by recycling vs landfilling, in kg CO2e per kg of
material (positive = benefit), the **EPA WARM** comparative convention.

- **WARM_\*** — U.S. EPA *Waste Reduction Model (WARM)* emission factors,
  mapped to the nearest WARM material category (corrugated containers, mixed
  paper, glass, mixed plastics, steel/aluminium, mixed electronics, food waste).
  WARM publishes MTCO2E per short ton; values here are converted to kg/kg
  (x ~1.102) and sign-flipped to "avoided". Electronics and large-appliance
  rows are approximate composites — verify against the current WARM tool for
  any reporting use.
- **nonrecoverable** — `Trash`: no recovery pathway, factor 0, `recyclable=false`.

## `material_factors.csv`

Per material: the primary-production burden avoided per kg recovered
(WEEE-LCA bottom-up method), plus a static price fallback used when the live
metals feed is unavailable.

- **NussEckelman2014** — Nuss P. & Eckelman M.J., *Life Cycle Assessment of
  Metals: A Scientific Synthesis*, PLoS ONE 9(7), 2014. Cumulative energy demand
  (MJ/kg) and global warming potential (kg CO2e/kg) for primary metal production.
- **IAI_ecoinvent** — International Aluminium Institute / ecoinvent for aluminium
  recycling energy savings (~200 MJ/kg vs primary).
- **worldsteel** — worldsteel LCI for steel/ferrous scrap recycling.
- **ecoinvent** — ecoinvent-class values for plastics and glass recycling
  savings (not metals-priced; contribute to LCA energy/CO2 only).

`fallback_usd_per_gram` are approximate mid-decade spot levels: gold ~$85/g,
silver ~$1/g, palladium ~$35/g, platinum ~$32/g, copper ~$9.5/kg,
aluminium ~$2.5/kg, ferrous scrap ~$0.4/kg. The live provider overrides these
when reachable.
