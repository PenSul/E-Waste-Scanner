"""Render a :class:`ScanResult` as a Digital Receipt (CSV and HTML).

The receipt is the user-facing artifact: a per-class table of counts, mass, and
recoverable value, plus the three environmental-impact figures and a prominent
disclaimer that the numbers are coarse, typical-device estimates. Rendering is
pure string-building so it can be unit-tested and offered as a download.
"""

from __future__ import annotations

import csv
import html
import io
from datetime import datetime, timezone

from ewaste.service_layer.scan_service import ScanResult

#: Shown on every receipt; mirrors reference/SOURCES.md and ADR-0002.
DISCLAIMER = (
    "Estimates only. Values come from typical-device material profiles times "
    "detected counts, priced at live metals quotes (static fallback when "
    "offline). Figures are coarse, order-of-magnitude estimates for awareness "
    "and triage, not a settlement-grade valuation or a certified life-cycle "
    "assessment."
)


def _now(generated_at: datetime | None) -> datetime:
    """Resolve the receipt timestamp (injectable for deterministic tests)."""
    return generated_at or datetime.now(timezone.utc)


def summary_metrics(result: ScanResult) -> dict[str, float | int]:
    """Headline figures for the dashboard and receipt footer."""
    impact = result.impact
    return {
        "items": result.total_count,
        "recoverable_value_usd": round(result.total_value.amount, 2),
        "mass_kg": round(impact.mass_kg, 3),
        "warm_co2e_kg_avoided": round(impact.co2e_kg_avoided, 3),
        "lca_energy_mj_saved": round(impact.energy_mj_saved, 1),
        "lca_co2e_kg_avoided": round(impact.lca_co2e_kg_avoided, 3),
    }


def to_csv(result: ScanResult, generated_at: datetime | None = None) -> str:
    """Render the receipt as CSV: a per-class table then summary rows."""
    stamp = _now(generated_at).isoformat(timespec="seconds")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["E-Waste Scanner receipt", stamp])
    writer.writerow([])
    writer.writerow(["waste_class", "count", "mass_kg", "recoverable_value_usd"])
    for line in result.lines:
        writer.writerow(
            [line.waste_class, line.count, round(line.mass_kg, 3), round(line.value.amount, 2)]
        )
    writer.writerow([])
    for key, value in summary_metrics(result).items():
        writer.writerow([key, value])
    if result.unknown_classes:
        writer.writerow([])
        writer.writerow(["unpriced_unknown_classes", ";".join(result.unknown_classes)])
    return buf.getvalue()


def _rows_html(result: ScanResult) -> str:
    """Build the per-class table body for the HTML receipt."""
    cells = []
    for line in result.lines:
        cells.append(
            "<tr>"
            f"<td>{html.escape(line.waste_class)}</td>"
            f"<td class='num'>{line.count}</td>"
            f"<td class='num'>{line.mass_kg:.3f}</td>"
            f"<td class='num'>${line.value.amount:,.2f}</td>"
            "</tr>"
        )
    return "\n".join(cells)


def to_html(result: ScanResult, generated_at: datetime | None = None) -> str:
    """Render a self-contained HTML receipt suitable for download or display."""
    stamp = _now(generated_at).isoformat(timespec="seconds")
    metrics = summary_metrics(result)
    unknown = ""
    if result.unknown_classes:
        names = html.escape(", ".join(result.unknown_classes))
        unknown = (
            f"<p class='warn'>No reference composition for: {names}. "
            "These are counted but not valued.</p>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>E-Waste Scanner receipt</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 40rem; margin: 2rem auto; color: #1a1a1a; }}
 h1 {{ font-size: 1.4rem; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
 th, td {{ border-bottom: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
 td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
 .total {{ font-weight: 700; font-size: 1.2rem; }}
 .metrics li {{ margin: 0.2rem 0; }}
 .warn {{ color: #8a5a00; }}
 .disclaimer {{ font-size: 0.8rem; color: #555; border-top: 1px solid #ddd; padding-top: 0.8rem; }}
</style>
</head>
<body>
<h1>E-Waste Scanner receipt</h1>
<p>Generated {html.escape(stamp)} - {metrics['items']} item(s) detected</p>
<p class="total">Recoverable value: ${metrics['recoverable_value_usd']:,.2f}</p>
<table>
<thead><tr><th>Class</th><th class="num">Count</th><th class="num">Mass (kg)</th><th class="num">Value (USD)</th></tr></thead>
<tbody>
{_rows_html(result)}
</tbody>
</table>
<h2>Environmental impact</h2>
<ul class="metrics">
<li>Total mass: {metrics['mass_kg']} kg</li>
<li>CO2e avoided (EPA WARM, recycle vs landfill): {metrics['warm_co2e_kg_avoided']} kg</li>
<li>Primary-production energy saved (WEEE-LCA): {metrics['lca_energy_mj_saved']} MJ</li>
<li>CO2 avoided (WEEE-LCA): {metrics['lca_co2e_kg_avoided']} kg</li>
</ul>
{unknown}
<p class="disclaimer">{html.escape(DISCLAIMER)}</p>
</body>
</html>
"""
