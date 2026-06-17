"""Dashboard sections: market table, haul value, eco-impact, line items.

Each function takes plain domain/service objects and renders one part of the
page, so the entrypoint just calls them in order.
"""

from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

from ewaste.domain.model import Material, MetalPrice
from ewaste.service_layer import receipt
from ewaste.service_layer.scan_service import ScanResult


def render_market(prices: Mapping[Material, MetalPrice]) -> None:
    """Show the live (or fallback) per-gram metal prices feeding the valuation."""
    st.subheader("Live commodity market")
    rows = []
    for material in Material:  # stable, canonical order
        price = prices.get(material)
        if price is None:
            continue
        live = price.source.startswith("yfinance")
        rows.append(
            {
                "Material": material.value.title(),
                "USD / gram": round(price.usd_per_gram, 4),
                "Source": "live (Yahoo)" if live else "fallback (cited)",
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


def render_summary(result: ScanResult) -> None:
    """Show the headline haul value and the three eco-impact metrics."""
    metrics = receipt.summary_metrics(result)
    st.subheader("Current haul value")
    top = st.columns(2)
    top[0].metric("Recoverable value", f"${metrics['recoverable_value_usd']:,.2f}")
    top[1].metric("Items detected", metrics["items"])

    st.subheader("Eco-impact meter")
    cols = st.columns(3)
    cols[0].metric("CO2e avoided (WARM)", f"{metrics['warm_co2e_kg_avoided']:.2f} kg")
    cols[1].metric("Energy saved (LCA)", f"{metrics['lca_energy_mj_saved']:.0f} MJ")
    cols[2].metric("Total mass", f"{metrics['mass_kg']:.2f} kg")


def render_line_items(result: ScanResult) -> None:
    """Show the per-class breakdown table."""
    st.subheader("Detected items")
    rows = [
        {
            "Class": line.waste_class,
            "Count": line.count,
            "Mass (kg)": round(line.mass_kg, 3),
            "Value (USD)": round(line.value.amount, 2),
        }
        for line in result.lines
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)
    if result.unknown_classes:
        st.warning(
            "No reference composition for: "
            + ", ".join(result.unknown_classes)
            + ". These are counted but not valued."
        )


def render_downloads(result: ScanResult) -> None:
    """Offer the Digital Receipt as HTML and CSV downloads."""
    st.subheader("Digital receipt")
    html_doc = receipt.to_html(result)
    csv_doc = receipt.to_csv(result)
    cols = st.columns(2)
    cols[0].download_button(
        "Download receipt (HTML)",
        data=html_doc,
        file_name="ewaste_receipt.html",
        mime="text/html",
    )
    cols[1].download_button(
        "Download receipt (CSV)",
        data=csv_doc,
        file_name="ewaste_receipt.csv",
        mime="text/csv",
    )
