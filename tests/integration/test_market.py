"""Integration tests: yfinance market adapter, driven by a fake fetcher."""

from __future__ import annotations

import pytest

from ewaste.adapters.market import (
    _POUND_G,
    _TONNE_G,
    _TROY_OUNCE_G,
    YFinanceMarketProvider,
)
from ewaste.domain.model import Material, MetalPrice

FALLBACKS = {
    Material.GOLD: MetalPrice(Material.GOLD, 85.0, source="fallback"),
    Material.COPPER: MetalPrice(Material.COPPER, 0.0095, source="fallback"),
    Material.FERROUS: MetalPrice(Material.FERROUS, 0.0004, source="fallback"),
}


def test_live_prices_are_converted_to_usd_per_gram():
    # Yahoo quotes: gold $/troy-oz, copper $/lb, aluminium $/tonne.
    quotes = {"GC=F": 3110.34768, "HG=F": 4.5359237, "ALI=F": 2500000.0}
    provider = YFinanceMarketProvider(
        FALLBACKS, fetcher=lambda sym: quotes.get(sym)
    )
    prices = provider.prices()
    assert prices[Material.GOLD].usd_per_gram == pytest.approx(3110.34768 / _TROY_OUNCE_G)
    assert prices[Material.GOLD].usd_per_gram == pytest.approx(100.0)
    assert prices[Material.COPPER].usd_per_gram == pytest.approx(4.5359237 / _POUND_G)
    assert prices[Material.ALUMINUM].usd_per_gram == pytest.approx(2500000.0 / _TONNE_G)
    assert prices[Material.GOLD].source == "yfinance:GC=F"


def test_missing_quote_falls_back_to_static_price():
    # fetcher returns nothing -> every priced metal with a fallback uses it.
    provider = YFinanceMarketProvider(FALLBACKS, fetcher=lambda sym: None)
    prices = provider.prices()
    assert prices[Material.GOLD].usd_per_gram == pytest.approx(85.0)
    assert prices[Material.GOLD].source == "fallback"
    # ferrous has no ticker at all, so it is always a fallback
    assert prices[Material.FERROUS].usd_per_gram == pytest.approx(0.0004)


def test_fetcher_exception_falls_back_without_raising():
    def boom(_sym: str) -> float | None:
        raise RuntimeError("network down")

    # a raising fetcher must not crash prices(); it degrades to the fallback.
    provider = YFinanceMarketProvider(FALLBACKS, fetcher=boom)
    prices = provider.prices()
    assert prices[Material.GOLD].source == "fallback"
    assert prices[Material.GOLD].usd_per_gram == pytest.approx(85.0)


def test_results_are_cached_within_ttl():
    calls: list[str] = []

    def counting(sym: str) -> float:
        calls.append(sym)
        return 3110.34768

    clock = iter([0.0, 0.0, 10.0, 10.0])  # well within ttl
    provider = YFinanceMarketProvider(
        FALLBACKS,
        fetcher=counting,
        ttl_seconds=3600,
        clock=lambda: next(clock),
    )
    provider.prices()
    first_call_count = len(calls)
    provider.prices()  # served from cache, no new fetches
    assert len(calls) == first_call_count
