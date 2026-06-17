"""Live metals pricing via yfinance, with a cited static fallback.

Yahoo Finance quotes commodity futures in mixed units (precious metals per troy
ounce, copper per pound, aluminium per tonne); this adapter converts each to USD
per gram. Results are cached for a TTL, and any ticker that fails to resolve
falls back to the static price from ``material_factors.csv`` so the app never
hard-fails when the network (or Community Cloud egress) is unavailable.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

from ewaste.domain.model import PRICED_MATERIALS, Material, MetalPrice
from ewaste.ports import MarketPriceProvider

#: Grams per quoted unit, used to convert a quote to USD per gram.
_TROY_OUNCE_G = 31.1034768
_POUND_G = 453.59237
_TONNE_G = 1_000_000.0


@dataclass(frozen=True)
class TickerSpec:
    """A Yahoo Finance symbol and the grams contained in one quoted unit."""

    symbol: str
    grams_per_unit: float


#: Default futures tickers per priced material. Ferrous scrap has no liquid,
#: reliably-quoted Yahoo future, so it is intentionally absent and always uses
#: the static fallback.
DEFAULT_TICKERS: dict[Material, TickerSpec] = {
    Material.GOLD: TickerSpec("GC=F", _TROY_OUNCE_G),
    Material.SILVER: TickerSpec("SI=F", _TROY_OUNCE_G),
    Material.PLATINUM: TickerSpec("PL=F", _TROY_OUNCE_G),
    Material.PALLADIUM: TickerSpec("PA=F", _TROY_OUNCE_G),
    Material.COPPER: TickerSpec("HG=F", _POUND_G),
    Material.ALUMINUM: TickerSpec("ALI=F", _TONNE_G),
}


def _yahoo_last_price(symbol: str) -> float | None:
    """Return the latest price for ``symbol`` via yfinance, or None on failure.

    Tries the lightweight ``fast_info.last_price`` first and falls back to the
    last close of a short daily history. Network and parsing errors are
    swallowed (the caller substitutes the static fallback).
    """
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        try:
            price = ticker.fast_info.last_price
            if price is not None and price == price and price > 0:  # not NaN
                return float(price)
        except Exception:
            pass
        history = ticker.history(period="5d", interval="1d")
        if not history.empty:
            closes = history["Close"].dropna()
            if not closes.empty:
                return float(closes.iloc[-1])
    except Exception:
        return None
    return None


class YFinanceMarketProvider(MarketPriceProvider):
    """Resolves USD-per-gram prices for the priced materials, cached with a TTL.

    ``fetcher`` is injectable so tests can drive prices without touching the
    network; ``clock`` is injectable so cache expiry can be tested deterministically.
    """

    def __init__(
        self,
        fallbacks: Mapping[Material, MetalPrice],
        *,
        tickers: Mapping[Material, TickerSpec] = DEFAULT_TICKERS,
        ttl_seconds: int = 3600,
        fetcher: Callable[[str], float | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fallbacks = dict(fallbacks)
        self._tickers = dict(tickers)
        self._ttl = ttl_seconds
        self._fetch = fetcher or _yahoo_last_price
        self._clock = clock
        self._cache: dict[Material, MetalPrice] | None = None
        self._cached_at: float | None = None

    def prices(self) -> dict[Material, MetalPrice]:
        """Return a price per priced material, live where possible else fallback."""
        if self._cache is not None and self._cached_at is not None:
            if self._clock() - self._cached_at < self._ttl:
                return dict(self._cache)

        asof = datetime.now(timezone.utc).isoformat(timespec="seconds")
        resolved: dict[Material, MetalPrice] = {}
        for material in PRICED_MATERIALS:
            live = self._live_price(material, asof)
            if live is not None:
                resolved[material] = live
            elif material in self._fallbacks:
                resolved[material] = self._fallbacks[material]

        self._cache = resolved
        self._cached_at = self._clock()
        return dict(resolved)

    def _live_price(self, material: Material, asof: str) -> MetalPrice | None:
        """Fetch and unit-convert one material's live price, or None."""
        spec = self._tickers.get(material)
        if spec is None:
            return None
        try:
            raw = self._fetch(spec.symbol)
        except Exception:
            raw = None
        if raw is None or raw <= 0:
            return None
        return MetalPrice(
            material=material,
            usd_per_gram=raw / spec.grams_per_unit,
            asof=asof,
            source=f"yfinance:{spec.symbol}",
        )
