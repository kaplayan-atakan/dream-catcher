"""Binance data fetching helpers for Phase 1 (rule-based bot).

These helpers share the aiohttp session that main/analyzer already manage. They
respect the request/retry settings in config and only expose the endpoints that
Phase 1 requires (24h tickers + multi-timeframe klines).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List

import aiohttp

import config

logger = logging.getLogger(__name__)

_TICKER_ENDPOINT = f"{config.BINANCE_BASE_URL}/api/v3/ticker/24hr"
_KLINES_ENDPOINT = f"{config.BINANCE_BASE_URL}/api/v3/klines"


async def fetch_24h_tickers(session: aiohttp.ClientSession) -> List[dict]:
    """Return the raw 24h ticker payload for all USDT pairs."""
    params = {"type": "FULL"}
    return await _request_with_retry(session, "GET", _TICKER_ENDPOINT, params=params)


def parse_ticker_data(raw_ticker: dict) -> Dict[str, float] | None:
    """Return normalized ticker data or None when the symbol is out of scope."""
    symbol = raw_ticker.get("symbol", "")
    if not symbol.endswith(config.SYMBOL_FILTER_SUFFIX):
        return None  # Non-USDT pairs are silently skipped upstream.

    price = float(raw_ticker.get("lastPrice", 0.0))
    quote_volume = float(raw_ticker.get("quoteVolume", 0.0))
    price_change_pct = float(raw_ticker.get("priceChangePercent", 0.0))

    return {
        "symbol": symbol,
        "price": price,
        "quote_volume": quote_volume,
        "price_change_pct": price_change_pct,
    }


async def fetch_multi_timeframe_klines(
    session: aiohttp.ClientSession, symbol: str
) -> Dict[str, List[dict]]:
    """Fetch OHLCV data for all configured timeframes for the given symbol."""
    results: Dict[str, List[dict]] = {}
    usdt_symbol = symbol if symbol.endswith(config.SYMBOL_FILTER_SUFFIX) else f"{symbol}{config.SYMBOL_FILTER_SUFFIX}"

    for interval in config.TIMEFRAMES:
        params = {
            "symbol": usdt_symbol,
            "interval": interval,
            "limit": 500,
        }
        klines = await _request_with_retry(session, "GET", _KLINES_ENDPOINT, params=params)
        parsed = [
            {
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "close_time": int(item[6]),
            }
            for item in klines
        ]
        results[interval] = parsed

    return results


async def _request_with_retry(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    params: Dict[str, str] | None = None,
) -> List[dict]:
    """Perform HTTP request with the retry/backoff policy from config."""
    retries = max(1, config.MAX_RETRIES)
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            async with session.request(method, url, params=params, timeout=config.REQUEST_TIMEOUT) as resp:
                resp.raise_for_status()
                return await resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Binance request failed (attempt %s/%s): %s", attempt, retries, exc)
            if attempt < retries:
                await asyncio.sleep(config.RETRY_BACKOFF)

    raise RuntimeError(f"Failed to fetch {url}: {last_exc}")
