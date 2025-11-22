"""Price action heuristics consumed by rules.compute_price_action_block.

The logic mirrors the client docs for Phase 1 and only surfaces the boolean
flags that rules.py expects. Any richer metrics should be added later as
backwards-compatible extras.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

import config


def analyze_price_action(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    ema20_values: Sequence[float],
) -> Dict[str, bool]:
    if not closes:
        return {key: False for key in _DEFAULT_KEYS}

    open_val = float(opens[-1])
    high_val = float(highs[-1])
    low_val = float(lows[-1])
    close_val = float(closes[-1])
    volume_val = float(volumes[-1])

    body = close_val - open_val
    range_total = max(high_val - low_val, 1e-8)
    body_pct = abs(body) / max(abs(open_val), 1e-8) * 100

    if body >= 0:
        lower_wick = open_val - low_val
    else:
        lower_wick = close_val - low_val
    lower_wick_ratio = lower_wick / range_total if range_total else 0

    long_lower_wick = lower_wick_ratio >= 0.4 and body_pct >= config.MIN_BODY_PCT / 2
    strong_green = body > 0 and body_pct >= config.MIN_BODY_PCT

    no_collapse = _check_no_collapse(closes)
    ema20_break = _check_ema20_break(closes, ema20_values)
    volume_spike = _check_volume_spike(volumes)
    min_volume = volume_val >= config.MIN_BAR_VOLUME_USDT

    return {
        "long_lower_wick": long_lower_wick,
        "strong_green": strong_green,
        "no_collapse": no_collapse,
        "ema20_break": ema20_break,
        "volume_spike": volume_spike,
        "min_volume": min_volume,
    }


_DEFAULT_KEYS = (
    "long_lower_wick",
    "strong_green",
    "no_collapse",
    "ema20_break",
    "volume_spike",
    "min_volume",
)


def _check_no_collapse(closes: Sequence[float]) -> bool:
    lookback = min(config.COLLAPSE_LOOKBACK_BARS, len(closes))
    if lookback < 2:
        return True
    window = closes[-lookback:]
    max_close = max(window)
    min_close = min(window)
    if max_close == 0:
        return True
    drop_pct = (max_close - min_close) / max_close * 100
    return drop_pct <= config.COLLAPSE_MAX_DROP_PCT


def _check_ema20_break(closes: Sequence[float], ema20_values: Sequence[float]) -> bool:
    if len(closes) < 2 or len(ema20_values) < 2:
        return False
    prev_close = closes[-2]
    curr_close = closes[-1]
    prev_ema = ema20_values[-2]
    curr_ema = ema20_values[-1]
    if any(np.isnan(val) for val in (prev_ema, curr_ema)):
        return False
    return prev_close <= prev_ema and curr_close > curr_ema


def _check_volume_spike(volumes: Sequence[float]) -> bool:
    if len(volumes) < config.VOLUME_LOOKBACK + 1:
        return False
    recent = volumes[-(config.VOLUME_LOOKBACK + 1) : -1]
    avg_volume = sum(recent) / len(recent)
    if avg_volume == 0:
        return False
    current = volumes[-1]
    return current >= avg_volume * config.VOLUME_SPIKE_MULTIPLIER
