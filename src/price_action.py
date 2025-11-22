"""Price action heuristics consumed by rules.compute_price_action_block.

The logic mirrors the client docs for Phase 1 and only surfaces the boolean
flags that rules.py expects. Any richer metrics should be added later as
backwards-compatible extras.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

import config


def analyze_price_action(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    ema20_values: Sequence[float],
) -> Dict[str, object]:
    if not closes:
        defaults = {key: False for key in _DEFAULT_KEYS}
        defaults["volume_spike_factor"] = 0.0
        defaults["details"] = {}
        return defaults

    open_val = float(opens[-1])
    high_val = float(highs[-1])
    low_val = float(lows[-1])
    close_val = float(closes[-1])
    volume_val = float(volumes[-1])

    body = close_val - open_val
    range_total = max(high_val - low_val, 1e-8)
    body_pct_vs_open = abs(body) / max(abs(open_val), 1e-8) * 100
    body_pct_of_range = abs(body) / range_total * 100 if range_total else 0

    if body >= 0:
        lower_wick = open_val - low_val
    else:
        lower_wick = close_val - low_val
    lower_wick_ratio = lower_wick / range_total if range_total else 0
    lower_wick_pct = lower_wick_ratio * 100

    long_lower_wick = lower_wick_ratio >= 0.4 and body_pct_vs_open >= config.MIN_BODY_PCT / 2
    strong_green = body > 0 and body_pct_vs_open >= config.MIN_BODY_PCT

    no_collapse, max_drop_pct = _check_no_collapse(closes)
    ema20_break, ema20_break_details = _check_ema20_break(closes, ema20_values)
    volume_spike, volume_spike_factor, avg_volume = _check_volume_spike(volumes)
    min_volume = volume_val >= config.MIN_BAR_VOLUME_USDT
    min_volume_multiple = (
        volume_val / config.MIN_BAR_VOLUME_USDT if config.MIN_BAR_VOLUME_USDT > 0 else 0
    )

    pa_details = {
        "body_pct_vs_open": body_pct_vs_open,
        "body_pct_of_range": body_pct_of_range,
        "lower_wick_pct": lower_wick_pct,
        "max_drop_pct": max_drop_pct,
        "ema20_break_details": ema20_break_details,
        "volume_spike_factor": volume_spike_factor,
        "avg_volume": avg_volume,
        "current_volume": volume_val,
        "min_volume_multiple": min_volume_multiple,
    }

    return {
        "long_lower_wick": long_lower_wick,
        "strong_green": strong_green,
        "no_collapse": no_collapse,
        "ema20_break": ema20_break,
        "volume_spike": volume_spike,
        "min_volume": min_volume,
        "volume_spike_factor": volume_spike_factor,
        "details": pa_details,
    }


_DEFAULT_KEYS = (
    "long_lower_wick",
    "strong_green",
    "no_collapse",
    "ema20_break",
    "volume_spike",
    "min_volume",
)


def _check_no_collapse(closes: Sequence[float]) -> Tuple[bool, float]:
    lookback = min(config.COLLAPSE_LOOKBACK_BARS, len(closes))
    if lookback < 2:
        return True, 0.0
    window = closes[-lookback:]
    max_close = max(window)
    min_close = min(window)
    if max_close == 0:
        return True, 0.0
    drop_pct = (max_close - min_close) / max_close * 100
    return drop_pct <= config.COLLAPSE_MAX_DROP_PCT, drop_pct


def _check_ema20_break(
    closes: Sequence[float], ema20_values: Sequence[float]
) -> Tuple[bool, Dict[str, float]]:
    if len(closes) < 2 or len(ema20_values) < 2:
        return False, {
            "prev_close": 0.0,
            "curr_close": 0.0,
            "prev_ema20": 0.0,
            "curr_ema20": 0.0,
        }
    prev_close = closes[-2]
    curr_close = closes[-1]
    prev_ema = ema20_values[-2]
    curr_ema = ema20_values[-1]
    if any(np.isnan(val) for val in (prev_ema, curr_ema)):
        return False, {
            "prev_close": float(prev_close),
            "curr_close": float(curr_close),
            "prev_ema20": float(prev_ema),
            "curr_ema20": float(curr_ema),
        }
    breakout = prev_close <= prev_ema and curr_close > curr_ema
    return breakout, {
        "prev_close": float(prev_close),
        "curr_close": float(curr_close),
        "prev_ema20": float(prev_ema),
        "curr_ema20": float(curr_ema),
    }


def _check_volume_spike(volumes: Sequence[float]) -> Tuple[bool, float, float]:
    if len(volumes) < config.VOLUME_LOOKBACK + 1:
        return False, 0.0, 0.0
    recent = volumes[-(config.VOLUME_LOOKBACK + 1) : -1]
    avg_volume = sum(recent) / len(recent)
    if avg_volume == 0:
        return False, 0.0, avg_volume
    current = volumes[-1]
    factor = current / avg_volume if avg_volume else 0.0
    return current >= avg_volume * config.VOLUME_SPIKE_MULTIPLIER, factor, avg_volume
