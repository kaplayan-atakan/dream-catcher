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
        defaults.update({
            "volume_spike_factor": 0.0,
            "details": {},
        })
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

    lower_wick = (min(open_val, close_val) - low_val)
    lower_wick_ratio = lower_wick / range_total if range_total else 0
    lower_wick_pct = lower_wick_ratio * 100

    long_lower_wick = (
        body > 0
        and lower_wick_ratio >= config.LONG_WICK_MIN_RATIO
        and body_pct_vs_open >= config.MIN_BODY_PCT / 2
    )

    is_green = close_val > open_val
    strong_green, very_strong_green, avg_body = _assess_green_candle(
        opens,
        closes,
        abs(body),
        body_pct_vs_open,
        is_green,
    )

    collapse_ok, max_drop_pct = _check_no_collapse(closes)
    ema_break, ema_retest, ema_details = _check_ema20_break_and_retest(
        closes, ema20_values
    )
    volume_spike, volume_spike_factor, avg_volume = _check_volume_spike(volumes)
    min_volume = volume_val >= config.MIN_BAR_VOLUME_USDT
    min_volume_multiple = (
        volume_val / config.MIN_BAR_VOLUME_USDT if config.MIN_BAR_VOLUME_USDT > 0 else 0
    )

    # === LATE SPIKE / OVEREXTENSION METRICS (Revizyon 1) ===
    ema_last = ema20_values[-1] if ema20_values and ema20_values[-1] is not np.nan else None
    dist_from_ema_pct = 0.0
    overextended_vs_ema = False
    if ema_last and ema_last > 0:
        dist_from_ema_pct = (close_val - ema_last) / ema_last * 100
        overextended_vs_ema = dist_from_ema_pct >= config.LATE_PUMP_EMA_DIST_PCT

    exhaustion_lookback = getattr(config, "EXHAUSTION_LOOKBACK", 8)
    runup_from_recent_low_pct = 0.0
    parabolic_runup = False
    if len(closes) >= exhaustion_lookback:
        recent_window = closes[-exhaustion_lookback:]
        recent_low = min(recent_window) if recent_window else close_val
        if recent_low and recent_low > 0:
            runup_from_recent_low_pct = (close_val / recent_low - 1.0) * 100
            parabolic_runup = runup_from_recent_low_pct >= config.LATE_PUMP_RUNUP_PCT

    # === REVERSAL CANDLE PATTERNS (Bottom-fishing) ===
    reversal_signals = detect_reversal_candles(opens, highs, lows, closes)

    # === VOLUME INCREASE ON BOUNCE ===
    volume_increase_on_bounce = False
    if len(volumes) >= 2:
        prev_vol = volumes[-2]
        curr_vol = volumes[-1]
        min_ratio = getattr(config, "MIN_VOLUME_INCREASE_RATIO", 1.3)
        if prev_vol > 0 and curr_vol >= prev_vol * min_ratio:
            volume_increase_on_bounce = True

    pa_details = {
        "body_pct_vs_open": body_pct_vs_open,
        "body_pct_of_range": body_pct_of_range,
        "lower_wick_pct": lower_wick_pct,
        "avg_body_lookup": avg_body,
        "max_drop_pct": max_drop_pct,
        "ema_break_details": ema_details,
        "volume_spike_factor": volume_spike_factor,
        "avg_volume": avg_volume,
        "current_volume": volume_val,
        "min_volume_multiple": min_volume_multiple,
        "overextended_vs_ema": overextended_vs_ema,
        "parabolic_runup": parabolic_runup,
        "dist_from_ema_pct": dist_from_ema_pct,
        "runup_from_recent_low_pct": runup_from_recent_low_pct,
        "reversal_patterns": reversal_signals,
        "has_reversal": reversal_signals.get("has_reversal", False),
        "volume_increase_on_bounce": volume_increase_on_bounce,
    }

    return {
        "long_lower_wick": bool(long_lower_wick),
        "strong_green": bool(strong_green),
        "very_strong_green": bool(very_strong_green),
        "collapse_ok": bool(collapse_ok),
        "no_collapse": bool(collapse_ok),  # backward-compatible alias
        "ema_breakout": bool(ema_break),
        "ema_retest": bool(ema_retest),
        "volume_spike": bool(volume_spike),
        "min_volume": bool(min_volume),
        "volume_spike_factor": float(volume_spike_factor or 0.0),
        "details": pa_details,
    }


_DEFAULT_KEYS = (
    "long_lower_wick",
    "strong_green",
    "very_strong_green",
    "collapse_ok",
    "no_collapse",
    "ema_breakout",
    "ema_retest",
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


def _assess_green_candle(
    opens: Sequence[float],
    closes: Sequence[float],
    current_body_abs: float,
    body_pct_vs_open: float,
    is_green: bool,
) -> Tuple[bool, bool, float]:
    lookback = min(config.STRONG_GREEN_LOOKBACK, len(opens))
    if lookback < 2:
        return body_pct_vs_open >= config.MIN_BODY_PCT and is_green, False, current_body_abs
    recent_opens = np.asarray(opens[-lookback:], dtype=float)
    recent_closes = np.asarray(closes[-lookback:], dtype=float)
    bodies = np.abs(recent_closes - recent_opens)
    avg_body = float(np.mean(bodies)) if bodies.size else 0.0
    strong_green = is_green and body_pct_vs_open >= config.MIN_BODY_PCT
    very_strong = (
        avg_body > 0
        and is_green
        and current_body_abs >= config.STRONG_GREEN_BODY_MULTIPLIER * avg_body
    )
    if very_strong:
        strong_green = True
    return strong_green, very_strong, avg_body


def _check_ema20_break_and_retest(
    closes: Sequence[float], ema20_values: Sequence[float]
) -> Tuple[bool, bool, Dict[str, float]]:
    if len(closes) < 3 or len(ema20_values) < 3:
        return False, False, {}
    lookback = min(config.EMA_BREAK_LOOKBACK, len(closes) - 1)
    start_idx = len(closes) - lookback
    breakout_idx = None
    for idx in range(max(1, start_idx), len(closes)):
        prev_close = closes[idx - 1]
        curr_close = closes[idx]
        prev_ema = ema20_values[idx - 1]
        curr_ema = ema20_values[idx]
        if any(np.isnan(val) for val in (prev_ema, curr_ema)):
            continue
        if prev_close <= prev_ema and curr_close > curr_ema:
            breakout_idx = idx
            break

    if breakout_idx is None:
        return False, False, {}

    retest = _check_retest(closes, ema20_values, breakout_idx)
    details = {
        "breakout_index": breakout_idx,
        "breakout_close": float(closes[breakout_idx]),
        "breakout_ema": float(ema20_values[breakout_idx]),
        "retest": retest,
    }
    return True, retest, details


def _check_retest(closes: Sequence[float], ema20_values: Sequence[float], breakout_idx: int) -> bool:
    tolerance = config.EMA_NEAR_TOLERANCE
    max_idx = min(len(closes), breakout_idx + config.EMA_RETEST_LOOKBACK)
    for idx in range(breakout_idx + 1, max_idx):
        ema = ema20_values[idx]
        close = closes[idx]
        if ema == 0 or np.isnan(ema):
            continue
        if abs(close - ema) / abs(ema) <= tolerance:
            if idx + 1 < len(closes) and closes[idx + 1] > ema20_values[idx + 1]:
                return True
    return False


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


def detect_reversal_candles(
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
) -> Dict[str, object]:
    """
    Detect basic bullish reversal patterns:
    - Hammer (long lower wick, small body, green candle)
    - Bullish Engulfing (red followed by larger green that engulfs)
    - Morning Star (3-candle: red, small body, green closing above midpoint)
    """
    result = {
        "hammer": False,
        "bullish_engulfing": False,
        "morning_star": False,
        "has_reversal": False,
    }

    if len(closes) < 3:
        return result

    o1, h1, l1, c1 = opens[-3], highs[-3], lows[-3], closes[-3]
    o2, h2, l2, c2 = opens[-2], highs[-2], lows[-2], closes[-2]
    o3, h3, l3, c3 = opens[-1], highs[-1], lows[-1], closes[-1]

    # Hammer detection (current bar)
    body3 = abs(c3 - o3)
    lower_wick3 = min(o3, c3) - l3
    upper_wick3 = h3 - max(o3, c3)
    total_range3 = h3 - l3

    is_hammer = (
        c3 > o3  # Green candle
        and total_range3 > 0
        and lower_wick3 > body3 * 2.0  # Lower wick at least 2x body
        and upper_wick3 < body3 * 0.3  # Small upper wick
    )

    # Bullish Engulfing (previous bar red, current bar green and engulfs)
    is_engulfing = (
        c2 < o2  # Previous candle red
        and c3 > o3  # Current candle green
        and o3 < c2  # Open below previous close
        and c3 > o2  # Close above previous open
    )

    # Morning Star (3-candle pattern)
    first_body = abs(c1 - o1)
    is_morning_star = (
        c1 < o1  # First candle red
        and first_body > 0
        and abs(c2 - o2) < first_body * 0.3  # Small middle candle
        and c3 > o3  # Last candle green
        and c3 > (o1 + c1) / 2  # Closes above midpoint of first candle
    )

    result.update({
        "hammer": bool(is_hammer),
        "bullish_engulfing": bool(is_engulfing),
        "morning_star": bool(is_morning_star),
        "has_reversal": bool(is_hammer or is_engulfing or is_morning_star),
    })

    return result


def check_support_levels(
    closes: Sequence[float],
    lows: Sequence[float],
    ema20_values: Sequence[float],
) -> Dict[str, object]:
    """
    Check for EMA20 and recent low support.
    Returns flags for near_ema20_support, bounced_from_support, and the support level.
    
    Args:
        closes: List of closing prices
        lows: List of low prices
        ema20_values: List of EMA20 values
        
    Returns:
        Dict containing:
        - near_ema20_support: True if price is near EMA20
        - bounced_from_support: True if price recently bounced from support
        - support_level: The detected support level
    """
    import numpy as np  # Local import to avoid circular dependency
    import config
    
    result = {
        "near_ema20_support": False,
        "bounced_from_support": False,
        "support_level": None,
    }

    lookback = getattr(config, "SUPPORT_LOOKBACK_BARS", 20)
    proximity_pct = getattr(config, "SUPPORT_EMA_PROXIMITY_PCT", 1.5) / 100.0

    if len(closes) < lookback or len(lows) < lookback:
        return result

    # Filter out NaN from ema20
    valid_ema = [e for e in list(ema20_values)[-lookback:] if e is not None and not np.isnan(e)]
    if not valid_ema:
        return result

    current_price = closes[-1]
    ema20_current = valid_ema[-1]

    # Near EMA20 support: within proximity_pct of EMA20
    if ema20_current > 0:
        near_ema20 = abs(current_price - ema20_current) / ema20_current < proximity_pct
        result["near_ema20_support"] = near_ema20

    # Support level from recent lows
    recent_lows = list(lows)[-lookback:]
    support_level = min(recent_lows)
    result["support_level"] = support_level

    # Bounced from support: price is now above support and recently touched it
    min_bounce_pct = getattr(config, "MIN_BOUNCE_FROM_SUPPORT", 0.5) / 100.0
    if support_level > 0:
        bounced = (
            current_price > support_level * (1 + min_bounce_pct)
            and min(list(lows)[-3:]) <= support_level * 1.01
        )
        result["bounced_from_support"] = bounced

    return result