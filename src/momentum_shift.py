"""Momentum reversal detection logic — dip catching engine.

This module detects early momentum reversals from oversold dips,
intended for integration in analyzer -> rules flow.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

import config


def detect_momentum_shift(
    rsi_values: List[float],
    stoch_k_values: List[float],
    macd_hist: List[float],
    current_rsi: float,
    price_change_pct: float,
) -> Dict[str, Any]:
    """
    Returns confidence flags for early momentum reversal.

    Checks:
    - RSI: was oversold recently and is now turning up
    - Stoch K: exiting low zone and rising
    - MACD histogram: negative phase turning positive

    Returns dict with boolean flags and a confidence_score (0-8).
    """
    result: Dict[str, Any] = {
        "rsi_oversold_recovery": False,
        "stoch_cross_up": False,
        "macd_hist_turning": False,
        "rsi_rising_3bars": False,
        "rsi_in_recovery_zone": False,
        "confidence_score": 0,
        "details": {},
    }

    if len(rsi_values) < 5:
        return result

    # Filter out NaN values
    rsi_clean = [r for r in rsi_values[-5:] if r is not None and not np.isnan(r)]
    if len(rsi_clean) < 3:
        return result

    # 1. RSI: oversold exit + rising
    rsi_recent = rsi_clean
    rsi_was_oversold = any(r < config.RSI_OVERSOLD_ZONE for r in rsi_recent[:-1])
    rsi_current = rsi_recent[-1]
    rsi_prev = rsi_recent[-2]
    rsi_prev2 = rsi_recent[-3] if len(rsi_recent) >= 3 else rsi_prev

    rsi_turning_up = rsi_current > rsi_prev > rsi_prev2
    rsi_rise_amount = rsi_current - rsi_prev2
    rsi_in_recovery_zone = config.RSI_RECOVERY_MIN <= rsi_current < config.RSI_OVERSOLD_EXIT

    result["rsi_oversold_recovery"] = rsi_was_oversold and rsi_turning_up
    result["rsi_rising_3bars"] = rsi_rise_amount >= config.MIN_RSI_RISE_LAST_3
    result["rsi_in_recovery_zone"] = rsi_in_recovery_zone
    result["details"]["rsi_rise_3bars"] = round(rsi_rise_amount, 1)

    # 2. Stochastic K: low zone exit and rising
    stoch_clean = [s for s in stoch_k_values[-5:] if s is not None and not np.isnan(s)]
    if len(stoch_clean) >= 3:
        stoch_was_low = any(s < config.STOCH_OVERSOLD_EXIT for s in stoch_clean[:-1])
        stoch_current = stoch_clean[-1]
        stoch_prev = stoch_clean[-2]
        stoch_cross_up = (
            stoch_current > stoch_prev
            and stoch_prev < config.STOCH_OVERSOLD_EXIT
        )
        result["stoch_cross_up"] = stoch_was_low and stoch_cross_up

    # 3. MACD histogram: negative phase then turning up
    bars = getattr(config, "MACD_HIST_NEG_TO_POS_BARS", 3)
    macd_clean = [m for m in macd_hist[-bars:] if m is not None and not np.isnan(m)]
    if len(macd_clean) == bars:
        neg_phase = all(v < 0 for v in macd_clean[:-1])
        current_turning = macd_clean[-1] > macd_clean[-2]
        result["macd_hist_turning"] = neg_phase and current_turning

    # Confidence scoring
    points = 0
    if result["rsi_oversold_recovery"]:
        points += 2
    if result["rsi_rising_3bars"]:
        points += 1
    if result["rsi_in_recovery_zone"]:
        points += 1
    if result["stoch_cross_up"]:
        points += 2
    if result["macd_hist_turning"]:
        points += 2

    # If price already moved a lot, reduce confidence (late entry risk)
    if price_change_pct is not None and price_change_pct > 5.0:
        points = max(0, points - 2)

    result["confidence_score"] = points
    return result
