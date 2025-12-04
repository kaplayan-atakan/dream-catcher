"""Technical indicator helpers used by analyzer.py (Phase 1).

All functions return full-length lists and use numpy NaNs when insufficient data
exists. Wherever analyzer.py does not pass an explicit period (e.g. MACD,
Stochastic RSI), this module pulls the values from config so that all tuning
remains centralized there. Implementations are deterministic and match standard
rule-based formulas aligned with the client docs.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

import config

NAN = np.nan


def ema(values: Sequence[float], period: int) -> List[float]:
    values = _to_float_array(values)
    length = len(values)
    result = [NAN] * length
    if period <= 0 or length == 0:
        return result

    start = _find_seed_window(values, period)
    if start is None:
        return result

    seed = float(np.mean(values[start : start + period]))
    idx = start + period - 1
    result[idx] = seed
    multiplier = 2 / (period + 1)
    ema_val = seed

    for i in range(idx + 1, length):
        val = values[i]
        if np.isnan(val):
            result[i] = NAN
            continue
        ema_val = (val - ema_val) * multiplier + ema_val
        result[i] = ema_val

    return result


def _sma(values: np.ndarray, period: int) -> np.ndarray:
    length = len(values)
    result = np.full(length, NAN)
    if period <= 0 or length < period:
        return result
    cumsum = np.cumsum(values, dtype=float)
    cumsum[period:] = cumsum[period:] - cumsum[:-period]
    result[period - 1 :] = cumsum[period - 1 :] / period
    return result


def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int,
) -> Tuple[List[float], List[float], List[float]]:
    h = _to_float_array(highs)
    l = _to_float_array(lows)
    c = _to_float_array(closes)
    length = len(c)

    adx_vals = [NAN] * length
    plus_di = [NAN] * length
    minus_di = [NAN] * length

    if period <= 0 or length < period + 1:
        return adx_vals, plus_di, minus_di

    tr_list: List[float] = []
    plus_dm_list: List[float] = []
    minus_dm_list: List[float] = []

    for i in range(1, length):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        tr_list.append(tr)

        up_move = h[i] - h[i - 1]
        down_move = l[i - 1] - l[i]
        plus_dm_list.append(up_move if (up_move > down_move and up_move > 0) else 0.0)
        minus_dm_list.append(down_move if (down_move > up_move and down_move > 0) else 0.0)

    atr = _wilder_smooth(tr_list, period)
    plus_smoothed = _wilder_smooth(plus_dm_list, period)
    minus_smoothed = _wilder_smooth(minus_dm_list, period)

    for i in range(period, length):
        atr_val = atr[i - 1]
        if atr_val == 0 or np.isnan(atr_val):
            continue
        plus = 100 * plus_smoothed[i - 1] / atr_val
        minus = 100 * minus_smoothed[i - 1] / atr_val
        plus_di[i] = plus
        minus_di[i] = minus

    dx_values: List[float] = []
    for i in range(period, length):
        p = plus_di[i]
        m = minus_di[i]
        if np.isnan(p) or np.isnan(m) or (p + m) == 0:
            dx_values.append(NAN)
        else:
            dx_values.append(100 * abs(p - m) / (p + m))

    adx_series = _wilder_smooth(dx_values, period)
    for i, val in enumerate(adx_series, start=period * 2):
        if i < length:
            adx_vals[i] = val

    return adx_vals, plus_di, minus_di


def _wilder_smooth(values: Sequence[float], period: int) -> List[float]:
    arr = _to_float_array(values)
    length = len(arr)
    result = [NAN] * length
    if period <= 0 or length == 0 or length < period:
        return result

    prev = float(np.sum(arr[:period]))
    result[period - 1] = prev / period
    for i in range(period, length):
        prev = prev - (prev / period) + arr[i]
        result[i] = prev / period
    return result


def macd(closes: Sequence[float]) -> Tuple[List[float], List[float], List[float]]:
    fast = ema(closes, config.MACD_FAST)
    slow = ema(closes, config.MACD_SLOW)
    macd_line = [
        (f - s) if _is_valid(f) and _is_valid(s) else NAN
        for f, s in zip(fast, slow)
    ]
    signal_line = ema(macd_line, config.MACD_SIGNAL)
    hist = [
        (m - s) if _is_valid(m) and _is_valid(s) else NAN
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, hist


def momentum(closes: Sequence[float], period: int) -> List[float]:
    values = _to_float_array(closes)
    length = len(values)
    result = [NAN] * length
    if period <= 0 or length <= period:
        return result
    for i in range(period, length):
        result[i] = values[i] - values[i - period]
    return result


def awesome_oscillator(highs: Sequence[float], lows: Sequence[float]) -> List[float]:
    median = (_to_float_array(highs) + _to_float_array(lows)) / 2
    sma5 = _sma(median, 5)
    sma34 = _sma(median, 34)
    return (sma5 - sma34).tolist()


def rsi(closes: Sequence[float], period: int) -> List[float]:
    values = _to_float_array(closes)
    length = len(values)
    result = [NAN] * length
    if period <= 0 or length <= period:
        return result

    deltas = np.diff(values)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = _wilder_smooth(gains, period)
    avg_loss = _wilder_smooth(losses, period)

    for i in range(period, length):
        lg = avg_gain[i - 1]
        ll = avg_loss[i - 1]
        if np.isnan(lg) or np.isnan(ll) or ll == 0:
            continue
        rs = lg / ll
        result[i] = 100 - (100 / (1 + rs))
    return result


def stochastic_k(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int,
) -> List[float]:
    h = _to_float_array(highs)
    l = _to_float_array(lows)
    c = _to_float_array(closes)
    length = len(c)
    result = [NAN] * length
    if period <= 0 or length < period:
        return result

    for i in range(period - 1, length):
        highest = float(np.max(h[i - period + 1 : i + 1]))
        lowest = float(np.min(l[i - period + 1 : i + 1]))
        denom = highest - lowest
        if denom == 0:
            result[i] = 50.0
        else:
            result[i] = 100 * (c[i] - lowest) / denom
    return result


def cci(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int,
) -> List[float]:
    h = _to_float_array(highs)
    l = _to_float_array(lows)
    c = _to_float_array(closes)
    tp = (h + l + c) / 3
    sma_tp = _sma(tp, period)
    length = len(tp)
    result = [NAN] * length
    if period <= 0 or length < period:
        return result

    for i in range(period - 1, length):
        window = tp[i - period + 1 : i + 1]
        mean_dev = np.mean(np.abs(window - sma_tp[i]))
        denom = 0.015 * mean_dev if mean_dev != 0 else None
        if denom:
            result[i] = (tp[i] - sma_tp[i]) / denom
    return result


def stochastic_rsi(closes: Sequence[float]) -> List[float]:
    rsi_values = rsi(closes, config.RSI_PERIOD)
    period = config.RSI_PERIOD
    arr = _to_float_array(rsi_values)
    length = len(arr)
    result = [NAN] * length
    if length < period:
        return result
    for i in range(period - 1, length):
        window = arr[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            continue
        min_val = np.min(valid)
        max_val = np.max(valid)
        denom = max_val - min_val
        result[i] = 0 if denom == 0 else 100 * (arr[i] - min_val) / denom
    return result


def williams_r(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int,
) -> List[float]:
    h = _to_float_array(highs)
    l = _to_float_array(lows)
    c = _to_float_array(closes)
    length = len(c)
    result = [NAN] * length
    if period <= 0 or length < period:
        return result
    for i in range(period - 1, length):
        highest = float(np.max(h[i - period + 1 : i + 1]))
        lowest = float(np.min(l[i - period + 1 : i + 1]))
        denom = highest - lowest
        result[i] = -100 if denom == 0 else -100 * (highest - c[i]) / denom
    return result


def ultimate_oscillator(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    periods: Tuple[int, int, int],
) -> List[float]:
    short, mid, long = periods
    h = _to_float_array(highs)
    l = _to_float_array(lows)
    c = _to_float_array(closes)
    bp = c - np.minimum(l, np.roll(c, 1))
    tr = np.maximum(h, np.roll(c, 1)) - np.minimum(l, np.roll(c, 1))
    bp[0] = c[0] - l[0]
    tr[0] = h[0] - l[0]

    avg_short = _rolling_sum(bp, short) / _rolling_sum(tr, short)
    avg_mid = _rolling_sum(bp, mid) / _rolling_sum(tr, mid)
    avg_long = _rolling_sum(bp, long) / _rolling_sum(tr, long)

    uo = 100 * ((4 * avg_short) + (2 * avg_mid) + avg_long) / 7
    return uo.tolist()


def obv(closes: Sequence[float], volumes: Sequence[float]) -> List[float]:
    c = _to_float_array(closes)
    v = _to_float_array(volumes)
    length = len(c)
    result = [0.0] * length
    for i in range(1, length):
        if c[i] > c[i - 1]:
            result[i] = result[i - 1] + v[i]
        elif c[i] < c[i - 1]:
            result[i] = result[i - 1] - v[i]
        else:
            result[i] = result[i - 1]
    return result


def bull_bear_power(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> Tuple[List[float], List[float]]:
    ema_vals = _to_float_array(ema(closes, config.EMA_FAST))
    h = _to_float_array(highs)
    l = _to_float_array(lows)
    bull = (h - ema_vals).tolist()
    bear = (l - ema_vals).tolist()
    return bull, bear


def is_obv_uptrend(obv_values: Sequence[float], lookback: int) -> bool:
    if lookback <= 0 or len(obv_values) < lookback:
        return False
    start = obv_values[-lookback]
    end = obv_values[-1]
    return _is_valid(start) and _is_valid(end) and end > start


def obv_change_percent(obv_values: Sequence[float], lookback: int) -> float:
    if lookback <= 0 or len(obv_values) < lookback:
        return 0.0
    start = obv_values[-lookback]
    end = obv_values[-1]
    if not (_is_valid(start) and _is_valid(end)):
        return 0.0
    baseline = abs(start) if abs(start) > 1e-8 else 0.0
    if baseline == 0:
        return 0.0
    return (end - start) / baseline * 100.0


# ---------- Helper utilities ----------

def _to_float_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray([float(v) if v is not None else NAN for v in values], dtype=float)


def _find_seed_window(values: np.ndarray, period: int) -> int | None:
    for i in range(0, len(values) - period + 1):
        window = values[i : i + period]
        if not np.any(np.isnan(window)):
            return i
    return None


def _rolling_sum(values: np.ndarray, period: int) -> np.ndarray:
    length = len(values)
    result = np.full(length, NAN)
    if period <= 0 or length < period:
        return result
    cumsum = np.cumsum(values, dtype=float)
    result[period - 1 :] = cumsum[period - 1 :] - np.concatenate(
        ([0.0], cumsum[:-period])
    )
    return result


def _is_valid(value: float) -> bool:
    return value is not None and not np.isnan(value)


def detect_momentum_shift(
    rsi_values: Sequence[float],
    stoch_values: Sequence[float],
    macd_hist: Sequence[float],
) -> dict:
    """
    Detect the early point of momentum reversal from a dip.

    Returns a dict with:
    - momentum_shift: True when RSI was oversold recently and is now turning up,
      Stochastic was low and is rising, MACD histogram is turning from negative.
    - Details about each component for debugging/logging.
    """
    result = {
        "momentum_shift": False,
        "rsi_was_oversold": False,
        "rsi_turning_up": False,
        "stoch_was_low": False,
        "stoch_rising": False,
        "macd_turning": False,
    }

    lookback = getattr(config, "MOMENTUM_SHIFT_LOOKBACK", 5)
    rsi_oversold_zone = getattr(config, "RSI_OVERSOLD_ZONE", 35)
    stoch_oversold = getattr(config, "STOCH_OVERSOLD", 20)
    stoch_exit = getattr(config, "STOCH_OVERSOLD_EXIT", 25)

    if len(rsi_values) < lookback or len(stoch_values) < lookback or len(macd_hist) < 3:
        return result

    # Filter out NaN values
    rsi_recent = [r for r in rsi_values[-lookback:] if _is_valid(r)]
    stoch_recent = [s for s in stoch_values[-lookback:] if _is_valid(s)]
    macd_recent = [m for m in macd_hist[-3:] if _is_valid(m)]

    if len(rsi_recent) < 3 or len(stoch_recent) < 3 or len(macd_recent) < 3:
        return result

    # RSI: oversold in recent past and now turning up
    rsi_current = rsi_recent[-1]
    rsi_prev = rsi_recent[-2]
    rsi_was_oversold = any(r < rsi_oversold_zone for r in rsi_recent[:-1])
    rsi_turning_up = rsi_current > rsi_prev and (len(rsi_recent) < 3 or rsi_prev > rsi_recent[-3])

    # Stochastic: exited oversold and rising
    stoch_current = stoch_recent[-1]
    stoch_was_low = any(s < stoch_oversold for s in stoch_recent[:-1])
    stoch_rising = stoch_current > stoch_recent[-2] and stoch_current > stoch_exit

    # MACD histogram: turning from negative to rising
    macd_turning = (
        macd_recent[-3] < 0
        and macd_recent[-2] < 0
        and macd_recent[-1] > macd_recent[-2]
    )

    result.update({
        "rsi_was_oversold": rsi_was_oversold,
        "rsi_turning_up": rsi_turning_up,
        "stoch_was_low": stoch_was_low,
        "stoch_rising": stoch_rising,
        "macd_turning": macd_turning,
        "momentum_shift": (
            rsi_was_oversold and rsi_turning_up
            and stoch_was_low and stoch_rising
            and macd_turning
        ),
    })

    return result


def detect_early_momentum_shift(
    rsi_values: Sequence[float],
    macd_hist_values: Sequence[float],
    stoch_k_values: Sequence[float],
    stoch_d_values: Sequence[float],
) -> dict:
    """
    Detect early momentum shift - signals BEFORE the move, not after.
    
    Conditions (ALL must be true for full detection):
    1. RSI rising for 3 bars AND RSI[-1] in 38-48 range (dip recovery zone)
    2. MACD histogram: hist[-3] < 0, hist[-2] < 0, hist[-1] > hist[-2] (turning up from negative)
    3. Stoch K: K[-1] > D[-1] AND K[-1] > 20 (exiting oversold)
    
    Returns:
        {
            "detected": bool,
            "confidence": int,  # 0-3 based on conditions met
            "rsi_rising": bool,
            "rsi_in_recovery_zone": bool,
            "macd_turning_up": bool,
            "stoch_bullish_cross": bool,
            "details": {...}
        }
    """
    result = {
        "detected": False,
        "confidence": 0,
        "rsi_rising": False,
        "rsi_in_recovery_zone": False,
        "macd_turning_up": False,
        "stoch_bullish_cross": False,
        "details": {}
    }
    
    rsi_arr = _to_float_array(rsi_values)
    macd_arr = _to_float_array(macd_hist_values)
    stoch_k_arr = _to_float_array(stoch_k_values)
    stoch_d_arr = _to_float_array(stoch_d_values)
    
    if len(rsi_arr) < 3 or len(macd_arr) < 3 or len(stoch_k_arr) < 1:
        return result
    
    # Get config values
    rsi_min = getattr(config, "EARLY_MOMENTUM_RSI_MIN", 38)
    rsi_max = getattr(config, "EARLY_MOMENTUM_RSI_MAX", 48)
    stoch_min = getattr(config, "EARLY_MOMENTUM_STOCH_MIN", 20)
    
    # Condition 1: RSI rising for 3 bars AND in recovery zone (38-48)
    rsi_1 = rsi_arr[-1]
    rsi_2 = rsi_arr[-2]
    rsi_3 = rsi_arr[-3]
    
    if not (_is_valid(rsi_1) and _is_valid(rsi_2) and _is_valid(rsi_3)):
        return result
    
    rsi_rising = rsi_1 > rsi_2 > rsi_3
    rsi_in_zone = rsi_min <= rsi_1 <= rsi_max
    
    result["rsi_rising"] = rsi_rising
    result["rsi_in_recovery_zone"] = rsi_in_zone
    result["details"]["rsi_current"] = round(float(rsi_1), 1)
    result["details"]["rsi_3bar_change"] = round(float(rsi_1 - rsi_3), 1)
    
    if rsi_rising and rsi_in_zone:
        result["confidence"] += 1
    
    # Condition 2: MACD histogram turning up from negative
    hist_1 = macd_arr[-1]
    hist_2 = macd_arr[-2]
    hist_3 = macd_arr[-3]
    
    if _is_valid(hist_1) and _is_valid(hist_2) and _is_valid(hist_3):
        macd_turning = hist_3 < 0 and hist_2 < 0 and hist_1 > hist_2
        result["macd_turning_up"] = macd_turning
        result["details"]["macd_hist_current"] = round(float(hist_1), 6)
        result["details"]["macd_hist_prev"] = round(float(hist_2), 6)
        
        if macd_turning:
            result["confidence"] += 1
    
    # Condition 3: Stoch K bullish cross (K > D and K > 20)
    k_1 = stoch_k_arr[-1]
    d_1 = stoch_d_arr[-1] if len(stoch_d_arr) >= 1 and _is_valid(stoch_d_arr[-1]) else 50.0
    
    if _is_valid(k_1):
        stoch_bullish = k_1 > d_1 and k_1 > stoch_min
        result["stoch_bullish_cross"] = stoch_bullish
        result["details"]["stoch_k"] = round(float(k_1), 1)
        result["details"]["stoch_d"] = round(float(d_1), 1)
        
        if stoch_bullish:
            result["confidence"] += 1
    
    # All 3 conditions = momentum shift detected
    result["detected"] = result["confidence"] >= 3
    
    return result


def detect_breakout(
    closes: Sequence[float],
    highs: Sequence[float],
    volumes: Sequence[float],
    ema20: float,
    lookback: int = 20,
) -> dict:
    """
    Detect resistance breakout - signals at the breakout moment, not after.
    
    Conditions (ALL must be true):
    1. last_close > max(high[-lookback:-1]) — resistance broken
    2. last_close > ema20 — short-term trend confirmed
    3. volume[-1] >= 1.2 * avg(volume[-lookback:-1]) — volume confirmation
    
    Returns:
        {
            "detected": bool,
            "breakout_level": float,
            "breakout_pct": float,  # % above resistance
            "volume_ratio": float,
            "details": {...}
        }
    """
    result = {
        "detected": False,
        "breakout_level": 0.0,
        "breakout_pct": 0.0,
        "volume_ratio": 0.0,
        "details": {}
    }
    
    closes_arr = _to_float_array(closes)
    highs_arr = _to_float_array(highs)
    volumes_arr = _to_float_array(volumes)
    
    if len(closes_arr) < lookback + 1 or len(highs_arr) < lookback + 1 or len(volumes_arr) < lookback + 1:
        return result
    
    last_close = closes_arr[-1]
    last_volume = volumes_arr[-1]
    
    if not _is_valid(last_close) or not _is_valid(last_volume):
        return result
    
    # Get volume multiplier from config
    vol_mult = getattr(config, "BREAKOUT_VOLUME_MULTIPLIER", 1.2)
    
    # Resistance level = max of previous highs (excluding current bar)
    prev_highs = highs_arr[-lookback-1:-1]
    valid_highs = prev_highs[~np.isnan(prev_highs)]
    if len(valid_highs) == 0:
        return result
    
    resistance = float(np.max(valid_highs))
    result["breakout_level"] = resistance
    result["details"]["resistance_20bar"] = round(resistance, 6)
    
    # Condition 1: Price broke resistance
    broke_resistance = last_close > resistance
    if broke_resistance:
        result["breakout_pct"] = round((last_close - resistance) / resistance * 100, 2)
    
    # Condition 2: Price above EMA20
    above_ema = last_close > ema20 if ema20 > 0 and _is_valid(ema20) else False
    result["details"]["above_ema20"] = above_ema
    result["details"]["ema20"] = round(ema20, 6) if _is_valid(ema20) else None
    
    # Condition 3: Volume confirmation
    prev_volumes = volumes_arr[-lookback-1:-1]
    valid_volumes = prev_volumes[~np.isnan(prev_volumes)]
    avg_volume = float(np.mean(valid_volumes)) if len(valid_volumes) > 0 else 0
    
    volume_ratio = last_volume / avg_volume if avg_volume > 0 else 0.0
    result["volume_ratio"] = round(volume_ratio, 2)
    result["details"]["avg_volume_20bar"] = round(avg_volume, 2)
    
    volume_confirmed = volume_ratio >= vol_mult
    result["details"]["volume_confirmed"] = volume_confirmed
    
    # All conditions met = breakout detected
    result["detected"] = broke_resistance and above_ema and volume_confirmed
    
    return result


def stochastic_d(k_values: Sequence[float], period: int = 3) -> List[float]:
    """Calculate Stochastic %D (SMA of %K)."""
    k_arr = _to_float_array(k_values)
    length = len(k_arr)
    result = [NAN] * length
    
    if period <= 0 or length < period:
        return result
    
    for i in range(period - 1, length):
        window = k_arr[i - period + 1 : i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == period:
            result[i] = float(np.mean(valid))
    
    return result
