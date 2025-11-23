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
