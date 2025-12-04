"""Signal Performance Analyzer for the Binance USDT Signal Bot.

This module analyzes what happens AFTER each signal fires:
- How much did price rise?
- How many bars until the peak?
- Did it hold or reverse quickly?

Uses the same signal generation logic as spot_backtest.py but focuses on
post-signal price behavior analysis rather than trade simulation.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import rules

# === CONSTANTS ===
MILLIS_PER_MINUTE = 60_000
MILLIS_PER_15M = 15 * MILLIS_PER_MINUTE
DEFAULT_DATA_DIR_15M = "data/precomputed_15m"
DEFAULT_DATA_DIR_1H = "data/precomputed_1h"
DEFAULT_OUTPUT_DIR = "results/signal_analysis"
DEFAULT_LOOKAHEAD_BARS = 48  # 48 × 15m = 12 hours
DEFAULT_MIN_WATCH_SCORE = 18
ERROR_LOG_PATH = Path("logs/error.log")

# Focus symbols for analysis
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


# === ZONE CLASSIFICATION HELPERS ===

def classify_rsi_zone(rsi: float) -> str:
    """Classify RSI value into meaningful zones."""
    if rsi < 30:
        return "deep_oversold"
    elif rsi < 35:
        return "oversold"
    elif rsi < 45:
        return "recovery"
    elif rsi < 55:
        return "neutral"
    elif rsi < 65:
        return "strong"
    else:
        return "overbought"


def classify_ema_zone(dist_pct: float) -> str:
    """Classify price distance from EMA20."""
    if dist_pct < -3.0:
        return "far_below"
    elif dist_pct < -1.0:
        return "below"
    elif dist_pct <= 1.0:
        return "near"
    elif dist_pct <= 3.0:
        return "above"
    else:
        return "far_above"


def classify_24h_zone(change_pct: float) -> str:
    """Classify 24h price change into zones."""
    if change_pct < -5.0:
        return "dump"
    elif change_pct < -2.0:
        return "down"
    elif change_pct <= 2.0:
        return "flat"
    elif change_pct <= 5.0:
        return "up"
    else:
        return "pump"


def classify_volume_zone(vol_ratio: float) -> str:
    """Classify volume relative to average."""
    if vol_ratio < 0.5:
        return "low"
    elif vol_ratio < 1.5:
        return "normal"
    elif vol_ratio < 2.5:
        return "high"
    else:
        return "spike"


def classify_macd_zone(value: float, rising: bool) -> str:
    """Classify MACD histogram state."""
    prefix = "positive" if value >= 0 else "negative"
    suffix = "rising" if rising else "falling"
    return f"{prefix}_{suffix}"


def count_consecutive_candles(opens: np.ndarray, closes: np.ndarray, idx: int, lookback: int = 10) -> Tuple[int, int]:
    """Count consecutive green and red candles ending at idx."""
    green_count = 0
    red_count = 0
    
    # Count consecutive green (close > open)
    for i in range(idx, max(idx - lookback, -1), -1):
        if closes[i] > opens[i]:
            green_count += 1
        else:
            break
    
    # Count consecutive red (close < open)
    for i in range(idx, max(idx - lookback, -1), -1):
        if closes[i] < opens[i]:
            red_count += 1
        else:
            break
    
    return green_count, red_count


@dataclass
class SignalPerformance:
    """Captures a signal and its post-signal performance metrics."""
    # === CORE FIELDS ===
    symbol: str
    signal_time: int  # millisecond timestamp
    signal_time_str: str
    signal_type: str  # WATCH_PREMIUM, STRONG_BUY, ULTRA_BUY
    entry_price: float
    score_core: int
    score_total: int
    
    # Block scores for analysis
    trend_score: int
    osc_score: int
    vol_score: int
    pa_score: int
    htf_bonus: int
    
    # Post-signal metrics
    max_rise_pct: float
    bars_to_peak: int
    max_drawdown_from_peak_pct: float
    final_return_pct: float
    held_above_entry_bars: int
    
    # Context at signal time
    rsi_at_signal: float
    change_24h_pct: float
    volume_24h: float
    
    # Signal reasons
    reasons: str = ""
    filter_notes: str = ""
    
    # === NEW CONTEXT FIELDS ===
    # RSI context
    rsi_zone: str = ""
    rsi_vs_config_min: float = 0.0
    rsi_change_3bar: float = 0.0
    
    # Trend context
    ema20_dist_pct: float = 0.0
    ema20_zone: str = ""
    consecutive_green: int = 0
    consecutive_red: int = 0
    
    # Momentum context
    macd_hist_value: float = 0.0
    macd_hist_zone: str = ""
    stoch_k_value: float = 0.0
    
    # Volume context
    volume_vs_avg: float = 0.0
    volume_zone: str = ""
    
    # Market context
    change_24h_zone: str = ""
    
    # Success flags (for easy filtering)
    win_1pct: bool = False
    win_2pct: bool = False
    win_3pct: bool = False
    win_5pct: bool = False

    def to_row(self) -> Dict[str, Any]:
        """Convert to dict for DataFrame export."""
        return {
            "symbol": self.symbol,
            "signal_time": self.signal_time,
            "signal_time_str": self.signal_time_str,
            "signal_type": self.signal_type,
            "entry_price": self.entry_price,
            "score_core": self.score_core,
            "score_total": self.score_total,
            "trend_score": self.trend_score,
            "osc_score": self.osc_score,
            "vol_score": self.vol_score,
            "pa_score": self.pa_score,
            "htf_bonus": self.htf_bonus,
            "max_rise_pct": self.max_rise_pct,
            "bars_to_peak": self.bars_to_peak,
            "max_drawdown_from_peak_pct": self.max_drawdown_from_peak_pct,
            "final_return_pct": self.final_return_pct,
            "held_above_entry_bars": self.held_above_entry_bars,
            "rsi_at_signal": self.rsi_at_signal,
            "change_24h_pct": self.change_24h_pct,
            "volume_24h": self.volume_24h,
            "reasons": self.reasons,
            "filter_notes": self.filter_notes,
            # New context fields
            "rsi_zone": self.rsi_zone,
            "rsi_vs_config_min": self.rsi_vs_config_min,
            "rsi_change_3bar": self.rsi_change_3bar,
            "ema20_dist_pct": self.ema20_dist_pct,
            "ema20_zone": self.ema20_zone,
            "consecutive_green": self.consecutive_green,
            "consecutive_red": self.consecutive_red,
            "macd_hist_value": self.macd_hist_value,
            "macd_hist_zone": self.macd_hist_zone,
            "stoch_k_value": self.stoch_k_value,
            "volume_vs_avg": self.volume_vs_avg,
            "volume_zone": self.volume_zone,
            "change_24h_zone": self.change_24h_zone,
            "win_1pct": self.win_1pct,
            "win_2pct": self.win_2pct,
            "win_3pct": self.win_3pct,
            "win_5pct": self.win_5pct,
        }


def log_error(module: str, message: str, *, symbol: Optional[str] = None, exc: Optional[BaseException] = None) -> None:
    """Append an error line to logs/error.log."""
    detail = message
    if exc is not None:
        detail = f"{message} | {exc.__class__.__name__}: {exc}"
    line = f"{datetime.now(timezone.utc).isoformat()} | {module} | {symbol or '-'} | {detail}\n"
    try:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        pass


def canonical_symbol(symbol: str) -> str:
    """Normalize symbol name by removing common suffixes."""
    name = symbol.upper()
    # Remove all known suffixes in order
    suffixes = ["_15M_FEATURES", "_1H_FEATURES", "_1M_FEATURES", "_1M", "_15M", "_1H", "_FEATURES"]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def discover_symbol_files(
    data_dir_15m: Path,
    data_dir_1h: Path,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Tuple[Path, Path]]:
    """Find matching 15m and 1h parquet files for requested symbols."""
    # Map canonical symbol names to file paths
    files_15m: Dict[str, Path] = {}
    for p in data_dir_15m.glob("*.parquet"):
        canon = canonical_symbol(p.stem)
        files_15m[canon] = p
    
    files_1h: Dict[str, Path] = {}
    for p in data_dir_1h.glob("*.parquet"):
        canon = canonical_symbol(p.stem)
        files_1h[canon] = p
    
    requested = [canonical_symbol(s) for s in symbols] if symbols else sorted(files_15m.keys())
    
    artifacts: Dict[str, Tuple[Path, Path]] = {}
    for sym in requested:
        path_15m = files_15m.get(sym)
        path_1h = files_1h.get(sym)
        if path_15m and path_1h:
            artifacts[sym] = (path_15m, path_1h)
        else:
            missing = []
            if not path_15m:
                missing.append("15m")
            if not path_1h:
                missing.append("1h")
            log_error("signal_analyzer.discovery", f"Missing {', '.join(missing)} data", symbol=sym)
    
    return artifacts


def load_parquet(path: Path, required_cols: List[str]) -> pd.DataFrame:
    """Load a parquet file and ensure required columns exist."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_parquet(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def prepare_15m_frame(df: pd.DataFrame, start_ts: Optional[int] = None, end_ts: Optional[int] = None) -> pd.DataFrame:
    """Enrich 15m frame with derived columns needed for signal generation."""
    filtered = df.copy()
    
    if start_ts is not None:
        filtered = filtered[filtered["timestamp"] >= start_ts]
    if end_ts is not None:
        filtered = filtered[filtered["timestamp"] <= end_ts]
    filtered = filtered.reset_index(drop=True)
    
    if filtered.empty:
        return filtered
    
    # 24h rolling quote volume
    filtered["quote_volume_24h"] = (
        (filtered["close"] * filtered["volume"])
        .rolling(96, min_periods=96)
        .sum()
        .fillna(0.0)
    )
    
    # 24h price change
    filtered["price_change_24h"] = (
        (filtered["close"] / filtered["close"].shift(96) - 1.0) * 100.0
    )
    
    # MA60
    filtered["ma60"] = (
        filtered["close"]
        .rolling(config.MA60_PERIOD, min_periods=config.MA60_PERIOD)
        .mean()
    )
    
    # MACD histogram rising flag
    macd_hist = filtered.get("macd_hist")
    if macd_hist is not None:
        lookback = config.MACD_HIST_RISING_BARS
        rising = np.zeros(len(filtered), dtype=bool)
        values = macd_hist.to_numpy(dtype=float, copy=False)
        for idx in range(lookback - 1, len(filtered)):
            window = values[idx - lookback + 1 : idx + 1]
            if not np.any(np.isnan(window)) and np.all(np.diff(window) > 0):
                rising[idx] = True
        filtered["macd_hist_rising_flag"] = rising
    
    # OBV change percent
    obv_series = filtered.get("obv")
    if obv_series is not None:
        obv_vals = obv_series.to_numpy(dtype=float, copy=False)
        obv_change_pct = np.zeros(len(filtered), dtype=float)
        obv_lookback = config.OBV_TREND_LOOKBACK
        for idx in range(obv_lookback, len(filtered)):
            start_val = obv_vals[idx - obv_lookback]
            end_val = obv_vals[idx]
            if start_val != 0:
                obv_change_pct[idx] = (end_val - start_val) / abs(start_val) * 100.0
        filtered["obv_change_pct"] = obv_change_pct
    
    # RSI momentum
    rsi_series = filtered.get("rsi")
    if rsi_series is not None:
        filtered["rsi_momentum"] = rsi_series.diff()
        filtered["rsi_momentum_avg"] = (
            filtered["rsi_momentum"]
            .rolling(config.RSI_MOMENTUM_LOOKBACK, min_periods=config.RSI_MOMENTUM_LOOKBACK)
            .mean()
        )
    
    return filtered


def prepare_1h_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich 1h frame with EMA slope."""
    enriched = df.copy()
    enriched = enriched.sort_values("timestamp").reset_index(drop=True)
    
    if "ema_fast" not in enriched.columns:
        raise ValueError("1h frame missing ema_fast column")
    
    lookback = config.HTF_EMA_SLOPE_LOOKBACK
    ema_vals = enriched["ema_fast"].to_numpy(dtype=float, copy=False)
    slope = np.zeros(len(enriched), dtype=float)
    
    for idx in range(lookback, len(enriched)):
        prev = ema_vals[idx - lookback]
        curr = ema_vals[idx]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[idx] = (curr / prev - 1.0) * 100.0
    
    enriched["ema20_slope_pct"] = slope
    return enriched


def build_htf_index(main_ts: np.ndarray, htf_ts: np.ndarray) -> np.ndarray:
    """Build index mapping each main timeframe bar to corresponding HTF bar."""
    idx = np.searchsorted(htf_ts, main_ts, side="right") - 1
    idx[idx < 0] = -1
    return idx


def analyze_post_signal_performance(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    entry_idx: int,
    entry_price: float,
    lookahead_bars: int = 48,
) -> Optional[Dict[str, Any]]:
    """
    Analyze price action after a signal fires.
    
    Args:
        closes: Array of close prices
        highs: Array of high prices
        lows: Array of low prices
        entry_idx: Index of the signal bar
        entry_price: Close price at signal (entry reference)
        lookahead_bars: Number of bars to analyze after signal
    
    Returns:
        Dict with performance metrics, or None if insufficient data
    """
    end_idx = min(entry_idx + lookahead_bars, len(closes) - 1)
    
    if end_idx <= entry_idx:
        return None
    
    # Get future bars (excluding entry bar itself)
    future_highs = highs[entry_idx + 1 : end_idx + 1]
    future_lows = lows[entry_idx + 1 : end_idx + 1]
    future_closes = closes[entry_idx + 1 : end_idx + 1]
    
    if len(future_highs) == 0:
        return None
    
    # Max rise (using highs)
    max_high = float(np.nanmax(future_highs))
    max_rise_pct = (max_high - entry_price) / entry_price * 100.0
    bars_to_peak = int(np.nanargmax(future_highs)) + 1
    
    # Drawdown from peak (after peak is reached)
    peak_idx = np.nanargmax(future_highs)
    max_drawdown_from_peak_pct = 0.0
    
    if peak_idx < len(future_lows) - 1:
        post_peak_lows = future_lows[peak_idx + 1 :]
        if len(post_peak_lows) > 0:
            min_after_peak = float(np.nanmin(post_peak_lows))
            if max_high > 0:
                max_drawdown_from_peak_pct = (max_high - min_after_peak) / max_high * 100.0
    
    # Final return
    final_close = float(future_closes[-1])
    final_return_pct = (final_close - entry_price) / entry_price * 100.0
    
    # Bars held above entry
    held_above_entry_bars = int(np.sum(future_closes > entry_price))
    
    return {
        "max_rise_pct": round(max_rise_pct, 4),
        "bars_to_peak": bars_to_peak,
        "max_drawdown_from_peak_pct": round(max_drawdown_from_peak_pct, 4),
        "final_return_pct": round(final_return_pct, 4),
        "held_above_entry_bars": held_above_entry_bars,
    }


def generate_signals_for_symbol(
    symbol: str,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    lookahead_bars: int = 48,
    min_watch_score: int = 18,
) -> List[SignalPerformance]:
    """
    Generate signals using rules.py logic and analyze post-signal performance.
    
    Replicates the signal generation from spot_backtest.py but tracks
    performance metrics instead of simulating trades.
    """
    if symbol in config.STABLE_SYMBOLS:
        return []
    
    signals: List[SignalPerformance] = []
    
    # Prepare data arrays
    main_ts = df_15m["timestamp"].to_numpy(dtype=np.int64)
    htf_ts = df_1h["timestamp"].to_numpy(dtype=np.int64)
    htf_indexer = build_htf_index(main_ts, htf_ts)
    
    closes = df_15m["close"].to_numpy(dtype=float, copy=False)
    highs = df_15m["high"].to_numpy(dtype=float, copy=False)
    lows = df_15m["low"].to_numpy(dtype=float, copy=False)
    opens = df_15m["open"].to_numpy(dtype=float, copy=False)
    volumes = df_15m["volume"].to_numpy(dtype=float, copy=False)
    rsi_values = df_15m["rsi"].to_numpy(dtype=float, copy=False)
    
    cooldown_ms = config.COOLDOWN_MINUTES * MILLIS_PER_MINUTE
    last_signal_ts = -cooldown_ms
    
    for idx in range(len(df_15m)):
        row = df_15m.iloc[idx]
        timestamp = int(row["timestamp"])
        
        # Cooldown check
        if timestamp - last_signal_ts < cooldown_ms:
            continue
        
        # Prefilter checks
        quote_vol = row.get("quote_volume_24h", 0)
        if quote_vol < config.MIN_24H_QUOTE_VOLUME:
            continue
        
        price = float(row["close"])
        if price < config.MIN_PRICE_USDT:
            continue
        
        change_24h = row.get("price_change_24h", np.nan)
        if math.isnan(change_24h) or not (config.MIN_24H_CHANGE <= change_24h <= config.MAX_24H_CHANGE):
            continue
        
        # HTF context
        htf_idx = htf_indexer[idx]
        if htf_idx < 0:
            continue
        
        htf_row = df_1h.iloc[htf_idx]
        htf_context = {
            "close_above_ema20": bool(htf_row["close"] > htf_row["ema_fast"]),
            "ema20_slope_pct": float(htf_row.get("ema20_slope_pct", 0.0)),
            "macd_hist": float(htf_row.get("macd_hist", 0.0)),
            "macd_line": float(htf_row.get("macd", 0.0)),
        }
        
        # Build PA signals dict from precomputed columns
        pa_signals = {
            "long_lower_wick": bool(row.get("pa_long_lower_wick", False)),
            "strong_green": bool(row.get("pa_strong_green", False)),
            "very_strong_green": bool(row.get("pa_very_strong_green", False)),
            "collapse_ok": bool(row.get("pa_collapse_ok", True)),
            "no_collapse": bool(row.get("pa_no_collapse", True)),
            "ema_breakout": bool(row.get("pa_ema_breakout", False)),
            "ema_retest": bool(row.get("pa_ema_retest", False)),
            "volume_spike": bool(row.get("pa_volume_spike", False)),
            "min_volume": bool(row.get("pa_min_volume", False)),
            "volume_spike_factor": float(row.get("pa_volume_spike_factor", 0.0)),
            "details": {},
        }
        
        # Compute block scores
        trend_block = rules.compute_trend_block(
            price=price,
            ema20=float(row["ema_fast"]) if not pd.isna(row["ema_fast"]) else None,
            ema50=float(row["ema_slow"]) if not pd.isna(row["ema_slow"]) else None,
            adx=float(row["adx"]) if not pd.isna(row["adx"]) else 0,
            plus_di=float(row["plus_di"]) if not pd.isna(row["plus_di"]) else 0,
            minus_di=float(row["minus_di"]) if not pd.isna(row["minus_di"]) else 0,
            macd_hist=float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0,
            macd_hist_rising=bool(row.get("macd_hist_rising_flag", False)),
            momentum=float(row["momentum"]) if not pd.isna(row["momentum"]) else 0,
            ao=float(row["awesome_osc"]) if not pd.isna(row["awesome_osc"]) else 0,
        )
        
        osc_block = rules.compute_osc_block(
            rsi_val=float(row["rsi"]) if not pd.isna(row["rsi"]) else 50,
            stoch_k=float(row["stoch_k"]) if not pd.isna(row["stoch_k"]) else 50,
            cci=float(row["cci"]) if not pd.isna(row["cci"]) else 0,
            stoch_rsi=float(row["stoch_rsi"]) if not pd.isna(row["stoch_rsi"]) else 50,
            williams_r=float(row["williams_r"]) if not pd.isna(row["williams_r"]) else -50,
            uo=float(row["ultimate_osc"]) if not pd.isna(row["ultimate_osc"]) else 50,
            stoch_rsi_prev=None,
            uo_prev=None,
        )
        
        vol_block = rules.compute_volume_block(
            bull_power=float(row["bull_power"]) if not pd.isna(row["bull_power"]) else 0,
            bear_power=float(row["bear_power"]) if not pd.isna(row["bear_power"]) else 0,
            volume_spike_factor=float(row.get("pa_volume_spike_factor", 0.0)),
            obv_change_pct=float(row.get("obv_change_pct", 0.0)),
        )
        
        pa_block = rules.compute_price_action_block(pa_signals)
        htf_block = rules.compute_htf_bonus(htf_context)
        
        # Meta for signal decision
        meta = {
            "price": price,
            "price_change_pct": float(change_24h),
            "quote_volume": float(quote_vol),
        }
        
        # Pre-signal context
        rsi_val = float(row["rsi"]) if not pd.isna(row["rsi"]) else 50
        pre_signal_context = {
            "last_close": price,
            "ema20_15m": float(row["ema_fast"]) if not pd.isna(row["ema_fast"]) else None,
            "ma60": float(row.get("ma60")) if not pd.isna(row.get("ma60")) else None,
            "macd_1h": htf_context.get("macd_line"),
            "macd_hist_1h": htf_context.get("macd_hist"),
            "rsi_value": rsi_val,
            "rsi_momentum_curr": float(row.get("rsi_momentum")) if not pd.isna(row.get("rsi_momentum")) else None,
            "rsi_momentum_avg": float(row.get("rsi_momentum_avg")) if not pd.isna(row.get("rsi_momentum_avg")) else None,
            "change_24h_pct": float(change_24h),
        }
        
        # Get signal result
        signal_result = rules.decide_signal_label(
            trend_block=trend_block,
            osc_block=osc_block,
            vol_block=vol_block,
            pa_block=pa_block,
            htf_block=htf_block,
            meta=meta,
            rsi_value=rsi_val,
            symbol=symbol,
            pre_signal_context=pre_signal_context,
        )
        
        # Classify signal type
        signal_type: Optional[str] = None
        dip_signal_label = getattr(config, "DIP_SIGNAL_LABEL", "DIP_ALERT")
        
        if signal_result.label == "ULTRA_BUY":
            signal_type = "ULTRA_BUY"
        elif signal_result.label == "STRONG_BUY":
            signal_type = "STRONG_BUY"
        elif signal_result.label == dip_signal_label:
            signal_type = "DIP_ALERT"
        elif signal_result.label == "WATCH" and signal_result.score_total >= min_watch_score:
            signal_type = "WATCH_PREMIUM"
        
        if signal_type is None:
            continue
        
        # Update cooldown
        last_signal_ts = timestamp
        
        # Analyze post-signal performance
        perf = analyze_post_signal_performance(
            closes=closes,
            highs=highs,
            lows=lows,
            entry_idx=idx,
            entry_price=price,
            lookahead_bars=lookahead_bars,
        )
        
        if perf is None:
            continue
        
        # Create signal performance record
        signal_time_str = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        
        # === ZONE CLASSIFICATION CONTEXT ===
        # RSI context
        rsi_zone = classify_rsi_zone(rsi_val)
        rsi_vs_config_min = round(rsi_val - config.RSI_STRONG_MIN, 2)
        
        # RSI 3-bar change
        rsi_change_3bar = 0.0
        if idx >= 3:
            rsi_3_bars_ago = rsi_values[idx - 3]
            if not np.isnan(rsi_3_bars_ago):
                rsi_change_3bar = round(rsi_val - rsi_3_bars_ago, 2)
        
        # EMA20 distance context
        ema20_val = float(row["ema_fast"]) if not pd.isna(row["ema_fast"]) else price
        ema20_dist_pct = round((price - ema20_val) / ema20_val * 100, 4) if ema20_val > 0 else 0.0
        ema20_zone = classify_ema_zone(ema20_dist_pct)
        
        # Consecutive green/red candles
        consecutive_green = 0
        consecutive_red = 0
        for k in range(idx, max(-1, idx - 10), -1):
            if closes[k] > opens[k]:
                consecutive_green += 1
            else:
                break
        for k in range(idx, max(-1, idx - 10), -1):
            if closes[k] < opens[k]:
                consecutive_red += 1
            else:
                break
        
        # MACD context
        macd_hist_value = float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else 0.0
        prev_macd_hist = 0.0
        if idx >= 1:
            prev_macd_hist = float(df_15m.iloc[idx - 1]["macd_hist"]) if not pd.isna(df_15m.iloc[idx - 1]["macd_hist"]) else 0.0
        macd_hist_zone = classify_macd_zone(macd_hist_value, prev_macd_hist)
        
        # Stochastic K value
        stoch_k_value = float(row["stoch_k"]) if not pd.isna(row["stoch_k"]) else 50.0
        
        # Volume context
        vol_avg_20 = np.nanmean(volumes[max(0, idx - 20):idx]) if idx >= 1 else volumes[idx]
        current_volume = volumes[idx]
        volume_vs_avg = round(current_volume / vol_avg_20, 2) if vol_avg_20 > 0 else 1.0
        volume_zone = classify_volume_zone(volume_vs_avg)
        
        # 24h change zone
        change_24h_zone = classify_24h_zone(float(change_24h))
        
        # Win thresholds
        win_1pct = perf["max_rise_pct"] >= 1.0
        win_2pct = perf["max_rise_pct"] >= 2.0
        win_3pct = perf["max_rise_pct"] >= 3.0
        win_5pct = perf["max_rise_pct"] >= 5.0
        
        sig = SignalPerformance(
            symbol=symbol,
            signal_time=timestamp,
            signal_time_str=signal_time_str,
            signal_type=signal_type,
            entry_price=price,
            score_core=signal_result.score_core,
            score_total=signal_result.score_total,
            trend_score=signal_result.trend_score,
            osc_score=signal_result.osc_score,
            vol_score=signal_result.vol_score,
            pa_score=signal_result.pa_score,
            htf_bonus=signal_result.htf_bonus,
            max_rise_pct=perf["max_rise_pct"],
            bars_to_peak=perf["bars_to_peak"],
            max_drawdown_from_peak_pct=perf["max_drawdown_from_peak_pct"],
            final_return_pct=perf["final_return_pct"],
            held_above_entry_bars=perf["held_above_entry_bars"],
            rsi_at_signal=rsi_val,
            change_24h_pct=float(change_24h),
            volume_24h=float(quote_vol),
            reasons="; ".join(signal_result.reasons or []),
            filter_notes="; ".join(signal_result.filter_notes or []) if signal_result.filter_notes else "",
            # Zone classification context
            rsi_zone=rsi_zone,
            rsi_vs_config_min=rsi_vs_config_min,
            rsi_change_3bar=rsi_change_3bar,
            ema20_dist_pct=ema20_dist_pct,
            ema20_zone=ema20_zone,
            consecutive_green=consecutive_green,
            consecutive_red=consecutive_red,
            macd_hist_value=macd_hist_value,
            macd_hist_zone=macd_hist_zone,
            stoch_k_value=stoch_k_value,
            volume_vs_avg=volume_vs_avg,
            volume_zone=volume_zone,
            change_24h_zone=change_24h_zone,
            win_1pct=win_1pct,
            win_2pct=win_2pct,
            win_3pct=win_3pct,
            win_5pct=win_5pct,
        )
        
        signals.append(sig)
    
    return signals


def compute_summary_statistics(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by signal type."""
    if signals_df.empty:
        return pd.DataFrame()
    
    summary_rows = []
    
    for signal_type in ["ULTRA_BUY", "STRONG_BUY", "WATCH_PREMIUM", "DIP_ALERT"]:
        subset = signals_df[signals_df["signal_type"] == signal_type]
        
        if subset.empty:
            continue
        
        count = len(subset)
        avg_max_rise = subset["max_rise_pct"].mean()
        avg_bars_to_peak = subset["bars_to_peak"].mean()
        avg_drawdown = subset["max_drawdown_from_peak_pct"].mean()
        avg_final_return = subset["final_return_pct"].mean()
        avg_held_bars = subset["held_above_entry_bars"].mean()
        avg_score = subset["score_total"].mean()
        
        # Win rates at different thresholds
        win_rate_1pct = (subset["max_rise_pct"] >= 1.0).mean() * 100
        win_rate_2pct = (subset["max_rise_pct"] >= 2.0).mean() * 100
        win_rate_3pct = (subset["max_rise_pct"] >= 3.0).mean() * 100
        
        # Positive final return rate
        positive_final_rate = (subset["final_return_pct"] > 0).mean() * 100
        
        summary_rows.append({
            "signal_type": signal_type,
            "count": count,
            "avg_score": round(avg_score, 1),
            "avg_max_rise_pct": round(avg_max_rise, 2),
            "avg_bars_to_peak": round(avg_bars_to_peak, 1),
            "avg_drawdown_pct": round(avg_drawdown, 2),
            "avg_final_return_pct": round(avg_final_return, 2),
            "avg_held_bars": round(avg_held_bars, 1),
            "win_rate_1pct": round(win_rate_1pct, 1),
            "win_rate_2pct": round(win_rate_2pct, 1),
            "win_rate_3pct": round(win_rate_3pct, 1),
            "positive_final_pct": round(positive_final_rate, 1),
        })
    
    return pd.DataFrame(summary_rows)


def compute_symbol_statistics(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Compute statistics grouped by symbol."""
    if signals_df.empty:
        return pd.DataFrame()
    
    symbol_rows = []
    
    for symbol in sorted(signals_df["symbol"].unique()):
        subset = signals_df[signals_df["symbol"] == symbol]
        
        count = len(subset)
        ultra_count = len(subset[subset["signal_type"] == "ULTRA_BUY"])
        strong_count = len(subset[subset["signal_type"] == "STRONG_BUY"])
        watch_count = len(subset[subset["signal_type"] == "WATCH_PREMIUM"])
        
        avg_max_rise = subset["max_rise_pct"].mean()
        avg_final_return = subset["final_return_pct"].mean()
        win_rate_2pct = (subset["max_rise_pct"] >= 2.0).mean() * 100
        
        symbol_rows.append({
            "symbol": symbol,
            "total_signals": count,
            "ultra_buy": ultra_count,
            "strong_buy": strong_count,
            "watch_premium": watch_count,
            "avg_max_rise_pct": round(avg_max_rise, 2),
            "avg_final_return_pct": round(avg_final_return, 2),
            "win_rate_2pct": round(win_rate_2pct, 1),
        })
    
    return pd.DataFrame(symbol_rows)


def analyze_success_patterns(df: pd.DataFrame, output_dir: Path) -> dict:
    """Analyze patterns in successful vs failed signals."""
    
    if df.empty:
        return {"total": 0, "wins": 0, "win_rate": 0}
    
    successful = df[df["win_2pct"] == True]
    failed = df[df["win_2pct"] == False]
    
    analysis = {
        "total": len(df),
        "wins": len(successful),
        "win_rate": round(len(successful) / len(df) * 100, 1) if len(df) > 0 else 0
    }
    
    # Group analyses
    group_cols = ["rsi_zone", "ema20_zone", "change_24h_zone", "macd_hist_zone", "volume_zone", "signal_type"]
    
    for col in group_cols:
        if col not in df.columns:
            continue
        grp = df.groupby(col).agg({
            "win_2pct": ["count", "sum", "mean"],
            "max_rise_pct": "mean"
        })
        grp.columns = ["count", "wins", "win_rate", "avg_rise"]
        grp["win_rate"] = (grp["win_rate"] * 100).round(1)
        grp["avg_rise"] = grp["avg_rise"].round(2)
        analysis[f"by_{col}"] = grp.to_dict("index")
    
    # Successful signal profile
    if len(successful) > 10:
        analysis["successful_profile"] = {
            "rsi_mean": round(successful["rsi_at_signal"].mean(), 1),
            "rsi_q25": round(successful["rsi_at_signal"].quantile(0.25), 0),
            "rsi_q75": round(successful["rsi_at_signal"].quantile(0.75), 0),
            "ema_dist_mean": round(successful["ema20_dist_pct"].mean(), 2),
            "change_24h_mean": round(successful["change_24h_pct"].mean(), 2),
        }
    
    # Config recommendations
    if len(successful) > 10:
        analysis["config_suggestions"] = {
            "RSI_STRONG_MIN": int(successful["rsi_at_signal"].quantile(0.25)),
            "RSI_STRONG_MAX": int(successful["rsi_at_signal"].quantile(0.75)),
            "PARABOLIC_EMA_DIST_PCT": round(successful["ema20_dist_pct"].quantile(0.95), 1),
        }
    
    # Save markdown report
    save_analysis_report(analysis, output_dir)
    
    return analysis


def save_analysis_report(analysis: dict, output_dir: Path):
    """Save analysis as markdown report."""
    
    with open(output_dir / "pattern_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Signal Success Pattern Analysis\n\n")
        f.write(f"**Total Signals:** {analysis['total']} | **Wins:** {analysis['wins']} | **Win Rate:** {analysis['win_rate']}%\n\n")
        
        # RSI Zone table
        if "by_rsi_zone" in analysis:
            f.write("## Win Rate by RSI Zone\n\n")
            f.write("| Zone | Count | Wins | Win Rate | Avg Rise |\n")
            f.write("|------|-------|------|----------|----------|\n")
            for zone, stats in analysis["by_rsi_zone"].items():
                f.write(f"| {zone} | {stats['count']} | {stats['wins']} | {stats['win_rate']}% | {stats['avg_rise']}% |\n")
            f.write("\n")
        
        # EMA Zone table
        if "by_ema20_zone" in analysis:
            f.write("## Win Rate by EMA20 Distance\n\n")
            f.write("| Zone | Count | Wins | Win Rate | Avg Rise |\n")
            f.write("|------|-------|------|----------|----------|\n")
            for zone, stats in analysis["by_ema20_zone"].items():
                f.write(f"| {zone} | {stats['count']} | {stats['wins']} | {stats['win_rate']}% | {stats['avg_rise']}% |\n")
            f.write("\n")
        
        # 24h Change Zone table
        if "by_change_24h_zone" in analysis:
            f.write("## Win Rate by 24h Change Zone\n\n")
            f.write("| Zone | Count | Wins | Win Rate | Avg Rise |\n")
            f.write("|------|-------|------|----------|----------|\n")
            for zone, stats in analysis["by_change_24h_zone"].items():
                f.write(f"| {zone} | {stats['count']} | {stats['wins']} | {stats['win_rate']}% | {stats['avg_rise']}% |\n")
            f.write("\n")
        
        # MACD Zone table
        if "by_macd_hist_zone" in analysis:
            f.write("## Win Rate by MACD Histogram Zone\n\n")
            f.write("| Zone | Count | Wins | Win Rate | Avg Rise |\n")
            f.write("|------|-------|------|----------|----------|\n")
            for zone, stats in analysis["by_macd_hist_zone"].items():
                f.write(f"| {zone} | {stats['count']} | {stats['wins']} | {stats['win_rate']}% | {stats['avg_rise']}% |\n")
            f.write("\n")
        
        # Volume Zone table
        if "by_volume_zone" in analysis:
            f.write("## Win Rate by Volume Zone\n\n")
            f.write("| Zone | Count | Wins | Win Rate | Avg Rise |\n")
            f.write("|------|-------|------|----------|----------|\n")
            for zone, stats in analysis["by_volume_zone"].items():
                f.write(f"| {zone} | {stats['count']} | {stats['wins']} | {stats['win_rate']}% | {stats['avg_rise']}% |\n")
            f.write("\n")
        
        # Signal Type table
        if "by_signal_type" in analysis:
            f.write("## Win Rate by Signal Type\n\n")
            f.write("| Type | Count | Wins | Win Rate | Avg Rise |\n")
            f.write("|------|-------|------|----------|----------|\n")
            for stype, stats in analysis["by_signal_type"].items():
                f.write(f"| {stype} | {stats['count']} | {stats['wins']} | {stats['win_rate']}% | {stats['avg_rise']}% |\n")
            f.write("\n")
        
        # Successful profile
        if "successful_profile" in analysis:
            sp = analysis["successful_profile"]
            f.write("## Successful Signal Profile\n\n")
            f.write(f"- RSI: {sp['rsi_mean']} (range: {sp['rsi_q25']}-{sp['rsi_q75']})\n")
            f.write(f"- EMA20 Distance: {sp['ema_dist_mean']}%\n")
            f.write(f"- 24h Change: {sp['change_24h_mean']}%\n\n")
        
        # Config suggestions
        if "config_suggestions" in analysis:
            f.write("## Config Recommendations\n\n")
            for param, val in analysis["config_suggestions"].items():
                f.write(f"- **{param}**: {val}\n")
    
    print(f"Pattern analysis saved to: {output_dir}/pattern_analysis.md")


def run_analysis(
    symbols: List[str],
    lookahead_bars: int = 48,
    min_watch_score: int = 18,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    data_dir_15m: str = DEFAULT_DATA_DIR_15M,
    data_dir_1h: str = DEFAULT_DATA_DIR_1H,
) -> pd.DataFrame:
    """
    Main entry point. Runs analysis for all symbols and produces reports.
    
    Args:
        symbols: List of symbols to analyze
        lookahead_bars: Number of bars to track after each signal
        min_watch_score: Minimum score_total for WATCH to be WATCH_PREMIUM
        output_dir: Directory for output files
        data_dir_15m: Directory containing 15m precomputed parquet files
        data_dir_1h: Directory containing 1h precomputed parquet files
    
    Returns:
        Summary DataFrame
    """
    print(f"\n{'='*60}")
    print("SIGNAL PERFORMANCE ANALYZER")
    print(f"{'='*60}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Lookahead: {lookahead_bars} bars ({lookahead_bars * 15 / 60:.1f} hours)")
    print(f"Min WATCH score for PREMIUM: {min_watch_score}")
    print(f"{'='*60}\n")
    
    # Discover files
    data_path_15m = Path(data_dir_15m)
    data_path_1h = Path(data_dir_1h)
    
    artifacts = discover_symbol_files(data_path_15m, data_path_1h, symbols)
    
    if not artifacts:
        print("ERROR: No symbols found with complete data!")
        return pd.DataFrame()
    
    print(f"Found {len(artifacts)} symbols with complete data\n")
    
    # Required columns for 15m data
    feature_cols = [
        "timestamp", "open", "high", "low", "close", "volume",
        "ema_fast", "ema_slow", "adx", "plus_di", "minus_di",
        "macd", "macd_signal", "macd_hist", "momentum", "awesome_osc",
        "rsi", "stoch_k", "cci", "stoch_rsi", "williams_r", "ultimate_osc",
        "obv", "bull_power", "bear_power",
        "pa_long_lower_wick", "pa_strong_green", "pa_very_strong_green",
        "pa_collapse_ok", "pa_no_collapse", "pa_ema_breakout", "pa_ema_retest",
        "pa_volume_spike", "pa_min_volume", "pa_volume_spike_factor",
    ]
    
    htf_cols = ["timestamp", "open", "high", "low", "close", "volume", "ema_fast", "macd", "macd_hist"]
    
    all_signals: List[Dict[str, Any]] = []
    
    for symbol, (path_15m, path_1h) in artifacts.items():
        print(f"Processing {symbol}...", end=" ")
        
        try:
            # Load data
            df_15m = load_parquet(path_15m, feature_cols)
            df_15m = prepare_15m_frame(df_15m)
            
            df_1h = load_parquet(path_1h, htf_cols)
            df_1h = prepare_1h_frame(df_1h)
            
            if df_15m.empty or df_1h.empty:
                print("SKIPPED (empty data)")
                continue
            
            # Generate and analyze signals
            signals = generate_signals_for_symbol(
                symbol=symbol,
                df_15m=df_15m,
                df_1h=df_1h,
                lookahead_bars=lookahead_bars,
                min_watch_score=min_watch_score,
            )
            
            print(f"{len(signals)} signals found")
            
            for sig in signals:
                all_signals.append(sig.to_row())
        
        except Exception as exc:
            log_error("signal_analyzer.run", f"Processing failed: {exc}", symbol=symbol, exc=exc)
            print(f"ERROR: {exc}")
    
    if not all_signals:
        print("\nNo signals generated!")
        return pd.DataFrame()
    
    # Create DataFrames
    signals_df = pd.DataFrame(all_signals)
    summary_df = compute_summary_statistics(signals_df)
    symbol_df = compute_symbol_statistics(signals_df)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed signals
    signals_path = output_path / "signals_detailed.csv"
    signals_df.to_csv(signals_path, index=False)
    print(f"\nDetailed signals saved to: {signals_path}")
    
    # Save summary
    summary_path = output_path / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")
    
    # Save symbol breakdown
    symbol_path = output_path / "by_symbol.csv"
    symbol_df.to_csv(symbol_path, index=False)
    print(f"Symbol breakdown saved to: {symbol_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY BY SIGNAL TYPE")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("BREAKDOWN BY SYMBOL")
    print(f"{'='*60}")
    print(symbol_df.to_string(index=False))
    
    # Print insights
    print(f"\n{'='*60}")
    print("KEY INSIGHTS")
    print(f"{'='*60}")
    
    total_signals = len(signals_df)
    print(f"Total signals analyzed: {total_signals}")
    
    if not summary_df.empty:
        best_type = summary_df.loc[summary_df["avg_max_rise_pct"].idxmax(), "signal_type"]
        best_rise = summary_df.loc[summary_df["avg_max_rise_pct"].idxmax(), "avg_max_rise_pct"]
        print(f"Best performing signal type: {best_type} (avg max rise: {best_rise:.2f}%)")
        
        best_win_type = summary_df.loc[summary_df["win_rate_2pct"].idxmax(), "signal_type"]
        best_win_rate = summary_df.loc[summary_df["win_rate_2pct"].idxmax(), "win_rate_2pct"]
        print(f"Highest 2% win rate: {best_win_type} ({best_win_rate:.1f}%)")
    
    if not symbol_df.empty:
        best_symbol = symbol_df.loc[symbol_df["avg_max_rise_pct"].idxmax(), "symbol"]
        best_symbol_rise = symbol_df.loc[symbol_df["avg_max_rise_pct"].idxmax(), "avg_max_rise_pct"]
        print(f"Best performing symbol: {best_symbol} (avg max rise: {best_symbol_rise:.2f}%)")
    
    # Run pattern analysis
    print("\n" + "="*50)
    print("PATTERN ANALYSIS")
    print("="*50)
    pattern_analysis = analyze_success_patterns(signals_df, output_path)
    
    if "successful_profile" in pattern_analysis:
        sp = pattern_analysis["successful_profile"]
        print(f"Successful RSI range: {sp['rsi_q25']}-{sp['rsi_q75']}")
    
    if "config_suggestions" in pattern_analysis:
        print("\nSuggested config changes:")
        for k, v in pattern_analysis["config_suggestions"].items():
            print(f"  {k} = {v}")
    
    return summary_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze signal performance for the Binance USDT Signal Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze default symbols (BTC, ETH, SOL, XRP)
  python -m signal_performance_analyzer

  # Analyze specific symbols
  python -m signal_performance_analyzer --symbols BTCUSDT ETHUSDT

  # Use longer lookahead window (24 hours)
  python -m signal_performance_analyzer --lookahead 96

  # Lower threshold for WATCH_PREMIUM
  python -m signal_performance_analyzer --min-watch-score 16
        """,
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Symbols to analyze (default: {', '.join(DEFAULT_SYMBOLS)})",
    )
    
    parser.add_argument(
        "--lookahead",
        type=int,
        default=DEFAULT_LOOKAHEAD_BARS,
        help=f"Bars to track after signal (default: {DEFAULT_LOOKAHEAD_BARS} = 12 hours)",
    )
    
    parser.add_argument(
        "--min-watch-score",
        type=int,
        default=DEFAULT_MIN_WATCH_SCORE,
        help=f"Min score_total for WATCH_PREMIUM (default: {DEFAULT_MIN_WATCH_SCORE})",
    )
    
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    
    parser.add_argument(
        "--data-dir-15m",
        default=DEFAULT_DATA_DIR_15M,
        help=f"15m precomputed data directory (default: {DEFAULT_DATA_DIR_15M})",
    )
    
    parser.add_argument(
        "--data-dir-1h",
        default=DEFAULT_DATA_DIR_1H,
        help=f"1h precomputed data directory (default: {DEFAULT_DATA_DIR_1H})",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    try:
        run_analysis(
            symbols=args.symbols,
            lookahead_bars=args.lookahead,
            min_watch_score=args.min_watch_score,
            output_dir=args.output_dir,
            data_dir_15m=args.data_dir_15m,
            data_dir_1h=args.data_dir_1h,
        )
    except Exception as exc:
        log_error("signal_analyzer.main", "Fatal error", exc=exc)
        raise


if __name__ == "__main__":
    main()
