"""Futures-mode backtesting utility for the Binance USDT Signal Bot.

This module consumes the precomputed feature files produced by src/pre_computer.py
and simulates STRONG_BUY / ULTRA_BUY strategies under deterministic
commission and slippage settings. The goal is to evaluate strategy variants
without touching the live trading loop.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Literal

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd

try:
    from . import config as default_config  # type: ignore
    from . import indicators  # type: ignore
    from . import price_action  # type: ignore
    from . import rules  # type: ignore
except ImportError:  # pragma: no cover
    import config as default_config  # type: ignore
    import indicators  # type: ignore
    import price_action  # type: ignore
    import rules  # type: ignore

DEFAULT_CONFIG_MODULE = "config_futures"
DATA_DIR_15M_DEFAULT = "data/precomputed_15m"
DATA_DIR_1H_DEFAULT = "data/precomputed_1h"
DATA_DIR_1M_DEFAULT = "data"
DEFAULT_NUM_CYCLES = 10
INITIAL_EQUITY = 10_000.0
COMMISSION_RATE = 0.0008  # 0.08% per side
SLIPPAGE_RATE = 0.0001  # 0.01% unfavorable price adjustment per fill
DEFAULT_POSITION_PCT = 1.0  # Use 100% of equity notionally per trade by default
MAX_BACKTEST_WORKERS = 10
RESULTS_DIR_DEFAULT = "results"
TRADES_SUBDIR = "trades"
SUMMARY_FILENAME = "summary.md"
MIN_TRADES_RANKING_DEFAULT = 20
BARS_PER_DAY_15M = 96
CYCLE_JITTER_PCT = 0.05
MILLIS_PER_MINUTE = 60_000
MILLIS_PER_15M = 15 * MILLIS_PER_MINUTE
CUSTOM_SCORE_FORMULA_TEMPLATE = (
    "score = 0.6 * total_pnl_pct + 10 * profit_factor - 0.3 * abs(max_drawdown_pct)"
)

DEFAULT_SCORE_BUCKETS: Tuple[Tuple[float, float, str], ...] = (
    (8.0, 9.99, "score_08_09"),
    (10.0, 11.99, "score_10_11"),
    (12.0, float("inf"), "score_12_plus"),
)

Direction = Literal["long", "short"]
DIRECTION_LONG: Direction = "long"
DIRECTION_SHORT: Direction = "short"

COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "timestamp": ("timestamp",),
    "open": ("open", "Open"),
    "high": ("high", "High"),
    "low": ("low", "Low"),
    "close": ("close", "Close"),
    "volume": ("volume", "Volume"),
    "ema_fast": ("ema_fast", "ema20", "ema_20"),
    "ema_slow": ("ema_slow", "ema50", "ema_50"),
    "adx": ("adx",),
    "plus_di": ("plus_di", "+di", "plus_di14"),
    "minus_di": ("minus_di", "-di", "minus_di14"),
    "macd_line": ("macd", "macd_line"),
    "macd_signal": ("macd_signal", "signal_line"),
    "macd_hist": ("macd_hist", "macd_histogram"),
    "momentum": ("momentum", "momentum_10"),
    "awesome_osc": ("awesome_osc", "ao"),
    "rsi": ("rsi", "rsi_14"),
    "stoch_k": ("stoch_k", "stochastic_k"),
    "cci": ("cci",),
    "stoch_rsi": ("stoch_rsi",),
    "williams_r": ("williams_r", "wpr"),
    "ultimate_osc": ("ultimate_osc", "uo"),
    "obv": ("obv",),
    "bull_power": ("bull_power",),
    "bear_power": ("bear_power",),
    "volume_spike_factor": ("volume_spike_factor", "vol_spike_factor"),
    "volume_spike_flag": ("volume_spike", "volume_spike_flag"),
}

REQUIRED_OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def canonical_symbol_key(raw: str) -> str:
    cleaned = raw.upper()
    for suffix in ("_1M", "_15M", "_1H", "-1M", "-15M", "-1H"):
        if cleaned.endswith(suffix):
            return cleaned[: -len(suffix)]
    return cleaned


def scan_feature_directory(data_dir: Path, timeframe: str) -> Dict[str, Path]:
    timeframe = timeframe.lower()
    suffix = f"_{timeframe}_features"
    mapping: Dict[str, Path] = {}
    for path in sorted(data_dir.glob("*.parquet")):
        stem = path.stem
        lowered = stem.lower()
        if lowered.endswith(suffix):
            base = stem[: -len(suffix)]
            mapping[canonical_symbol_key(base)] = path
    if not mapping:
        raise FileNotFoundError(f"No parquet features found for timeframe {timeframe} under {data_dir}")
    return mapping


def scan_one_minute_directory(data_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sorted(data_dir.glob("*_1m.parquet")):
        stem = path.stem
        lowered = stem.lower()
        if lowered.endswith("_1m"):
            base = stem[: -len("_1m")]
            mapping[canonical_symbol_key(base)] = path
    if not mapping:
        raise FileNotFoundError(f"No 1m parquet files found under {data_dir}")
    return mapping


def list_symbol_artifacts(
    data_dir_15m: Path,
    data_dir_1h: Path,
    data_dir_1m: Path,
    symbols: Optional[Sequence[str]],
) -> Dict[str, Tuple[Path, Path, Path]]:
    files_15m = scan_feature_directory(data_dir_15m, "15m")
    files_1h = scan_feature_directory(data_dir_1h, "1h")
    files_1m = scan_one_minute_directory(data_dir_1m)
    requested = [canonical_symbol_key(sym) for sym in symbols] if symbols else list(files_15m.keys())
    artifacts: Dict[str, Tuple[Path, Path, Path]] = {}
    for symbol in requested:
        if symbol not in files_15m:
            raise FileNotFoundError(f"Missing 15m features for {symbol} in {data_dir_15m}")
        if symbol not in files_1h:
            raise FileNotFoundError(f"Missing 1h features for {symbol} in {data_dir_1h}")
        if symbol not in files_1m:
            raise FileNotFoundError(f"Missing 1m raw data for {symbol} in {data_dir_1m}")
        artifacts[symbol] = (files_15m[symbol], files_1h[symbol], files_1m[symbol])
    return artifacts

TRADE_CSV_COLUMNS = [
    "symbol",
    "strategy",
    "direction",
    "cycle_index",
    "tp_pct",
    "sl_pct",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "pnl_abs",
    "pnl_pct",
    "hold_bars",
    "exit_reason",
    "signal_label",
    "score_value",
    "score_bucket",
    "risk_tag",
]


@dataclass(frozen=True)
class FuturesStrategyConfig:
    name: str
    min_total_score: float
    ultra_threshold: float
    use_ultra_only: bool
    tp_grid_base_long: Sequence[float]
    sl_grid_base_long: Sequence[float]
    tp_grid_base_short: Sequence[float]
    sl_grid_base_short: Sequence[float]
    max_hold_bars: int
    position_size_pct: float = DEFAULT_POSITION_PCT


FUTURES_STRATEGY_CONFIGS: Dict[str, FuturesStrategyConfig] = {
    "fut_safe": FuturesStrategyConfig(
        name="fut_safe",
        min_total_score=12.0,
        ultra_threshold=14.5,
        use_ultra_only=False,
        tp_grid_base_long=(0.02, 0.025, 0.03),
        sl_grid_base_long=(0.01, 0.0125, 0.015),
        tp_grid_base_short=(0.015, 0.02, 0.03),
        sl_grid_base_short=(0.03, 0.04, 0.05),
        max_hold_bars=360,
    ),
    "fut_balanced": FuturesStrategyConfig(
        name="fut_balanced",
        min_total_score=11.5,
        ultra_threshold=14.0,
        use_ultra_only=False,
        tp_grid_base_long=(0.03, 0.035, 0.04),
        sl_grid_base_long=(0.0125, 0.0175, 0.02),
        tp_grid_base_short=(0.04, 0.05, 0.06),
        sl_grid_base_short=(0.02, 0.025, 0.03),
        max_hold_bars=480,
    ),
    "fut_aggressive": FuturesStrategyConfig(
        name="fut_aggressive",
        min_total_score=11.0,
        ultra_threshold=13.5,
        use_ultra_only=True,
        tp_grid_base_long=(0.04, 0.045, 0.055),
        sl_grid_base_long=(0.015, 0.02, 0.0275),
        tp_grid_base_short=(0.05, 0.06, 0.08),
        sl_grid_base_short=(0.025, 0.03, 0.04),
        max_hold_bars=600,
    ),
}


@dataclass
class Trade:
    symbol: str
    params_name: str
    direction: Direction
    cycle_index: int
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    holding_bars: int
    exit_reason: str
    signal_label: str
    score_value: float
    score_bucket: str
    risk_tag: Optional[str] = None


@dataclass
class BacktestResult:
    symbol: str
    params_name: str
    direction: Direction
    tp_pct: float
    sl_pct: float
    cycle_index: int
    trades: List[Trade]
    final_equity: float
    total_return_pct: float
    win_rate_pct: float
    avg_return_pct: float
    profit_factor: float
    max_drawdown_pct: float

    def to_summary_row(self) -> Dict[str, float]:
        return {
            "symbol": self.symbol,
            "params": self.params_name,
            "direction": self.direction,
            "cycle": self.cycle_index,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "trades": len(self.trades),
            "win_rate_pct": self.win_rate_pct,
            "avg_return_pct": self.avg_return_pct,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
        }


@dataclass(frozen=True)
class BacktestJob:
    symbol: str
    feature_15m_path: str
    feature_1h_path: str
    one_min_path: str
    cfg_module: str
    start: Optional[str]
    end: Optional[str]
    score_buckets: Tuple[Tuple[float, float, str], ...]
    results_dir: str
    strategies: Tuple[FuturesStrategyConfig, ...]
    num_cycles: int
    direction: Direction


@dataclass
class FeatureBundle:
    frame_15m: pd.DataFrame
    frame_1h: pd.DataFrame
    frame_1m: pd.DataFrame
    macd_hist_rising: pd.Series
    obv_change_pct: pd.Series
    price_change_pct: pd.Series
    quote_volume_24h: pd.Series
    htf_indexer: List[int]
    one_minute_ts: np.ndarray
    one_minute_open: np.ndarray
    one_minute_high: np.ndarray
    one_minute_low: np.ndarray
    one_minute_close: np.ndarray


def clamp_workers(value: int) -> int:
    if value <= 1:
        return 1
    return min(value, MAX_BACKTEST_WORKERS)


def parse_score_buckets(spec: str) -> Tuple[Tuple[float, float, str], ...]:
    if not spec:
        return DEFAULT_SCORE_BUCKETS
    buckets: List[Tuple[float, float, str]] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            range_part, label = raw.split(":", 1)
            label = label.strip() or raw
        else:
            range_part = raw
            label = raw.replace("+", "_plus").replace("-", "_")
        if "+" in range_part:
            low = float(range_part.replace("+", ""))
            high = float("inf")
        elif "-" in range_part:
            low_str, high_str = range_part.split("-", 1)
            low = float(low_str)
            high = float(high_str)
        else:
            low = float(range_part)
            high = float(range_part)
        buckets.append((low, high, label))
    return tuple(buckets)


def bucket_for_score(score: float, buckets: Sequence[Tuple[float, float, str]]) -> str:
    if score is None or np.isnan(score):
        return "score_unknown"
    for low, high, label in buckets:
        if low <= score <= high:
            return label
    return "score_unbucketed"


def resolve_config(module_name: str):
    if not module_name:
        module_name = DEFAULT_CONFIG_MODULE
    candidates = []
    if module_name.startswith("src."):
        candidates.append(module_name)
    else:
        candidates.append(f"src.{module_name}")
        candidates.append(module_name)

    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise ImportError(f"Unable to import config module {module_name}")


def list_symbol_files(data_dir: Path, symbols: Optional[Sequence[str]]) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    if symbols:
        for symbol in symbols:
            symbol_key = symbol.upper()
            candidates = [
                data_dir / f"{symbol_key}.parquet",
                data_dir / f"{symbol_key}_features.parquet",
                data_dir / f"{symbol_key}.npz",
                data_dir / f"{symbol_key}_features.npz",
            ]
            for candidate in candidates:
                if candidate.exists():
                    files[symbol_key] = candidate
                    break
            else:
                raise FileNotFoundError(f"No feature file found for {symbol} in {data_dir}")
        return files

    for path in data_dir.iterdir():
        if path.suffix.lower() not in {".parquet", ".npz"}:
            continue
        files[path.stem.upper()] = path
    if not files:
        raise FileNotFoundError(f"No parquet/npz feature files found in {data_dir}")
    return files


def load_feature_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    elif path.suffix.lower() == ".npz":
        with np.load(path) as data:
            frame = pd.DataFrame({k: data[k] for k in data.files})
    else:
        raise ValueError(f"Unsupported feature file format: {path.suffix}")
    if "timestamp" not in frame.columns:
        raise ValueError(f"Feature file {path} lacks a timestamp column")
    frame = frame.copy()
    frame.sort_values("timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def normalize_timestamp_ms(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series([], index=series.index, dtype="int64")
    if pd.api.types.is_datetime64_any_dtype(series):
        cleaned = series.dt.tz_localize(None) if getattr(series.dtype, "tz", None) else series
        return (cleaned.astype("int64", copy=False) // 1_000_000).astype(np.int64)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Timestamp column contains non-numeric values")
    numeric = numeric.astype(np.int64)
    max_ts = int(numeric.max()) if len(numeric) else 0
    if max_ts > 10**15:
        numeric = numeric // 1_000_000
    elif 0 < max_ts < 10**11:
        numeric = numeric * 1000
    return numeric.astype(np.int64)


def load_one_minute_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing 1m parquet file: {path}")
    frame = pd.read_parquet(path)
    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"1m parquet file {path} missing columns: {missing}")
    ordered = frame.loc[:, REQUIRED_OHLCV_COLUMNS].copy()
    ordered.sort_values("timestamp", inplace=True)
    ordered.reset_index(drop=True, inplace=True)
    for column in ("open", "high", "low", "close", "volume"):
        ordered[column] = ordered[column].astype(float)
    ordered["timestamp"] = normalize_timestamp_ms(ordered["timestamp"])
    return ordered


def apply_date_range(frame: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if not start and not end:
        return frame
    ts = frame["timestamp"].astype(np.int64)
    mask = pd.Series(True, index=frame.index)
    if start:
        start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        mask &= ts >= start_ms
    if end:
        end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        mask &= ts <= end_ms
    return frame.loc[mask].reset_index(drop=True)


def get_series(frame: pd.DataFrame, logical_name: str) -> pd.Series:
    aliases = COLUMN_ALIASES.get(logical_name, ())
    for alias in aliases:
        if alias in frame.columns:
            if logical_name == "timestamp":
                return frame[alias].astype(np.int64)
            return frame[alias].astype(float)
    dtype = np.int64 if logical_name == "timestamp" else float
    fill_value = 0 if logical_name == "timestamp" else np.nan
    return pd.Series(fill_value, index=frame.index, dtype=dtype)


def get_bool_series(frame: pd.DataFrame, logical_name: str) -> pd.Series:
    aliases = COLUMN_ALIASES.get(logical_name, ())
    for alias in aliases:
        if alias in frame.columns:
            series = frame[alias]
            if series.dtype == bool:
                return series
            return series.astype(float).fillna(0.0) > 0.0
    return pd.Series(False, index=frame.index)


def bind_runtime_modules(cfg) -> None:
    try:
        indicators.config = cfg  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    try:
        price_action.config = cfg  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    try:
        rules.config = cfg  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def determine_price_action_lookback(cfg) -> int:
    collapse_lb = getattr(cfg, "COLLAPSE_LOOKBACK_BARS", 96)
    green_lb = getattr(cfg, "STRONG_GREEN_LOOKBACK", 20)
    volume_lb = getattr(cfg, "VOLUME_LOOKBACK", 20) + 1
    ema_lb = getattr(cfg, "EMA_BREAK_LOOKBACK", 10) + getattr(cfg, "EMA_RETEST_LOOKBACK", 10) + 2
    return max(collapse_lb, green_lb, volume_lb, ema_lb)


def compute_macd_hist_rising_series(series: pd.Series, lookback: int) -> pd.Series:
    if lookback <= 1:
        return pd.Series(False, index=series.index)
    values = series.astype(float).to_numpy()
    flags = np.zeros_like(values, dtype=bool)
    for idx in range(lookback - 1, len(values)):
        window = values[idx - lookback + 1 : idx + 1]
        if np.isnan(window).any():
            continue
        diffs = np.diff(window)
        if np.all(diffs > 0):
            flags[idx] = True
    return pd.Series(flags, index=series.index)


def compute_obv_change_pct_series(series: pd.Series, lookback: int) -> pd.Series:
    values = series.astype(float).to_numpy()
    result = np.full_like(values, np.nan, dtype=float)
    if lookback <= 0:
        return pd.Series(result, index=series.index)
    for idx in range(lookback, len(values)):
        start = values[idx - lookback]
        end = values[idx]
        if np.isnan(start) or np.isnan(end) or abs(start) < 1e-8:
            continue
        result[idx] = (end - start) / abs(start) * 100.0
    return pd.Series(result, index=series.index)


def compute_price_change_pct_series(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    base = series.astype(float)
    shifted = base.shift(window)
    pct = (base / shifted - 1.0) * 100.0
    pct.replace([np.inf, -np.inf], np.nan, inplace=True)
    return pct.fillna(0.0)


def compute_quote_volume_series(closes: pd.Series, volumes: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        return pd.Series(np.zeros(len(closes)), index=closes.index)
    notional = closes.astype(float) * volumes.astype(float)
    return notional.rolling(window, min_periods=window).sum().fillna(0.0)


HTF_DEFAULT_CONTEXT = {
    "close_above_ema20": False,
    "ema20_slope_pct": 0.0,
    "macd_hist": 0.0,
}


def prepare_htf_frame(frame_1h: pd.DataFrame, cfg) -> pd.DataFrame:
    if frame_1h.empty:
        return frame_1h.loc[:, :].copy()
    htf = frame_1h.copy()
    htf.sort_values("timestamp", inplace=True)
    htf.reset_index(drop=True, inplace=True)
    if "ema_fast" not in htf.columns:
        htf["ema_fast"] = np.nan
    ema_series = htf["ema_fast"].astype(float)
    slope_lb = max(1, int(getattr(cfg, "HTF_EMA_SLOPE_LOOKBACK", 5)))
    shifted = ema_series.shift(slope_lb)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = ((ema_series / shifted) - 1.0) * 100.0
    slopes.replace([np.inf, -np.inf], np.nan, inplace=True)
    htf["ema20_slope_pct"] = slopes.fillna(0.0)
    close_series = htf["close"].astype(float)
    htf["close_above_ema20"] = (close_series > ema_series).fillna(False)
    if "macd_hist" not in htf.columns:
        htf["macd_hist"] = 0.0
    return htf


def build_htf_indexer(frame_15m: pd.DataFrame, frame_1h: pd.DataFrame) -> List[int]:
    if frame_1h.empty or frame_15m.empty:
        return [-1] * len(frame_15m)
    ref = frame_15m.loc[:, ["timestamp"]].copy()
    ref["row_index"] = np.arange(len(ref))
    htf = frame_1h.loc[:, ["timestamp"]].copy()
    htf["htf_index"] = np.arange(len(htf))
    aligned = pd.merge_asof(
        ref.sort_values("timestamp"),
        htf,
        on="timestamp",
        direction="backward",
    )
    aligned.sort_values("row_index", inplace=True)
    indices = aligned["htf_index"].fillna(-1).astype(int)
    return indices.tolist()


def build_htf_context_from_row(frame_1h: pd.DataFrame, row_idx: int) -> Dict[str, Any]:
    if frame_1h.empty or row_idx < 0:
        return dict(HTF_DEFAULT_CONTEXT)
    row = frame_1h.iloc[row_idx]
    return {
        "close_above_ema20": bool(row.get("close_above_ema20", False)),
        "ema20_slope_pct": float(row.get("ema20_slope_pct", 0.0) or 0.0),
        "macd_hist": float(row.get("macd_hist", 0.0) or 0.0),
    }


PA_BOOL_COLUMNS = (
    "pa_long_lower_wick",
    "pa_strong_green",
    "pa_very_strong_green",
    "pa_collapse_ok",
    "pa_no_collapse",
    "pa_ema_breakout",
    "pa_ema_retest",
    "pa_volume_spike",
    "pa_min_volume",
)


def price_action_columns_available(frame: pd.DataFrame) -> bool:
    return all(column in frame.columns for column in PA_BOOL_COLUMNS + ("pa_volume_spike_factor",))


def extract_price_action_signals(frame: pd.DataFrame, idx: int, cfg) -> Dict[str, Any]:
    if price_action_columns_available(frame):
        snapshot = frame.iloc[idx]
        return {
            "long_lower_wick": bool(snapshot["pa_long_lower_wick"]),
            "strong_green": bool(snapshot["pa_strong_green"]),
            "very_strong_green": bool(snapshot["pa_very_strong_green"]),
            "collapse_ok": bool(snapshot["pa_collapse_ok"]),
            "no_collapse": bool(snapshot["pa_no_collapse"]),
            "ema_breakout": bool(snapshot["pa_ema_breakout"]),
            "ema_retest": bool(snapshot["pa_ema_retest"]),
            "volume_spike": bool(snapshot["pa_volume_spike"]),
            "min_volume": bool(snapshot["pa_min_volume"]),
            "volume_spike_factor": float(snapshot.get("pa_volume_spike_factor", 0.0) or 0.0),
            "details": {},
        }

    lookback = determine_price_action_lookback(cfg)
    start = max(0, idx - lookback)
    window = frame.iloc[start : idx + 1]
    pa = price_action.analyze_price_action(
        window["open"].tolist(),
        window["high"].tolist(),
        window["low"].tolist(),
        window["close"].tolist(),
        window["volume"].tolist(),
        window.get("ema_fast", window["close"]).tolist(),
    )
    pa.setdefault("details", {})
    return pa


def _cooldown_blocked(timestamp_ms: int, last_signal_ts: Optional[int], cfg) -> bool:
    cooldown_minutes = getattr(cfg, "COOLDOWN_MINUTES", 0)
    if not cooldown_minutes or last_signal_ts is None:
        return False
    cooldown_ms = int(cooldown_minutes) * 60_000
    return (timestamp_ms - last_signal_ts) < cooldown_ms


def compute_signal_from_15m_1h_row(
    symbol: str,
    idx: int,
    bundle: FeatureBundle,
    cfg,
    last_signal_ts: Optional[int],
) -> Optional[rules.SignalResult]:
    frame = bundle.frame_15m
    if idx >= len(frame):
        return None
    row = frame.iloc[idx]
    price = float(row.get("close", np.nan))
    if np.isnan(price) or price <= 0:
        return None
    timestamp_ms = int(row.get("timestamp", 0))

    quote_volume = float(bundle.quote_volume_24h.iloc[idx]) if not np.isnan(bundle.quote_volume_24h.iloc[idx]) else 0.0
    price_change_pct = float(bundle.price_change_pct.iloc[idx]) if not np.isnan(bundle.price_change_pct.iloc[idx]) else 0.0

    if quote_volume < getattr(cfg, "MIN_24H_QUOTE_VOLUME", 0):
        return None
    if price < getattr(cfg, "MIN_PRICE_USDT", 0):
        return None
    if not (
        getattr(cfg, "MIN_24H_CHANGE", -100.0)
        <= price_change_pct
        <= getattr(cfg, "MAX_24H_CHANGE", 100.0)
    ):
        return None
    if _cooldown_blocked(timestamp_ms, last_signal_ts, cfg):
        return None

    ema_fast = float(row.get("ema_fast", np.nan))
    ema_slow = float(row.get("ema_slow", np.nan))
    if np.isnan(ema_fast) or np.isnan(ema_slow):
        return None

    adx = float(row.get("adx", np.nan))
    plus_di = float(row.get("plus_di", np.nan))
    minus_di = float(row.get("minus_di", np.nan))
    macd_hist = float(row.get("macd_hist", np.nan))
    momentum = float(row.get("momentum", np.nan))
    ao = float(row.get("awesome_osc", np.nan))
    rsi_val = float(row.get("rsi", np.nan))
    stoch_k = float(row.get("stoch_k", np.nan))
    cci = float(row.get("cci", np.nan))
    stoch_rsi = float(row.get("stoch_rsi", np.nan))
    williams_r = float(row.get("williams_r", np.nan))
    uo_val = float(row.get("ultimate_osc", np.nan))
    bull_power = float(row.get("bull_power", np.nan))
    bear_power = float(row.get("bear_power", np.nan))

    stoch_rsi_prev = None
    uo_prev = None
    if idx > 0:
        prev_row = frame.iloc[idx - 1]
        stoch_rsi_prev = float(prev_row.get("stoch_rsi", np.nan))
        if np.isnan(stoch_rsi_prev):
            stoch_rsi_prev = None
        uo_prev = float(prev_row.get("ultimate_osc", np.nan))
        if np.isnan(uo_prev):
            uo_prev = None

    macd_rising_val = bundle.macd_hist_rising.iloc[idx] if not bundle.macd_hist_rising.empty else False
    macd_hist_rising = bool(macd_rising_val) if not pd.isna(macd_rising_val) else False
    obv_change_pct = float(bundle.obv_change_pct.iloc[idx]) if not np.isnan(bundle.obv_change_pct.iloc[idx]) else 0.0
    pa_signals = extract_price_action_signals(frame, idx, cfg)

    trend_block = rules.compute_trend_block(
        price=price,
        ema20=ema_fast,
        ema50=ema_slow,
        adx=adx,
        plus_di=plus_di,
        minus_di=minus_di,
        macd_hist=macd_hist,
        macd_hist_rising=macd_hist_rising,
        momentum=momentum,
        ao=ao,
    )
    osc_block = rules.compute_osc_block(
        rsi_val=rsi_val,
        stoch_k=stoch_k,
        cci=cci,
        stoch_rsi=stoch_rsi,
        williams_r=williams_r,
        uo=uo_val,
        stoch_rsi_prev=stoch_rsi_prev,
        uo_prev=uo_prev,
    )
    vol_block = rules.compute_volume_block(
        bull_power=bull_power,
        bear_power=bear_power,
        volume_spike_factor=pa_signals.get("volume_spike_factor"),
        obv_change_pct=obv_change_pct,
    )
    pa_block = rules.compute_price_action_block(pa_signals)

    htf_row_idx = bundle.htf_indexer[idx] if idx < len(bundle.htf_indexer) else -1
    htf_context = build_htf_context_from_row(bundle.frame_1h, htf_row_idx)
    htf_block = rules.compute_htf_bonus(htf_context)

    meta = {
        "price": price,
        "price_change_pct": price_change_pct,
        "quote_volume": quote_volume,
    }

    signal = rules.decide_signal_label(
        trend_block=trend_block,
        osc_block=osc_block,
        vol_block=vol_block,
        pa_block=pa_block,
        htf_block=htf_block,
        meta=meta,
        rsi_value=rsi_val,
        symbol=symbol,
    )
    if signal.label not in {"STRONG_BUY", "ULTRA_BUY"}:
        return None
    signal.price = price
    signal.price_change_pct = price_change_pct
    signal.quote_volume = quote_volume
    signal.htf_price_above_ema = bool(htf_context.get("close_above_ema20"))
    signal.mtf_trend_confirmed = signal.htf_price_above_ema
    return signal




def get_futures_strategy_configs(names: Optional[Sequence[str]]) -> List[FuturesStrategyConfig]:
    if not names:
        return list(FUTURES_STRATEGY_CONFIGS.values())
    configs: List[FuturesStrategyConfig] = []
    for name in names:
        key = name.strip()
        if key not in FUTURES_STRATEGY_CONFIGS:
            raise KeyError(f"Unknown strategy config: {key}")
        configs.append(FUTURES_STRATEGY_CONFIGS[key])
    return configs


def deterministic_cycle_seed(strategy_name: str, cycle_index: int) -> int:
    digest = hashlib.sha256(f"{strategy_name}:{cycle_index}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def generate_tp_sl_grid_for_cycle(
    strategy: FuturesStrategyConfig,
    cycle_index: int,
    direction: Direction,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    seed = deterministic_cycle_seed(f"{strategy.name}_{direction}", cycle_index)
    rng = random.Random(seed)

    if direction == DIRECTION_LONG:
        base_tp = strategy.tp_grid_base_long
        base_sl = strategy.sl_grid_base_long
    else:
        base_tp = strategy.tp_grid_base_short
        base_sl = strategy.sl_grid_base_short

    def jitter(values: Sequence[float]) -> Tuple[float, ...]:
        mutated: List[float] = []
        for value in values:
            delta = (rng.random() * 2 - 1) * CYCLE_JITTER_PCT
            mutated.append(round(max(0.0005, value * (1 + delta)), 5))
        rng.shuffle(mutated)
        return tuple(mutated)

    return jitter(base_tp), jitter(base_sl)


def should_enter_trade(signal: rules.SignalResult, strategy: FuturesStrategyConfig) -> bool:
    if signal.label not in {"STRONG_BUY", "ULTRA_BUY"}:
        return False
    if signal.score_total < strategy.min_total_score:
        return False
    if strategy.use_ultra_only and signal.label != "ULTRA_BUY":
        return False
    if signal.label == "ULTRA_BUY" and signal.score_total < strategy.ultra_threshold:
        return False
    return True


def compute_hold_bars(entry_time_ms: int, exit_time_ms: int) -> int:
    elapsed = max(0, exit_time_ms - entry_time_ms)
    if elapsed == 0:
        return 1
    return max(1, int(math.ceil(elapsed / MILLIS_PER_15M)))


def apply_entry_slippage(price: float, direction: Direction) -> float:
    if direction == DIRECTION_LONG:
        return price * (1 + SLIPPAGE_RATE)
    return price * (1 - SLIPPAGE_RATE)


def apply_exit_slippage(price: float, direction: Direction) -> float:
    if direction == DIRECTION_LONG:
        return price * (1 - SLIPPAGE_RATE)
    return price * (1 + SLIPPAGE_RATE)


def advance_position_until_timestamp(
    position: Dict[str, Any],
    one_min_ts: np.ndarray,
    one_min_high: np.ndarray,
    one_min_low: np.ndarray,
    one_min_close: np.ndarray,
    target_ts: int,
) -> Optional[Dict[str, Any]]:
    idx = position.get("next_1m_idx", 0)
    length = len(one_min_ts)
    stop_price = position["stop_price"]
    target_price = position["target_price"]
    max_exit_ts = position["max_exit_ts"]
    direction = position["direction"]
    while idx < length and int(one_min_ts[idx]) <= target_ts:
        current_ts = int(one_min_ts[idx])
        close_price = float(one_min_close[idx])
        if current_ts >= max_exit_ts:
            exit_price = apply_exit_slippage(close_price, direction)
            position["next_1m_idx"] = idx + 1
            return {"exit_price": exit_price, "exit_time": current_ts, "exit_reason": "MAX_HOLD"}
        low = float(one_min_low[idx])
        high = float(one_min_high[idx])
        if direction == DIRECTION_LONG:
            touched_sl = not np.isnan(low) and low <= stop_price
            touched_tp = not np.isnan(high) and high >= target_price
        else:
            touched_sl = not np.isnan(high) and high >= stop_price
            touched_tp = not np.isnan(low) and low <= target_price
        if touched_sl or touched_tp:
            if touched_sl:
                exit_reason = "SL"
                exit_price_raw = stop_price
            else:
                exit_reason = "TP"
                exit_price_raw = target_price
            if touched_sl and touched_tp:
                exit_reason = "SL"
                exit_price_raw = stop_price
            exit_price = apply_exit_slippage(exit_price_raw, direction)
            position["next_1m_idx"] = idx + 1
            return {"exit_price": exit_price, "exit_time": current_ts, "exit_reason": exit_reason}
        idx += 1
    position["next_1m_idx"] = idx
    return None


def force_close_position(
    position: Dict[str, Any],
    one_min_ts: np.ndarray,
    one_min_close: np.ndarray,
    reason: str = "DATA_END",
) -> Optional[Dict[str, Any]]:
    if not len(one_min_ts):
        return None
    last_idx = min(position.get("next_1m_idx", len(one_min_ts) - 1), len(one_min_ts) - 1)
    exit_time = int(one_min_ts[last_idx])
    exit_price_raw = float(one_min_close[last_idx])
    exit_price = apply_exit_slippage(exit_price_raw, position["direction"])
    position["next_1m_idx"] = len(one_min_ts)
    return {"exit_price": exit_price, "exit_time": exit_time, "exit_reason": reason}


def build_feature_bundle(frame_15m: pd.DataFrame, frame_1h: pd.DataFrame, frame_1m: pd.DataFrame, cfg) -> FeatureBundle:
    frame_1h_prepped = prepare_htf_frame(frame_1h, cfg)
    macd_hist = get_series(frame_15m, "macd_hist")
    macd_rising = compute_macd_hist_rising_series(macd_hist, getattr(cfg, "MACD_HIST_RISING_BARS", 3))
    obv_series = get_series(frame_15m, "obv")
    obv_change = compute_obv_change_pct_series(obv_series, getattr(cfg, "OBV_TREND_LOOKBACK", 10))
    price_change = compute_price_change_pct_series(get_series(frame_15m, "close"), BARS_PER_DAY_15M)
    quote_volume = compute_quote_volume_series(get_series(frame_15m, "close"), get_series(frame_15m, "volume"), BARS_PER_DAY_15M)
    htf_indexer = build_htf_indexer(frame_15m, frame_1h_prepped)
    frame_1m_sorted = frame_1m.copy()
    frame_1m_sorted.sort_values("timestamp", inplace=True)
    frame_1m_sorted.reset_index(drop=True, inplace=True)
    one_min_ts = frame_1m_sorted["timestamp"].astype(np.int64).to_numpy()
    one_min_open = frame_1m_sorted["open"].astype(float).to_numpy()
    one_min_high = frame_1m_sorted["high"].astype(float).to_numpy()
    one_min_low = frame_1m_sorted["low"].astype(float).to_numpy()
    one_min_close = frame_1m_sorted["close"].astype(float).to_numpy()
    return FeatureBundle(
        frame_15m=frame_15m,
        frame_1h=frame_1h_prepped,
        frame_1m=frame_1m_sorted,
        macd_hist_rising=macd_rising,
        obv_change_pct=obv_change,
        price_change_pct=price_change,
        quote_volume_24h=quote_volume,
        htf_indexer=htf_indexer,
        one_minute_ts=one_min_ts,
        one_minute_open=one_min_open,
        one_minute_high=one_min_high,
        one_minute_low=one_min_low,
        one_minute_close=one_min_close,
    )


def simulate_symbol(
    symbol: str,
    bundle: FeatureBundle,
    strategy: FuturesStrategyConfig,
    cfg,
    tp_pct: float,
    sl_pct: float,
    score_buckets: Sequence[Tuple[float, float, str]],
    cycle_index: int,
    direction: Direction,
) -> Tuple[BacktestResult, List[Trade]]:
    frame = bundle.frame_15m
    if frame.empty or not len(bundle.one_minute_ts):
        empty = BacktestResult(
            symbol=symbol,
            params_name=strategy.name,
            direction=direction,
            tp_pct=tp_pct,
            sl_pct=sl_pct,
            cycle_index=cycle_index,
            trades=[],
            final_equity=INITIAL_EQUITY,
            total_return_pct=0.0,
            win_rate_pct=0.0,
            avg_return_pct=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
        )
        return empty, []

    timestamps = get_series(frame, "timestamp")

    equity = INITIAL_EQUITY
    equity_curve: List[float] = []
    trades: List[Trade] = []
    position: Optional[Dict[str, Any]] = None
    last_signal_ts: Optional[int] = None
    direction_multiplier = 1 if direction == DIRECTION_LONG else -1

    for idx in range(len(frame)):
        current_ts = int(timestamps.iloc[idx])
        if position is not None and idx >= position["next_15m_idx"]:
            exit_event = advance_position_until_timestamp(
                position,
                bundle.one_minute_ts,
                bundle.one_minute_high,
                bundle.one_minute_low,
                bundle.one_minute_close,
                current_ts,
            )
            if exit_event:
                qty = position["qty"]
                exit_price = float(exit_event["exit_price"])
                exit_commission = qty * exit_price * COMMISSION_RATE
                gross_pnl = (exit_price - position["entry_price"]) * qty * direction_multiplier
                net_pnl = gross_pnl - exit_commission
                equity += net_pnl
                hold_bars = compute_hold_bars(position["entry_time"], exit_event["exit_time"])
                trades.append(
                    Trade(
                        symbol=symbol,
                        params_name=strategy.name,
                        direction=direction,
                        cycle_index=cycle_index,
                        entry_time=int(position["entry_time"]),
                        exit_time=int(exit_event["exit_time"]),
                        entry_price=float(position["entry_price"]),
                        exit_price=float(exit_price),
                        pnl=float(net_pnl),
                        pnl_pct=float(net_pnl / (position["entry_price"] * qty) * 100.0),
                        holding_bars=hold_bars,
                        exit_reason=exit_event["exit_reason"],
                        signal_label=position["signal_label"],
                        score_value=float(position["score_value"]),
                        score_bucket=position["score_bucket"],
                        risk_tag=position.get("risk_tag"),
                    )
                )
                position = None
            else:
                position["next_15m_idx"] = idx + 1

        equity_curve.append(equity)

        if position is not None or idx >= len(frame) - 1:
            continue

        signal = compute_signal_from_15m_1h_row(symbol, idx, bundle, cfg, last_signal_ts)
        if not signal:
            continue
        last_signal_ts = int(timestamps.iloc[idx])
        if not should_enter_trade(signal, strategy):
            continue
        score_value = float(signal.score_total)
        if np.isnan(score_value):
            continue

        entry_idx_1m = int(np.searchsorted(bundle.one_minute_ts, current_ts, side="right"))
        if entry_idx_1m >= len(bundle.one_minute_ts):
            continue
        entry_open = bundle.one_minute_open[entry_idx_1m]
        if not np.isfinite(entry_open) or entry_open <= 0:
            continue
        entry_time_ms = int(bundle.one_minute_ts[entry_idx_1m])
        entry_price = float(apply_entry_slippage(entry_open, direction))
        qty = (equity * strategy.position_size_pct) / entry_price
        if qty <= 0:
            continue
        entry_commission = qty * entry_price * COMMISSION_RATE
        equity -= entry_commission
        score_bucket = bucket_for_score(score_value, score_buckets)
        max_exit_ts = entry_time_ms + int(strategy.max_hold_bars) * MILLIS_PER_15M
        if direction == DIRECTION_LONG:
            target_price = entry_price * (1 + tp_pct)
            stop_price = entry_price * (1 - sl_pct)
        else:
            target_price = entry_price * (1 - tp_pct)
            stop_price = entry_price * (1 + sl_pct)
        position = {
            "entry_price": entry_price,
            "qty": qty,
            "entry_time": entry_time_ms,
            "target_price": target_price,
            "stop_price": stop_price,
            "signal_label": signal.label,
            "score_value": score_value,
            "score_bucket": score_bucket,
            "risk_tag": signal.risk_tag,
            "next_1m_idx": entry_idx_1m,
            "next_15m_idx": idx + 1,
            "max_exit_ts": max_exit_ts,
            "direction": direction,
        }

    if position is not None:
        fallback_ts = int(bundle.one_minute_ts[-1])
        exit_event = advance_position_until_timestamp(
            position,
            bundle.one_minute_ts,
            bundle.one_minute_high,
            bundle.one_minute_low,
            bundle.one_minute_close,
            fallback_ts,
        )
        if exit_event is None:
            exit_event = force_close_position(position, bundle.one_minute_ts, bundle.one_minute_close)
        if exit_event is not None:
            qty = position["qty"]
            exit_price = float(exit_event["exit_price"])
            exit_commission = qty * exit_price * COMMISSION_RATE
            gross_pnl = (exit_price - position["entry_price"]) * qty * direction_multiplier
            net_pnl = gross_pnl - exit_commission
            equity += net_pnl
            hold_bars = compute_hold_bars(position["entry_time"], exit_event["exit_time"])
            trades.append(
                Trade(
                    symbol=symbol,
                    params_name=strategy.name,
                    direction=direction,
                    cycle_index=cycle_index,
                    entry_time=int(position["entry_time"]),
                    exit_time=int(exit_event["exit_time"]),
                    entry_price=float(position["entry_price"]),
                    exit_price=float(exit_price),
                    pnl=float(net_pnl),
                    pnl_pct=float(net_pnl / (position["entry_price"] * qty) * 100.0),
                    holding_bars=hold_bars,
                    exit_reason=exit_event.get("exit_reason", "DATA_END"),
                    signal_label=position["signal_label"],
                    score_value=float(position["score_value"]),
                    score_bucket=position["score_bucket"],
                    risk_tag=position.get("risk_tag"),
                )
            )
            equity_curve.append(equity)
        position = None

    stats = build_statistics(symbol, strategy.name, direction, trades, equity_curve, tp_pct, sl_pct, cycle_index)
    return stats, trades


def build_statistics(
    symbol: str,
    strategy_name: str,
    direction: Direction,
    trades: List[Trade],
    equity_curve: Sequence[float],
    tp_pct: float,
    sl_pct: float,
    cycle_index: int,
) -> BacktestResult:
    metrics = compute_performance_metrics(trades, equity_curve)
    return BacktestResult(
        symbol=symbol,
        params_name=strategy_name,
        direction=direction,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        cycle_index=cycle_index,
        trades=trades,
        final_equity=metrics["final_equity"],
        total_return_pct=metrics["total_return_pct"],
        win_rate_pct=metrics["win_rate_pct"],
        avg_return_pct=metrics["avg_return_pct"],
        profit_factor=metrics["profit_factor"],
        max_drawdown_pct=metrics["max_drawdown_pct"],
    )


def compute_performance_metrics(
    trades: Sequence[Trade],
    equity_curve: Sequence[float],
) -> Dict[str, float]:
    equity_curve = list(equity_curve) if equity_curve else [INITIAL_EQUITY]
    final_equity = equity_curve[-1]
    total_return_pct = (final_equity / INITIAL_EQUITY - 1) * 100.0

    if trades:
        pnl_values = [t.pnl for t in trades]
        profits = [p for p in pnl_values if p > 0]
        losses = [-p for p in pnl_values if p < 0]
        if losses:
            profit_factor = sum(profits) / sum(losses) if profits else 0.0
        else:
            profit_factor = float("inf") if profits else 0.0
        win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades) * 100.0
        avg_return = float(np.mean([t.pnl_pct for t in trades]))
    else:
        profit_factor = 0.0
        win_rate = 0.0
        avg_return = 0.0

    equity_arr = np.asarray(equity_curve, dtype=float)
    peaks = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - peaks) / peaks
    max_drawdown_pct = float(drawdowns.min() * 100.0)

    return {
        "final_equity": final_equity,
        "total_return_pct": total_return_pct,
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_return,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown_pct,
    }


def write_trades_csv(
    trades_dir: Path,
    tp_pct: float,
    sl_pct: float,
    trades: Sequence[Trade],
    direction: Direction,
) -> Path:
    trades_dir.mkdir(parents=True, exist_ok=True)
    tp_tag = int(round(tp_pct * 10_000))
    sl_tag = int(round(sl_pct * 10_000))
    filename = f"tp{tp_tag}_sl{sl_tag}_{direction}.csv"
    output_path = trades_dir / filename

    if not trades:
        return output_path

    rows = [
        {
            "symbol": trade.symbol,
            "strategy": trade.params_name,
            "direction": trade.direction,
            "cycle_index": trade.cycle_index,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl_abs": trade.pnl,
            "pnl_pct": trade.pnl_pct,
            "hold_bars": trade.holding_bars,
            "exit_reason": trade.exit_reason,
            "signal_label": trade.signal_label,
            "score_value": trade.score_value,
            "score_bucket": trade.score_bucket,
            "risk_tag": trade.risk_tag,
        }
        for trade in trades
    ]
    df = pd.DataFrame(rows, columns=TRADE_CSV_COLUMNS)
    header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=header, index=False)
    return output_path


def build_overall_stat_entry(
    result: BacktestResult,
    trades: Sequence[Trade],
    tp_pct: float,
    sl_pct: float,
    csv_path: Path,
) -> Dict[str, Any]:
    wins = sum(1 for trade in trades if trade.pnl > 0)
    losses = sum(1 for trade in trades if trade.pnl < 0)
    sum_trade_pnl_pct = float(sum(trade.pnl_pct for trade in trades))
    gross_profit_abs = float(sum(trade.pnl for trade in trades if trade.pnl > 0))
    gross_loss_abs = float(sum(-trade.pnl for trade in trades if trade.pnl < 0))
    net_pnl_abs = gross_profit_abs - gross_loss_abs
    return {
        "strategy": result.params_name,
        "direction": result.direction,
        "symbol": result.symbol,
        "cycle_index": result.cycle_index,
        "tp_pct": tp_pct,
        "sl_pct": sl_pct,
        "num_trades": len(trades),
        "win_rate_pct": result.win_rate_pct,
        "total_pnl_pct": result.total_return_pct,
        "avg_pnl_pct": result.avg_return_pct,
        "profit_factor": result.profit_factor,
        "max_drawdown_pct": result.max_drawdown_pct,
        "final_equity": result.final_equity,
        "trades_csv": str(csv_path),
        "wins_count": wins,
        "losses_count": losses,
        "sum_trade_pnl_pct": sum_trade_pnl_pct,
        "gross_profit_abs": gross_profit_abs,
        "gross_loss_abs": gross_loss_abs,
        "net_pnl_abs": net_pnl_abs,
        "initial_equity": INITIAL_EQUITY,
    }


def _execute_backtest_job(job: BacktestJob) -> List[Dict[str, Any]]:
    cfg = resolve_config(job.cfg_module)
    bind_runtime_modules(cfg)
    frame_15m = load_feature_frame(Path(job.feature_15m_path))
    frame_15m = apply_date_range(frame_15m, job.start, job.end)
    frame_1h = load_feature_frame(Path(job.feature_1h_path))
    frame_1h = apply_date_range(frame_1h, job.start, job.end)
    frame_1m = load_one_minute_frame(Path(job.one_min_path))
    frame_1m = apply_date_range(frame_1m, job.start, job.end)
    if frame_15m.empty:
        return []
    if frame_1m.empty:
        return []

    bundle = build_feature_bundle(frame_15m, frame_1h, frame_1m, cfg)
    stats: List[Dict[str, Any]] = []
    trades_dir = Path(job.results_dir) / TRADES_SUBDIR

    for strategy in job.strategies:
        for cycle in range(job.num_cycles):
            tp_grid, sl_grid = generate_tp_sl_grid_for_cycle(strategy, cycle, job.direction)
            for tp_pct in tp_grid:
                for sl_pct in sl_grid:
                    overall, trades = simulate_symbol(
                        job.symbol,
                        bundle,
                        strategy,
                        cfg,
                        tp_pct,
                        sl_pct,
                        job.score_buckets,
                        cycle,
                        job.direction,
                    )
                    csv_path = write_trades_csv(trades_dir, tp_pct, sl_pct, trades, job.direction)
                    stats.append(build_overall_stat_entry(overall, trades, tp_pct, sl_pct, csv_path))
    return stats


def aggregate_overall_stats(stats: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not stats:
        return []
    grouped: Dict[Tuple[str, float, float, Direction], Dict[str, Any]] = {}
    for entry in stats:
        key = (entry["strategy"], entry["tp_pct"], entry["sl_pct"], entry.get("direction", DIRECTION_LONG))
        bucket = grouped.get(key)
        if bucket is None:
            bucket = {
                "strategy": entry["strategy"],
                "direction": entry.get("direction", DIRECTION_LONG),
                "tp_pct": entry["tp_pct"],
                "sl_pct": entry["sl_pct"],
                "num_trades": 0,
                "wins": 0,
                "losses": 0,
                "sum_trade_pnl_pct": 0.0,
                "gross_profit_abs": 0.0,
                "gross_loss_abs": 0.0,
                "net_pnl_abs": 0.0,
                "members": 0,
                "max_drawdown_pct": None,
                "symbols": set(),
                "cycles": set(),
            }
            grouped[key] = bucket
        bucket["num_trades"] += entry.get("num_trades", 0)
        bucket["wins"] += entry.get("wins_count", 0)
        bucket["losses"] += entry.get("losses_count", 0)
        bucket["sum_trade_pnl_pct"] += entry.get("sum_trade_pnl_pct", 0.0)
        bucket["gross_profit_abs"] += entry.get("gross_profit_abs", 0.0)
        bucket["gross_loss_abs"] += entry.get("gross_loss_abs", 0.0)
        bucket["net_pnl_abs"] += entry.get("net_pnl_abs", 0.0)
        bucket["members"] += 1
        drawdown = entry.get("max_drawdown_pct")
        if drawdown is not None:
            if bucket["max_drawdown_pct"] is None:
                bucket["max_drawdown_pct"] = float(drawdown)
            else:
                bucket["max_drawdown_pct"] = min(bucket["max_drawdown_pct"], float(drawdown))
        symbol = entry.get("symbol")
        if symbol:
            bucket["symbols"].add(symbol)
        cycle_idx = entry.get("cycle_index")
        if cycle_idx is not None:
            bucket["cycles"].add(cycle_idx)

    aggregated: List[Dict[str, Any]] = []
    for bucket in grouped.values():
        total_trades = bucket["num_trades"]
        wins = bucket["wins"]
        losses = bucket["losses"]
        sum_trade_pnl_pct = bucket["sum_trade_pnl_pct"]
        gross_profit = bucket["gross_profit_abs"]
        gross_loss = bucket["gross_loss_abs"]
        net_pnl_abs = bucket["net_pnl_abs"]
        members = bucket["members"]
        initial_capital = INITIAL_EQUITY * members
        total_pnl_pct = (net_pnl_abs / initial_capital * 100.0) if initial_capital else 0.0
        avg_return_pct = (sum_trade_pnl_pct / total_trades) if total_trades else 0.0
        win_rate_pct = (wins / total_trades * 100.0) if total_trades else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss if gross_profit else 0.0
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0
        max_drawdown_pct = bucket["max_drawdown_pct"] if bucket["max_drawdown_pct"] is not None else 0.0
        aggregated.append(
            {
                "strategy": bucket["strategy"],
                "direction": bucket["direction"],
                "tp_pct": bucket["tp_pct"],
                "sl_pct": bucket["sl_pct"],
                "num_trades": total_trades,
                "win_rate_pct": win_rate_pct,
                "avg_return_pct": avg_return_pct,
                "profit_factor": profit_factor,
                "max_drawdown_pct": max_drawdown_pct,
                "final_equity": initial_capital + net_pnl_abs,
                "total_pnl_pct": total_pnl_pct,
                "symbols_covered": len(bucket["symbols"]),
                "cycles_covered": len(bucket["cycles"]),
                "wins_count": wins,
                "losses_count": losses,
                "gross_profit_abs": gross_profit,
                "gross_loss_abs": gross_loss,
            }
        )
    return aggregated


def run_backtests_controller(
    data_dir_15m: Path,
    data_dir_1h: Path,
    data_dir_1m: Path,
    symbols: Optional[Sequence[str]],
    strategies: Sequence[FuturesStrategyConfig],
    cfg_module: str,
    start: Optional[str],
    end: Optional[str],
    score_buckets: Tuple[Tuple[float, float, str], ...],
    results_dir: Path,
    max_workers: int,
    num_cycles: int,
    direction: Direction,
) -> List[Dict[str, Any]]:
    symbol_artifacts = list_symbol_artifacts(data_dir_15m, data_dir_1h, data_dir_1m, symbols)
    if not symbol_artifacts:
        return []

    jobs: List[BacktestJob] = []
    for symbol, (path_15m, path_1h, path_1m) in symbol_artifacts.items():
        jobs.append(
            BacktestJob(
                symbol=symbol,
                feature_15m_path=str(path_15m),
                feature_1h_path=str(path_1h),
                one_min_path=str(path_1m),
                cfg_module=cfg_module,
                start=start,
                end=end,
                score_buckets=score_buckets,
                results_dir=str(results_dir),
                strategies=tuple(strategies),
                num_cycles=num_cycles,
                direction=direction,
            )
        )

    if not jobs:
        return []

    max_workers = clamp_workers(max_workers)
    stats: List[Dict[str, Any]] = []

    if max_workers == 1:
        for job in jobs:
            stats.extend(_execute_backtest_job(job))
            print(
                f"[DONE] {job.symbol} ({len(job.strategies)} strategies x {job.num_cycles} cycles)"
            )
        return stats

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_execute_backtest_job, job): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            job_stats = future.result()
            stats.extend(job_stats)
            print(
                f"[DONE] {job.symbol} ({len(job.strategies)} strategies x {job.num_cycles} cycles)"
            )
    return stats


def combo_score(stat: Dict[str, Any], min_trades: int) -> float:
    if stat["num_trades"] < max(1, min_trades):
        return float("-inf")
    total_pnl = stat["total_pnl_pct"]
    profit_factor = stat["profit_factor"] if np.isfinite(stat["profit_factor"]) else 0.0
    drawdown = abs(stat["max_drawdown_pct"])
    return total_pnl * 0.6 + profit_factor * 10.0 - drawdown * 0.3


def build_rankings(
    stats: Sequence[Dict[str, Any]],
    min_trades: int,
) -> Dict[str, List[Dict[str, Any]]]:
    top_pnl = sorted(stats, key=lambda entry: entry["total_pnl_pct"], reverse=True)[:5]
    worst_pnl = sorted(stats, key=lambda entry: entry["total_pnl_pct"])[:5]

    eligible = [entry for entry in stats if entry["num_trades"] >= max(1, min_trades)]
    top_win = sorted(eligible, key=lambda entry: entry["win_rate_pct"], reverse=True)[:5]
    worst_win = sorted(eligible, key=lambda entry: entry["win_rate_pct"])[:5]

    scored: List[Dict[str, Any]] = []
    for entry in stats:
        score_value = combo_score(entry, min_trades)
        if score_value == float("-inf"):
            continue
        enriched = dict(entry)
        enriched["combo_score"] = score_value
        scored.append(enriched)
    top_score = sorted(scored, key=lambda entry: entry["combo_score"], reverse=True)[:5]
    worst_score = sorted(scored, key=lambda entry: entry["combo_score"])[:5]

    return {
        "pnl_top": top_pnl,
        "pnl_worst": worst_pnl,
        "win_top": top_win,
        "win_worst": worst_win,
        "combo_top": top_score,
        "combo_worst": worst_score,
    }


def format_ratio(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_summary_markdown(
    output_path: Path,
    meta: Dict[str, str],
    rankings: Dict[str, List[Dict[str, Any]]],
    min_trades: int,
) -> None:
    lines = [
        "# Futures Backtest Summary",
        f"- 15m data: `{meta['data_dir_15m']}`",
        f"- 1h data: `{meta['data_dir_1h']}`",
        f"- 1m data: `{meta['data_dir_1m']}`",
        f"- Strategies: {meta['strategies']}",
        f"- Symbols: {meta['symbols']}",
        f"- Cycles per strategy: {meta['num_cycles']}",
        f"- Direction: {meta['direction'].upper()}",
        f"- Date run: {meta['generated_at']}",
    ]

    def add_section(title: str, entries: List[Dict[str, Any]]) -> None:
        lines.append("")
        lines.append(f"## {title}")
        if not entries:
            lines.append("_No qualifying combinations._")
            return
        for idx, entry in enumerate(entries, start=1):
            lines.append(
                (
                    f"{idx}. {entry['direction']} | {entry['strategy']} | TP {format_ratio(entry['tp_pct'])} | SL {format_ratio(entry['sl_pct'])} | "
                    f"trades={entry['num_trades']} | win={entry['win_rate_pct']:.2f}% | pnl={entry['total_pnl_pct']:.2f}% | "
                    f"pf={entry['profit_factor']:.2f} | maxDD={entry['max_drawdown_pct']:.2f}% | "
                    f"symbols={entry.get('symbols_covered', 0)} | cycles={entry.get('cycles_covered', 0)}"
                )
            )

    add_section("Top 5 by Total PnL", rankings.get("pnl_top", []))
    add_section("Worst 5 by Total PnL", rankings.get("pnl_worst", []))
    add_section("Top 5 by Win Rate", rankings.get("win_top", []))
    add_section("Worst 5 by Win Rate", rankings.get("win_worst", []))
    lines.append("")
    lines.append(
        f"> {CUSTOM_SCORE_FORMULA_TEMPLATE} (min {max(1, min_trades)} trades)"
    )
    add_section("Top 5 by Composite Score", rankings.get("combo_top", []))
    add_section("Worst 5 by Composite Score", rankings.get("combo_worst", []))

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_symbol_list(raw: str) -> Optional[List[str]]:
    if not raw or raw.strip().lower() in {"", "all"}:
        return None
    return [token.strip().upper() for token in raw.split(",") if token.strip()]


def parse_strategy_names(raw: str) -> Optional[List[str]]:
    if not raw:
        return None
    return [token.strip() for token in raw.split(",") if token.strip()]


def determine_strategies_from_args(args) -> List[FuturesStrategyConfig]:
    selected: List[str] = []
    if getattr(args, "all_strategies", False):
        selected = list(FUTURES_STRATEGY_CONFIGS.keys())
    else:
        if getattr(args, "fut_safe", False):
            selected.append("fut_safe")
        if getattr(args, "fut_balanced", False):
            selected.append("fut_balanced")
        if getattr(args, "fut_aggressive", False):
            selected.append("fut_aggressive")
    if not selected:
        legacy = parse_strategy_names(getattr(args, "strategies", "") or getattr(args, "strategy_names", ""))
        if legacy:
            selected = legacy
    return get_futures_strategy_configs(selected if selected else None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Futures-mode backtester")
    parser.add_argument(
        "--data-dir-15m",
        default=DATA_DIR_15M_DEFAULT,
        help="Directory containing 15m feature parquet files",
    )
    parser.add_argument(
        "--data-dir-1h",
        default=DATA_DIR_1H_DEFAULT,
        help="Directory containing 1h feature parquet files",
    )
    parser.add_argument(
        "--data-dir-1m",
        default=DATA_DIR_1M_DEFAULT,
        help="Directory containing raw 1m OHLCV parquet files",
    )
    parser.add_argument(
        "--data-dir",
        help="(Deprecated) fallback directory for 15m features",
    )
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbol list or 'all' for every file",
    )
    parser.add_argument(
        "--strategies",
        default="",
        help="Comma-separated strategy names (default: all futures presets)",
    )
    parser.add_argument(
        "--strategy-names",
        default="",
        help=argparse.SUPPRESS,
    )
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument(
        "--long",
        dest="long_mode",
        action="store_true",
        help="Run backtest in long mode (default)",
    )
    direction_group.add_argument(
        "--short",
        dest="short_mode",
        action="store_true",
        help="Run backtest in short mode (contrarian)",
    )
    parser.add_argument("--fut-safe", dest="fut_safe", action="store_true", help="Include fut_safe strategy")
    parser.add_argument(
        "--fut-balanced",
        dest="fut_balanced",
        action="store_true",
        help="Include fut_balanced strategy",
    )
    parser.add_argument(
        "--fut-aggressive",
        dest="fut_aggressive",
        action="store_true",
        help="Include fut_aggressive strategy",
    )
    parser.add_argument(
        "--all",
        dest="all_strategies",
        action="store_true",
        help="Include all futures strategies (default)",
    )
    parser.add_argument("--config-module", default=DEFAULT_CONFIG_MODULE, help="Config module to import")
    parser.add_argument("--start", help="UTC start date, e.g. 2023-01-01", default="")
    parser.add_argument("--end", help="UTC end date, e.g. 2023-06-01", default="")
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR_DEFAULT,
        help="Directory where trade CSVs and summary.md will be stored",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum parallel workers (1-10) for backtests",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=DEFAULT_NUM_CYCLES,
        help="Number of deterministic TP/SL randomization cycles per strategy",
    )
    parser.add_argument(
        "--min-trades-ranking",
        type=int,
        default=MIN_TRADES_RANKING_DEFAULT,
        help="Minimum trades required for win-rate/custom rankings",
    )
    parser.add_argument(
        "--overall-csv",
        default="",
        help="Optional path to save overall statistics as CSV",
    )
    parser.add_argument(
        "--score-buckets",
        default="",
        help="Custom score buckets, e.g. '8-9,10-11,12+:hi' (low-high[:label])",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir_15m_arg = args.data_dir_15m or args.data_dir
    if not data_dir_15m_arg:
        raise ValueError("--data-dir-15m must be provided")
    data_dir_15m = Path(data_dir_15m_arg).expanduser().resolve()
    data_dir_1h = Path(args.data_dir_1h).expanduser().resolve()
    data_dir_1m = Path(args.data_dir_1m).expanduser().resolve()
    if not data_dir_15m.exists():
        raise FileNotFoundError(f"15m data directory does not exist: {data_dir_15m}")
    if not data_dir_1h.exists():
        raise FileNotFoundError(f"1h data directory does not exist: {data_dir_1h}")
    if not data_dir_1m.exists():
        raise FileNotFoundError(f"1m data directory does not exist: {data_dir_1m}")

    symbols = parse_symbol_list(args.symbols)
    strategies = determine_strategies_from_args(args)
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    score_buckets = parse_score_buckets(args.score_buckets)
    start = args.start or None
    end = args.end or None
    max_workers = clamp_workers(args.max_workers)
    num_cycles = max(1, int(args.num_cycles))
    direction = DIRECTION_SHORT if getattr(args, "short_mode", False) else DIRECTION_LONG

    overall_stats = run_backtests_controller(
        data_dir_15m,
        data_dir_1h,
        data_dir_1m,
        symbols,
        strategies,
        args.config_module,
        start,
        end,
        score_buckets,
        results_dir,
        max_workers,
        num_cycles,
        direction,
    )

    if not overall_stats:
        print("No backtest results produced (check filters or data)")
        return

    aggregated_stats = aggregate_overall_stats(overall_stats)
    if not aggregated_stats:
        print("No aggregated statistics produced")
        return

    df = pd.DataFrame(aggregated_stats)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False, justify="center"))

    if args.overall_csv:
        overall_path = Path(args.overall_csv).expanduser().resolve()
        overall_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(overall_path, index=False)
        print(f"Overall stats saved to {overall_path}")

    rankings = build_rankings(aggregated_stats, args.min_trades_ranking)
    meta = {
        "data_dir_15m": str(data_dir_15m),
        "data_dir_1h": str(data_dir_1h),
        "data_dir_1m": str(data_dir_1m),
        "strategies": ", ".join(strategy.name for strategy in strategies) or "NONE",
        "symbols": ", ".join(symbols) if symbols else "ALL",
        "num_cycles": str(num_cycles),
        "direction": direction,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    summary_path = results_dir / SUMMARY_FILENAME
    write_summary_markdown(summary_path, meta, rankings, args.min_trades_ranking)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
