"""Futures-mode backtesting utility for the Binance USDT Signal Bot.

This module consumes the precomputed feature files produced by src/pre_computer.py
and simulates simplified STRONG_BUY / ULTRA_BUY strategies under deterministic
commission and slippage settings. The goal is to evaluate strategy variants
without touching the live trading loop.
"""
from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_CONFIG_MODULE = "config_futures"
INITIAL_EQUITY = 10_000.0
COMMISSION_RATE = 0.0008  # 0.08% per side
SLIPPAGE_RATE = 0.0001  # 0.01% unfavorable price adjustment per fill
DEFAULT_POSITION_PCT = 1.0  # Use 100% of equity notionally per trade by default

DEFAULT_SCORE_BUCKETS: Tuple[Tuple[float, float, str], ...] = (
    (8.0, 9.99, "score_08_09"),
    (10.0, 11.99, "score_10_11"),
    (12.0, float("inf"), "score_12_plus"),
)
DEFAULT_TP_GRID = (0.02, 0.03, 0.04)
DEFAULT_SL_GRID = (0.01, 0.015, 0.02)

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


def parse_float_list(value: str) -> List[float]:
    if not value:
        return []
    floats: List[float] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        floats.append(float(token))
    return floats


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


@dataclass(frozen=True)
class StrategyParams:
    name: str
    min_total_score: float
    ultra_threshold: float
    use_ultra_only: bool
    tp_pct: float
    sl_pct: float
    max_hold_bars: int
    position_size_pct: float = DEFAULT_POSITION_PCT


@dataclass
class Trade:
    symbol: str
    params_name: str
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


@dataclass
class BacktestResult:
    symbol: str
    params_name: str
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
            "trades": len(self.trades),
            "win_rate_pct": self.win_rate_pct,
            "avg_return_pct": self.avg_return_pct,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
        }


PRESET_PARAMS: Tuple[StrategyParams, ...] = (
    StrategyParams(
        name="fut_v1",
        min_total_score=10.0,
        ultra_threshold=13.0,
        use_ultra_only=False,
        tp_pct=0.03,
        sl_pct=0.015,
        max_hold_bars=240,
    ),
    StrategyParams(
        name="fut_v2_ultra",
        min_total_score=11.0,
        ultra_threshold=14.0,
        use_ultra_only=True,
        tp_pct=0.04,
        sl_pct=0.02,
        max_hold_bars=360,
    ),
    StrategyParams(
        name="fut_v3_fast",
        min_total_score=9.5,
        ultra_threshold=12.5,
        use_ultra_only=False,
        tp_pct=0.025,
        sl_pct=0.012,
        max_hold_bars=180,
    ),
)


def resolve_config(module_name: str):
    if not module_name:
        module_name = DEFAULT_CONFIG_MODULE
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        if not module_name.startswith("src."):
            return importlib.import_module(f"src.{module_name}")
        raise


def list_symbol_files(data_dir: Path, symbols: Optional[Sequence[str]]) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    if symbols:
        for symbol in symbols:
            for suffix in (".parquet", ".npz"):
                candidate = data_dir / f"{symbol}{suffix}"
                if candidate.exists():
                    files[symbol] = candidate
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


def compute_total_scores(frame: pd.DataFrame, cfg) -> pd.Series:
    close = get_series(frame, "close")
    high = get_series(frame, "high")
    low = get_series(frame, "low")
    volume = get_series(frame, "volume")
    ema_fast = get_series(frame, "ema_fast")
    ema_slow = get_series(frame, "ema_slow")
    adx = get_series(frame, "adx")
    plus_di = get_series(frame, "plus_di")
    minus_di = get_series(frame, "minus_di")
    macd_hist = get_series(frame, "macd_hist")
    momentum = get_series(frame, "momentum")
    ao = get_series(frame, "awesome_osc")
    rsi = get_series(frame, "rsi")
    stoch_k = get_series(frame, "stoch_k")
    cci = get_series(frame, "cci")
    stoch_rsi = get_series(frame, "stoch_rsi")
    williams_r = get_series(frame, "williams_r")
    ultimate_osc = get_series(frame, "ultimate_osc")
    obv = get_series(frame, "obv")
    bull_power = get_series(frame, "bull_power")
    bear_power = get_series(frame, "bear_power")
    volume_spike_factor = get_series(frame, "volume_spike_factor")
    volume_spike_flag = get_bool_series(frame, "volume_spike_flag")

    score = pd.Series(0.0, index=frame.index, dtype=float)

    score += ((close > ema_fast) & (ema_fast > ema_slow)).astype(float) * 2.0
    score += ((close > ema_fast) & (ema_slow.notna())).astype(float) * 0.5
    score += (adx >= getattr(cfg, "ADX_STRONG_TREND", 20.0)).astype(float)
    score += (plus_di > minus_di).astype(float)
    score += (macd_hist > 0).astype(float)
    score += (macd_hist.diff() > 0).astype(float) * 0.5
    score += (momentum > 0).astype(float)
    score += (ao > 0).astype(float) * 0.5

    score += (
        (rsi.between(getattr(cfg, "RSI_HEALTHY_MIN", 45), getattr(cfg, "RSI_HEALTHY_MAX", 65)))
    ).astype(float)
    score += (stoch_k > 50).astype(float) * 0.5
    score += (cci > 0).astype(float) * 0.5
    score += (stoch_rsi > 50).astype(float) * 0.5
    score += (williams_r > -80).astype(float) * 0.5
    score += (ultimate_osc > getattr(cfg, "UO_BULLISH", 50)).astype(float)

    obv_lookback = getattr(cfg, "OBV_TREND_LOOKBACK", 10)
    obv_change = obv - obv.shift(obv_lookback)
    score += (obv_change > 0).astype(float)
    score += ((bull_power - bear_power) > 0).astype(float)

    if not volume_spike_factor.isna().all():
        score += (volume_spike_factor >= getattr(cfg, "VOLUME_SPIKE_MULTIPLIER", 1.5)).astype(float)
    else:
        score += volume_spike_flag.astype(float)

    vol_window = int(getattr(cfg, "VOLUME_LOOKBACK", 20))
    if vol_window > 1:
        avg_volume = volume.rolling(vol_window).mean()
        score += ((volume > avg_volume * getattr(cfg, "VOLUME_SPIKE_MULTIPLIER", 1.5))).astype(float) * 0.5

    score[(ema_fast.isna()) | (ema_slow.isna())] = np.nan
    return score


def label_signals(scores: pd.Series, params: StrategyParams) -> pd.Series:
    labels = []
    for value in scores.fillna(-np.inf):
        if value < params.min_total_score:
            labels.append("NONE")
            continue
        label = "ULTRA_BUY" if value >= params.ultra_threshold else "STRONG_BUY"
        if params.use_ultra_only and label != "ULTRA_BUY":
            labels.append("NONE")
        else:
            labels.append(label)
    return pd.Series(labels, index=scores.index)


def get_param_presets(names: Optional[Sequence[str]]) -> List[StrategyParams]:
    if not names:
        return list(PRESET_PARAMS)
    preset_map = {p.name: p for p in PRESET_PARAMS}
    selected: List[StrategyParams] = []
    for name in names:
        if name not in preset_map:
            raise KeyError(f"Unknown strategy preset: {name}")
        selected.append(preset_map[name])
    return selected


def simulate_symbol(
    symbol: str,
    frame: pd.DataFrame,
    params: StrategyParams,
    cfg,
    tp_pct: float,
    sl_pct: float,
    score_buckets: Sequence[Tuple[float, float, str]],
) -> Tuple[BacktestResult, List[Trade]]:
    if frame.empty:
        empty = BacktestResult(
            symbol=symbol,
            params_name=params.name,
            trades=[],
            final_equity=INITIAL_EQUITY,
            total_return_pct=0.0,
            win_rate_pct=0.0,
            avg_return_pct=0.0,
            profit_factor=0.0,
            max_drawdown_pct=0.0,
        )
        return empty, []

    total_scores = compute_total_scores(frame, cfg)
    signals = label_signals(total_scores, params)

    timestamps = get_series(frame, "timestamp")
    opens = get_series(frame, "open")
    highs = get_series(frame, "high")
    lows = get_series(frame, "low")
    closes = get_series(frame, "close")

    equity = INITIAL_EQUITY
    equity_curve: List[float] = []
    trades: List[Trade] = []
    position: Optional[Dict[str, Any]] = None

    for idx in range(len(frame)):
        # Manage open position first
        if position and idx >= int(position["entry_idx"]):
            exit_price, exit_reason = evaluate_exit(idx, highs, lows, closes, position)
            if exit_price is not None:
                qty = position["qty"]
                exit_commission = qty * exit_price * COMMISSION_RATE
                gross_pnl = (exit_price - position["entry_price"]) * qty
                net_pnl = gross_pnl - exit_commission
                equity += net_pnl

                trades.append(
                    Trade(
                        symbol=symbol,
                        params_name=params.name,
                        entry_time=int(position["entry_time"]),
                        exit_time=int(timestamps.iloc[idx]),
                        entry_price=float(position["entry_price"]),
                        exit_price=float(exit_price),
                        pnl=float(net_pnl),
                        pnl_pct=float(net_pnl / (position["entry_price"] * qty) * 100.0),
                        holding_bars=int(idx - position["entry_idx"] + 1),
                        exit_reason=exit_reason,
                        signal_label=position["signal_label"],
                        score_value=float(position["score_value"]),
                        score_bucket=position["score_bucket"],
                    )
                )
                position = None

        equity_curve.append(equity)

        if position is None and idx < len(frame) - 1:
            signal = signals.iloc[idx]
            if signal in {"STRONG_BUY", "ULTRA_BUY"}:
                score_value = float(total_scores.iloc[idx])
                if np.isnan(score_value):
                    continue
                entry_idx = idx + 1
                entry_price_raw = opens.iloc[entry_idx]
                if np.isnan(entry_price_raw) or entry_price_raw <= 0:
                    continue
                entry_price = float(entry_price_raw * (1 + SLIPPAGE_RATE))
                qty = (equity * params.position_size_pct) / entry_price
                if qty <= 0:
                    continue
                entry_commission = qty * entry_price * COMMISSION_RATE
                equity -= entry_commission
                score_bucket = bucket_for_score(score_value, score_buckets)
                position = {
                    "entry_idx": entry_idx,
                    "entry_price": entry_price,
                    "qty": qty,
                    "entry_time": timestamps.iloc[entry_idx],
                    "target_price": entry_price * (1 + tp_pct),
                    "stop_price": entry_price * (1 - sl_pct),
                    "max_exit_idx": min(entry_idx + params.max_hold_bars, len(frame) - 1),
                    "signal_label": signal,
                    "score_value": score_value,
                    "score_bucket": score_bucket,
                }

    # Force close any remaining position at the final close
    if position is not None:
        last_idx = len(frame) - 1
        final_price = float(closes.iloc[last_idx] * (1 - SLIPPAGE_RATE))
        qty = position["qty"]
        exit_commission = qty * final_price * COMMISSION_RATE
        gross_pnl = (final_price - position["entry_price"]) * qty
        net_pnl = gross_pnl - exit_commission
        equity += net_pnl
        trades.append(
            Trade(
                symbol=symbol,
                params_name=params.name,
                entry_time=int(position["entry_time"]),
                exit_time=int(timestamps.iloc[last_idx]),
                entry_price=float(position["entry_price"]),
                exit_price=float(final_price),
                pnl=float(net_pnl),
                pnl_pct=float(net_pnl / (position["entry_price"] * qty) * 100.0),
                holding_bars=int(last_idx - position["entry_idx"] + 1),
                exit_reason="forced",
                signal_label=position["signal_label"],
                score_value=float(position["score_value"]),
                score_bucket=position["score_bucket"],
            )
        )
        equity_curve.append(equity)

    stats = build_statistics(symbol, params, trades, equity_curve)
    return stats, trades


def evaluate_exit(
    idx: int,
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    position: Dict[str, Any],
) -> Tuple[Optional[float], str]:
    if idx < position["entry_idx"]:
        return None, ""
    high = highs.iloc[idx]
    low = lows.iloc[idx]
    close = closes.iloc[idx]
    target = position["target_price"]
    stop = position["stop_price"]

    if idx >= position["max_exit_idx"]:
        exit_price_raw = close
        return float(exit_price_raw * (1 - SLIPPAGE_RATE)), "time"

    if not np.isnan(low) and low <= stop:
        exit_price_raw = stop
        return float(exit_price_raw * (1 - SLIPPAGE_RATE)), "stop"

    if not np.isnan(high) and high >= target:
        exit_price_raw = target
        return float(exit_price_raw * (1 - SLIPPAGE_RATE)), "tp"

    return None, ""


def build_statistics(
    symbol: str,
    params: StrategyParams,
    trades: List[Trade],
    equity_curve: Sequence[float],
) -> BacktestResult:
    metrics = compute_performance_metrics(trades, equity_curve)
    return BacktestResult(
        symbol=symbol,
        params_name=params.name,
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


def build_equity_curve_from_trades(trades: Sequence[Trade]) -> List[float]:
    equity = INITIAL_EQUITY
    curve = [equity]
    for trade in sorted(trades, key=lambda t: t.exit_time):
        equity += trade.pnl
        curve.append(equity)
    return curve


def build_bucket_rows(
    symbol: str,
    params_name: str,
    trades: List[Trade],
    score_buckets: Sequence[Tuple[float, float, str]],
    tp_pct: float,
    sl_pct: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    bucket_labels = [bucket[2] for bucket in score_buckets]
    grouped: Dict[str, List[Trade]] = {label: [] for label in bucket_labels}
    extra_labels: Dict[str, List[Trade]] = {}

    for trade in trades:
        label = trade.score_bucket
        if label in grouped:
            grouped[label].append(trade)
        else:
            extra_labels.setdefault(label, []).append(trade)

    for label, trade_list in grouped.items():
        if not trade_list:
            continue
        curve = build_equity_curve_from_trades(trade_list)
        metrics = compute_performance_metrics(trade_list, curve)
        rows.append(
            {
                "symbol": symbol,
                "params": params_name,
                "score_bucket": label,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "trades": len(trade_list),
                "win_rate_pct": metrics["win_rate_pct"],
                "avg_return_pct": metrics["avg_return_pct"],
                "profit_factor": metrics["profit_factor"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "final_equity": metrics["final_equity"],
                "total_return_pct": metrics["total_return_pct"],
            }
        )

    for label in sorted(extra_labels.keys()):
        trade_list = extra_labels[label]
        if not trade_list:
            continue
        curve = build_equity_curve_from_trades(trade_list)
        metrics = compute_performance_metrics(trade_list, curve)
        rows.append(
            {
                "symbol": symbol,
                "params": params_name,
                "score_bucket": label,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "trades": len(trade_list),
                "win_rate_pct": metrics["win_rate_pct"],
                "avg_return_pct": metrics["avg_return_pct"],
                "profit_factor": metrics["profit_factor"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "final_equity": metrics["final_equity"],
                "total_return_pct": metrics["total_return_pct"],
            }
        )

    return rows


def run_backtests(
    data_dir: Path,
    symbols: Optional[Sequence[str]],
    params: Sequence[StrategyParams],
    cfg,
    start: Optional[str],
    end: Optional[str],
    enable_grid: bool,
    tp_grid: Sequence[float],
    sl_grid: Sequence[float],
    score_buckets: Sequence[Tuple[float, float, str]],
) -> pd.DataFrame:
    symbol_files = list_symbol_files(data_dir, symbols)
    summaries: List[Dict[str, Any]] = []

    for symbol, path in symbol_files.items():
        frame = load_feature_frame(path)
        frame = apply_date_range(frame, start, end)
        for param_set in params:
            tp_values = list(tp_grid) if enable_grid and tp_grid else [param_set.tp_pct]
            sl_values = list(sl_grid) if enable_grid and sl_grid else [param_set.sl_pct]
            for tp_pct in tp_values:
                for sl_pct in sl_values:
                    overall, trades = simulate_symbol(
                        symbol,
                        frame,
                        param_set,
                        cfg,
                        tp_pct,
                        sl_pct,
                        score_buckets,
                    )
                    row = overall.to_summary_row()
                    row.update(
                        {
                            "score_bucket": "ALL",
                            "tp_pct": tp_pct,
                            "sl_pct": sl_pct,
                        }
                    )
                    summaries.append(row)
                    summaries.extend(
                        build_bucket_rows(
                            symbol,
                            param_set.name,
                            trades,
                            score_buckets,
                            tp_pct,
                            sl_pct,
                        )
                    )
    return pd.DataFrame(summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Futures-mode backtester")
    parser.add_argument("--data-dir", default="data/precomputed", help="Directory with precomputed parquet/npz files")
    parser.add_argument("--symbols", help="Comma-separated symbol list; default uses all files", default="")
    parser.add_argument("--param-presets", help="Comma-separated preset names", default="")
    parser.add_argument("--config-module", default=DEFAULT_CONFIG_MODULE, help="Config module to import")
    parser.add_argument("--start", help="UTC start date, e.g. 2023-01-01", default="")
    parser.add_argument("--end", help="UTC end date, e.g. 2023-06-01", default="")
    parser.add_argument("--output", help="Optional CSV output path", default="")
    parser.add_argument(
        "--enable-tp-sl-grid",
        action="store_true",
        help="Enable TP/SL grid search instead of single preset values",
    )
    parser.add_argument(
        "--tp-grid",
        default="",
        help="Comma separated TP percentages (e.g. 0.02,0.03,0.04) when grid search is enabled",
    )
    parser.add_argument(
        "--sl-grid",
        default="",
        help="Comma separated SL percentages (e.g. 0.01,0.015) when grid search is enabled",
    )
    parser.add_argument(
        "--score-buckets",
        default="",
        help="Custom score buckets, e.g. '8-9,10-11,12+:hi' (low-high[:label])",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    symbol_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    params = get_param_presets([p.strip() for p in args.param_presets.split(",") if p.strip()])
    cfg = resolve_config(args.config_module)
    start = args.start or None
    end = args.end or None

    tp_grid = parse_float_list(args.tp_grid)
    sl_grid = parse_float_list(args.sl_grid)
    enable_grid = bool(args.enable_tp_sl_grid or tp_grid or sl_grid)
    if enable_grid:
        if not tp_grid:
            tp_grid = list(DEFAULT_TP_GRID)
        if not sl_grid:
            sl_grid = list(DEFAULT_SL_GRID)
    score_buckets = parse_score_buckets(args.score_buckets)

    summary_df = run_backtests(
        data_dir,
        symbol_list or None,
        params,
        cfg,
        start,
        end,
        enable_grid,
        tp_grid,
        sl_grid,
        score_buckets,
    )
    if summary_df.empty:
        print("No backtest results produced (check filters or data)")
        return

    pd.set_option("display.max_columns", None)
    print(summary_df.to_string(index=False, justify="center"))

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        print(f"Summary saved to {output_path}")


if __name__ == "__main__":
    main()
