"""Spot-mode backtesting utility for the Binance USDT Signal Bot.

This module reuses the live signal engine (15m + 1h scoring stack) to
simulate spot trades under deterministic risk/reward presets.
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config
import rules

MILLIS_PER_MINUTE = 60_000
MILLIS_PER_15M = 15 * MILLIS_PER_MINUTE
DEFAULT_DATA_DIR_15M = "data/precomputed_15m"
DEFAULT_DATA_DIR_1H = "data/precomputed_1h"
DEFAULT_DATA_DIR_1M = "data"
DEFAULT_RESULTS_DIR = "results/spot_backtest"
SPOT_TRADES_FILENAME = "spot_trades.xlsx"
SPOT_SUMMARY_FILENAME = "spot_summary.xlsx"
DEFAULT_POSITION_NOTIONAL = 1_000.0
DEFAULT_FEE_RATE = 0.0005  # 0.05% per side
DEFAULT_SLIPPAGE_RATE = 0.0001  # 0.01% per side
MAX_WORKERS_CAP = 8
ERROR_LOG_PATH = Path("logs/error.log")


@dataclass(frozen=True)
class SpotRRConfig:
    name: str
    tp_pct: float
    sl_pct: float
    max_hold_bars: int


# Percentages as decimals (0.005 = +0.50%).
SPOT_RR_CONFIGS: Tuple[SpotRRConfig, ...] = (
    SpotRRConfig("RR_50_10", tp_pct=0.0050, sl_pct=0.0010, max_hold_bars=96),
    SpotRRConfig("RR_30_10", tp_pct=0.0030, sl_pct=0.0010, max_hold_bars=96),
    SpotRRConfig("RR_10_10", tp_pct=0.0010, sl_pct=0.0010, max_hold_bars=96),
)


@dataclass
class SpotTrade:
    trade_id: str
    symbol: str
    rr_name: str
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    quantity: float
    direction: str
    gross_pnl: float
    fee: float
    slippage: float
    net_pnl: float
    return_pct: float
    duration_seconds: float
    max_drawdown_pct: float
    entry_signal_type: str
    entry_reason: str
    exit_reason: str
    core_score: float
    trend_score: int
    osc_score: int
    vol_score: int
    pa_score: int
    htf_bonus: int
    rsi_15m: float
    rsi_momentum: float
    ma60: float
    macd_1h: float

    def to_row(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "rr_name": self.rr_name,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "direction": self.direction,
            "gross_pnl": self.gross_pnl,
            "fee": self.fee,
            "slippage": self.slippage,
            "net_pnl": self.net_pnl,
            "return_pct": self.return_pct,
            "duration_seconds": self.duration_seconds,
            "max_drawdown_pct": self.max_drawdown_pct,
            "entry_signal_type": self.entry_signal_type,
            "entry_reason": self.entry_reason,
            "exit_reason": self.exit_reason,
            "core_score": self.core_score,
            "trend_score": self.trend_score,
            "osc_score": self.osc_score,
            "vol_score": self.vol_score,
            "pa_score": self.pa_score,
            "htf_bonus": self.htf_bonus,
            "rsi_15m": self.rsi_15m,
            "rsi_momentum": self.rsi_momentum,
            "ma60": self.ma60,
            "macd_1h": self.macd_1h,
        }


@dataclass
class SummaryRow:
    symbol: str
    rr_name: str
    n_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    expectancy_pct: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    sortino: float
    avg_holding_time_seconds: float
    exposure_pct: float
    trades_per_month: float
    liquidity_metric: float
    profitability_score: float
    risk_score: float
    consistency_score: float
    efficiency_score: float
    liquidity_score: float
    total_score: float

    def to_row(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "rr_name": self.rr_name,
            "n_trades": self.n_trades,
            "win_rate": self.win_rate,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "profit_factor": self.profit_factor,
            "expectancy_pct": self.expectancy_pct,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "avg_holding_time_seconds": self.avg_holding_time_seconds,
            "exposure_pct": self.exposure_pct,
            "trades_per_month": self.trades_per_month,
            "liquidity_metric": self.liquidity_metric,
            "profitability_score": self.profitability_score,
            "risk_score": self.risk_score,
            "consistency_score": self.consistency_score,
            "efficiency_score": self.efficiency_score,
            "liquidity_score": self.liquidity_score,
            "total_score": self.total_score,
        }


TRADE_COLUMNS = [field.name for field in fields(SpotTrade)]
SUMMARY_COLUMNS = [field.name for field in fields(SummaryRow)]


@dataclass(frozen=True)
class SpotRunSettings:
    start_ts: Optional[int]
    end_ts: Optional[int]
    rr_configs: Tuple[SpotRRConfig, ...]
    position_notional: float
    fee_rate: float
    slippage_rate: float


@dataclass(frozen=True)
class SpotJob:
    symbol: str
    path_15m: str
    path_1h: str
    path_1m: str
    settings: SpotRunSettings


def log_error(module: str, message: str, *, symbol: Optional[str] = None, exc: Optional[BaseException] = None) -> None:
    """Append an error line to logs/error.log per repository standard."""
    detail = message
    if exc is not None:
        detail = f"{message} | {exc.__class__.__name__}: {exc}"
    line = f"{datetime.utcnow().isoformat()} | {module} | {symbol or '-'} | {detail}\n"
    try:
        ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spot backtest runner for the Binance USDT Signal Bot")
    parser.add_argument("--data-dir-15m", default=DEFAULT_DATA_DIR_15M, help="Directory with 15m feature parquet files")
    parser.add_argument("--data-dir-1h", default=DEFAULT_DATA_DIR_1H, help="Directory with 1h feature parquet files")
    parser.add_argument("--data-dir-1m", default=DEFAULT_DATA_DIR_1M, help="Directory with 1m OHLCV parquet files")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directory for output CSVs")
    parser.add_argument("--symbols", default="", help="Comma separated list of symbols to backtest (default: auto-discover)")
    parser.add_argument("--rr-profiles", default="", help="Comma separated RR profile names (default: all)")
    parser.add_argument("--start-date", default="", help="Inclusive start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="", help="Inclusive end date (YYYY-MM-DD)")
    parser.add_argument("--position-notional", type=float, default=DEFAULT_POSITION_NOTIONAL, help="Quote notional per trade")
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="Fee rate per side")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="Slippage per side")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel workers (ProcessPool)")
    return parser.parse_args()


def parse_symbols(raw: str) -> Optional[List[str]]:
    if not raw:
        return None
    symbols = [token.strip().upper() for token in raw.split(",") if token.strip()]
    return symbols or None


def parse_date(raw: str) -> Optional[int]:
    if not raw:
        return None
    dt = datetime.strptime(raw, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def clamp_workers(value: int) -> int:
    if value <= 1:
        return 1
    return min(value, MAX_WORKERS_CAP)


def resolve_rr_configs(raw: str) -> Tuple[SpotRRConfig, ...]:
    if not raw:
        return SPOT_RR_CONFIGS
    lookup = {cfg.name.upper(): cfg for cfg in SPOT_RR_CONFIGS}
    selected: List[SpotRRConfig] = []
    for token in raw.split(","):
        key = token.strip().upper()
        if not key:
            continue
        cfg = lookup.get(key)
        if not cfg:
            raise ValueError(f"Unknown RR profile: {token}")
        selected.append(cfg)
    if not selected:
        raise ValueError("No valid RR profiles selected")
    return tuple(selected)


def canonical_symbol(symbol: str) -> str:
    lowered = symbol.upper()
    for suffix in ("_15M_FEATURES", "_1H_FEATURES", "_1M", "_15M", "_1H", "_FEATURES"):
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def discover_symbol_artifacts(
    data_dir_15m: Path,
    data_dir_1h: Path,
    data_dir_1m: Path,
    symbols: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[Path, Path, Path]]:
    artifacts: Dict[str, Tuple[Path, Path, Path]] = {}
    files_15m = {canonical_symbol(path.stem): path for path in data_dir_15m.glob("*.parquet")}
    files_1h = {canonical_symbol(path.stem): path for path in data_dir_1h.glob("*.parquet")}
    files_1m = {canonical_symbol(path.stem): path for path in data_dir_1m.glob("*.parquet")}
    requested = [canonical_symbol(sym) for sym in symbols] if symbols else sorted(files_15m.keys())
    missing_records: List[str] = []
    for symbol in requested:
        path_15m = files_15m.get(symbol)
        path_1h = files_1h.get(symbol)
        path_1m = files_1m.get(symbol)
        if not all((path_15m, path_1h, path_1m)):
            missing_bits = []
            if not path_15m:
                missing_bits.append("15m features")
            if not path_1h:
                missing_bits.append("1h features")
            if not path_1m:
                missing_bits.append("1m data")
            detail = f"Missing {', '.join(missing_bits)}"
            log_error("spot_backtest.discovery", detail, symbol=symbol)
            missing_records.append(f"{symbol}: {detail}")
            continue
        artifacts[symbol] = (path_15m, path_1h, path_1m)
    if missing_records:
        message = "Missing required data for symbols:\n" + "\n".join(missing_records)
        print(message)
        raise RuntimeError(message)
    return artifacts


def load_parquet(path: Path, required: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    ordered = frame.loc[:, required].copy()
    ordered.sort_values("timestamp", inplace=True)
    ordered.reset_index(drop=True, inplace=True)
    return ordered


def prepare_15m_frame(frame: pd.DataFrame, start_ts: Optional[int], end_ts: Optional[int]) -> pd.DataFrame:
    filtered = frame.copy()
    if start_ts is not None:
        filtered = filtered[filtered["timestamp"] >= start_ts]
    if end_ts is not None:
        filtered = filtered[filtered["timestamp"] <= end_ts]
    filtered.reset_index(drop=True, inplace=True)
    if filtered.empty:
        return filtered

    filtered["quote_volume_24h"] = (
        (filtered["close"] * filtered["volume"]).rolling(96, min_periods=96).sum().fillna(0.0)
    )
    filtered["price_change_24h"] = (
        (filtered["close"] / filtered["close"].shift(96) - 1.0).mul(100.0)
    )
    filtered["ma60"] = filtered["close"].rolling(config.MA60_PERIOD, min_periods=config.MA60_PERIOD).mean()

    macd_hist = filtered.get("macd_hist")
    if macd_hist is None:
        raise ValueError("15m frame missing macd_hist column")
    lookback = config.MACD_HIST_RISING_BARS
    rising = np.zeros(len(filtered), dtype=bool)
    values = macd_hist.to_numpy(dtype=float, copy=False)
    for idx in range(lookback - 1, len(filtered)):
        window = values[idx - lookback + 1 : idx + 1]
        if np.any(np.isnan(window)):
            continue
        if np.all(np.diff(window) > 0):
            rising[idx] = True
    filtered["macd_hist_rising_flag"] = rising

    obv_series = filtered.get("obv")
    if obv_series is None:
        raise ValueError("15m frame missing obv column")
    obv_vals = obv_series.to_numpy(dtype=float, copy=False)
    obv_change_pct = np.zeros(len(filtered), dtype=float)
    obv_lookback = config.OBV_TREND_LOOKBACK
    for idx in range(len(filtered)):
        prior = idx - obv_lookback
        if prior < 0:
            continue
        start_val = obv_vals[prior]
        end_val = obv_vals[idx]
        if start_val == 0:
            continue
        obv_change_pct[idx] = (end_val - start_val) / abs(start_val) * 100.0
    filtered["obv_change_pct"] = obv_change_pct

    rsi_series = filtered.get("rsi")
    if rsi_series is None:
        raise ValueError("15m frame missing rsi column")
    filtered["rsi_momentum"] = rsi_series.diff()
    filtered["rsi_momentum_avg"] = (
        filtered["rsi_momentum"].rolling(config.RSI_MOMENTUM_LOOKBACK, min_periods=config.RSI_MOMENTUM_LOOKBACK).mean()
    )
    return filtered


def prepare_1h_frame(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched.sort_values("timestamp", inplace=True)
    enriched.reset_index(drop=True, inplace=True)
    if "ema_fast" not in enriched.columns:
        raise ValueError("1h frame missing ema_fast column")
    lookback = config.HTF_EMA_SLOPE_LOOKBACK
    ema_vals = enriched["ema_fast"].to_numpy(dtype=float, copy=False)
    slope = np.zeros(len(enriched), dtype=float)
    for idx in range(lookback, len(enriched)):
        prev = ema_vals[idx - lookback]
        curr = ema_vals[idx]
        if prev == 0 or np.isnan(prev) or np.isnan(curr):
            continue
        slope[idx] = (curr / prev - 1.0) * 100.0
    enriched["ema20_slope_pct"] = slope
    return enriched


def build_htf_index(main_ts: np.ndarray, htf_ts: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(htf_ts, main_ts, side="right") - 1
    idx[idx < 0] = -1
    return idx


def evaluate_post_signal_block(frame: pd.DataFrame, idx: int) -> Optional[str]:
    entry_row = frame.iloc[idx]
    entry_price = float(entry_row["close"])
    horizon = min(len(frame) - 1, idx + config.POST_SIGNAL_MONITOR_BARS)
    if idx + 1 > horizon:
        return None
    next_row = frame.iloc[idx + 1]
    if next_row["close"] <= next_row["open"]:
        return "FIRST_BAR_NOT_GREEN"
    target_price = entry_price * (1 + config.POST_SIGNAL_TARGET_PCT / 100.0)
    high_window = frame.iloc[idx + 1 : horizon + 1]["high"]
    if high_window.max(skipna=True) >= target_price:
        return None
    return "TARGET_NOT_MET"


def normalize_positive(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return float(max(0.0, min(1.0, value / cap)) * 100.0)


def normalize_inverse(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    clipped = max(0.0, min(cap, value))
    return float((1.0 - clipped / cap) * 100.0)


def summarize_trades(
    symbol: str,
    rr_name: str,
    trades: Sequence[SpotTrade],
    period_start: int,
    period_end: int,
    liquidity_metric: float,
) -> Optional[SummaryRow]:
    if not trades:
        return None
    durations = [t.duration_seconds for t in trades]
    returns = np.array([t.return_pct / 100.0 for t in trades], dtype=float)
    wins = [t.return_pct for t in trades if t.return_pct > 0]
    losses = [t.return_pct for t in trades if t.return_pct < 0]
    gross_profit = sum(max(t.net_pnl, 0.0) for t in trades)
    gross_loss = sum(min(t.net_pnl, 0.0) for t in trades)
    total_return_pct = sum(t.return_pct for t in trades)
    win_rate = (len(wins) / len(trades)) * 100.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf") if gross_profit > 0 else 0.0
    expectancy_pct = total_return_pct / len(trades)
    avg_hold = float(np.mean(durations)) if durations else 0.0
    exposure_seconds = float(sum(durations))
    total_period_seconds = max(1.0, (period_end - period_start) / 1000.0)
    exposure_pct = exposure_seconds / total_period_seconds * 100.0
    period_months = max(1e-6, total_period_seconds / (30 * 24 * 3600))
    trades_per_month = len(trades) / period_months

    equity = 1.0
    peak = 1.0
    max_drawdown_pct = 0.0
    for ret in returns:
        equity *= 1.0 + ret
        if equity <= 0:
            equity = 1e-6
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak * 100.0
        max_drawdown_pct = max(max_drawdown_pct, drawdown)

    sharpe = 0.0
    sortino = 0.0
    if len(trades) > 1:
        mean_ret = returns.mean()
        std_ret = returns.std(ddof=1)
        if std_ret > 0:
            sharpe = mean_ret / std_ret * math.sqrt(len(trades))
        downside = returns[returns < 0]
        if downside.size > 1:
            downside_std = float(downside.std(ddof=1))
        elif downside.size == 1:
            downside_std = abs(float(downside[0]))
        else:
            downside_std = 0.0
        if downside_std > 0:
            sortino = mean_ret / downside_std * math.sqrt(len(trades))

    efficiency_metric = trades_per_month * expectancy_pct
    profitability_score = normalize_positive(total_return_pct, 200.0)
    risk_score = 0.5 * normalize_inverse(max_drawdown_pct, 60.0) + 0.5 * normalize_positive(max(profit_factor - 1.0, 0.0) * 25.0, 100.0)
    consistency_score = normalize_positive(win_rate, 100.0)
    efficiency_score = normalize_positive(efficiency_metric, 200.0)
    liquidity_score = normalize_positive(liquidity_metric, 2000.0)
    total_score = (
        0.4 * profitability_score
        + 0.2 * risk_score
        + 0.2 * consistency_score
        + 0.1 * efficiency_score
        + 0.1 * liquidity_score
    )

    return SummaryRow(
        symbol=symbol,
        rr_name=rr_name,
        n_trades=len(trades),
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
        expectancy_pct=expectancy_pct,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe=sharpe,
        sortino=sortino,
        avg_holding_time_seconds=avg_hold,
        exposure_pct=exposure_pct,
        trades_per_month=trades_per_month,
        liquidity_metric=liquidity_metric,
        profitability_score=profitability_score,
        risk_score=risk_score,
        consistency_score=consistency_score,
        efficiency_score=efficiency_score,
        liquidity_score=liquidity_score,
        total_score=total_score,
    )


def simulate_trade(
    symbol: str,
    rr_cfg: SpotRRConfig,
    one_minute: pd.DataFrame,
    signal_ts: int,
    signal_label: str,
    signal_reason: str,
    signal_result: rules.SignalResult,
    position_notional: float,
    fee_rate: float,
    slippage_rate: float,
    rsi_momentum: float,
    ma60: float,
    macd_1h: float,
) -> Optional[SpotTrade]:
    timestamps = one_minute["timestamp"].to_numpy(dtype=np.int64)
    entry_idx = np.searchsorted(timestamps, signal_ts, side="right")
    if entry_idx >= len(one_minute):
        return None
    entry_price = float(one_minute.iloc[entry_idx]["open"])
    if entry_price <= 0:
        return None
    entry_time = int(timestamps[entry_idx])
    quantity = position_notional / entry_price
    tp_price = entry_price * (1 + rr_cfg.tp_pct)
    sl_price = entry_price * (1 - rr_cfg.sl_pct)
    max_hold_ts = signal_ts + rr_cfg.max_hold_bars * MILLIS_PER_15M

    exit_price = entry_price
    exit_reason = "MAX_HOLD"
    exit_idx = entry_idx
    max_dd_pct = 0.0

    highs = one_minute["high"].to_numpy(dtype=float, copy=False)
    lows = one_minute["low"].to_numpy(dtype=float, copy=False)
    closes = one_minute["close"].to_numpy(dtype=float, copy=False)

    for idx in range(entry_idx, len(one_minute)):
        ts = int(timestamps[idx])
        bar_high = highs[idx]
        bar_low = lows[idx]
        drawdown_pct = (bar_low - entry_price) / entry_price * 100.0
        max_dd_pct = min(max_dd_pct, drawdown_pct)

        hit_sl = bar_low <= sl_price
        hit_tp = bar_high >= tp_price
        if hit_sl and hit_tp:
            exit_price = sl_price
            exit_reason = "SL"
            exit_idx = idx
            break
        if hit_sl:
            exit_price = sl_price
            exit_reason = "SL"
            exit_idx = idx
            break
        if hit_tp:
            exit_price = tp_price
            exit_reason = "TP"
            exit_idx = idx
            break
        if ts >= max_hold_ts:
            exit_price = closes[idx]
            exit_reason = "MAX_HOLD"
            exit_idx = idx
            break
    else:
        exit_idx = len(one_minute) - 1
        exit_price = float(closes[exit_idx])

    exit_time = int(timestamps[exit_idx])
    duration_seconds = (exit_time - entry_time) / 1000.0
    gross = (exit_price - entry_price) * quantity
    fee = (entry_price + exit_price) * quantity * fee_rate
    slippage = (entry_price + exit_price) * quantity * slippage_rate
    net = gross - fee - slippage
    ret_pct = (net / (entry_price * quantity)) * 100.0
    trade_id = f"{symbol}-{rr_cfg.name}-{entry_time}"
    return SpotTrade(
        trade_id=trade_id,
        symbol=symbol,
        rr_name=rr_cfg.name,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        direction="long",
        gross_pnl=gross,
        fee=fee,
        slippage=slippage,
        net_pnl=net,
        return_pct=ret_pct,
        duration_seconds=duration_seconds,
        max_drawdown_pct=abs(max_dd_pct),
        entry_signal_type=signal_label,
        entry_reason=signal_reason,
        exit_reason=exit_reason,
        core_score=signal_result.score_core,
        trend_score=signal_result.trend_score,
        osc_score=signal_result.osc_score,
        vol_score=signal_result.vol_score,
        pa_score=signal_result.pa_score,
        htf_bonus=signal_result.htf_bonus,
        rsi_15m=signal_result.rsi,
        rsi_momentum=rsi_momentum,
        ma60=ma60,
        macd_1h=macd_1h,
    )


def run_symbol_job(job: SpotJob) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if job.symbol in config.STABLE_SYMBOLS:
        return [], []
    try:
        path_15m = Path(job.path_15m)
        path_1h = Path(job.path_1h)
        path_1m = Path(job.path_1m)
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        feature_cols = required_cols + [
            "ema_fast",
            "ema_slow",
            "adx",
            "plus_di",
            "minus_di",
            "macd",
            "macd_signal",
            "macd_hist",
            "momentum",
            "awesome_osc",
            "rsi",
            "stoch_k",
            "cci",
            "stoch_rsi",
            "williams_r",
            "ultimate_osc",
            "obv",
            "bull_power",
            "bear_power",
            "pa_long_lower_wick",
            "pa_strong_green",
            "pa_very_strong_green",
            "pa_collapse_ok",
            "pa_no_collapse",
            "pa_ema_breakout",
            "pa_ema_retest",
            "pa_volume_spike",
            "pa_min_volume",
            "pa_volume_spike_factor",
        ]
        frame_15m = load_parquet(path_15m, feature_cols)
        frame_15m = prepare_15m_frame(frame_15m, job.settings.start_ts, job.settings.end_ts)
        if frame_15m.empty:
            return [], []
        frame_1h = load_parquet(path_1h, required_cols + ["ema_fast", "macd", "macd_hist"])
        frame_1h = prepare_1h_frame(frame_1h)
        if frame_1h.empty:
            return [], []
        frame_1m = load_parquet(path_1m, required_cols)
        if frame_1m.empty:
            return [], []

        main_ts = frame_15m["timestamp"].to_numpy(dtype=np.int64)
        htf_ts = frame_1h["timestamp"].to_numpy(dtype=np.int64)
        htf_indexer = build_htf_index(main_ts, htf_ts)
        cooldown_ms = config.COOLDOWN_MINUTES * MILLIS_PER_MINUTE
        last_signal_ts = -cooldown_ms
        blocked_until_ts = -1

        trades: List[SpotTrade] = []
        rr_buckets: Dict[str, List[SpotTrade]] = {cfg.name: [] for cfg in job.settings.rr_configs}
        avg_quote = float(frame_15m["quote_volume_24h"].mean(skipna=True))
        liquidity_metric = 0.0 if math.isnan(avg_quote) else avg_quote / 1_000_000.0

        for idx in range(len(frame_15m)):
            row = frame_15m.iloc[idx]
            timestamp = int(row["timestamp"])
            if timestamp - last_signal_ts < cooldown_ms:
                continue
            if timestamp < blocked_until_ts:
                continue
            if row["quote_volume_24h"] < config.MIN_24H_QUOTE_VOLUME:
                continue
            if row["close"] < config.MIN_PRICE_USDT:
                continue
            change = row["price_change_24h"]
            if math.isnan(change) or change < config.MIN_24H_CHANGE or change > config.MAX_24H_CHANGE:
                continue
            htf_idx = htf_indexer[idx]
            if htf_idx < 0:
                continue
            htf_row = frame_1h.iloc[htf_idx]
            htf_context = {
                "close_above_ema20": bool(htf_row["close"] > htf_row["ema_fast"]),
                "ema20_slope_pct": float(htf_row.get("ema20_slope_pct", 0.0)),
                "macd_hist": float(htf_row.get("macd_hist", 0.0)),
                "macd_line": float(htf_row.get("macd", 0.0)),
            }

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

            trend_block = rules.compute_trend_block(
                price=float(row["close"]),
                ema20=float(row["ema_fast"]),
                ema50=float(row["ema_slow"]),
                adx=float(row["adx"]),
                plus_di=float(row["plus_di"]),
                minus_di=float(row["minus_di"]),
                macd_hist=float(row["macd_hist"]),
                macd_hist_rising=bool(row["macd_hist_rising_flag"]),
                momentum=float(row["momentum"]),
                ao=float(row["awesome_osc"]),
            )
            osc_block = rules.compute_osc_block(
                rsi_val=float(row["rsi"]),
                stoch_k=float(row["stoch_k"]),
                cci=float(row["cci"]),
                stoch_rsi=float(row["stoch_rsi"]),
                williams_r=float(row["williams_r"]),
                uo=float(row["ultimate_osc"]),
                stoch_rsi_prev=None,
                uo_prev=None,
            )
            vol_block = rules.compute_volume_block(
                bull_power=float(row["bull_power"]),
                bear_power=float(row["bear_power"]),
                volume_spike_factor=float(row.get("pa_volume_spike_factor", 0.0)),
                obv_change_pct=float(row.get("obv_change_pct", 0.0)),
            )
            pa_block = rules.compute_price_action_block(pa_signals)
            htf_block = rules.compute_htf_bonus(htf_context)

            meta = {
                "price": float(row["close"]),
                "price_change_pct": float(change),
                "quote_volume": float(row["quote_volume_24h"]),
            }
            pre_signal_context = {
                "last_close": float(row["close"]),
                "ma60": float(row.get("ma60", np.nan)),
                "macd_1h": htf_context.get("macd_line"),
                "rsi_value": float(row["rsi"]),
                "rsi_momentum_curr": float(row.get("rsi_momentum", np.nan)),
                "rsi_momentum_avg": float(row.get("rsi_momentum_avg", np.nan)),
            }
            signal_result = rules.decide_signal_label(
                trend_block=trend_block,
                osc_block=osc_block,
                vol_block=vol_block,
                pa_block=pa_block,
                htf_block=htf_block,
                meta=meta,
                rsi_value=float(row["rsi"]),
                symbol=job.symbol,
                pre_signal_context=pre_signal_context,
            )
            label = signal_result.label
            if label not in {"STRONG_BUY", "ULTRA_BUY"}:
                continue

            last_signal_ts = timestamp
            block_reason = evaluate_post_signal_block(frame_15m, idx)
            if block_reason:
                blocked_until_ts = timestamp + config.POST_SIGNAL_BLOCK_MINUTES * MILLIS_PER_MINUTE

            entry_reason = ", ".join(signal_result.reasons or [])
            if signal_result.filter_notes:
                note_str = "; ".join(signal_result.filter_notes)
                entry_reason = f"{entry_reason} | Filters: {note_str}" if entry_reason else f"Filters: {note_str}"
            rsi_momentum_val = float(row.get("rsi_momentum", np.nan))
            ma60_val = float(row.get("ma60", np.nan))
            macd_1h_val = float(htf_context.get("macd_line", np.nan))
            for rr_cfg in job.settings.rr_configs:
                trade = simulate_trade(
                    symbol=job.symbol,
                    rr_cfg=rr_cfg,
                    one_minute=frame_1m,
                    signal_ts=timestamp,
                    signal_label=label,
                    signal_reason=entry_reason,
                    signal_result=signal_result,
                    position_notional=job.settings.position_notional,
                    fee_rate=job.settings.fee_rate,
                    slippage_rate=job.settings.slippage_rate,
                    rsi_momentum=rsi_momentum_val,
                    ma60=ma60_val,
                    macd_1h=macd_1h_val,
                )
                if trade:
                    trades.append(trade)
                    rr_buckets[rr_cfg.name].append(trade)

        if not trades:
            return [], []

        start_ts = int(frame_15m["timestamp"].iloc[0])
        end_ts = int(frame_15m["timestamp"].iloc[-1])
        summary_rows: List[Dict[str, Any]] = []
        for rr_name, bucket in rr_buckets.items():
            summary = summarize_trades(job.symbol, rr_name, bucket, start_ts, end_ts, liquidity_metric)
            if summary:
                summary_rows.append(summary.to_row())
        trade_rows = [trade.to_row() for trade in trades]
        return trade_rows, summary_rows
    except Exception as exc:
        log_error("spot_backtest.run_symbol_job", "Processing failed", symbol=job.symbol, exc=exc)
        print(f"Job failed for {job.symbol}: {exc}")
        return [], []


def write_xlsx(
    path: Path,
    rows: Iterable[Dict[str, Any]],
    *,
    columns: Optional[Sequence[str]] = None,
) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame(columns=columns)
    df.to_excel(path, index=False)


def main() -> None:
    try:
        args = parse_args()
        symbols = parse_symbols(args.symbols)
        rr_configs = resolve_rr_configs(args.rr_profiles)
        start_ts = parse_date(args.start_date)
        end_ts = parse_date(args.end_date)
        if start_ts and end_ts and end_ts < start_ts:
            raise ValueError("end-date must be >= start-date")

        data_dir_15m = Path(args.data_dir_15m)
        data_dir_1h = Path(args.data_dir_1h)
        data_dir_1m = Path(args.data_dir_1m)
        artifacts = discover_symbol_artifacts(data_dir_15m, data_dir_1h, data_dir_1m, symbols)
        if not artifacts:
            message = "No symbols discovered with complete data; check data directories or --symbols filter."
            print(message)
            raise RuntimeError(message)
        settings = SpotRunSettings(
            start_ts=start_ts,
            end_ts=end_ts,
            rr_configs=rr_configs,
            position_notional=args.position_notional,
            fee_rate=args.fee_rate,
            slippage_rate=args.slippage_rate,
        )

        jobs = [
            SpotJob(
                symbol=symbol,
                path_15m=str(paths[0]),
                path_1h=str(paths[1]),
                path_1m=str(paths[2]),
                settings=settings,
            )
            for symbol, paths in artifacts.items()
        ]

        all_trade_rows: List[Dict[str, Any]] = []
        all_summary_rows: List[Dict[str, Any]] = []
        max_workers = clamp_workers(args.max_workers)

        if max_workers == 1:
            for job in jobs:
                trade_rows, summary_rows = run_symbol_job(job)
                all_trade_rows.extend(trade_rows)
                all_summary_rows.extend(summary_rows)
                print(f"Completed {job.symbol}: trades={len(trade_rows)} summaries={len(summary_rows)}")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_symbol_job, job): job.symbol for job in jobs}
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        trade_rows, summary_rows = future.result()
                    except Exception as exc:  # pragma: no cover
                        log_error("spot_backtest.executor", "Worker failed", symbol=symbol, exc=exc)
                        print(f"Job failed for {symbol}: {exc}")
                        continue
                    all_trade_rows.extend(trade_rows)
                    all_summary_rows.extend(summary_rows)
                    print(f"Completed {symbol}: trades={len(trade_rows)} summaries={len(summary_rows)}")

        results_dir = Path(args.results_dir)
        trades_path = results_dir / SPOT_TRADES_FILENAME
        summary_path = results_dir / SPOT_SUMMARY_FILENAME
        write_xlsx(trades_path, all_trade_rows, columns=TRADE_COLUMNS)
        write_xlsx(summary_path, all_summary_rows, columns=SUMMARY_COLUMNS)
        print(f"Spot trade log written to {trades_path}")
        print(f"Spot summary written to {summary_path}")
    except Exception as exc:  # pragma: no cover - bubble fatal errors to caller after logging
        log_error("spot_backtest.main", "Fatal error", exc=exc)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()
