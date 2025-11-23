"""Offline indicator pre-computation for historical parquet data.

This module mirrors the live bot's indicator stack so bulk datasets can be
annotated once and reused for research, backtests, or futures-mode studies.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import indicators
import config as default_config

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
DEFAULT_CONFIG_MODULE = "config"


def load_ohlcv(path: Path) -> pd.DataFrame:
    """Load and sanitize OHLCV data from a parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet file: {path}")
    frame = pd.read_parquet(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Parquet file is missing columns: {missing}")

    ordered = frame.loc[:, REQUIRED_COLUMNS].copy()
    ordered.sort_values("timestamp", inplace=True)
    ordered.reset_index(drop=True, inplace=True)
    for column in ("open", "high", "low", "close", "volume"):
        ordered[column] = ordered[column].astype(float)
    ordered["timestamp"] = ordered["timestamp"].astype(np.int64)
    return ordered


def compute_indicators(frame: pd.DataFrame, cfg=default_config) -> pd.DataFrame:
    """Compute indicator columns using the same formulas as the live bot."""
    opens = frame["open"].tolist()
    highs = frame["high"].tolist()
    lows = frame["low"].tolist()
    closes = frame["close"].tolist()
    volumes = frame["volume"].tolist()

    enriched = frame.copy()

    def assign(name: str, values: Iterable[float]) -> None:
        enriched[name] = pd.Series(list(values), dtype="float64")

    assign("ema_fast", indicators.ema(closes, cfg.EMA_FAST))
    assign("ema_slow", indicators.ema(closes, cfg.EMA_SLOW))

    adx_vals, plus_di, minus_di = indicators.adx(highs, lows, closes, cfg.ADX_PERIOD)
    assign("adx", adx_vals)
    assign("plus_di", plus_di)
    assign("minus_di", minus_di)

    macd_line, macd_signal, macd_hist = indicators.macd(closes)
    assign("macd", macd_line)
    assign("macd_signal", macd_signal)
    assign("macd_hist", macd_hist)

    assign("momentum", indicators.momentum(closes, cfg.MOMENTUM_PERIOD))
    assign("awesome_osc", indicators.awesome_oscillator(highs, lows))
    assign("rsi", indicators.rsi(closes, cfg.RSI_PERIOD))
    assign("stoch_k", indicators.stochastic_k(highs, lows, closes, cfg.STOCH_K_PERIOD))
    assign("cci", indicators.cci(highs, lows, closes, cfg.CCI_PERIOD))
    assign("stoch_rsi", indicators.stochastic_rsi(closes))
    assign("williams_r", indicators.williams_r(highs, lows, closes, cfg.WILLIAMS_PERIOD))
    assign("ultimate_osc", indicators.ultimate_oscillator(highs, lows, closes, cfg.UO_PERIODS))
    assign("obv", indicators.obv(closes, volumes))

    bull_power, bear_power = indicators.bull_bear_power(highs, lows, closes)
    assign("bull_power", bull_power)
    assign("bear_power", bear_power)

    return enriched


def save_enriched(frame: pd.DataFrame, output_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def resolve_config(module_name: str):
    if not module_name:
        return default_config
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        if not module_name.startswith("src."):
            return importlib.import_module(f"src.{module_name}")
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute indicator columns for a parquet file")
    parser.add_argument("--input", "-i", required=True, help="Path to the input parquet file")
    parser.add_argument(
        "--output",
        "-o",
        help="Optional output parquet path. Defaults to <input>_features.parquet",
    )
    parser.add_argument(
        "--config-module",
        default=DEFAULT_CONFIG_MODULE,
        help="Config module to import (e.g. config or config_futures)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file",
    )
    return parser.parse_args()


def run_from_cli() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_features.parquet")
    )

    cfg = resolve_config(args.config_module)
    frame = load_ohlcv(input_path)
    enriched = compute_indicators(frame, cfg)
    save_enriched(enriched, output_path, overwrite=args.overwrite)
    print(f"Indicators saved to {output_path}")


if __name__ == "__main__":
    run_from_cli()
