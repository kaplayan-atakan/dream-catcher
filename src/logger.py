"""Logging utilities used by main.py.

Phase 1 requires console logging plus a CSV sink for emitted signals. The CSV
schema is intentionally minimal and backward-compatible so later phases can add
more columns without breaking existing analyses.
"""
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import config

_DEFAULT_COLUMNS = [
    "timestamp",
    "symbol",
    "label",
    "trend_score",
    "osc_score",
    "vol_score",
    "pa_score",
    "score_core",
    "htf_bonus",
    "total_score",
    "risk_tag",
    "rsi",
    "price",
    "change_24h",
    "quote_vol_24h",
    "reasons",
    "htf_notes",
    "vol_spike_factor",
]


def setup_logger() -> logging.Logger:
    """Configure and return the shared application logger."""
    logger = logging.getLogger("binance_usdt_signal_bot")
    if logger.handlers:
        return logger

    level = getattr(logging, str(config.LOG_LEVEL).upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_signal_to_csv(
    path: str,
    signal: Dict[str, object],
    extra_fields: Optional[Dict[str, object]] = None,
) -> None:
    """Append a signal row to CSV, creating the file with headers if needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.isfile(path)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": signal.get("symbol"),
        "label": signal.get("label"),
        "trend_score": signal.get("trend_score"),
        "osc_score": signal.get("osc_score"),
        "vol_score": signal.get("vol_score"),
        "pa_score": signal.get("pa_score"),
        "score_core": signal.get("score_core", signal.get("total_score")),
        "htf_bonus": signal.get("htf_bonus"),
        "total_score": signal.get("score_total", signal.get("total_score")),
        "risk_tag": signal.get("risk_tag"),
        "rsi": signal.get("rsi"),
        "price": signal.get("price"),
        "change_24h": signal.get("price_change_pct"),
        "quote_vol_24h": signal.get("quote_volume"),
        "reasons": "; ".join(signal.get("reasons", [])) if signal.get("reasons") else "",
    }

    trend_details = signal.get("trend_details") or {}
    htf_details = signal.get("htf_details") or {}
    vol_details = signal.get("vol_details") or {}

    htf_bits = []
    if htf_details.get("close_above_ema20"):
        htf_bits.append("1h>EMA20")
    if htf_details.get("ema20_slope_pct"):
        htf_bits.append(f"1h-slope={float(htf_details['ema20_slope_pct']):+.1f}%")
    if htf_details.get("macd_hist") is not None:
        htf_bits.append(f"1h-MACD={float(htf_details['macd_hist']):+.3f}")

    if signal.get("fourh_alignment_ok"):
        htf_bits.append("4h-stack")
    elif signal.get("fourh_price_above_ema20"):
        htf_bits.append("4h>EMA20")

    slope = signal.get("fourh_ema20_slope_pct") or trend_details.get("fourh_ema20_slope_pct")
    if slope:
        htf_bits.append(f"4h-slope={float(slope):+.1f}%")

    row["htf_notes"] = " | ".join(htf_bits)

    spike_factor = vol_details.get("volume_spike_factor")
    row["vol_spike_factor"] = round(float(spike_factor), 2) if spike_factor else ""

    if extra_fields:
        row.update(extra_fields)

    columns = list(dict.fromkeys(_DEFAULT_COLUMNS + list(extra_fields or {})))

    with open(path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
