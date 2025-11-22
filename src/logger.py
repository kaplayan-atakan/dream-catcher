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
    "total_score",
    "rsi",
    "price",
    "change_24h",
    "quote_vol_24h",
    "reasons",
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
        "total_score": signal.get("total_score"),
        "rsi": signal.get("rsi"),
        "price": signal.get("price"),
        "change_24h": signal.get("price_change_pct"),
        "quote_vol_24h": signal.get("quote_volume"),
        "reasons": "; ".join(signal.get("reasons", [])) if signal.get("reasons") else "",
    }

    if extra_fields:
        row.update(extra_fields)

    columns = list(dict.fromkeys(_DEFAULT_COLUMNS + list(extra_fields or {})))

    with open(path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
