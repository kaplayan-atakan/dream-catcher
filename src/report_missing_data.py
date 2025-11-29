"""Utility to report which symbols lack 1m/15m/1h data triplets.

The script scans the configured data directories, groups symbols by missing
components, and writes a plain-text summary so we can regenerate just the
needed artifacts instead of recomputing everything.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

DEFAULT_DIR_1M = "data"
DEFAULT_DIR_15M = "data/precomputed_15m"
DEFAULT_DIR_1H = "data/precomputed_1h"
DEFAULT_REPORT_PATH = "results/missing_data_report.txt"


def canonical_symbol(stem: str) -> str:
    upper = stem.upper()
    for suffix in ("_15M_FEATURES", "_1H_FEATURES", "_1M", "_15M", "_1H", "_FEATURES"):
        if upper.endswith(suffix):
            return upper[: -len(suffix)]
    return upper


def collect_symbols(directory: Path, suffix: str) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        mapping[canonical_symbol(path.stem)] = path
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report symbols missing 1m/15m/1h data triplets")
    parser.add_argument("--dir-1m", default=DEFAULT_DIR_1M, help="Directory holding *_1m.parquet files")
    parser.add_argument("--dir-15m", default=DEFAULT_DIR_15M, help="Directory holding *_15m_features.parquet files")
    parser.add_argument("--dir-1h", default=DEFAULT_DIR_1H, help="Directory holding *_1h_features.parquet files")
    parser.add_argument(
        "--report-path",
        default=DEFAULT_REPORT_PATH,
        help="Where to write the missing-data report (text file)",
    )
    return parser.parse_args()


def summarize(missing: Sequence[str], label: str) -> str:
    if not missing:
        return f"{label}: none\n\n"
    lines = "\n".join(missing)
    return f"{label} ({len(missing)} symbols):\n{lines}\n\n"


def main() -> None:
    args = parse_args()
    dir_1m = Path(args.dir_1m)
    dir_15m = Path(args.dir_15m)
    dir_1h = Path(args.dir_1h)
    report_path = Path(args.report_path)

    symbols_1m = collect_symbols(dir_1m, "_1m.parquet")
    symbols_15m = collect_symbols(dir_15m, "_15m_features.parquet")
    symbols_1h = collect_symbols(dir_1h, "_1h_features.parquet")

    universe = set(symbols_15m) | set(symbols_1h) | set(symbols_1m)
    missing_1m_only: List[str] = []
    missing_15m_only: List[str] = []
    missing_1h_only: List[str] = []
    missing_two: List[str] = []
    missing_all: List[str] = []

    for symbol in sorted(universe):
        has_1m = symbol in symbols_1m
        has_15m = symbol in symbols_15m
        has_1h = symbol in symbols_1h
        missing_parts: List[str] = []
        if not has_1m:
            missing_parts.append("1m")
        if not has_15m:
            missing_parts.append("15m")
        if not has_1h:
            missing_parts.append("1h")
        if not missing_parts:
            continue
        label = f"{symbol}: missing {', '.join(missing_parts)}"
        missing_count = len(missing_parts)
        if missing_count == 1:
            part = missing_parts[0]
            if part == "1m":
                missing_1m_only.append(label)
            elif part == "15m":
                missing_15m_only.append(label)
            elif part == "1h":
                missing_1h_only.append(label)
        elif missing_count == 2:
            missing_two.append(label)
        else:
            missing_all.append(label)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        summarize(missing_1m_only, "Missing only 1m"),
        summarize(missing_15m_only, "Missing only 15m"),
        summarize(missing_1h_only, "Missing only 1h"),
        summarize(missing_two, "Missing exactly two components"),
        summarize(missing_all, "Missing all components"),
    ]
    report_path.write_text("".join(sections), encoding="utf-8")
    print(f"Report written to {report_path} (total {len(universe)} symbols scanned)")


if __name__ == "__main__":  # pragma: no cover
    main()
