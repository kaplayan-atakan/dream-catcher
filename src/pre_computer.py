"""Offline indicator pre-computation for historical parquet data.

This module mirrors the live bot's indicator stack so bulk datasets can be
annotated once and reused for research, backtests, or futures-mode studies.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import numpy as np
import pandas as pd

try:
    from . import indicators
    from . import config as default_config
    from . import price_action
except ImportError:  # pragma: no cover - fallback for direct script execution
    import indicators  # type: ignore
    import config as default_config  # type: ignore
    import price_action  # type: ignore

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
DEFAULT_CONFIG_MODULE = "config"
DEFAULT_OUTPUT_SUBDIR = "features"
MAX_PARALLEL_WORKERS = 10
SUPPORTED_TIMEFRAMES = ("1m", "15m", "1h")
DEFAULT_TIMEFRAMES = ("15m", "1h")
TIMEFRAME_RESAMPLE_RULES = {
    "1m": None,
    "15m": "15min",
    "1h": "1h",
}
DEFAULT_TIMEFRAME_OUTPUT_SUBDIRS = {
    "1m": DEFAULT_OUTPUT_SUBDIR,
    "15m": "precomputed_15m",
    "1h": "precomputed_1h",
}
PA_COLUMN_MAP = {
    "long_lower_wick": "pa_long_lower_wick",
    "strong_green": "pa_strong_green",
    "very_strong_green": "pa_very_strong_green",
    "collapse_ok": "pa_collapse_ok",
    "no_collapse": "pa_no_collapse",
    "ema_breakout": "pa_ema_breakout",
    "ema_retest": "pa_ema_retest",
    "volume_spike": "pa_volume_spike",
    "min_volume": "pa_min_volume",
}
PRICE_ACTION_LOOKBACK_DEFAULT = max(
    default_config.COLLAPSE_LOOKBACK_BARS,
    default_config.STRONG_GREEN_LOOKBACK,
    default_config.VOLUME_LOOKBACK + 1,
    default_config.EMA_BREAK_LOOKBACK + default_config.EMA_RETEST_LOOKBACK + 2,
)


def normalize_timeframe(token: str) -> str:
    value = token.strip().lower()
    if value not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe requested: {token!r}")
    return value


def parse_timeframes(raw: str) -> Tuple[str, ...]:
    if not raw:
        return tuple(DEFAULT_TIMEFRAMES)
    tokens = [normalize_timeframe(part) for part in raw.split(",") if part.strip()]
    if not tokens:
        return tuple(DEFAULT_TIMEFRAMES)
    seen = []
    for tf in tokens:
        if tf not in seen:
            seen.append(tf)
    return tuple(seen)


def strip_timeframe_suffix(symbol: str) -> str:
    lowered = symbol.lower()
    for suffix in ("_1m", "_15m", "_1h"):
        if lowered.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def canonical_symbol(symbol: str) -> str:
    return strip_timeframe_suffix(symbol).upper()


def default_output_dir_for_timeframe(
    input_dir: Path,
    timeframe: str,
    override: Optional[str],
) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    timeframe = normalize_timeframe(timeframe)
    subdir = DEFAULT_TIMEFRAME_OUTPUT_SUBDIRS.get(timeframe, f"features_{timeframe}")
    if timeframe == "1m":
        return (input_dir / subdir).resolve()
    base = input_dir.parent if input_dir.parent != input_dir else input_dir
    return (base / subdir).resolve()


def build_timeframe_output_path(symbol: str, timeframe: str, output_dir: Path) -> Path:
    timeframe = normalize_timeframe(timeframe)
    clean_symbol = canonical_symbol(symbol)
    filename = f"{clean_symbol}_{timeframe}_features.parquet"
    return output_dir / filename


def bind_indicator_modules(cfg) -> None:
    try:
        indicators.config = cfg  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - best effort
        pass
    try:
        price_action.config = cfg  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - best effort
        pass


def clamp_workers(value: int) -> int:
    """Clamp the worker count to the supported range."""
    if value <= 1:
        return 1
    return min(value, MAX_PARALLEL_WORKERS)


def resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    timeframe = normalize_timeframe(timeframe)
    rule = TIMEFRAME_RESAMPLE_RULES.get(timeframe)
    if not rule:
        return frame.loc[:, REQUIRED_COLUMNS].copy()

    if frame.empty:
        return frame.loc[:, REQUIRED_COLUMNS].copy()

    df = frame.loc[:, REQUIRED_COLUMNS].copy()
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    grouped = (
        df.set_index("ts")
        .resample(rule, label="right", closed="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
    )
    grouped.dropna(subset=["open", "high", "low", "close", "volume"], inplace=True)
    grouped = grouped.reset_index().rename(columns={"ts": "timestamp"})
    if grouped.empty:
        return grouped.loc[:, REQUIRED_COLUMNS]
    grouped["timestamp"] = (
        grouped["timestamp"].astype("int64", copy=False) // 1_000_000
    ).astype(np.int64)
    ordered = grouped.loc[:, REQUIRED_COLUMNS].copy()
    ordered.sort_values("timestamp", inplace=True)
    ordered.reset_index(drop=True, inplace=True)
    return ordered


def determine_price_action_lookback(cfg) -> int:
    return max(
        getattr(cfg, "COLLAPSE_LOOKBACK_BARS", PRICE_ACTION_LOOKBACK_DEFAULT),
        getattr(cfg, "STRONG_GREEN_LOOKBACK", PRICE_ACTION_LOOKBACK_DEFAULT),
        getattr(cfg, "VOLUME_LOOKBACK", default_config.VOLUME_LOOKBACK) + 1,
        getattr(cfg, "EMA_BREAK_LOOKBACK", default_config.EMA_BREAK_LOOKBACK)
        + getattr(cfg, "EMA_RETEST_LOOKBACK", default_config.EMA_RETEST_LOOKBACK)
        + 2,
    )


def append_price_action_columns(enriched: pd.DataFrame, cfg) -> None:
    if enriched.empty or "ema_fast" not in enriched.columns:
        return
    lookback = determine_price_action_lookback(cfg)

    opens = enriched["open"].tolist()
    highs = enriched["high"].tolist()
    lows = enriched["low"].tolist()
    closes = enriched["close"].tolist()
    volumes = enriched["volume"].tolist()
    ema_fast = enriched["ema_fast"].tolist()

    bool_columns = {col: [] for col in PA_COLUMN_MAP.values()}
    volume_factor: List[float] = []

    for idx in range(len(enriched)):
        start = max(0, idx - lookback)
        pa_snapshot = price_action.analyze_price_action(
            opens[start : idx + 1],
            highs[start : idx + 1],
            lows[start : idx + 1],
            closes[start : idx + 1],
            volumes[start : idx + 1],
            ema_fast[start : idx + 1],
        )
        for key, column in PA_COLUMN_MAP.items():
            bool_columns[column].append(bool(pa_snapshot.get(key, False)))
        volume_factor.append(float(pa_snapshot.get("volume_spike_factor") or 0.0))

    for column, values in bool_columns.items():
        enriched[column] = pd.Series(values, dtype="bool")
    enriched["pa_volume_spike_factor"] = pd.Series(volume_factor, dtype="float64")


def normalize_timestamp_ms(series: pd.Series) -> pd.Series:
    """Return timestamps as int milliseconds regardless of source dtype."""
    if series.empty:
        return pd.Series([], index=series.index, dtype="int64")

    if pd.api.types.is_datetime64_any_dtype(series):
        dt_series = series
        if getattr(series.dtype, "tz", None) is not None:
            dt_series = series.dt.tz_localize(None)
        return (dt_series.astype("int64", copy=False) // 1_000_000).astype(np.int64)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Timestamp column contains non-numeric values")
    numeric = numeric.astype(np.int64)

    max_ts = int(numeric.max()) if len(numeric) else 0
    if max_ts > 10**15:
        numeric = numeric // 1_000_000  # downscale from ns to ms
    elif 0 < max_ts < 10**11:
        numeric = numeric * 1000  # upscale from seconds to ms

    return numeric.astype(np.int64)


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
    ordered["timestamp"] = normalize_timestamp_ms(ordered["timestamp"])
    return ordered


def compute_indicators(
    frame: pd.DataFrame,
    cfg=default_config,
    *,
    include_price_action: bool = True,
) -> pd.DataFrame:
    """Compute indicator columns using the same formulas as the live bot."""
    bind_indicator_modules(cfg)
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

    if include_price_action:
        append_price_action_columns(enriched, cfg)

    return enriched


def save_enriched(frame: pd.DataFrame, output_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def resolve_config(module_name: str):
    if not module_name:
        return default_config
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


def _ensure_config_module(cfg_or_name: Union[str, object]):
    if isinstance(cfg_or_name, str):
        return resolve_config(cfg_or_name)
    return cfg_or_name


def discover_input_files(input_dir: Path, symbols: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """Return a mapping of SYMBOL -> parquet path for the requested directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files: Dict[str, Path] = {}
    if symbols:
        for symbol in symbols:
            symbol_clean = symbol.strip().upper()
            if not symbol_clean:
                continue
            candidate = input_dir / f"{symbol_clean}.parquet"
            if not candidate.exists():
                raise FileNotFoundError(f"Parquet file for {symbol_clean} not found in {input_dir}")
            files[symbol_clean] = candidate
        return files

    for path in sorted(input_dir.glob("*.parquet")):
        files[path.stem.upper()] = path

    if not files:
        raise FileNotFoundError(f"No parquet files found under {input_dir}")
    return files


def process_symbol_timeframes_loaded(
    symbol: str,
    input_path: Path,
    timeframes: Sequence[str],
    output_dirs: Dict[str, Path],
    cfg_module=default_config,
    *,
    overwrite: bool = False,
) -> Dict[str, str]:
    if not timeframes:
        raise ValueError("At least one timeframe must be requested")

    frame_1m = load_ohlcv(input_path)
    cache: Dict[str, pd.DataFrame] = {"1m": frame_1m}
    outputs: Dict[str, str] = {}

    for timeframe in timeframes:
        tf = normalize_timeframe(timeframe)
        base_frame = cache.get(tf)
        if base_frame is None:
            base_frame = resample_ohlcv(frame_1m, tf)
            cache[tf] = base_frame
        if base_frame.empty:
            raise ValueError(f"No data points remain for {symbol} on timeframe {tf}")
        enriched = compute_indicators(
            base_frame,
            cfg_module,
            include_price_action=(tf == "15m"),
        )
        target_root = output_dirs[tf]
        if target_root.suffix.lower() == ".parquet":
            output_path = target_root
        else:
            target_root.mkdir(parents=True, exist_ok=True)
            output_path = build_timeframe_output_path(symbol, tf, target_root)
        save_enriched(enriched, output_path, overwrite=overwrite)
        outputs[tf] = str(output_path)
    return outputs


def process_symbol_1m_to_15m_1h(
    in_path: Path,
    out_dir_15m: Path,
    out_dir_1h: Path,
    *,
    symbol: Optional[str] = None,
    config_module: Union[str, object] = DEFAULT_CONFIG_MODULE,
    overwrite: bool = False,
) -> Dict[str, str]:
    cfg = _ensure_config_module(config_module)
    symbol_name = symbol or canonical_symbol(Path(in_path).stem)
    output_dirs = {"15m": Path(out_dir_15m), "1h": Path(out_dir_1h)}
    return process_symbol_timeframes_loaded(
        symbol_name,
        Path(in_path),
        ("15m", "1h"),
        output_dirs,
        cfg_module=cfg,
        overwrite=overwrite,
    )


def process_all_symbols_1m_to_15m_1h(
    input_dir: Path,
    out_dir_15m: Path,
    out_dir_1h: Path,
    *,
    symbols: Optional[Sequence[str]] = None,
    max_workers: int = 1,
    config_module: str = DEFAULT_CONFIG_MODULE,
    overwrite: bool = False,
) -> None:
    output_dirs = {"15m": Path(out_dir_15m), "1h": Path(out_dir_1h)}
    process_all_symbols_parallel(
        input_dir,
        output_dirs=output_dirs,
        timeframes=("15m", "1h"),
        symbols=symbols,
        max_workers=max_workers,
        config_module=config_module,
        overwrite=overwrite,
    )


def _parallel_process_symbol(
    args: Tuple[str, str, Tuple[str, ...], Dict[str, str], str, bool]
) -> Dict[str, object]:
    symbol, input_path_str, timeframes, output_dirs_map, config_module, overwrite = args
    try:
        cfg = resolve_config(config_module)
        output_dirs = {tf: Path(path) for tf, path in output_dirs_map.items()}
        outputs = process_symbol_timeframes_loaded(
            symbol,
            Path(input_path_str),
            timeframes,
            output_dirs,
            cfg_module=cfg,
            overwrite=overwrite,
        )
        return {"symbol": symbol, "status": "ok", "outputs": outputs}
    except Exception as exc:  # noqa: BLE001
        return {"symbol": symbol, "status": "error", "error": str(exc)}


def process_all_symbols_parallel(
    input_dir: Path,
    *,
    output_dirs: Dict[str, Path],
    timeframes: Sequence[str],
    symbols: Optional[Sequence[str]] = None,
    max_workers: int = 1,
    config_module: str = DEFAULT_CONFIG_MODULE,
    overwrite: bool = False,
) -> None:
    """Process every requested symbol using up to max_workers processes."""
    symbol_map = discover_input_files(input_dir, symbols)
    if not symbol_map:
        print("No input parquet files discovered; nothing to do.")
        return

    if not timeframes:
        raise ValueError("At least one timeframe must be specified")
    normalized_timeframes = tuple(normalize_timeframe(tf) for tf in timeframes)
    for tf in normalized_timeframes:
        if tf not in output_dirs:
            raise ValueError(f"Missing output directory configuration for timeframe {tf}")
    resolved_output_dirs = {tf: path.resolve() for tf, path in output_dirs.items()}
    for path in resolved_output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    max_workers = clamp_workers(max_workers)
    errors: Dict[str, str] = {}
    processed = 0

    if max_workers == 1:
        cfg = resolve_config(config_module)
        bind_indicator_modules(cfg)
        for symbol, input_path in symbol_map.items():
            try:
                outputs = process_symbol_timeframes_loaded(
                    symbol,
                    input_path,
                    normalized_timeframes,
                    resolved_output_dirs,
                    cfg_module=cfg,
                    overwrite=overwrite,
                )
                processed += 1
                artifacts = ", ".join(Path(path).name for path in outputs.values())
                print(f"[OK] {symbol} -> {artifacts}")
            except Exception as exc:  # noqa: BLE001
                errors[symbol] = str(exc)
                print(f"[ERR] {symbol}: {exc}")
    else:
        tasks = []
        for symbol, input_path in symbol_map.items():
            tasks.append(
                (
                    symbol,
                    str(input_path),
                    normalized_timeframes,
                    {tf: str(path) for tf, path in resolved_output_dirs.items()},
                    config_module,
                    overwrite,
                )
            )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_parallel_process_symbol, task): task[0] for task in tasks}
            for future in as_completed(future_map):
                result = future.result()
                symbol = result["symbol"]
                if result["status"] == "ok":
                    processed += 1
                    outputs = result.get("outputs", {})
                    artifacts = ", ".join(Path(path).name for path in outputs.values())
                    print(f"[OK] {symbol} -> {artifacts}")
                else:
                    errors[symbol] = result.get("error", "unknown error")
                    print(f"[ERR] {symbol}: {errors[symbol]}")

    print(
        f"Completed indicator generation: {processed} success, {len(errors)} failure(s)."
    )
    if errors:
        for symbol, message in errors.items():
            print(f" - {symbol}: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-compute indicator columns for parquet files")
    parser.add_argument("--input", "-i", help="Single parquet file to process")
    parser.add_argument(
        "--output",
        "-o",
        help="Output parquet path for single-file mode; defaults to <input>_features.parquet",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing multiple raw parquet files for bulk processing",
    )
    parser.add_argument(
        "--input-dir-1m",
        dest="input_dir_1m",
        help="Alias for --input-dir specifically for raw 1m parquet directories",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store feature files when using --input-dir (default: <input-dir>/features)",
    )
    parser.add_argument(
        "--output-dir-15m",
        help="Directory for 15m feature outputs when using --input-dir (default: <parent>/precomputed_15m)",
    )
    parser.add_argument(
        "--output-dir-1h",
        help="Directory for 1h feature outputs when using --input-dir (default: <parent>/precomputed_1h)",
    )
    parser.add_argument(
        "--symbols",
        help="Comma-separated list of symbols (file stems) to process when using --input-dir",
        default="",
    )
    parser.add_argument(
        "--config-module",
        default=DEFAULT_CONFIG_MODULE,
        help="Config module to import (e.g. config or config_futures)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum parallel workers (1-10) for bulk processing",
    )
    parser.add_argument(
        "--timeframes",
        default=",".join(DEFAULT_TIMEFRAMES),
        help="Comma-separated list of timeframes to produce (subset of 1m,15m,1h)",
    )
    parser.add_argument(
        "--output-15m",
        help="Output parquet path for single-file 15m mode (defaults beside input)",
    )
    parser.add_argument(
        "--output-1h",
        help="Output parquet path for single-file 1h mode (defaults beside input)",
    )
    return parser.parse_args()


def run_from_cli() -> None:
    args = parse_args()
    max_workers = clamp_workers(args.max_workers)
    timeframes = parse_timeframes(args.timeframes)

    input_dir_arg = args.input_dir_1m or args.input_dir
    if input_dir_arg:
        input_dir = Path(input_dir_arg).expanduser().resolve()
        simple_pair = set(timeframes) == {"15m", "1h"}
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        if simple_pair:
            out_15m = default_output_dir_for_timeframe(input_dir, "15m", args.output_dir_15m)
            out_1h = default_output_dir_for_timeframe(input_dir, "1h", args.output_dir_1h)
            process_all_symbols_1m_to_15m_1h(
                input_dir,
                out_15m,
                out_1h,
                symbols=symbols or None,
                max_workers=max_workers,
                config_module=args.config_module,
                overwrite=args.overwrite,
            )
            return
        output_dirs: Dict[str, Path] = {}
        for tf in timeframes:
            if tf == "1m":
                output_dirs[tf] = default_output_dir_for_timeframe(input_dir, tf, args.output_dir)
            elif tf == "15m":
                output_dirs[tf] = default_output_dir_for_timeframe(input_dir, tf, args.output_dir_15m)
            elif tf == "1h":
                output_dirs[tf] = default_output_dir_for_timeframe(input_dir, tf, args.output_dir_1h)
            else:
                raise ValueError(f"Unsupported timeframe requested: {tf}")
        process_all_symbols_parallel(
            input_dir,
            symbols=symbols or None,
            output_dirs=output_dirs,
            timeframes=timeframes,
            max_workers=max_workers,
            config_module=args.config_module,
            overwrite=args.overwrite,
        )
        return

    if not args.input:
        raise ValueError("Either --input or --input-dir must be provided")

    input_path = Path(args.input).expanduser().resolve()
    cfg = resolve_config(args.config_module)
    output_targets: Dict[str, Path] = {}
    for tf in timeframes:
        if tf == "1m":
            override = args.output
        elif tf == "15m":
            override = args.output_15m
        elif tf == "1h":
            override = args.output_1h
        else:
            override = None
        if override:
            target_path = Path(override).expanduser().resolve()
        else:
            target_path = build_timeframe_output_path(input_path.stem, tf, input_path.parent)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        output_targets[tf] = target_path

    outputs = process_symbol_timeframes_loaded(
        input_path.stem,
        input_path,
        timeframes,
        output_targets,
        cfg_module=cfg,
        overwrite=args.overwrite,
    )
    emitted = ", ".join(str(Path(path)) for path in outputs.values())
    print(f"Indicators saved to {emitted}")


if __name__ == "__main__":
    run_from_cli()
