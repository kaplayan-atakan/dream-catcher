# Binance USDT Signal Bot – AI Guidance

## Scope and Constraints
- Treat `README-phase1.md`, `README-phase2.md`, `README-phase3.md`, `README.md`, `docs/client-message.md`, and `docs/gosterge-ozet.md` as the scope contract. If a behavior is missing there (or in `main.py`, `analyzer.py`, `rules.py`, `config.py`), leave it unimplemented or stub it with a TODO referencing the absent spec.
- Keep all code deterministic and rule-based; no ML/AI, randomness, or new signal types beyond the documented STRONG_BUY/ULTRA_BUY workflow.
- Only express future ideas or opinions in comments. Active code must mirror the existing architecture; do not widen scope or add new configuration unless the docs explicitly require it.
- When touching helper modules (`data_fetcher.py`, `indicators.py`, `price_action.py`, `telegram_bot.py`, `logger.py`), follow the current call signatures and behaviors expected by `analyzer.py`/`main.py`. Optional hooks for future phases are fine only if they are backward compatible and documented in comments.

## Architecture at a Glance
- Entry point is `src/main.py`: `main_loop()` keeps a long-lived `aiohttp` session, calls `scan_market()`, logs results, enforces cooldown, and optionally pushes Telegram alerts.
- `scan_market()` pulls `/api/v3/ticker/24hr`, applies hard prefilters (volume, price floor, 24h change, cooldown) before queuing candidates for analysis. Respect `MAX_SYMBOLS_PER_SCAN`, concurrency semaphore (10), and batch sleep when adding new network work.
- Per-symbol analysis lives in `src/analyzer.py`. It reconfirms prefilters, downloads klines for every `config.TIMEFRAMES`, derives OHLCV arrays, computes indicators, runs price-action logic, then feeds block scores into `rules.decide_signal_label()`.

## Indicators, Price Action, and Scoring
- Indicator helpers (expected in `src/indicators.py`) must align with analyzer signatures: return full series, use numpy NaNs, and keep lookbacks configurable via `config.py` (EMA20/50, ADX 14, MACD 12-26-9, etc.).
- Price-action heuristics belong in `src/price_action.py`. Return the booleans consumed by `rules.compute_price_action_block` (`long_lower_wick`, `strong_green`, `no_collapse`, `ema20_break`, `volume_spike`, `min_volume`). When you add richer metrics (e.g., wick/body ratios) keep the dict extensible.
- `src/rules.py` is strictly rule-based: each block returns a `BlockScore` with top 3 reason strings, and `SignalResult` only becomes `STRONG_BUY`/`ULTRA_BUY` if combined thresholds plus HTF confirmations pass. Preserve this deterministic scoring when extending logic and resist adding new blocks or endpoints unless mandated by the docs.

## Prefilter, Cooldown, and Data Fetching
- Prefilter thresholds are centralized in `src/config.py` (5M USDT volume, price ≥ 0.02, daily change –15% to +20%). Always reuse those constants rather than hard-coding.
- Cooldown uses `last_signal_times` in `main.py`; before emitting another signal for the same symbol ensure `datetime.utcnow() - last_signal_times[symbol]` exceeds `COOLDOWN_MINUTES`.
- Network helpers (`src/data_fetcher.py`) should share the aiohttp session, honor `REQUEST_TIMEOUT`, `MAX_RETRIES`, and pull klines for `15m`, `1h`, `4h`. Return lists of dicts shaped exactly as analyzer expects (`open`, `high`, `low`, `close`, `volume`)—do not introduce extra endpoints or payload fields beyond the documented needs.

## Async, Logging, and Outputs
- Concurrency is bounded via `asyncio.Semaphore(10)` inside `process_symbol_batch`; keep API-bound work within that guard to prevent Binance bans.
- Logging goes through `log_module` (see `logger.setup_logger()`); signals must be serialized with `log_module.log_signal_to_csv` to `signals_log.csv`. Extend CSV schemas carefully so downstream analyses stay intact.
- Error handling standard: every recoverable exception should be caught and appended to `logs/error.log` as `ISO8601 | module | symbol(optional) | message`. Treat the file as append-only text encoded in UTF-8; never delete or truncate it from automation so post-mortem reviews stay intact. Use helpers (`log_error()` where available) instead of ad-hoc prints.
- When any step errors, immediately fix the issue and rerun the exact failing section/command until it completes successfully; never leave an errored workflow without re-execution confirmation.
- Spot backtest artifacts must be written as XLSX files (`spot_trades.xlsx`, `spot_summary.xlsx`) rather than CSV so analysts can consume them directly in spreadsheets.
- Telegram support (in `src/telegram_bot.py`) is optional but wired from `main.py`. Provide `format_signal_message` that highlights block scores and reasons; respect `config.ENABLE_TELEGRAM`.

## Working Effectively
- Run the bot with `python -m src.main` from the repo root so relative imports resolve.
- When adding new indicators or tweaks, route tunables through `config.py` to keep strategy adjustments non-code where possible, but only introduce new config knobs if the scope docs demand them.
- Maintain pure Python + `aiohttp`/`numpy` stack; the README explicitly forbids ML heuristics. Document any strategy deviations in `docs/` comments and keep executable code aligned with the current phases.
- Missing modules (`data_fetcher.py`, `indicators.py`, `price_action.py`, `telegram_bot.py`, `logger.py`) are expected deliverables—match the expectations spelled out in `README.md` and keep APIs stable because existing code already imports them.
- Suggestions for Phase 2/3 enhancements belong in comments/TODOs unless those features are explicitly activated.

### Response Style and Final Replies
- Provide file blocks exactly when required by the GitHub environment, without preambles like "Preparing final response" or similar boilerplate.
- Keep prose short and focused on what changed and why; non-code narration should be minimal even if the code payload is large.
- Do not repeat the same file contents multiple times in one reply unless the user explicitly asks for it.
- It is acceptable for code/file blocks to be long, but avoid verbose meta commentary outside those blocks.
