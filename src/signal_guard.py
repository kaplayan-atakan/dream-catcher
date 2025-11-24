"""Post-signal validation and blocking helpers for STRONG/ULTRA signals."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

import config
import data_fetcher

logger = logging.getLogger(__name__)

BAR_INTERVAL_MS = config.MAIN_TIMEFRAME_MINUTES * 60 * 1000


@dataclass
class ActiveSignalState:
    """Runtime context for a STRONG/ULTRA signal awaiting validation."""

    symbol: str
    label: str
    price: float
    target_price: float
    emitted_at: datetime
    signal_close_time: int
    deadline_close_time: int
    first_bar_deadline: int
    first_bar_checked: bool = False
    first_bar_passed: bool = False


class SignalMonitor:
    """Tracks active signals and enforces post-signal filters (Filters 2 & 3)."""

    def __init__(self) -> None:
        self.active: Dict[str, ActiveSignalState] = {}
        self.blocked_until: Dict[str, datetime] = {}
        self.block_reasons: Dict[str, str] = {}

    def is_symbol_blocked(self, symbol: str) -> bool:
        now = datetime.utcnow()
        expiry = self.blocked_until.get(symbol)
        if expiry and now < expiry:
            return True
        if expiry and now >= expiry:
            self.blocked_until.pop(symbol, None)
            self.block_reasons.pop(symbol, None)
        return False

    def get_block_reason(self, symbol: str) -> Optional[str]:
        if not self.is_symbol_blocked(symbol):
            return None
        return self.block_reasons.get(symbol)

    def register_signal(self, signal: dict) -> None:
        """Start monitoring a freshly emitted STRONG/ULTRA signal."""
        symbol = signal.get("symbol")
        price = float(signal.get("price") or 0.0)
        close_time = signal.get("bar_close_time")
        if not symbol or not close_time or price <= 0:
            return

        target_multiplier = 1 + (config.POST_SIGNAL_TARGET_PCT / 100.0)
        target_price = price * target_multiplier
        deadline_close_time = close_time + (config.POST_SIGNAL_MONITOR_BARS * BAR_INTERVAL_MS)
        first_bar_deadline = close_time + BAR_INTERVAL_MS

        self.active[symbol] = ActiveSignalState(
            symbol=symbol,
            label=signal.get("label", "STRONG_BUY"),
            price=price,
            target_price=target_price,
            emitted_at=datetime.utcnow(),
            signal_close_time=int(close_time),
            deadline_close_time=int(deadline_close_time),
            first_bar_deadline=int(first_bar_deadline),
        )
        logger.info(
            "Post-signal monitor registered for %s (target %.4f by %s)",
            symbol,
            target_price,
            datetime.fromtimestamp(deadline_close_time / 1000.0),
        )

    async def evaluate_active_signals(self, session) -> None:
        """Poll Binance and update each active signal's validation status."""
        if not self.active:
            return

        for symbol in list(self.active.keys()):
            state = self.active.get(symbol)
            if not state:
                continue
            try:
                klines = await data_fetcher.fetch_multi_timeframe_klines(session, symbol)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to refresh klines for %s: %s", symbol, exc)
                continue

            bars = klines.get(config.MAIN_TIMEFRAME) or []
            if not bars:
                continue

            self._evaluate_first_bar(state, bars)
            if symbol not in self.active:
                continue

            self._evaluate_price_target(state, bars)

    def _evaluate_first_bar(self, state: ActiveSignalState, bars: list[dict]) -> None:
        if state.first_bar_checked:
            return

        next_bar = next((bar for bar in bars if bar.get("close_time") > state.signal_close_time), None)
        if not next_bar:
            return

        state.first_bar_checked = True
        if next_bar.get("close", 0.0) > next_bar.get("open", 0.0):
            state.first_bar_passed = True
            return

        self._mark_failure(state.symbol, "First 15m bar failed to close green")

    def _evaluate_price_target(self, state: ActiveSignalState, bars: list[dict]) -> None:
        for bar in bars:
            close_time = int(bar.get("close_time") or 0)
            if close_time <= state.signal_close_time:
                continue
            if close_time > state.deadline_close_time:
                break
            high_price = float(bar.get("high") or 0.0)
            if high_price >= state.target_price:
                logger.info("Post-signal target met for %s (+1.5%% within window)", state.symbol)
                self.active.pop(state.symbol, None)
                return

        latest_close_time = int(bars[-1].get("close_time") or 0)
        if latest_close_time > state.deadline_close_time:
            self._mark_failure(state.symbol, "+1.5% target not reached within 12 bars")

    def _mark_failure(self, symbol: str, reason: str) -> None:
        self.active.pop(symbol, None)
        block_until = datetime.utcnow() + timedelta(minutes=config.POST_SIGNAL_BLOCK_MINUTES)
        self.blocked_until[symbol] = block_until
        self.block_reasons[symbol] = reason
        logger.warning(
            "Post-signal block applied to %s until %s (%s)",
            symbol,
            block_until.strftime("%H:%M"),
            reason,
        )