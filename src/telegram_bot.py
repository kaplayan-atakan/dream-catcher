"""Telegram formatting/sending helpers for Phase 1.

Actual sending is gated by config.ENABLE_TELEGRAM so the bot can run without
credentials during development.
"""
from __future__ import annotations

import logging
from typing import Dict

import aiohttp

import config

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org"
_LABEL_EMOJI = {
    "ULTRA_BUY": "🚀",
    "STRONG_BUY": "📈",
}
_DEFAULT_EMOJI = "🔔"


def format_signal_message(signal: Dict[str, object]) -> str:
    """Return a compact multiline summary of the signal."""
    price = signal.get("price")
    change = signal.get("price_change_pct")
    quote_vol = signal.get("quote_volume")
    label = signal.get("label", "NONE")
    symbol = signal.get("symbol", "UNKNOWN")
    emoji = _LABEL_EMOJI.get(label, _DEFAULT_EMOJI)
    lines = [
        f"{emoji} {label} - {symbol}",
        "• Price: {price:.4f} USDT | 24h: {change:+.2f}% | Vol: {vol:,.0f} USDT".format(
            price=float(price) if price is not None else 0.0,
            change=float(change) if change is not None else 0.0,
            vol=float(quote_vol) if quote_vol is not None else 0.0,
        ),
        "• Scores: Trend={trend}, Osc={osc}, Vol={volb}, PA={pa}, Total={total}".format(
            trend=signal.get("trend_score", 0),
            osc=signal.get("osc_score", 0),
            volb=signal.get("vol_score", 0),
            pa=signal.get("pa_score", 0),
            total=signal.get("total_score", 0),
        ),
    ]
    reasons = signal.get("reasons") or []
    if reasons:
        lines.append("• Top reasons:")
        for reason in reasons[:3]:
            lines.append(f"  - {reason}")
    return "\n".join(lines)


async def send_telegram_message(text: str) -> None:
    """Send `text` to the configured Telegram chat if enabled."""
    if not config.ENABLE_TELEGRAM:
        return
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram enabled but credentials missing; skipping send")
        return

    url = f"{_TELEGRAM_API}/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
        # Plain text avoids Markdown escaping issues that caused HTTP 400 errors.
    }

    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram send failed: %s", exc)
