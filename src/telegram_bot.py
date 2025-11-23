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
    label = signal.get("label", "NO_SIGNAL")
    symbol = signal.get("symbol", "UNKNOWN")
    emoji = _LABEL_EMOJI.get(label, _DEFAULT_EMOJI)
    trend_details = signal.get("trend_details") or {}
    vol_details = signal.get("vol_details") or {}
    core_score = signal.get("score_core", signal.get("total_score", 0))
    htf_bonus = signal.get("htf_bonus", 0)
    total_score = signal.get("score_total", signal.get("total_score", 0))
    lines = [
        f"{emoji} {label} - {symbol}",
        "• Price: {price:.4f} USDT | 24h: {change:+.2f}% | Vol: {vol:,.0f} USDT".format(
            price=float(price) if price is not None else 0.0,
            change=float(change) if change is not None else 0.0,
            vol=float(quote_vol) if quote_vol is not None else 0.0,
        ),
        "• Scores: Trend={trend}, Osc={osc}, Vol={volb}, PA={pa} | Core={core} HTF+{htf} Total={total}".format(
            trend=signal.get("trend_score", 0),
            osc=signal.get("osc_score", 0),
            volb=signal.get("vol_score", 0),
            pa=signal.get("pa_score", 0),
            core=core_score,
            htf=htf_bonus,
            total=total_score,
        ),
    ]

    htf_bits = []
    htf_details = signal.get("htf_details") or {}
    if htf_details.get("close_above_ema20"):
        htf_bits.append("1h above EMA20")
    if htf_details.get("ema20_slope_pct"):
        htf_bits.append(f"1h EMA20 slope {float(htf_details['ema20_slope_pct']):+.1f}%")
    if htf_details.get("macd_hist") is not None:
        htf_bits.append(f"1h MACD {float(htf_details['macd_hist']):+.3f}")

    if signal.get("fourh_alignment_ok"):
        htf_bits.append("4h EMA stack ok")
    elif signal.get("fourh_price_above_ema20"):
        htf_bits.append("4h above EMA20")

    slope = signal.get("fourh_ema20_slope_pct") or trend_details.get("fourh_ema20_slope_pct")
    if slope:
        htf_bits.append(f"4h EMA20 slope {float(slope):+.1f}%")

    if htf_bits:
        lines.append("• HTF: " + ", ".join(htf_bits[:3]))

    vol_bits = []
    if vol_details.get("volume_spike"):
        spike_factor = vol_details.get("volume_spike_factor")
        if spike_factor:
            vol_bits.append(f"Spike {float(spike_factor):.1f}x avg")
        else:
            vol_bits.append("Volume spike vs avg")

    obv_change = vol_details.get("obv_change_pct")
    if obv_change:
        vol_bits.append(f"OBV {float(obv_change):+.1f}%/{config.OBV_TREND_LOOKBACK} bars")

    bull_power = vol_details.get("bull_power")
    bear_power = vol_details.get("bear_power")
    if bull_power is not None and bear_power is not None:
        vol_bits.append(f"Bull {float(bull_power):.4f} | Bear {float(bear_power):.4f}")

    if vol_bits:
        lines.append("• Volume: " + ", ".join(vol_bits[:3]))

    risk_tag = signal.get("risk_tag")
    if risk_tag and risk_tag != "NORMAL":
        lines.append(f"• Risk: {risk_tag}")

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
