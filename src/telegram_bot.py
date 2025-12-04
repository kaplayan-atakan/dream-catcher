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


def _escape_markdown(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2"""
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text


def format_signal_message(
    signal: Dict[str, object],
    display_label: str | None = None,
    info_note: str | None = None,
) -> str:
    """Return a beautifully formatted signal message with optional display label overrides."""
    price = signal.get("price")
    change = signal.get("price_change_pct")
    quote_vol = signal.get("quote_volume")
    original_label = signal.get("label", "NO_SIGNAL")
    label = display_label or original_label
    symbol = signal.get("symbol", "UNKNOWN")
    
    # Get scores
    trend_score = signal.get("trend_score", 0)
    osc_score = signal.get("osc_score", 0)
    vol_score = signal.get("vol_score", 0)
    pa_score = signal.get("pa_score", 0)
    core_score = signal.get("score_core", signal.get("total_score", 0))
    htf_bonus = signal.get("htf_bonus", 0)
    total_score = signal.get("score_total", signal.get("total_score", 0))
    
    # Find highest score category
    score_categories = {
        "Trend": trend_score,
        "Osc": osc_score,
        "Vol": vol_score,
        "PA": pa_score
    }
    max_category = max(score_categories.items(), key=lambda x: x[1])
    
    # Different styling based on signal type
    if label == "ULTRA_BUY":
        header_emoji = "🚀🔥"
        divider = "━━━━━━━━━━━━━━━━━━━━"
        signal_type = "*ULTRA BUY*"
    elif label == "STRONG_BUY":
        header_emoji = "📈💎"
        divider = "━━━━━━━━━━━━━━━━━━━━"
        signal_type = "*STRONG BUY*"
    elif label == config.WATCH_PREMIUM_TG_LABEL:
        header_emoji = "🔔🌟"
        divider = "━━━━━━━━━━━━━━━━━━━━"
        signal_type = "*WATCH_PREMIUM*"
    else:  # WATCH
        header_emoji = "👀📊"
        divider = "━━━━━━━━━━━━━━━━━━━━"
        signal_type = "*WATCH*"
    
    # Escape special characters for MarkdownV2
    symbol_escaped = _escape_markdown(symbol)
    
    # Build message
    lines = [
        f"{header_emoji} {signal_type}",
        f"*{symbol_escaped}*",
        divider,
    ]
    
    # Price and change info
    price_val = float(price) if price is not None else 0.0
    change_val = float(change) if change is not None else 0.0
    vol_val = float(quote_vol) if quote_vol is not None else 0.0
    
    change_emoji = "🟢" if change_val >= 0 else "🔴"
    price_str = _escape_markdown(f"${price_val:.6f}")
    change_str = _escape_markdown(f"{change_val:+.2f}%")
    vol_str = _escape_markdown(f"${vol_val:,.0f}")
    
    lines.append(f"💰 Price: {price_str}")
    lines.append(f"{change_emoji} 24h Change: {change_str}")
    lines.append(f"📊 Volume: {vol_str}")
    lines.append(divider)
    
    # Scores section with bold total and highest category
    lines.append("*📈 SCORES*")
    
    # Individual scores with highlighting for max category
    for cat_name, cat_score in score_categories.items():
        if cat_name == max_category[0]:
            lines.append(f"  *{_escape_markdown(cat_name)}*: *{cat_score}* ⭐")
        else:
            lines.append(f"  {_escape_markdown(cat_name)}: {cat_score}")
    
    lines.append("")
    lines.append(f"Core Score: {core_score}")
    lines.append(f"HTF Bonus: \\+{htf_bonus}")
    lines.append(f"*TOTAL: {total_score}* 🎯")
    lines.append(divider)
    if label == config.WATCH_PREMIUM_TG_LABEL or info_note:
        detail_score = total_score if total_score is not None else 0
        detail_score_text = _escape_markdown(str(detail_score))
        label_text = _escape_markdown(label)
        lines.append(divider)
        lines.append(f"• Score: {detail_score_text} | Label: {label_text} (informational)")
        note = info_note or "Early alert only — not actionable. STRONG/ULTRA rules unchanged."
        lines.append(f"• Note: {_escape_markdown(note)}")
        lines.append("")
    
    # HTF details
    trend_details = signal.get("trend_details") or {}
    htf_bits = []
    htf_details = signal.get("htf_details") or {}
    
    if htf_details.get("close_above_ema20"):
        htf_bits.append("1h above EMA20")
    if htf_details.get("ema20_slope_pct"):
        slope_val = float(htf_details['ema20_slope_pct'])
        htf_bits.append(f"1h EMA20 slope {slope_val:+.1f}%")
    if htf_details.get("macd_hist") is not None:
        macd_val = float(htf_details['macd_hist'])
        htf_bits.append(f"1h MACD {macd_val:+.3f}")

    if signal.get("fourh_alignment_ok"):
        htf_bits.append("4h EMA stack ok")
    elif signal.get("fourh_price_above_ema20"):
        htf_bits.append("4h above EMA20")

    slope = signal.get("fourh_ema20_slope_pct") or trend_details.get("fourh_ema20_slope_pct")
    if slope:
        slope_val = float(slope)
        htf_bits.append(f"4h EMA20 slope {slope_val:+.1f}%")

    if htf_bits:
        lines.append("*🔍 Higher Timeframes*")
        for bit in htf_bits[:3]:
            lines.append(f"  • {_escape_markdown(bit)}")
        lines.append("")
    
    # Volume details
    vol_details = signal.get("vol_details") or {}
    vol_bits = []
    
    if vol_details.get("volume_spike"):
        spike_factor = vol_details.get("volume_spike_factor")
        if spike_factor:
            vol_bits.append(f"Spike {float(spike_factor):.1f}x avg")
        else:
            vol_bits.append("Volume spike vs avg")

    obv_change = vol_details.get("obv_change_pct")
    if obv_change:
        obv_val = float(obv_change)
        vol_bits.append(f"OBV {obv_val:+.1f}%/{config.OBV_TREND_LOOKBACK} bars")

    bull_power = vol_details.get("bull_power")
    bear_power = vol_details.get("bear_power")
    if bull_power is not None and bear_power is not None:
        vol_bits.append(f"Bull {float(bull_power):.4f} | Bear {float(bear_power):.4f}")

    if vol_bits:
        lines.append("*💪 Volume Analysis*")
        for bit in vol_bits[:3]:
            lines.append(f"  • {_escape_markdown(bit)}")
        lines.append("")

    # Risk tag
    risk_tag = signal.get("risk_tag")
    if risk_tag and risk_tag != "NORMAL":
        lines.append(f"⚠️ *Risk*: {_escape_markdown(risk_tag)}")
        lines.append("")

    # Top reasons
    reasons = signal.get("reasons") or []
    if reasons:
        lines.append("*✅ Key Reasons*")
        for reason in reasons[:3]:
            lines.append(f"  \\- {_escape_markdown(reason)}")
    
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
        "parse_mode": "MarkdownV2",
    }

    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram send failed: %s", exc)


async def send_simple_message(text: str) -> None:
    """Send a simple text message without Markdown formatting."""
    if not config.ENABLE_TELEGRAM:
        return
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        logger.warning("Telegram enabled but credentials missing; skipping send")
        return

    url = f"{_TELEGRAM_API}/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": config.TELEGRAM_CHAT_ID,
        "text": text,
    }

    timeout = aiohttp.ClientTimeout(total=15)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram send failed: %s", exc)
