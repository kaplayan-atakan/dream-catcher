"""Rule-based scoring aligned with the revised Phase 3 spec."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

import config


def _clamp(value: int, ceiling: int = 5) -> int:
    return min(value, ceiling)


def _ema_similarity(ema_fast: Optional[float], ema_slow: Optional[float]) -> bool:
    if ema_fast in (None, 0) or ema_slow in (None, 0):
        return False
    diff_ratio = abs(ema_fast - ema_slow) / max(abs(ema_slow), 1e-8)
    return diff_ratio <= config.EMA_SIMILARITY_TOLERANCE


# === ZONE CLASSIFICATION HELPERS (V5) ===
def get_rsi_zone(rsi: float) -> str:
    """Classify RSI into zones."""
    if rsi < 30:
        return "deep_oversold"
    elif rsi < 35:
        return "oversold"
    elif rsi < 45:
        return "recovery"
    elif rsi < 55:
        return "neutral"
    elif rsi < 65:
        return "strong"
    else:
        return "overbought"


def get_ema_zone(price: float, ema20: Optional[float]) -> Tuple[str, float]:
    """Classify EMA distance into zones. Returns (zone, distance_pct)."""
    if ema20 is None or ema20 <= 0:
        return "unknown", 0.0
    
    dist_pct = (price - ema20) / ema20 * 100
    
    if dist_pct < -3.0:
        return "far_below", dist_pct
    elif dist_pct < -1.0:
        return "below", dist_pct
    elif dist_pct <= 1.0:
        return "near", dist_pct
    elif dist_pct <= 3.0:
        return "above", dist_pct
    else:
        return "far_above", dist_pct


def get_24h_zone(change_pct: float) -> str:
    """Classify 24h change into zones."""
    if change_pct < -5.0:
        return "dump"
    elif change_pct < -2.0:
        return "down"
    elif change_pct <= 2.0:
        return "flat"
    elif change_pct <= 5.0:
        return "up"
    else:
        return "pump"


def check_dip_hunter_signal(
    rsi_value: float,
    price: float,
    ema20: Optional[float],
    change_24h_pct: float,
    base_score: int,
) -> Tuple[bool, str, int, List[str]]:
    """
    Check if conditions match DIP_HUNTER mode.
    
    Returns:
        (is_dip_signal, label, adjusted_score, reasons)
    """
    if not getattr(config, "ENABLE_DIP_HUNTER", False):
        return False, "", 0, []
    
    reasons = []
    bonus = 0
    
    # Get zones
    rsi_zone = get_rsi_zone(rsi_value)
    ema_zone, ema_dist = get_ema_zone(price, ema20)
    change_zone = get_24h_zone(change_24h_pct)
    
    # Check DIP_HUNTER conditions
    # 1. RSI must be oversold
    dip_rsi_max = getattr(config, "DIP_RSI_MAX", 35)
    if rsi_value > dip_rsi_max:
        return False, "", 0, []
    
    # 2. Price must be below EMA20
    if ema_zone not in ("below", "far_below"):
        return False, "", 0, []
    
    dip_ema_dist_min = getattr(config, "DIP_EMA_DIST_MIN_PCT", -1.0)
    if ema_dist > dip_ema_dist_min:
        return False, "", 0, []
    
    # 3. 24h change must be dump
    dip_24h_max = getattr(config, "DIP_24H_CHANGE_MAX", -5.0)
    if change_24h_pct > dip_24h_max:
        return False, "", 0, []
    
    # All conditions met - this is a DIP signal
    reasons.append(f"🎯 DIP_HUNTER: RSI={rsi_value:.0f} ({rsi_zone})")
    reasons.append(f"📉 EMA20 distance: {ema_dist:.1f}% ({ema_zone})")
    reasons.append(f"📊 24h change: {change_24h_pct:.1f}% ({change_zone})")
    
    # Apply bonuses
    dip_oversold_bonus = getattr(config, "DIP_OVERSOLD_BONUS", 3)
    dip_deep_dump_bonus = getattr(config, "DIP_DEEP_DUMP_BONUS", 2)
    dip_ema_far_below_bonus = getattr(config, "DIP_EMA_FAR_BELOW_BONUS", 2)
    
    if rsi_value < 30:
        bonus += dip_oversold_bonus
        reasons.append(f"🔥 Deep oversold bonus +{dip_oversold_bonus}")
    
    if change_24h_pct < -8.0:
        bonus += dip_deep_dump_bonus
        reasons.append(f"🔥 Deep dump bonus +{dip_deep_dump_bonus}")
    
    if ema_dist < -3.0:
        bonus += dip_ema_far_below_bonus
        reasons.append(f"🔥 Far below EMA bonus +{dip_ema_far_below_bonus}")
    
    # Calculate final score
    final_score = base_score + bonus
    
    # Check minimum score
    dip_min_score = getattr(config, "DIP_MIN_SCORE", 6)
    if final_score < dip_min_score:
        return False, "", 0, []
    
    label = getattr(config, "DIP_SIGNAL_LABEL", "DIP_ALERT")
    
    return True, label, final_score, reasons

@dataclass
class BlockScore:
    score: int
    reasons: List[str]
    details: Optional[Dict[str, Any]] = None


@dataclass
class SignalResult:
    symbol: str
    trend_score: int
    osc_score: int
    vol_score: int
    pa_score: int
    htf_bonus: int
    score_core: int
    score_total: int
    label: str
    reasons: List[str]
    rsi: float
    price: float = 0.0
    price_change_pct: float = 0.0
    quote_volume: float = 0.0
    risk_tag: Optional[str] = None
    trend_details: Optional[Dict[str, Any]] = None
    osc_details: Optional[Dict[str, Any]] = None
    vol_details: Optional[Dict[str, Any]] = None
    pa_details: Optional[Dict[str, Any]] = None
    htf_details: Optional[Dict[str, Any]] = None
    mtf_trend_confirmed: bool = False
    htf_price_above_ema: bool = False
    fourh_price_above_ema20: bool = False
    fourh_alignment_ok: bool = False
    fourh_ema20_slope_pct: float = 0.0
    bar_close_time: Optional[int] = None
    filter_notes: Optional[List[str]] = None
    total_score: int = field(init=False)

    def __post_init__(self) -> None:
        self.total_score = self.score_total


def compute_trend_block(
    price: float,
    ema20: Optional[float],
    ema50: Optional[float],
    adx: float,
    plus_di: float,
    minus_di: float,
    macd_hist: float,
    macd_hist_rising: bool,
    momentum: float,
    ao: float,
) -> BlockScore:
    """Trend block implements the 0–5 rubric from the revised spec."""
    adx_pts = 0
    if adx >= 25:
        adx_pts = 2
    elif adx >= 20:
        adx_pts = 1

    di_pts = 1 if plus_di > minus_di else 0

    price_above_ema20 = ema20 is not None and price > ema20
    ema_stack = ema20 is not None and ema50 is not None and price > ema20 > ema50
    ema_close = price_above_ema20 and _ema_similarity(ema20, ema50)
    ema_pts = 2 if ema_stack else (1 if ema_close else 0)

    macd_pts = 1 if (macd_hist > 0 and macd_hist_rising) else 0
    ao_mom_pts = 1 if (momentum > 0 and ao > 0) else 0

    raw_score = adx_pts + di_pts + ema_pts + macd_pts + ao_mom_pts
    score = _clamp(raw_score)

    reasons: List[str] = []
    if adx_pts:
        strength = "strong" if adx_pts == 2 else "moderate"
        reasons.append(f"Trend: ADX {adx:.1f} {strength} with DI+>DI-")
    elif di_pts:
        reasons.append("Trend: DI+ leading DI-")

    if ema_stack:
        reasons.append("Trend: Price>EMA20>EMA50 stack intact")
    elif ema_close:
        reasons.append("Trend: Price above EMA20 while EMA20≈EMA50")

    if macd_pts:
        reasons.append("Trend: MACD histogram positive & rising")
    if ao_mom_pts:
        reasons.append("Trend: Momentum and AO both positive")

    details = {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "ema20": ema20,
        "ema50": ema50,
        "price_above_ema20": price_above_ema20,
        "ema_stack_ok": ema_stack,
        "macd_hist": macd_hist,
        "macd_hist_rising": macd_hist_rising,
        "momentum": momentum,
        "ao": ao,
    }
    return BlockScore(score=int(score), reasons=reasons[:4], details=details)


def compute_osc_block(
    rsi_val: float,
    stoch_k: float,
    cci: float,
    stoch_rsi: float,
    williams_r: float,
    uo: float,
    stoch_rsi_prev: Optional[float] = None,
    uo_prev: Optional[float] = None,
) -> BlockScore:
    """Oscillator block using the 0–5 rubric."""
    rsi_pts = 0
    if config.RSI_STRONG_MIN <= rsi_val <= config.RSI_STRONG_MAX:
        rsi_pts = 2
    elif (
        config.RSI_BUFFER_MIN <= rsi_val < config.RSI_STRONG_MIN
        or config.RSI_STRONG_MAX < rsi_val <= config.RSI_BUFFER_MAX
    ):
        rsi_pts = 1

    stoch_pts = 1 if stoch_k > config.STOCH_K_MIDLINE else 0
    cci_pts = 1 if cci > config.CCI_STRONG_THRESHOLD else 0

    stoch_rsi_bull = (
        stoch_rsi > config.STOCH_RSI_BULL_LEVEL
        and stoch_rsi_prev is not None
        and stoch_rsi > stoch_rsi_prev
    )
    uo_rising = uo_prev is not None and (uo - uo_prev) >= config.UO_RISING_MIN_DELTA
    combo_bull = stoch_rsi_bull or (williams_r > config.WILLIAMS_BULLISH and uo_rising)
    other_pts = 1 if combo_bull else 0

    score = _clamp(rsi_pts + stoch_pts + cci_pts + other_pts)

    reasons: List[str] = []
    if rsi_pts == 2:
        reasons.append(f"Osc: RSI {rsi_val:.1f} in 50-65 sweet spot")
    elif rsi_pts == 1:
        reasons.append(f"Osc: RSI {rsi_val:.1f} holding mid band")

    if stoch_pts:
        reasons.append(f"Osc: StochK {stoch_k:.1f} > 50")
    if cci_pts:
        reasons.append(f"Osc: CCI {cci:.0f} above {config.CCI_STRONG_THRESHOLD}")
    if other_pts:
        if stoch_rsi_bull:
            reasons.append("Osc: StochRSI above 50 and rising")
        else:
            reasons.append("Osc: Williams%R + UO rising confirmation")

    details = {
        "rsi": rsi_val,
        "stoch_k": stoch_k,
        "cci": cci,
        "stoch_rsi": stoch_rsi,
        "stoch_rsi_prev": stoch_rsi_prev,
        "williams_r": williams_r,
        "uo": uo,
        "uo_prev": uo_prev,
    }
    return BlockScore(score=int(score), reasons=reasons[:4], details=details)


def compute_volume_block(
    bull_power: float,
    bear_power: float,
    volume_spike_factor: Optional[float],
    obv_change_pct: float,
) -> BlockScore:
    """Volume & power block: spike + OBV + bull bear power."""
    spike_pts = 0
    if volume_spike_factor is not None:
        if volume_spike_factor >= config.VOLUME_SPIKE_STRONG:
            spike_pts = 2
        elif volume_spike_factor >= config.VOLUME_SPIKE_MEDIUM:
            spike_pts = 1

    obv_pts = 0
    if obv_change_pct >= config.OBV_UPTREND_MIN_PCT:
        obv_pts = 2
    elif obv_change_pct >= config.OBV_SIDEWAYS_MIN_PCT:
        obv_pts = 1

    power_pts = 1 if (bull_power > 0 and bear_power < 0) else 0

    score = _clamp(spike_pts + obv_pts + power_pts)

    reasons: List[str] = []
    if spike_pts == 2:
        reasons.append(f"Vol: Volume spike {volume_spike_factor:.1f}x avg (strong)")
    elif spike_pts == 1:
        reasons.append(f"Vol: Volume spike {volume_spike_factor:.1f}x avg")

    if obv_pts == 2:
        reasons.append(
            f"Vol: OBV up {obv_change_pct:.1f}%/{config.OBV_TREND_LOOKBACK} bars"
        )
    elif obv_pts == 1:
        reasons.append("Vol: OBV tilting upward")

    if power_pts:
        reasons.append(
            f"Vol: Bull power {bull_power:.4f} vs Bear {bear_power:.4f} (bull bias)"
        )

    details = {
        "bull_power": bull_power,
        "bear_power": bear_power,
        "volume_spike_factor": volume_spike_factor,
        "obv_change_pct": obv_change_pct,
    }
    return BlockScore(score=int(score), reasons=reasons[:4], details=details)


def compute_price_action_block(pa_signals: dict) -> BlockScore:
    """Price-action scoring with collapse gate and EMA breakout tiers."""
    details = dict(pa_signals.get("details") or {})
    collapse_ok = pa_signals.get("collapse_ok", True)
    if not collapse_ok:
        drop_pct = details.get("max_drop_pct")
        if drop_pct is not None:
            reasons = [
                f"PA: Collapse {drop_pct:.1f}% in last {config.COLLAPSE_LOOKBACK_BARS} candles"
            ]
        else:
            reasons = ["PA: Recent collapse blocks score"]
        return BlockScore(score=0, reasons=reasons, details=details)

    ema_pts = 0
    if pa_signals.get("ema_breakout"):
        ema_pts += 1
    if pa_signals.get("ema_retest"):
        ema_pts += 1

    green_pts = 0
    if pa_signals.get("very_strong_green"):
        green_pts = 2
    elif pa_signals.get("strong_green"):
        green_pts = 1

    wick_pts = 1 if pa_signals.get("long_lower_wick") else 0

    score = _clamp(ema_pts + green_pts + wick_pts)

    reasons: List[str] = []
    if pa_signals.get("ema_breakout"):
        reasons.append("PA: EMA20 breakout confirmed")
    if pa_signals.get("ema_retest"):
        reasons.append("PA: EMA20 retest + bounce")
    if pa_signals.get("very_strong_green"):
        reasons.append("PA: Very strong green impulse candle")
    elif pa_signals.get("strong_green"):
        reasons.append("PA: Strong green candle backing move")
    if pa_signals.get("long_lower_wick"):
        reasons.append("PA: Long lower wick support sweep")

    return BlockScore(score=int(score), reasons=reasons[:4], details=details)


def compute_htf_bonus(htf_context: Dict[str, Any]) -> BlockScore:
    """1h higher-timeframe bonus: +1 each for close>EMA20, positive slope, MACD >= 0."""
    close_above_ema = bool(htf_context.get("close_above_ema20"))
    ema_slope_pct = float(htf_context.get("ema20_slope_pct", 0.0) or 0.0)
    macd_hist = float(htf_context.get("macd_hist", 0.0) or 0.0)

    bonus = 0
    reasons: List[str] = []
    if close_above_ema:
        bonus += 1
        reasons.append("HTF: 1h close above EMA20")
    if ema_slope_pct > config.HTF_SLOPE_MIN_PCT:
        bonus += 1
        reasons.append(f"HTF: EMA20 slope {ema_slope_pct:+.2f}% rising")
    if macd_hist >= 0:
        bonus += 1
        reasons.append("HTF: 1h MACD histogram >= 0")

    details = {
        "close_above_ema20": close_above_ema,
        "ema20_slope_pct": ema_slope_pct,
        "macd_hist": macd_hist,
    }
    return BlockScore(score=_clamp(bonus, ceiling=3), reasons=reasons[:3], details=details)


def detect_risk(
    meta: Optional[Dict[str, Any]],
    trend_score: int,
    vol_score: int,
    osc_score: int,
    pa_score: int,
) -> Optional[str]:
    meta = meta or {}
    change = meta.get("price_change_pct")
    if change is not None and change > config.RISK_LATE_PUMP_CHANGE:
        return "LATE_PUMP"
    if vol_score >= config.RISK_VOL_STRONG and trend_score <= config.RISK_TREND_WEAK:
        return "PUMP_DUMP_RISK"
    return "NORMAL"


# ============ BLOW-OFF TOP GUARD FUNCTIONS ============

def is_parabolic_extension(context: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Parabolic Extension Filter (Blow-off Top Guard #1)
    
    Detects when price has run too far from EMA20 in a parabolic move.
    Conditions:
    - Price >= EMA20 + PARABOLIC_EMA_DIST_PCT
    - Last N closes are ALL rising consecutively
    - Total run of last N bars >= PARABOLIC_RUN_PCT
    - 24h change >= PARABOLIC_24H_MIN_PCT
    """
    if not getattr(config, "ENABLE_PARABOLIC_EXTENSION_FILTER", False):
        return False, ""

    close = context.get("last_close")
    ema20 = context.get("ema20_15m")
    closes = context.get("recent_closes_15m") or []
    change_24h = context.get("change_24h_pct", 0.0)

    if close is None or ema20 is None or ema20 <= 0:
        return False, ""

    n_bars = getattr(config, "PARABOLIC_CONSECUTIVE_BARS", 5)
    if len(closes) < n_bars:
        return False, ""

    # Check distance from EMA20
    pct_from_ema = (close - ema20) / ema20 * 100.0
    if pct_from_ema < getattr(config, "PARABOLIC_EMA_DIST_PCT", 5.0):
        return False, ""

    # Check consecutive rising closes
    recent = closes[-n_bars:]
    all_rising = all(recent[i] > recent[i - 1] for i in range(1, len(recent)))
    if not all_rising:
        return False, ""

    # Check total run percentage
    total_run = (recent[-1] - recent[0]) / recent[0] * 100.0 if recent[0] > 0 else 0.0
    if total_run < getattr(config, "PARABOLIC_RUN_PCT", 4.0):
        return False, ""

    # Check 24h change threshold
    if change_24h < getattr(config, "PARABOLIC_24H_MIN_PCT", 8.0):
        return False, ""

    note = f"Filter: Parabolic extension (+{pct_from_ema:.1f}% from EMA20, {n_bars} rising bars, +{total_run:.1f}% run)"
    return True, note


def is_blowoff_candle(context: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Blow-off Candle Filter (Blow-off Top Guard #2)
    
    Detects a single exhaustion spike candle.
    Conditions:
    - Candle range >= BLOWOFF_RANGE_MULTIPLIER × median range
    - Close near high (upper wick <= BLOWOFF_UPPER_WICK_MAX_PCT of range)
    - Candle is green (close > open)
    - Volume >= BLOWOFF_VOLUME_MULTIPLIER × avg volume
    """
    if not getattr(config, "ENABLE_BLOWOFF_CANDLE_FILTER", False):
        return False, ""

    highs = context.get("high_15m") or []
    lows = context.get("low_15m") or []
    closes = context.get("recent_closes_15m") or []
    opens = context.get("opens_15m") or []
    volumes = context.get("volumes_15m") or []

    lb = getattr(config, "BLOWOFF_LOOKBACK", 20)
    if len(highs) < lb or len(lows) < lb or len(volumes) < lb or len(opens) < 1 or len(closes) < 1:
        return False, ""

    h = highs[-1]
    l = lows[-1]
    c = closes[-1]
    o = opens[-1]
    v = volumes[-1]

    rng = h - l
    if rng <= 0:
        return False, ""

    # Median range of last N bars
    ranges = [highs[-(i+1)] - lows[-(i+1)] for i in range(min(lb, len(highs)))]
    if not ranges:
        return False, ""
    median_rng = float(np.median(ranges))
    if median_rng <= 0:
        return False, ""

    range_mult = getattr(config, "BLOWOFF_RANGE_MULTIPLIER", 2.5)
    if rng < range_mult * median_rng:
        return False, ""

    # Close near the high (small upper wick)
    upper_wick_ratio = (h - c) / rng
    wick_max = getattr(config, "BLOWOFF_UPPER_WICK_MAX_PCT", 0.20)
    if upper_wick_ratio > wick_max:
        return False, ""

    # Must be green candle
    if c <= o:
        return False, ""

    # Volume spike check
    avg_vol = sum(volumes[-lb:]) / lb if len(volumes) >= lb else sum(volumes) / max(len(volumes), 1)
    vol_mult = getattr(config, "BLOWOFF_VOLUME_MULTIPLIER", 3.0)
    if avg_vol <= 0 or v < vol_mult * avg_vol:
        return False, ""

    range_factor = rng / median_rng
    vol_factor = v / avg_vol if avg_vol > 0 else 0
    note = f"Filter: Blow-off top candle (range {range_factor:.1f}x median, volume {vol_factor:.1f}x avg)"
    return True, note


def apply_late_pump_hard_stop(label: str, context: Dict[str, Any], notes: List[str]) -> str:
    """
    Late Pump Hard Stop (Blow-off Top Guard #3)
    
    If 24h change >= LATE_PUMP_24H_THRESHOLD, block STRONG/ULTRA entirely.
    The move is likely exhausted at this point.
    """
    if not getattr(config, "ENABLE_LATE_PUMP_HARD_STOP", False):
        return label

    if label not in {"STRONG_BUY", "ULTRA_BUY"}:
        return label

    change_24h = context.get("change_24h_pct", 0.0)
    threshold = getattr(config, "LATE_PUMP_24H_THRESHOLD", 15.0)

    if change_24h >= threshold:
        notes.append(f"Filter: Late pump hard stop (24h +{change_24h:.1f}% >= {threshold}%)")
        return "WATCH"

    return label


def decide_signal_label(
    trend_block: BlockScore,
    osc_block: BlockScore,
    vol_block: BlockScore,
    pa_block: BlockScore,
    htf_block: Optional[BlockScore],
    meta: Optional[Dict[str, Any]],
    *,
    rsi_value: float,
    symbol: str,
    pre_signal_context: Optional[Dict[str, Any]] = None,
    momentum_rev: Optional[Dict[str, Any]] = None,
    reversal_pa: Optional[Dict[str, Any]] = None,
    support_data: Optional[Dict[str, Any]] = None,
) -> SignalResult:
    """Aggregate block scores and label outcome per revised thresholds."""
    trend_component = trend_block.score if config.ENABLE_TREND_BLOCK else 0
    osc_component = osc_block.score if config.ENABLE_OSC_BLOCK else 0
    vol_component = vol_block.score if config.ENABLE_VOLUME_BLOCK else 0
    pa_component = pa_block.score if config.ENABLE_PRICE_ACTION_BLOCK else 0
    htf_bonus = htf_block.score if htf_block else 0

    score_core = trend_component + osc_component + vol_component + pa_component
    # score_total will be recalculated after dip_bonus is applied

    reasons: List[str] = []
    if config.ENABLE_TREND_BLOCK:
        reasons.extend(trend_block.reasons)
    if config.ENABLE_OSC_BLOCK:
        reasons.extend(osc_block.reasons)
    if config.ENABLE_VOLUME_BLOCK:
        reasons.extend(vol_block.reasons)
    if config.ENABLE_PRICE_ACTION_BLOCK:
        reasons.extend(pa_block.reasons)
    if htf_block:
        reasons.extend(htf_block.reasons)

    # === BOTTOM-FISHING: Calculate dip reversal bonus ===
    reversal_bonus = 0
    reversal_reasons: List[str] = []

    mr = momentum_rev or {}
    rp = reversal_pa or {}
    sd = support_data or {}

    if getattr(config, "ENABLE_BOTTOM_FISHING", False):
        # Momentum reversal bonuses
        if mr:
            if mr.get("rsi_oversold_recovery"):
                reversal_bonus += 2
                reversal_reasons.append("RSI recovering from oversold")
            if mr.get("stoch_cross_up"):
                reversal_bonus += 2
                reversal_reasons.append("Stoch rising from low zone")
            if mr.get("macd_hist_turning"):
                reversal_bonus += 2
                reversal_reasons.append("MACD histogram turning up from negative")
            if mr.get("rsi_rising_3bars"):
                reversal_bonus += 1
                rrise = mr.get("details", {}).get("rsi_rise_3bars")
                if rrise is not None:
                    reversal_reasons.append(f"RSI +{rrise:.1f} over last 3 bars")

        # Reversal candle pattern bonuses
        if rp.get("has_reversal"):
            reversal_bonus += 2
            patterns = []
            if rp.get("hammer"):
                patterns.append("hammer")
            if rp.get("bullish_engulfing"):
                patterns.append("bullish_engulfing")
            if rp.get("morning_star"):
                patterns.append("morning_star")
            if patterns:
                reversal_reasons.append("Reversal candles: " + ", ".join(patterns))

        # Support level bonuses
        if sd:
            if sd.get("bounced_from_support"):
                reversal_bonus += 1
                reversal_reasons.append("Bounce from recent support")
            if sd.get("near_ema20_support"):
                reversal_bonus += 1
                reversal_reasons.append("Near EMA20 support")

    # Add reversal bonus to total score
    score_total = score_core + htf_bonus + reversal_bonus

    # Add reversal reasons to main reasons list
    reasons.extend(reversal_reasons[:2])  # Limit to avoid clutter

    # === DIP_HUNTER CHECK (V5) ===
    # Check before regular label logic - DIP signals bypass normal RSI filters
    if getattr(config, "ENABLE_DIP_HUNTER", False):
        ctx = pre_signal_context or {}
        price = ctx.get("last_close", 0)
        ema20 = ctx.get("ema20_15m", 0)
        change_24h = ctx.get("change_24h_pct", 0)
        
        is_dip, dip_label, dip_score, dip_reasons = check_dip_hunter_signal(
            rsi_value=rsi_value,
            price=price,
            ema20=ema20,
            change_24h_pct=change_24h,
            base_score=score_core,
        )
        
        if is_dip:
            reasons.extend(dip_reasons)
            filter_notes: List[str] = []
            return SignalResult(
                symbol=symbol,
                trend_score=trend_component,
                osc_score=osc_component,
                vol_score=vol_component,
                pa_score=pa_component,
                htf_bonus=htf_bonus,
                score_core=score_core,
                score_total=dip_score,
                label=dip_label,
                reasons=reasons[:7],  # Allow more reasons for DIP
                rsi=rsi_value,
                price=meta.get("price") if meta else 0.0,
                price_change_pct=meta.get("price_change_pct") if meta else 0.0,
                quote_volume=meta.get("quote_volume") if meta else 0.0,
                risk_tag="DIP_HUNTER",
                trend_details=trend_block.details,
                osc_details=osc_block.details,
                vol_details=vol_block.details,
                pa_details=pa_block.details,
                htf_details=htf_block.details if htf_block else None,
                filter_notes=filter_notes,
            )

    label = "NO_SIGNAL"
    if score_core < config.CORE_SCORE_WATCH_MIN:
        label = "NO_SIGNAL"
    elif score_core < config.CORE_SCORE_STRONG_MIN:
        label = "WATCH"
    elif (
        score_core < config.CORE_SCORE_ULTRA_MIN
        and trend_component >= config.TREND_MIN_FOR_STRONG
        and vol_component >= config.VOL_MIN_FOR_STRONG
    ):
        label = "STRONG_BUY"
    elif (
        score_core >= config.CORE_SCORE_ULTRA_MIN
        and trend_component >= config.TREND_MIN_FOR_ULTRA
        and osc_component >= config.OSC_MIN_FOR_ULTRA
        and vol_component >= config.VOL_MIN_FOR_ULTRA
        and htf_bonus >= config.HTF_MIN_FOR_ULTRA
    ):
        label = "ULTRA_BUY"

    # === TOP FILTER: Downgrade overbought / late entries ===
    if getattr(config, "ENABLE_TOP_FILTER", False):
        rsi_top = getattr(config, "RSI_TOP_THRESHOLD", 70)
        if rsi_value > 65:
            if label == "ULTRA_BUY":
                label = "STRONG_BUY"
                reasons.append("RSI >65: momentum likely exhausted (ULTRA → STRONG)")
            elif label == "STRONG_BUY":
                label = "WATCH"
                reasons.append("RSI >65: momentum likely exhausted (STRONG → WATCH)")

        price_change_pct = meta.get("price_change_pct") if isinstance(meta, dict) else None
        if price_change_pct is not None and price_change_pct > 8.0:
            if label in {"ULTRA_BUY", "STRONG_BUY"}:
                label = "WATCH"
                reasons.append("24h change > +8%: move likely overextended")

    # If bottom-fishing is enabled but momentum_rev is weak, downgrade strong labels
    if getattr(config, "ENABLE_BOTTOM_FISHING", False) and mr:
        if mr.get("confidence_score", 0) < 3 and label in {"ULTRA_BUY", "STRONG_BUY"}:
            label = "WATCH"
            reasons.append("Dip momentum weak (confidence <3)")

    filter_notes: List[str] = []
    label, filter_notes = _apply_pre_signal_filters(
        label=label,
        score_core=score_core,
        rsi_value=rsi_value,
        context=pre_signal_context or {},
    )

    risk_tag = detect_risk(meta, trend_component, vol_component, osc_component, pa_component)

    return SignalResult(
        symbol=symbol,
        trend_score=trend_component,
        osc_score=osc_component,
        vol_score=vol_component,
        pa_score=pa_component,
        htf_bonus=htf_bonus,
        score_core=score_core,
        score_total=score_total,
        label=label,
        reasons=reasons[:5],
        rsi=rsi_value,
        price=meta.get("price") if meta else 0.0,
        price_change_pct=meta.get("price_change_pct") if meta else 0.0,
        quote_volume=meta.get("quote_volume") if meta else 0.0,
        risk_tag=risk_tag,
        trend_details=trend_block.details,
        osc_details=osc_block.details,
        vol_details=vol_block.details,
        pa_details=pa_block.details,
        htf_details=htf_block.details if htf_block else None,
        filter_notes=filter_notes or None,
    )


def _apply_pre_signal_filters(
    *,
    label: str,
    score_core: int,
    rsi_value: float,
    context: Dict[str, Any],
) -> Tuple[str, List[str]]:
    """Downgrade STRONG/ULTRA labels when guardrails fail."""
    if label not in {"STRONG_BUY", "ULTRA_BUY"}:
        return label, []

    notes: List[str] = []
    close_price = context.get("last_close")
    ema20_15m = context.get("ema20_15m")

    # === SSR EMA20 GATE (price must be above EMA20 on 15m for STRONG/ULTRA) ===
    if getattr(config, "ENABLE_SSR_EMA20_GATE", True):
        if close_price is not None and ema20_15m is not None:
            if close_price < ema20_15m:
                notes.append("Filter: SSR blocked because price is below EMA20 on 15m")
                downgraded = "WATCH" if score_core >= config.CORE_SCORE_WATCH_MIN else "NO_SIGNAL"
                return downgraded, notes

    # === BLOW-OFF TOP GUARD #1: PARABOLIC EXTENSION FILTER ===
    is_parabolic, parabolic_note = is_parabolic_extension(context)
    if is_parabolic:
        notes.append(parabolic_note)
        downgraded = "WATCH" if score_core >= config.CORE_SCORE_WATCH_MIN else "NO_SIGNAL"
        return downgraded, notes

    # === BLOW-OFF TOP GUARD #2: BLOW-OFF CANDLE FILTER ===
    is_blowoff, blowoff_note = is_blowoff_candle(context)
    if is_blowoff:
        notes.append(blowoff_note)
        downgraded = "WATCH" if score_core >= config.CORE_SCORE_WATCH_MIN else "NO_SIGNAL"
        return downgraded, notes

    # === BLOW-OFF TOP GUARD #3: LATE PUMP HARD STOP ===
    label = apply_late_pump_hard_stop(label, context, notes)
    if label == "WATCH":
        return label, notes

    # === EXISTING PRE-SIGNAL FILTERS ===
    ma60 = context.get("ma60")
    if close_price is not None and ma60 is not None and close_price < ma60:
        notes.append("Filter: Price below MA60 on 15m")

    macd_hist_1h = context.get("macd_hist_1h")
    if macd_hist_1h is not None and macd_hist_1h < config.MACD_1H_HIST_MIN_VALUE:
        notes.append("Filter: 1h MACD histogram below zero")

    if rsi_value is not None and rsi_value <= config.RSI_PRE_FILTER_THRESHOLD:
        notes.append(
            f"Filter: RSI {rsi_value:.1f} ≤ {config.RSI_PRE_FILTER_THRESHOLD:.0f}"
        )

    rsi_momentum_curr = context.get("rsi_momentum_curr")
    rsi_momentum_avg = context.get("rsi_momentum_avg")
    if (
        rsi_momentum_curr is not None
        and rsi_momentum_avg is not None
        and rsi_momentum_curr <= rsi_momentum_avg * config.RSI_MOMENTUM_MIN_MULTIPLIER
    ):
        notes.append("Filter: RSI momentum not exceeding 10-bar avg")

    # === REVIZYON 1: LATE SPIKE / OVEREXTENSION GUARD ===
    if getattr(config, "ENABLE_LATE_SPIKE_FILTER", False):
        pa_details = context.get("pa_details") or {}
        overextended = pa_details.get("overextended_vs_ema", False)
        dist_pct = pa_details.get("dist_from_ema_pct", 0.0)
        parabolic = pa_details.get("parabolic_runup", False)
        runup_pct = pa_details.get("runup_from_recent_low_pct", 0.0)

        if getattr(config, "ENABLE_OVEREXTENSION_FILTER", True) and overextended:
            notes.append(f"Filter: Overextended (+{dist_pct:.2f}% vs EMA20)")
        if getattr(config, "ENABLE_PARABOLIC_SPIKE_FILTER", True) and parabolic:
            notes.append(f"Filter: Parabolic runup (+{runup_pct:.2f}% from low)")

    # === REVIZYON 2.1: CANDLE DIRECTION GUARD (last 15m bar must be green) ===
    if getattr(config, "ENABLE_CANDLE_DIRECTION_FILTER", False):
        last_open = context.get("last_open_15m")
        last_close_15m = context.get("last_close_15m")
        if last_open is not None and last_close_15m is not None:
            if last_close_15m <= last_open:
                notes.append("Filter: Last 15m candle is not green (no follow-through)")

    # === REVIZYON 2.2: MOMENTUM TURNING FILTER (price + RSI + MACD histogram must turn up) ===
    if getattr(config, "ENABLE_MOMENTUM_TURNING_FILTER", False):
        closes_15m = context.get("recent_closes_15m") or []
        rsi_series = context.get("recent_rsi_15m") or []
        macd_hist_series = context.get("recent_macd_hist_1h") or []

        if len(closes_15m) >= 2 and len(rsi_series) >= 2 and len(macd_hist_series) >= 2:
            price_turns_up = closes_15m[-1] > closes_15m[-2]
            rsi_turns_up = rsi_series[-1] > rsi_series[-2]
            macd_hist_turns_up = macd_hist_series[-1] > macd_hist_series[-2]

            if not (price_turns_up and rsi_turns_up and macd_hist_turns_up):
                notes.append("Filter: Momentum not coherently turning up (price/RSI/MACD hist)")

    # === REVIZYON 2.3: LOCAL BOTTOM DETECTION ===
    if getattr(config, "ENABLE_LOCAL_BOTTOM_FILTER", False):
        closes_15m = context.get("recent_closes_15m") or []
        lookback = getattr(config, "LOCAL_BOTTOM_LOOKBACK", 8)
        if len(closes_15m) >= lookback:
            last_n = closes_15m[-lookback:]
            if len(last_n) >= 2:
                penultimate = last_n[-2]
                last_close_val = last_n[-1]
                if penultimate != min(last_n):
                    notes.append(f"Filter: No local bottom detected in last {lookback} bars")
                elif last_close_val <= penultimate:
                    notes.append("Filter: No bounce after local bottom")

    # === DOWNGRADE IF ANY FILTER TRIGGERED ===
    if not notes:
        return label, []

    downgraded = "WATCH" if score_core >= config.CORE_SCORE_WATCH_MIN else "NO_SIGNAL"
    return downgraded, notes