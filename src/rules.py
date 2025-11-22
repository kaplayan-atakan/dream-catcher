"""
Rule-based Scoring System - COMPLETE IMPLEMENTATION
NO ML/AI - Pure rule-based logic
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import config

# Phase 3: Score registries keep weights centralized so future configuration
# knobs can move into config.py without rewriting block logic.
TREND_WEIGHTS = {
    "adx_strong": 2.0,
    "adx_moderate": 1.0,
    "ema_alignment": 1.0,
    "ema_reclaim": 0.5,
    "macd_positive": 0.5,
    "macd_rising_bonus": 1.5,
    "momentum_confluence": 1.0,
    "momentum_partial": 0.5,
    "mtf_confirm": 1.0,
}

OSC_WEIGHTS = {
    "rsi_healthy": 1.0,
    "rsi_recovery": 0.5,
    "stoch_bullish": 1.0,
    "stoch_mid": 0.5,
    "cci_strong": 1.0,
    "cci_positive": 0.5,
    "stoch_rsi_high": 0.5,
    "stoch_rsi_rising": 0.5,
    "williams_bullish": 1.0,
    "williams_neutral": 0.5,
    "uo_bullish": 1.0,
}

VOLUME_WEIGHTS = {
    "obv_uptrend": 1.5,
    "volume_spike": 1.0,
    "bull_domination": 1.5,
    "bull_positive": 0.5,
    "obv_volume_confluence": 0.5,
}

PRICE_ACTION_WEIGHTS = {
    "long_lower_wick": 1.5,
    "strong_green": 1.0,
    "no_collapse": 1.0,
    "ema20_break": 1.5,
    "volume_confirm": 1.0,
    "min_volume_only": 0.5,
}

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
    total_score: int
    label: str
    reasons: List[str]
    rsi: float
    price: float = 0
    price_change_pct: float = 0
    quote_volume: float = 0
    trend_details: Optional[Dict[str, Any]] = None
    osc_details: Optional[Dict[str, Any]] = None
    vol_details: Optional[Dict[str, Any]] = None
    pa_details: Optional[Dict[str, Any]] = None
    mtf_trend_confirmed: bool = False
    htf_price_above_ema: bool = False
    fourh_price_above_ema20: bool = False
    fourh_alignment_ok: bool = False
    fourh_ema20_slope_pct: float = 0.0


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
    mtf_trend: bool,
    htf_price_above_ema: bool = False,
    fourh_price_above_ema20: bool = False,
    fourh_alignment: bool = False,
    fourh_ema20_slope_pct: float = 0.0,
) -> BlockScore:
    """
    Trend Block Scoring - REAL RULES
    Components: ADX, DI+/DI-, MACD, Momentum, AO, EMA alignment
    """
    score = 0
    reasons: List[str] = []
    price_above_ema20 = ema20 is not None and price > ema20
    ema_alignment_ok = (
        ema20 is not None and ema50 is not None and price > ema20 > ema50
    )

    details = {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "macd_hist": macd_hist,
        "macd_hist_rising": macd_hist_rising,
        "momentum": momentum,
        "ao": ao,
        "ema20": ema20,
        "ema50": ema50,
        "price_above_ema20": price_above_ema20,
        "ema_alignment_ok": ema_alignment_ok,
        "mtf_trend_confirmed": mtf_trend,
        "htf_price_above_ema": htf_price_above_ema,
        "fourh_price_above_ema20": fourh_price_above_ema20,
        "fourh_ema_alignment_ok": fourh_alignment,
        "fourh_ema20_slope_pct": fourh_ema20_slope_pct,
    }

    # Rule 1: ADX Strong Trend with Bullish DI
    if adx >= config.ADX_STRONG_TREND and plus_di > minus_di:
        score += TREND_WEIGHTS["adx_strong"]
        reasons.append(f"Trend: ADX={adx:.1f} strong, DI+>DI-")
    elif adx >= config.ADX_STRONG_TREND * 0.7 and plus_di > minus_di:
        score += TREND_WEIGHTS["adx_moderate"]
        reasons.append(f"Trend: ADX={adx:.1f} moderate, DI+>DI-")

    # Rule 2: EMA Alignment (Price > EMA20 > EMA50)
    if ema_alignment_ok:
        score += TREND_WEIGHTS["ema_alignment"]
        reasons.append("Trend: Price>EMA20>EMA50 alignment")
    elif price_above_ema20:
        score += TREND_WEIGHTS["ema_reclaim"]
        reasons.append("Trend: Price reclaimed EMA20 support")

    # Rule 3: MACD Histogram
    if macd_hist > 0:
        if macd_hist_rising:
            score += TREND_WEIGHTS["macd_rising_bonus"]
            reasons.append("Trend: MACD histogram positive & rising")
        else:
            score += TREND_WEIGHTS["macd_positive"]
            reasons.append("Trend: MACD histogram positive")

    # Rule 4: Momentum & AO Confluence
    if momentum > 0 and ao > 0:
        score += TREND_WEIGHTS["momentum_confluence"]
        reasons.append(f"Trend: Momentum {momentum:.2f} & AO {ao:.4f} both bullish")
    elif momentum > 0 or ao > 0:
        score += TREND_WEIGHTS["momentum_partial"]
        reasons.append("Trend: One of Momentum/AO turning positive")

    # Rule 5: Multi-timeframe Confirmation
    if mtf_trend:
        score += TREND_WEIGHTS["mtf_confirm"]
        reasons.append("Trend: 1h EMA stack confirms uptrend")
    elif htf_price_above_ema:
        reasons.append("Trend: 1h price above EMA20 (watching for alignment)")

    # Rule 6: 4h context (informational bonus only for Phase 3 transparency)
    if fourh_alignment:
        reasons.append("Trend: 4h EMA stack aligned with bulls")
    elif fourh_price_above_ema20:
        reasons.append("Trend: 4h price reclaimed EMA20 support")

    return BlockScore(int(score), reasons[:4], details)


def compute_osc_block(rsi_val: float, stoch_k: float, cci: float,
                      stoch_rsi: float, williams_r: float, uo: float) -> BlockScore:
    """
    Oscillator Block Scoring - REAL RULES
    Components: RSI, Stoch K, CCI, Stoch RSI, Williams %R, UO
    """
    score = 0
    reasons: List[str] = []
    details = {
        "rsi": rsi_val,
        "stoch_k": stoch_k,
        "cci": cci,
        "stoch_rsi": stoch_rsi,
        "williams_r": williams_r,
        "uo": uo,
    }

    # Rule 1: RSI in Healthy Zone
    rsi_healthy = config.RSI_HEALTHY_MIN <= rsi_val <= config.RSI_HEALTHY_MAX
    details["rsi_healthy"] = rsi_healthy
    if rsi_healthy:
        score += OSC_WEIGHTS["rsi_healthy"]
        reasons.append(
            f"Osc: RSI={rsi_val:.1f} in 45-65 healthy range (sustainable trend)"
        )
    elif 30 <= rsi_val < config.RSI_HEALTHY_MIN:
        score += OSC_WEIGHTS["rsi_recovery"]
        reasons.append(f"Osc: RSI={rsi_val:.1f} emerging from oversold zone")

    # Rule 2: Stochastic K
    stoch_bullish = config.STOCH_OVERSOLD < stoch_k < config.STOCH_OVERBOUGHT and stoch_k >= 50
    details["stoch_bullish"] = stoch_bullish
    if stoch_bullish:
        score += OSC_WEIGHTS["stoch_bullish"]
        reasons.append(f"Osc: StochK={stoch_k:.1f} above 50 (bullish range)")
    elif config.STOCH_OVERSOLD < stoch_k < 50:
        score += OSC_WEIGHTS["stoch_mid"]
        reasons.append(f"Osc: StochK={stoch_k:.1f} mid-range consolidation")
    elif stoch_k <= config.STOCH_OVERSOLD:
        reasons.append(f"Osc: StochK={stoch_k:.1f} oversold, watching for turn")

    # Rule 3: CCI
    cci_strong = cci > 100
    details["cci_strong_bull"] = cci_strong
    if cci_strong:
        score += OSC_WEIGHTS["cci_strong"]
        reasons.append(f"Osc: CCI={cci:.1f} strong bullish momentum")
    elif cci > 0:
        score += OSC_WEIGHTS["cci_positive"]
        reasons.append(f"Osc: CCI={cci:.1f} above zero")

    # Rule 4: Stochastic RSI
    stoch_rsi_rising = stoch_rsi > 20
    details["stoch_rsi_above_lower"] = stoch_rsi_rising
    if stoch_rsi > 80:
        score += OSC_WEIGHTS["stoch_rsi_high"]
        reasons.append(f"Osc: StochRSI={stoch_rsi:.1f} staying strong above 80")
    elif stoch_rsi_rising:
        score += OSC_WEIGHTS["stoch_rsi_rising"]
        reasons.append(f"Osc: StochRSI={stoch_rsi:.1f} rising from lower band")

    # Rule 5: Williams %R
    williams_bullish = williams_r > config.WILLIAMS_BULLISH
    details["williams_bullish"] = williams_bullish
    if williams_bullish:
        score += OSC_WEIGHTS["williams_bullish"]
        reasons.append(f"Osc: Williams%R={williams_r:.1f} bullish accumulation")
    elif williams_r > -80:
        score += OSC_WEIGHTS["williams_neutral"]
        reasons.append(f"Osc: Williams%R={williams_r:.1f} neutral range")

    # Rule 6: Ultimate Oscillator
    if uo > config.UO_BULLISH:
        score += OSC_WEIGHTS["uo_bullish"]
        reasons.append(f"Osc: UO={uo:.1f} bullish drive")

    return BlockScore(int(score), reasons[:4], details)


def compute_volume_block(
    obv_trend: bool,
    bull_power: float,
    bear_power: float,
    volume_spike: bool,
    volume_spike_factor: Optional[float] = None,
    obv_change_pct: float = 0.0,
) -> BlockScore:
    """
    Volume/Power Block Scoring - REAL RULES
    Components: OBV trend, Bull/Bear Power, Volume Spike
    """
    score = 0
    reasons: List[str] = []
    details = {
        "obv_trend": obv_trend,
        "obv_change_pct": obv_change_pct,
        "bull_power": bull_power,
        "bear_power": bear_power,
        "volume_spike": volume_spike,
        "volume_spike_factor": volume_spike_factor,
    }

    # Rule 1: OBV Uptrend
    if obv_trend:
        score += VOLUME_WEIGHTS["obv_uptrend"]
        change_txt = (
            f" ({obv_change_pct:+.1f}% over {config.OBV_TREND_LOOKBACK} bars)"
            if obv_change_pct
            else ""
        )
        reasons.append(
            f"Vol: OBV in uptrend over last {config.OBV_TREND_LOOKBACK} bars" + change_txt
        )

    # Rule 2: Volume Spike
    if volume_spike:
        score += VOLUME_WEIGHTS["volume_spike"]
        factor_txt = f" {volume_spike_factor:.1f}x" if volume_spike_factor else ""
        reasons.append(
            f"Vol: Volume spike{factor_txt} above {config.VOLUME_LOOKBACK}-bar average"
        )

    # Rule 3: Bull/Bear Power Analysis
    if bull_power > 0 and bear_power < 0:
        score += VOLUME_WEIGHTS["bull_domination"]
        reasons.append(
            f"Vol: Bulls dominating (Bull {bull_power:.4f} / Bear {bear_power:.4f})"
        )
    elif bull_power > 0:
        score += VOLUME_WEIGHTS["bull_positive"]
        reasons.append(f"Vol: Bull power positive ({bull_power:.4f})")

    # Rule 4: Combined Volume Strength
    if obv_trend and volume_spike:
        score += VOLUME_WEIGHTS["obv_volume_confluence"]  # Bonus for confluence
        reasons.append("Vol: OBV + volume spike confluence")

    return BlockScore(int(score), reasons[:4], details)


def compute_price_action_block(pa_signals: dict) -> BlockScore:
    """
    Price Action Block Scoring - REAL RULES
    Components: Candle patterns, volume confirmation, no collapse, EMA break
    """
    score = 0
    reasons: List[str] = []
    details = dict(pa_signals.get("details") or {})
    volume_spike_factor = pa_signals.get("volume_spike_factor")

    # Rule 1: Long Lower Wick (Hammer pattern)
    if pa_signals.get('long_lower_wick', False):
        score += PRICE_ACTION_WEIGHTS["long_lower_wick"]
        wick_pct = details.get("lower_wick_pct")
        if wick_pct is not None:
            reasons.append(
                f"PA: Hammer-type candle (lower wick {wick_pct:.1f}% of range)"
            )
        else:
            reasons.append("PA: Hammer-type candle (long lower wick)")
    
    # Rule 2: Strong Green Candle
    if pa_signals.get('strong_green', False):
        score += PRICE_ACTION_WEIGHTS["strong_green"]
        body_pct = details.get("body_pct_of_range")
        if body_pct is not None:
            reasons.append(
                f"PA: Strong green body ({body_pct:.1f}% of range) closing near highs"
            )
        else:
            reasons.append("PA: Strong green candle")
    
    # Rule 3: No Recent Collapse
    if pa_signals.get('no_collapse', False):
        score += PRICE_ACTION_WEIGHTS["no_collapse"]
        drop_pct = details.get("max_drop_pct")
        if drop_pct is not None:
            reasons.append(
                f"PA: No sharp dump in last {config.COLLAPSE_LOOKBACK_BARS} candles (max drop {drop_pct:.1f}%)"
            )
        else:
            reasons.append("PA: No major dumps recently")
    
    # Rule 4: EMA20 Breakout
    if pa_signals.get('ema20_break', False):
        score += PRICE_ACTION_WEIGHTS["ema20_break"]
        reasons.append("PA: EMA20 breakout after closing below prior bar")
    
    # Rule 5: Volume Confirmation
    if pa_signals.get('volume_spike', False) and pa_signals.get('min_volume', False):
        score += PRICE_ACTION_WEIGHTS["volume_confirm"]
        vol_txt = (
            f"{volume_spike_factor:.1f}x spike"
            if volume_spike_factor is not None
            else "volume spike"
        )
        min_mult = details.get("min_volume_multiple")
        if min_mult is not None and min_mult > 0:
            reasons.append(
                f"PA: {vol_txt} with liquidity {min_mult:.1f}x above minimum"
            )
        else:
            reasons.append("PA: Volume confirms price action")
    elif pa_signals.get('min_volume', False):
        score += PRICE_ACTION_WEIGHTS["min_volume_only"]
        reasons.append("PA: Minimum volume met")

    return BlockScore(int(score), reasons[:4], details)


def decide_signal_label(trend_block: BlockScore, osc_block: BlockScore,
                       vol_block: BlockScore, pa_block: BlockScore,
                       rsi_value: float, htf_trend_ok: bool,
                       symbol: str) -> SignalResult:
    """
    Final Signal Decision Logic
    Returns SignalResult with label: NONE, STRONG_BUY, or ULTRA_BUY
    """
    trend_component = trend_block.score if config.ENABLE_TREND_BLOCK else 0
    osc_component = osc_block.score if config.ENABLE_OSC_BLOCK else 0
    vol_component = vol_block.score if config.ENABLE_VOLUME_BLOCK else 0
    pa_component = pa_block.score if config.ENABLE_PRICE_ACTION_BLOCK else 0

    total_score = trend_component + osc_component + vol_component + pa_component
    vol_pa_combined = vol_component + pa_component
    
    # Combine all reasons but respect block toggles for transparency
    all_reasons: List[str] = []
    if config.ENABLE_TREND_BLOCK:
        all_reasons.extend(trend_block.reasons)
    if config.ENABLE_OSC_BLOCK:
        all_reasons.extend(osc_block.reasons)
    if config.ENABLE_VOLUME_BLOCK:
        all_reasons.extend(vol_block.reasons)
    if config.ENABLE_PRICE_ACTION_BLOCK:
        all_reasons.extend(pa_block.reasons)
    
    # Default to no signal
    label = "NONE"
    
    # ULTRA_BUY Conditions (Strictest)
    if (total_score >= config.ULTRA_BUY_SCORE and
        trend_component >= config.ULTRA_BUY_MIN_TREND and
        osc_component >= config.ULTRA_BUY_MIN_OSC and
        vol_pa_combined >= config.ULTRA_BUY_MIN_VOL_PA and
        rsi_value <= config.ULTRA_BUY_MAX_RSI and
        htf_trend_ok):  # Multi-timeframe confirmation required
        label = "ULTRA_BUY"
    
    # STRONG_BUY Conditions
    elif (total_score >= config.STRONG_BUY_SCORE and
          trend_component >= config.STRONG_BUY_MIN_TREND and
          osc_component >= config.STRONG_BUY_MIN_OSC and
          vol_pa_combined >= config.STRONG_BUY_MIN_VOL_PA):
        label = "STRONG_BUY"
    
    return SignalResult(
        symbol=symbol,
        trend_score=trend_component,
        osc_score=osc_component,
        vol_score=vol_component,
        pa_score=pa_component,
        total_score=total_score,
        label=label,
        reasons=all_reasons[:5],  # Top 5 reasons
        rsi=rsi_value,
        trend_details=trend_block.details,
        osc_details=osc_block.details,
        vol_details=vol_block.details,
        pa_details=pa_block.details,
        mtf_trend_confirmed=bool(
            (trend_block.details or {}).get("mtf_trend_confirmed", False)
        ),
        htf_price_above_ema=htf_trend_ok,
        fourh_price_above_ema20=bool(
            (trend_block.details or {}).get("fourh_price_above_ema20", False)
        ),
        fourh_alignment_ok=bool(
            (trend_block.details or {}).get("fourh_ema_alignment_ok", False)
        ),
        fourh_ema20_slope_pct=float(
            (trend_block.details or {}).get("fourh_ema20_slope_pct", 0.0) or 0.0
        ),
    )