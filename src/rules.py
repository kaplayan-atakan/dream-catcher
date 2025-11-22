"""
Rule-based Scoring System - COMPLETE IMPLEMENTATION
NO ML/AI - Pure rule-based logic
"""
from dataclasses import dataclass, asdict
from typing import List, Optional
import config

@dataclass
class BlockScore:
    score: int
    reasons: List[str]

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


def compute_trend_block(price: float, ema20: Optional[float], ema50: Optional[float],
                        adx: float, plus_di: float, minus_di: float,
                        macd_hist: float, macd_hist_rising: bool,
                        momentum: float, ao: float, mtf_trend: bool) -> BlockScore:
    """
    Trend Block Scoring - REAL RULES
    Components: ADX, DI+/DI-, MACD, Momentum, AO, EMA alignment
    """
    score = 0
    reasons = []
    
    # Rule 1: ADX Strong Trend with Bullish DI
    if adx >= config.ADX_STRONG_TREND and plus_di > minus_di:
        score += 2
        reasons.append(f"Trend: ADX={adx:.1f} strong, DI+>DI-")
    elif adx >= config.ADX_STRONG_TREND * 0.7 and plus_di > minus_di:
        score += 1
        reasons.append(f"Trend: ADX={adx:.1f} moderate, DI+>DI-")
    
    # Rule 2: EMA Alignment (Price > EMA20 > EMA50)
    if ema20 and ema50:
        if price > ema20 > ema50:
            score += 1
            reasons.append("Trend: Price>EMA20>EMA50 alignment")
        elif price > ema20:
            score += 0.5
            reasons.append("Trend: Price>EMA20")
    
    # Rule 3: MACD Histogram
    if macd_hist > 0:
        if macd_hist_rising:
            score += 1.5
            reasons.append("Trend: MACD hist positive & rising")
        else:
            score += 0.5
            reasons.append("Trend: MACD hist positive")
    
    # Rule 4: Momentum & AO Confluence
    if momentum > 0 and ao > 0:
        score += 1
        reasons.append(f"Trend: Momentum({momentum:.2f}) & AO({ao:.4f}) bullish")
    elif momentum > 0 or ao > 0:
        score += 0.5
        reasons.append("Trend: Partial momentum bullish")
    
    # Rule 5: Multi-timeframe Confirmation
    if mtf_trend:
        score += 1
        reasons.append("Trend: HTF (1h) trend confirms")
    
    return BlockScore(int(score), reasons[:3])  # Top 3 reasons


def compute_osc_block(rsi_val: float, stoch_k: float, cci: float,
                      stoch_rsi: float, williams_r: float, uo: float) -> BlockScore:
    """
    Oscillator Block Scoring - REAL RULES
    Components: RSI, Stoch K, CCI, Stoch RSI, Williams %R, UO
    """
    score = 0
    reasons = []
    
    # Rule 1: RSI in Healthy Zone
    if config.RSI_HEALTHY_MIN <= rsi_val <= config.RSI_HEALTHY_MAX:
        score += 1
        reasons.append(f"Osc: RSI={rsi_val:.1f} in healthy zone")
    elif 30 <= rsi_val < config.RSI_HEALTHY_MIN:
        score += 0.5
        reasons.append(f"Osc: RSI={rsi_val:.1f} oversold bounce")
    
    # Rule 2: Stochastic K
    if config.STOCH_OVERSOLD < stoch_k < config.STOCH_OVERBOUGHT:
        if stoch_k > 50:
            score += 1
            reasons.append(f"Osc: StochK={stoch_k:.1f} bullish")
        else:
            score += 0.5
            reasons.append(f"Osc: StochK={stoch_k:.1f} neutral")
    elif stoch_k <= config.STOCH_OVERSOLD and stoch_k > stoch_k - 5:  # Turning up from oversold
        score += 1
        reasons.append(f"Osc: StochK={stoch_k:.1f} oversold reversal")
    
    # Rule 3: CCI
    if cci > 100:
        score += 1
        reasons.append(f"Osc: CCI={cci:.1f} strong bullish")
    elif cci > 0:
        score += 0.5
        reasons.append(f"Osc: CCI={cci:.1f} positive")
    
    # Rule 4: Stochastic RSI
    if stoch_rsi > 80:
        score += 0.5
        reasons.append(f"Osc: StochRSI={stoch_rsi:.1f} strong")
    elif 20 < stoch_rsi <= 80:
        score += 0.5
        reasons.append(f"Osc: StochRSI={stoch_rsi:.1f} rising")
    
    # Rule 5: Williams %R
    if williams_r > config.WILLIAMS_BULLISH:
        score += 1
        reasons.append(f"Osc: Williams%R={williams_r:.1f} bullish")
    elif williams_r > -80:
        score += 0.5
        reasons.append(f"Osc: Williams%R={williams_r:.1f} neutral")
    
    # Rule 6: Ultimate Oscillator
    if uo > config.UO_BULLISH:
        score += 1
        reasons.append(f"Osc: UO={uo:.1f} bullish")
    
    return BlockScore(int(score), reasons[:3])


def compute_volume_block(obv_trend: bool, bull_power: float, bear_power: float,
                         volume_spike: bool) -> BlockScore:
    """
    Volume/Power Block Scoring - REAL RULES
    Components: OBV trend, Bull/Bear Power, Volume Spike
    """
    score = 0
    reasons = []
    
    # Rule 1: OBV Uptrend
    if obv_trend:
        score += 1.5
        reasons.append("Vol: OBV in uptrend")
    
    # Rule 2: Volume Spike
    if volume_spike:
        score += 1
        reasons.append("Vol: Volume spike detected")
    
    # Rule 3: Bull/Bear Power Analysis
    if bull_power > 0 and bear_power < 0:
        score += 1.5
        reasons.append(f"Vol: Bulls dominating (B:{bull_power:.4f}/B:{bear_power:.4f})")
    elif bull_power > 0:
        score += 0.5
        reasons.append(f"Vol: Bull power positive ({bull_power:.4f})")
    
    # Rule 4: Combined Volume Strength
    if obv_trend and volume_spike:
        score += 0.5  # Bonus for confluence
        reasons.append("Vol: OBV+Volume spike confluence")
    
    return BlockScore(int(score), reasons[:3])


def compute_price_action_block(pa_signals: dict) -> BlockScore:
    """
    Price Action Block Scoring - REAL RULES
    Components: Candle patterns, volume confirmation, no collapse, EMA break
    """
    score = 0
    reasons = []
    
    # Rule 1: Long Lower Wick (Hammer pattern)
    if pa_signals.get('long_lower_wick', False):
        score += 1.5
        reasons.append("PA: Long lower wick (hammer)")
    
    # Rule 2: Strong Green Candle
    if pa_signals.get('strong_green', False):
        score += 1
        reasons.append("PA: Strong green candle")
    
    # Rule 3: No Recent Collapse
    if pa_signals.get('no_collapse', False):
        score += 1
        reasons.append("PA: No major dumps recently")
    
    # Rule 4: EMA20 Breakout
    if pa_signals.get('ema20_break', False):
        score += 1.5
        reasons.append("PA: Price broke above EMA20")
    
    # Rule 5: Volume Confirmation
    if pa_signals.get('volume_spike', False) and pa_signals.get('min_volume', False):
        score += 1
        reasons.append("PA: Volume confirms price action")
    elif pa_signals.get('min_volume', False):
        score += 0.5
        reasons.append("PA: Minimum volume met")
    
    return BlockScore(int(score), reasons[:3])


def decide_signal_label(trend_block: BlockScore, osc_block: BlockScore,
                       vol_block: BlockScore, pa_block: BlockScore,
                       rsi_value: float, htf_trend_ok: bool,
                       symbol: str) -> SignalResult:
    """
    Final Signal Decision Logic
    Returns SignalResult with label: NONE, STRONG_BUY, or ULTRA_BUY
    """
    total_score = trend_block.score + osc_block.score + vol_block.score + pa_block.score
    vol_pa_combined = vol_block.score + pa_block.score
    
    # Combine all reasons
    all_reasons = trend_block.reasons + osc_block.reasons + vol_block.reasons + pa_block.reasons
    
    # Default to no signal
    label = "NONE"
    
    # ULTRA_BUY Conditions (Strictest)
    if (total_score >= config.ULTRA_BUY_SCORE and
        trend_block.score >= config.ULTRA_BUY_MIN_TREND and
        osc_block.score >= config.ULTRA_BUY_MIN_OSC and
        vol_pa_combined >= config.ULTRA_BUY_MIN_VOL_PA and
        rsi_value <= config.ULTRA_BUY_MAX_RSI and
        htf_trend_ok):  # Multi-timeframe confirmation required
        label = "ULTRA_BUY"
    
    # STRONG_BUY Conditions
    elif (total_score >= config.STRONG_BUY_SCORE and
          trend_block.score >= config.STRONG_BUY_MIN_TREND and
          osc_block.score >= config.STRONG_BUY_MIN_OSC and
          vol_pa_combined >= config.STRONG_BUY_MIN_VOL_PA):
        label = "STRONG_BUY"
    
    return SignalResult(
        symbol=symbol,
        trend_score=trend_block.score,
        osc_score=osc_block.score,
        vol_score=vol_block.score,
        pa_score=pa_block.score,
        total_score=total_score,
        label=label,
        reasons=all_reasons[:5],  # Top 5 reasons
        rsi=rsi_value
    )