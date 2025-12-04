"""
Symbol Analyzer Module - REAL INDICATORS & CALCULATIONS
"""
import logging
from typing import Optional

import numpy as np

import indicators
import price_action
import momentum_shift
import rules
import data_fetcher
import config

logger = logging.getLogger(__name__)


async def analyze_symbol(session, symbol_data: dict) -> Optional[dict]:
    """
    Analyze a single symbol with REAL indicators
    Includes prefilters and multi-timeframe analysis
    """
    try:
        symbol = symbol_data['symbol']
        
        # PREFILTER CHECK (24h data)
        if symbol_data['quote_volume'] < config.MIN_24H_QUOTE_VOLUME:
            return None
        if symbol_data['price'] < config.MIN_PRICE_USDT:
            return None
        if not (config.MIN_24H_CHANGE <= symbol_data['price_change_pct'] <= config.MAX_24H_CHANGE):
            return None
        
        # Fetch multi-timeframe klines
        klines_data = await data_fetcher.fetch_multi_timeframe_klines(session, symbol)
        
        if not klines_data or config.MAIN_TIMEFRAME not in klines_data:
            logger.debug(f"No kline data for {symbol}")
            return None
        
        # Extract main timeframe data
        klines = klines_data[config.MAIN_TIMEFRAME]
        if len(klines) < 200:  # Need enough data for all indicators
            logger.debug(f"Insufficient kline data for {symbol}: {len(klines)} bars")
            return None
        
        # Extract OHLCV arrays
        opens = [k['open'] for k in klines]
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        closes = [k['close'] for k in klines]
        volumes = [k['volume'] for k in klines]
        
        # ============ CALCULATE ALL REAL INDICATORS ============
        
        # Moving Averages
        ema20_values = indicators.ema(closes, config.EMA_FAST)
        ema50_values = indicators.ema(closes, config.EMA_SLOW)
        ema60_values = indicators.ema(closes, config.MA60_PERIOD)
        
        # Trend Indicators
        adx_values, plus_di_values, minus_di_values = indicators.adx(
            highs, lows, closes, config.ADX_PERIOD
        )
        macd_line, signal_line, macd_hist = indicators.macd(closes)
        momentum_values = indicators.momentum(closes, config.MOMENTUM_PERIOD)
        ao_values = indicators.awesome_oscillator(highs, lows)
        
        # Oscillators
        rsi_values = indicators.rsi(closes, config.RSI_PERIOD)
        stoch_k_values = indicators.stochastic_k(highs, lows, closes, config.STOCH_K_PERIOD)
        cci_values = indicators.cci(highs, lows, closes, config.CCI_PERIOD)
        stoch_rsi_values = indicators.stochastic_rsi(closes)
        williams_r_values = indicators.williams_r(highs, lows, closes, config.WILLIAMS_PERIOD)
        uo_values = indicators.ultimate_oscillator(highs, lows, closes, config.UO_PERIODS)
        
        # Volume Indicators
        obv_values = indicators.obv(closes, volumes)
        bull_power_values, bear_power_values = indicators.bull_bear_power(highs, lows, closes)
        
        # Get LATEST VALUES (last bar)
        last_close = closes[-1]
        last_ema20 = ema20_values[-1] if ema20_values[-1] is not np.nan else None
        last_ema50 = ema50_values[-1] if ema50_values[-1] is not np.nan else None
        last_ma60 = ema60_values[-1] if ema60_values[-1] is not np.nan else None
        
        # Trend indicators - latest
        last_adx = adx_values[-1] if adx_values[-1] is not np.nan else 0
        last_plus_di = plus_di_values[-1] if plus_di_values[-1] is not np.nan else 0
        last_minus_di = minus_di_values[-1] if minus_di_values[-1] is not np.nan else 0
        last_macd_hist = macd_hist[-1] if macd_hist[-1] is not np.nan else 0
        last_momentum = momentum_values[-1] if momentum_values[-1] is not np.nan else 0
        last_ao = ao_values[-1] if ao_values[-1] is not np.nan else 0
        
        # Check MACD histogram rising (last 3 bars)
        macd_hist_rising = False
        if len(macd_hist) >= config.MACD_HIST_RISING_BARS:
            valid_hist = [h for h in macd_hist[-config.MACD_HIST_RISING_BARS:] if h is not np.nan]
            if len(valid_hist) == config.MACD_HIST_RISING_BARS:
                macd_hist_rising = all(valid_hist[i] < valid_hist[i+1] for i in range(len(valid_hist)-1))
        
        # Oscillators - latest
        last_rsi = rsi_values[-1] if rsi_values[-1] is not np.nan else 50
        last_stoch_k = stoch_k_values[-1] if stoch_k_values[-1] is not np.nan else 50
        last_cci = cci_values[-1] if cci_values[-1] is not np.nan else 0
        last_stoch_rsi = stoch_rsi_values[-1] if stoch_rsi_values[-1] is not np.nan else 50
        last_williams_r = williams_r_values[-1] if williams_r_values[-1] is not np.nan else -50
        last_uo = uo_values[-1] if uo_values[-1] is not np.nan else 50
        last_stoch_rsi_prev = None
        if len(stoch_rsi_values) >= 2 and stoch_rsi_values[-2] is not np.nan:
            last_stoch_rsi_prev = stoch_rsi_values[-2]
        last_uo_prev = None
        if len(uo_values) >= 2 and uo_values[-2] is not np.nan:
            last_uo_prev = uo_values[-2]

        rsi_momentum_values = [np.nan] * len(rsi_values)
        for idx in range(1, len(rsi_values)):
            curr = rsi_values[idx]
            prev = rsi_values[idx - 1]
            if np.isnan(curr) or np.isnan(prev):
                continue
            rsi_momentum_values[idx] = curr - prev

        rsi_momentum_current = None
        if rsi_momentum_values and not np.isnan(rsi_momentum_values[-1]):
            rsi_momentum_current = rsi_momentum_values[-1]

        rsi_momentum_avg = None
        lookback = config.RSI_MOMENTUM_LOOKBACK
        if lookback > 0 and len(rsi_momentum_values) > lookback:
            window = [
                val
                for val in rsi_momentum_values[-(lookback + 1) : -1]
                if not np.isnan(val)
            ]
            if window:
                rsi_momentum_avg = float(np.mean(window))
        
        # Volume indicators - latest
        obv_change_pct = indicators.obv_change_percent(obv_values, config.OBV_TREND_LOOKBACK)
        last_bull_power = bull_power_values[-1] if bull_power_values[-1] is not np.nan else 0
        last_bear_power = bear_power_values[-1] if bear_power_values[-1] is not np.nan else 0
        
        # ============ MULTI-TIMEFRAME ANALYSIS ============
        htf_context = {
            "close_above_ema20": False,
            "ema20_slope_pct": 0.0,
            "macd_hist": 0.0,
        }
        ht4_price_above_ema20 = False
        ht4_alignment = False
        ht4_ema20_slope_pct = 0.0
        
        if '1h' in klines_data and len(klines_data['1h']) >= 30:
            htf_closes = [k['close'] for k in klines_data['1h']]
            htf_ema20 = indicators.ema(htf_closes, config.EMA_FAST)
            if htf_ema20 and htf_ema20[-1] is not np.nan:
                latest_ema20 = htf_ema20[-1]
                htf_context["close_above_ema20"] = htf_closes[-1] > latest_ema20
                slope_lookback = config.HTF_EMA_SLOPE_LOOKBACK
                if len(htf_ema20) > slope_lookback and htf_ema20[-(slope_lookback + 1)] is not np.nan:
                    past_ema = htf_ema20[-(slope_lookback + 1)]
                    if past_ema:
                        htf_context["ema20_slope_pct"] = ((latest_ema20 / past_ema) - 1) * 100
            htf_macd_line, _, htf_macd_hist = indicators.macd(htf_closes)
            if htf_macd_hist and htf_macd_hist[-1] is not np.nan:
                htf_context["macd_hist"] = htf_macd_hist[-1]
            if htf_macd_line and htf_macd_line[-1] is not np.nan:
                htf_context["macd_line"] = htf_macd_line[-1]
        
        fourh_context = {}
        if '4h' in klines_data and len(klines_data['4h']) >= 50:
            h4_closes = [k['close'] for k in klines_data['4h']]
            h4_ema20 = indicators.ema(h4_closes, 20)
            h4_ema50 = indicators.ema(h4_closes, 50)
            latest_ema20 = h4_ema20[-1]
            latest_ema50 = h4_ema50[-1]
            latest_close_h4 = h4_closes[-1]

            if latest_ema20 is not np.nan:
                ht4_price_above_ema20 = latest_close_h4 > latest_ema20

            if latest_ema20 is not np.nan and latest_ema50 is not np.nan:
                ht4_alignment = latest_close_h4 > latest_ema20 > latest_ema50

            slope_lookback = 5
            if len(h4_ema20) > slope_lookback and latest_ema20 is not np.nan:
                past_ema = h4_ema20[-(slope_lookback + 1)]
                if past_ema is not np.nan and past_ema != 0:
                    ht4_ema20_slope_pct = ((latest_ema20 / past_ema) - 1) * 100

            fourh_context = {
                "fourh_last_close": latest_close_h4,
                "fourh_ema20": latest_ema20 if latest_ema20 is not np.nan else None,
                "fourh_ema50": latest_ema50 if latest_ema50 is not np.nan else None,
                "fourh_price_above_ema20": ht4_price_above_ema20,
                "fourh_ema_alignment_ok": ht4_alignment,
                "fourh_ema20_slope_pct": ht4_ema20_slope_pct,
            }

        # ============ PRICE ACTION ANALYSIS ============
        pa_signals = price_action.analyze_price_action(
            opens, highs, lows, closes, volumes, ema20_values
        )
        volume_spike_factor = pa_signals.get('volume_spike_factor')
        
        # ============ COMPUTE BLOCK SCORES ============
        
        # TREND BLOCK
        trend_block = rules.compute_trend_block(
            price=last_close,
            ema20=last_ema20,
            ema50=last_ema50,
            adx=last_adx,
            plus_di=last_plus_di,
            minus_di=last_minus_di,
            macd_hist=last_macd_hist,
            macd_hist_rising=macd_hist_rising,
            momentum=last_momentum,
            ao=last_ao,
        )
        if trend_block.details is None:
            trend_block.details = {}
        trend_block.details.update({
            "fourh_price_above_ema20": ht4_price_above_ema20,
            "fourh_ema_alignment_ok": ht4_alignment,
            "fourh_ema20_slope_pct": ht4_ema20_slope_pct,
        })
        if fourh_context:
            trend_block.details.update(fourh_context)
        
        # OSCILLATOR BLOCK
        osc_block = rules.compute_osc_block(
            rsi_val=last_rsi,
            stoch_k=last_stoch_k,
            cci=last_cci,
            stoch_rsi=last_stoch_rsi,
            williams_r=last_williams_r,
            uo=last_uo,
            stoch_rsi_prev=last_stoch_rsi_prev,
            uo_prev=last_uo_prev,
        )
        
        # VOLUME BLOCK
        vol_block = rules.compute_volume_block(
            bull_power=last_bull_power,
            bear_power=last_bear_power,
            volume_spike_factor=volume_spike_factor,
            obv_change_pct=obv_change_pct,
        )
        
        # PRICE ACTION BLOCK
        pa_block = rules.compute_price_action_block(pa_signals)

        # HTF BONUS BLOCK
        htf_block = rules.compute_htf_bonus(htf_context)
        
        # ============ SIGNAL DECISION ============
        meta = {
            "price": symbol_data['price'],
            "price_change_pct": symbol_data['price_change_pct'],
            "quote_volume": symbol_data['quote_volume'],
        }

        # Build recent series for Revizyon 2 filters
        recent_closes_15m = closes[-12:] if len(closes) >= 12 else list(closes)
        recent_rsi_15m = [
            r for r in rsi_values[-3:]
            if r is not None and not np.isnan(r)
        ] if len(rsi_values) >= 3 else []

        # 1h MACD histogram series (last 3 bars if available)
        recent_macd_hist_1h = []
        if '1h' in klines_data and len(klines_data['1h']) >= 30:
            htf_closes_for_hist = [k['close'] for k in klines_data['1h']]
            _, _, htf_hist_series = indicators.macd(htf_closes_for_hist)
            if htf_hist_series:
                recent_macd_hist_1h = [
                    h for h in htf_hist_series[-3:]
                    if h is not None and not np.isnan(h)
                ]

        # === BOTTOM-FISHING: Momentum shift detection ===
        momentum_rev = {}
        if getattr(config, "ENABLE_MOMENTUM_SHIFT_DETECTION", False):
            momentum_rev = momentum_shift.detect_momentum_shift(
                rsi_values=rsi_values,
                stoch_k_values=stoch_k_values,
                macd_hist=macd_hist,
                current_rsi=last_rsi,
                price_change_pct=symbol_data['price_change_pct'],
            )
            logger.debug(
                "[%s] Momentum reversal confidence: %s",
                symbol,
                momentum_rev.get("confidence_score"),
            )

        # === EARLY MOMENTUM DETECTION (V6) ===
        early_momentum = {}
        if getattr(config, "ENABLE_EARLY_MOMENTUM_DETECTION", False):
            # Calculate Stochastic D for early momentum detection
            stoch_d_values = indicators.stochastic_d(stoch_k_values, period=3)
            
            early_momentum = indicators.detect_early_momentum_shift(
                rsi_values=rsi_values[-10:] if len(rsi_values) >= 10 else rsi_values,
                macd_hist_values=macd_hist[-10:] if len(macd_hist) >= 10 else macd_hist,
                stoch_k_values=stoch_k_values[-10:] if len(stoch_k_values) >= 10 else stoch_k_values,
                stoch_d_values=stoch_d_values[-10:] if len(stoch_d_values) >= 10 else stoch_d_values,
            )
            if early_momentum.get("detected"):
                logger.debug(
                    "[%s] Early momentum shift detected (confidence: %s/3)",
                    symbol,
                    early_momentum.get("confidence"),
                )

        # === BREAKOUT DETECTION (V6) ===
        breakout = {}
        if getattr(config, "ENABLE_BREAKOUT_DETECTION", False):
            breakout = indicators.detect_breakout(
                closes=closes,
                highs=highs,
                volumes=volumes,
                ema20=last_ema20 if last_ema20 is not None else 0,
                lookback=getattr(config, "BREAKOUT_LOOKBACK_BARS", 20),
            )
            if breakout.get("detected"):
                logger.debug(
                    "[%s] Breakout detected (+%.1f%% above resistance, vol %.1fx)",
                    symbol,
                    breakout.get("breakout_pct", 0),
                    breakout.get("volume_ratio", 0),
                )

        # === BOTTOM-FISHING: Reversal candles (already in pa_signals) ===
        reversal_pa = price_action.detect_reversal_candles(opens, highs, lows, closes)

        # === BOTTOM-FISHING: Support level checks ===
        support_data = price_action.check_support_levels(closes, lows, ema20_values)

        pre_signal_context = {
            "last_close": last_close,
            "last_open_15m": opens[-1] if opens else None,
            "last_close_15m": last_close,
            "ema20_15m": last_ema20,
            "ma60": last_ma60,
            "macd_1h": htf_context.get("macd_line"),
            "macd_hist_1h": htf_context.get("macd_hist"),
            "rsi_value": last_rsi,
            "rsi_momentum_curr": rsi_momentum_current,
            "rsi_momentum_avg": rsi_momentum_avg,
            "pa_details": pa_block.details,
            "recent_closes_15m": recent_closes_15m,
            "recent_rsi_15m": recent_rsi_15m,
            "recent_macd_hist_1h": recent_macd_hist_1h,
            "momentum_rev": momentum_rev,
            "reversal_pa": reversal_pa,
            "support_data": support_data,
            # Extended OHLCV arrays for blow-off / parabolic filters
            "opens_15m": opens[-30:] if len(opens) >= 30 else list(opens),
            "high_15m": highs[-30:] if len(highs) >= 30 else list(highs),
            "low_15m": lows[-30:] if len(lows) >= 30 else list(lows),
            "volumes_15m": volumes[-30:] if len(volumes) >= 30 else list(volumes),
            "change_24h_pct": symbol_data.get("price_change_pct", 0.0),
            # V6: Early momentum and breakout detection
            "early_momentum": early_momentum,
            "breakout": breakout,
            # V6: Additional context for WATCH_PREMIUM early trigger
            "macd_hist_rising": macd_hist_rising,
            "stoch_rising": (
                stoch_k_values[-1] > stoch_k_values[-2]
                if len(stoch_k_values) >= 2
                and stoch_k_values[-1] is not np.nan
                and stoch_k_values[-2] is not np.nan
                else False
            ),
            "ema20_slope_pct": htf_context.get("ema20_slope_pct", 0.0),
        }

        signal_result = rules.decide_signal_label(
            trend_block=trend_block,
            osc_block=osc_block,
            vol_block=vol_block,
            pa_block=pa_block,
            htf_block=htf_block,
            meta=meta,
            rsi_value=last_rsi,
            symbol=symbol,
            pre_signal_context=pre_signal_context,
            momentum_rev=momentum_rev,
            reversal_pa=reversal_pa,
            support_data=support_data,
        )
        
        # Add market data to result
        signal_result.price = symbol_data['price']
        signal_result.price_change_pct = symbol_data['price_change_pct']
        signal_result.quote_volume = symbol_data['quote_volume']
        signal_result.htf_price_above_ema = bool(htf_context.get("close_above_ema20"))
        signal_result.fourh_price_above_ema20 = ht4_price_above_ema20
        signal_result.fourh_alignment_ok = ht4_alignment
        signal_result.fourh_ema20_slope_pct = ht4_ema20_slope_pct
        signal_result.bar_close_time = klines[-1].get('close_time')
        
        # Return only if we have a signal
        return signal_result.__dict__ if signal_result.label != "NO_SIGNAL" else None
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol_data.get('symbol', 'Unknown')}: {e}")
        return None