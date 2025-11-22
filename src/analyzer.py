"""
Symbol Analyzer Module - REAL INDICATORS & CALCULATIONS
"""
import logging
from typing import Optional, Dict

import numpy as np

import indicators
import price_action
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
        
        # Volume indicators - latest
        obv_uptrend = indicators.is_obv_uptrend(obv_values, config.OBV_TREND_LOOKBACK)
        last_bull_power = bull_power_values[-1] if bull_power_values[-1] is not np.nan else 0
        last_bear_power = bear_power_values[-1] if bear_power_values[-1] is not np.nan else 0
        
        # ============ MULTI-TIMEFRAME ANALYSIS ============
        mtf_trend_confirmed = False
        htf_price_above_ema = False
        
        if '1h' in klines_data and len(klines_data['1h']) >= 50:
            htf_closes = [k['close'] for k in klines_data['1h']]
            htf_ema20 = indicators.ema(htf_closes, 20)
            
            if htf_ema20[-1] is not np.nan:
                htf_price_above_ema = htf_closes[-1] > htf_ema20[-1]
                
                # Check if 1h is also in uptrend
                htf_ema50 = indicators.ema(htf_closes, 50)
                if htf_ema50[-1] is not np.nan:
                    mtf_trend_confirmed = (htf_closes[-1] > htf_ema20[-1] > htf_ema50[-1])
        
        # ============ PRICE ACTION ANALYSIS ============
        pa_signals = price_action.analyze_price_action(
            opens, highs, lows, closes, volumes, ema20_values
        )
        
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
            mtf_trend=mtf_trend_confirmed
        )
        
        # OSCILLATOR BLOCK
        osc_block = rules.compute_osc_block(
            rsi_val=last_rsi,
            stoch_k=last_stoch_k,
            cci=last_cci,
            stoch_rsi=last_stoch_rsi,
            williams_r=last_williams_r,
            uo=last_uo
        )
        
        # VOLUME BLOCK
        vol_block = rules.compute_volume_block(
            obv_trend=obv_uptrend,
            bull_power=last_bull_power,
            bear_power=last_bear_power,
            volume_spike=pa_signals.get('volume_spike', False)
        )
        
        # PRICE ACTION BLOCK
        pa_block = rules.compute_price_action_block(pa_signals)
        
        # ============ SIGNAL DECISION ============
        signal_result = rules.decide_signal_label(
            trend_block=trend_block,
            osc_block=osc_block,
            vol_block=vol_block,
            pa_block=pa_block,
            rsi_value=last_rsi,
            htf_trend_ok=htf_price_above_ema,  # For ULTRA_BUY
            symbol=symbol
        )
        
        # Add market data to result
        signal_result.price = symbol_data['price']
        signal_result.price_change_pct = symbol_data['price_change_pct']
        signal_result.quote_volume = symbol_data['quote_volume']
        
        # Return only if we have a signal
        return signal_result.__dict__ if signal_result.label != "NONE" else None
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol_data.get('symbol', 'Unknown')}: {e}")
        return None