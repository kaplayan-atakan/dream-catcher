"""
Binance USDT Signal Bot - Main Module
Complete implementation with prefilters, cooldown, and error handling
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional

# Import modules
import config
import data_fetcher
import analyzer
import telegram_bot
import logger as log_module
from signal_guard import SignalMonitor, SignalFailureEvent

# DIP Trade Tracking (V8)
from dip_tracker import get_tracker, add_dip_trade, check_dip_prices, TradeStatus
from dip_notifications import format_status_change_notification

# Setup logging
logger = log_module.setup_logger()

# Global state
last_signal_times: Dict[str, datetime] = {}  # Legacy - kept for backwards compatibility
active_symbols: Set[str] = set()
signal_monitor = SignalMonitor()

# === SEPARATE COOLDOWN TRACKERS ===
# Five independent cooldown dictionaries for signal isolation
_cooldown_strong_ultra: Dict[str, datetime] = {}    # STRONG_BUY + ULTRA_BUY (shared)
_cooldown_watch_premium: Dict[str, datetime] = {}   # WATCH_PREMIUM only
_cooldown_dip_alert: Dict[str, datetime] = {}       # DIP_ALERT only (independent)
_cooldown_momentum_alert: Dict[str, datetime] = {}  # MOMENTUM_ALERT only (independent)
_cooldown_pump_alert: Dict[str, datetime] = {}      # PUMP_ALERT only (independent)


def is_in_cooldown(symbol: str, signal_type: str) -> bool:
    """
    Check if symbol is in cooldown for given signal type.
    
    Cooldown rules:
    - STRONG_BUY and ULTRA_BUY share cooldown
    - WATCH_PREMIUM has separate cooldown (does not block STRONG/ULTRA)
    - DIP_ALERT is completely independent (never blocked by others)
    - MOMENTUM_ALERT is completely independent (never blocked by others)
    - PUMP_ALERT is completely independent (never blocked by others)
    """
    now = datetime.now()
    
    if signal_type == "DIP_ALERT":
        # DIP_ALERT is completely independent
        if symbol in _cooldown_dip_alert:
            if now < _cooldown_dip_alert[symbol]:
                return True
        return False
    
    elif signal_type == "MOMENTUM_ALERT":
        # MOMENTUM_ALERT is completely independent
        if symbol in _cooldown_momentum_alert:
            if now < _cooldown_momentum_alert[symbol]:
                return True
        return False
    
    elif signal_type == "PUMP_ALERT":
        # PUMP_ALERT is completely independent
        if symbol in _cooldown_pump_alert:
            if now < _cooldown_pump_alert[symbol]:
                return True
        return False
    
    elif signal_type == "WATCH_PREMIUM":
        # WATCH_PREMIUM has its own cooldown
        if symbol in _cooldown_watch_premium:
            if now < _cooldown_watch_premium[symbol]:
                return True
        return False
    
    elif signal_type in ("STRONG_BUY", "ULTRA_BUY"):
        # STRONG and ULTRA share cooldown
        if symbol in _cooldown_strong_ultra:
            if now < _cooldown_strong_ultra[symbol]:
                return True
        return False
    
    return False


def set_cooldown(symbol: str, signal_type: str) -> None:
    """
    Set cooldown for symbol after sending signal.
    
    Cooldown rules:
    - STRONG_BUY/ULTRA_BUY: Sets shared cooldown, does NOT affect others
    - WATCH_PREMIUM: Sets only WATCH_PREMIUM cooldown
    - DIP_ALERT: Sets only DIP_ALERT cooldown
    - MOMENTUM_ALERT: Sets only MOMENTUM_ALERT cooldown
    - PUMP_ALERT: Sets only PUMP_ALERT cooldown
    """
    now = datetime.now()
    
    if signal_type == "DIP_ALERT":
        # DIP_ALERT only sets its own cooldown
        minutes = getattr(config, "DIP_COOLDOWN_MINUTES", 45)
        _cooldown_dip_alert[symbol] = now + timedelta(minutes=minutes)
        logger.debug(f"DIP_ALERT cooldown set for {symbol}: {minutes} min")
    
    elif signal_type == "MOMENTUM_ALERT":
        # MOMENTUM_ALERT only sets its own cooldown
        minutes = getattr(config, "MOMENTUM_COOLDOWN_MINUTES", 45)
        _cooldown_momentum_alert[symbol] = now + timedelta(minutes=minutes)
        logger.debug(f"MOMENTUM_ALERT cooldown set for {symbol}: {minutes} min")
    
    elif signal_type == "PUMP_ALERT":
        # PUMP_ALERT only sets its own cooldown
        minutes = getattr(config, "PUMP_COOLDOWN_MINUTES", 45)
        _cooldown_pump_alert[symbol] = now + timedelta(minutes=minutes)
        logger.debug(f"PUMP_ALERT cooldown set for {symbol}: {minutes} min")
    
    elif signal_type == "WATCH_PREMIUM":
        # WATCH_PREMIUM only sets its own cooldown
        minutes = getattr(config, "WATCH_PREMIUM_COOLDOWN_MINUTES", 30)
        _cooldown_watch_premium[symbol] = now + timedelta(minutes=minutes)
        logger.debug(f"WATCH_PREMIUM cooldown set for {symbol}: {minutes} min")
    
    elif signal_type in ("STRONG_BUY", "ULTRA_BUY"):
        # STRONG/ULTRA set shared cooldown only
        minutes = getattr(config, "COOLDOWN_MINUTES", 60)
        _cooldown_strong_ultra[symbol] = now + timedelta(minutes=minutes)
        logger.debug(f"{signal_type} cooldown set for {symbol}: {minutes} min")


def cleanup_expired_cooldowns() -> None:
    """Remove expired cooldown entries to prevent memory bloat."""
    now = datetime.now()
    
    for cooldown_dict in (
        _cooldown_strong_ultra, 
        _cooldown_watch_premium, 
        _cooldown_dip_alert,
        _cooldown_momentum_alert,
        _cooldown_pump_alert,
    ):
        expired = [sym for sym, exp_time in cooldown_dict.items() if now >= exp_time]
        for sym in expired:
            del cooldown_dict[sym]


def _describe_failure_event(event: SignalFailureEvent) -> str:
    if event.reason_code == "first_bar_failed":
        core_msg = "first 15m bar after signal did not confirm (close<=open)"
        status = "CANCELLED"
    elif event.reason_code == "follow_through_timeout":
        core_msg = "+1.5% target not reached within 12x15m bars"
        status = "INVALIDATED"
    else:
        core_msg = event.description
        status = "FAILED"
    expiry_str = event.block_expires_at.strftime("%H:%M UTC")
    return f"[{event.symbol}] {event.label} signal {status}: {core_msg}. Blocked until {expiry_str}."


async def _handle_post_signal_failures(events: List[SignalFailureEvent], in_warmup: bool) -> None:
    if not events:
        return
    for event in events:
        message = _describe_failure_event(event)
        logger.warning(message)
        if config.ENABLE_TELEGRAM and not in_warmup:
            try:
                # Use send_simple_message for plain text failure alerts (no Markdown)
                await telegram_bot.send_simple_message(message)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send Telegram failure alert for %s: %s", event.symbol, exc)


def _build_demo_signal(label: str) -> Dict[str, object]:
    """Return a canned signal payload for Telegram previews."""
    profiles = {
        "STRONG_BUY": {
            "symbol": "DEMO_STRONG",
            "price": 1.245,
            "change": 3.2,
            "volume": 48_000_000,
            "trend": 5,
            "osc": 4,
            "vol": 3,
            "pa": 1,
            "core": 12,
            "htf": 3,
            "total": 15,
        },
        "ULTRA_BUY": {
            "symbol": "DEMO_ULTRA",
            "price": 3.578,
            "change": 6.4,
            "volume": 95_000_000,
            "trend": 5,
            "osc": 4,
            "vol": 4,
            "pa": 2,
            "core": 15,
            "htf": 4,
            "total": 19,
        },
    }

    profile = profiles.get(label, profiles["WATCH"])
    return {
        "symbol": profile["symbol"],
        "label": label,
        "price": profile["price"],
        "price_change_pct": profile["change"],
        "quote_volume": profile["volume"],
        "trend_score": profile["trend"],
        "osc_score": profile["osc"],
        "vol_score": profile["vol"],
        "pa_score": profile["pa"],
        "score_core": profile["core"],
        "htf_bonus": profile["htf"],
        "score_total": profile["total"],
        "trend_details": {"fourh_ema20_slope_pct": 1.2},
        "htf_details": {
            "close_above_ema20": True,
            "ema20_slope_pct": 0.9,
            "macd_hist": 0.004,
        },
        "vol_details": {
            "volume_spike": True,
            "volume_spike_factor": 1.6,
            "obv_change_pct": 2.5,
            "bull_power": 0.0032,
            "bear_power": -0.0011,
        },
        "risk_tag": None,
        "reasons": [
            "Strong trend alignment",
            "Volume expansion confirmed",
            "Momentum shift in progress",
        ],
    }


def _log_prefilter_density(total_pairs: int, kept_pairs: int) -> None:
    """Emit Phase 3 transparency hints about how strict the current filters are."""
    if total_pairs <= 0:
        return
    keep_ratio = kept_pairs / total_pairs
    logger.info(
        "Prefilter keep ratio: %.2f%% (%d/%d)", keep_ratio * 100, kept_pairs, total_pairs
    )

    # Hooks for future dynamic thresholds – for now we only surface guidance to operators.
    if keep_ratio > 0.20:
        logger.info(
            "Prefilter note: High candidate density detected; future runs may tighten volume/change gates"
        )
    elif keep_ratio < 0.02:
        logger.info(
            "Prefilter note: Very few candidates surviving; future runs may relax thresholds slightly"
        )


async def process_symbol_batch(session: aiohttp.ClientSession, 
                              symbols: List[dict], 
                              semaphore: asyncio.Semaphore) -> List[dict]:
    """Process a batch of symbols with concurrency control"""
    async def analyze_with_limit(symbol_data):
        async with semaphore:
            try:
                return await analyzer.analyze_symbol(session, symbol_data)
            except Exception as e:
                logger.error(f"Error analyzing {symbol_data.get('symbol')}: {e}")
                return None
    
    tasks = [analyze_with_limit(s) for s in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out errors and None results
    valid_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task exception: {result}")
        elif result is not None:
            valid_results.append(result)
    
    return valid_results


async def scan_market(session: aiohttp.ClientSession) -> List[dict]:
    """
    Main market scanning function
    Applies prefilters and generates signals
    """
    try:
        # Step 1: Fetch 24h tickers
        logger.info("Fetching 24h ticker data...")
        tickers = await data_fetcher.fetch_24h_tickers(session)
        
        if not tickers:
            logger.warning("No tickers fetched from Binance")
            return []
        
        logger.info(f"Fetched {len(tickers)} USDT pairs")
        
        # Step 2: Parse and apply PREFILTERS
        filtered_symbols = []
        prefilter_stats = {
            "non_usdt": 0,
            "volume": 0,
            "price": 0,
            "change": 0,
        }
        
        for ticker in tickers:
            try:
                # Parse ticker data
                parsed = data_fetcher.parse_ticker_data(ticker)
                if parsed is None:
                    prefilter_stats["non_usdt"] += 1
                    logger.debug("Skipping non-USDT ticker")
                    continue

                symbol = parsed['symbol']
                if symbol in config.STABLE_SYMBOLS:
                    logger.debug("Skipping stablecoin %s", symbol)
                    continue
                
                # PREFILTER 1: Minimum 24h volume
                if parsed['quote_volume'] < config.MIN_24H_QUOTE_VOLUME:
                    prefilter_stats["volume"] += 1
                    continue
                
                # PREFILTER 2: Minimum price
                if parsed['price'] < config.MIN_PRICE_USDT:
                    prefilter_stats["price"] += 1
                    continue
                
                # PREFILTER 3: 24h price change range
                if not (config.MIN_24H_CHANGE <= parsed['price_change_pct'] <= config.MAX_24H_CHANGE):
                    prefilter_stats["change"] += 1
                    continue
                
                # NOTE: Cooldown is now checked per-signal-type AFTER analysis
                # This allows a symbol to generate DIP_ALERT even if STRONG_BUY is in cooldown
                
                filtered_symbols.append(parsed)
                
            except Exception as e:
                logger.error(f"Error parsing ticker: {e}")
                continue
        
        logger.info(
            "Prefilter summary - kept: %d | volume: %d | price: %d | change: %d | non-USDT: %d",
            len(filtered_symbols),
            prefilter_stats["volume"],
            prefilter_stats["price"],
            prefilter_stats["change"],
            prefilter_stats["non_usdt"],
        )
        _log_prefilter_density(len(tickers), len(filtered_symbols))
        
        # Step 3: Limit symbols if configured
        if config.MAX_SYMBOLS_PER_SCAN and len(filtered_symbols) > config.MAX_SYMBOLS_PER_SCAN:
            # Sort by volume and take top N
            filtered_symbols.sort(key=lambda x: x['quote_volume'], reverse=True)
            filtered_symbols = filtered_symbols[:config.MAX_SYMBOLS_PER_SCAN]
            logger.info(f"Limited to top {config.MAX_SYMBOLS_PER_SCAN} symbols by volume")
        
        # Step 4: Analyze symbols with concurrency control
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent API calls
        
        # Process in batches to avoid overwhelming the API
        batch_size = 20
        all_signals = []
        
        for i in range(0, len(filtered_symbols), batch_size):
            batch = filtered_symbols[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(filtered_symbols)-1)//batch_size + 1}")
            
            batch_results = await process_symbol_batch(session, batch, semaphore)
            all_signals.extend(batch_results)
            
            # Small delay between batches
            if i + batch_size < len(filtered_symbols):
                await asyncio.sleep(1)
        
        # Step 5: Filter for actual signals (exclude NO_SIGNAL)
        valid_signals = [s for s in all_signals if s and s.get('label') != 'NO_SIGNAL']
        
        return valid_signals
        
    except Exception as e:
        logger.error(f"Market scan error: {e}")
        return []


async def fetch_current_prices(session: aiohttp.ClientSession, symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Fetch current prices with OHLC data for a list of symbols from Binance.
    Uses parallel requests for efficiency.
    
    Args:
        session: aiohttp session
        symbols: List of symbol names (e.g., ["BTCUSDT", "ETHUSDT"])
        
    Returns:
        Dict mapping symbol -> {"price": float, "high": float, "low": float}
    """
    if not symbols:
        return {}
    
    prices = {}
    
    try:
        # First, get basic prices from ticker endpoint (fast, batch)
        url = f"{config.BINANCE_BASE_URL}/api/v3/ticker/price"
        
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                symbol_set = set(symbols)
                for item in data:
                    sym = item.get('symbol', '')
                    if sym in symbol_set:
                        try:
                            price = float(item.get('price', 0))
                            # Initialize with just price (will be enhanced with OHLC below)
                            prices[sym] = {"price": price, "high": price, "low": price}
                        except (ValueError, TypeError):
                            pass
            else:
                logger.warning(f"Failed to fetch prices: HTTP {response.status}")
        
        # Then, fetch 1m klines for each symbol to get accurate high/low
        # Use parallel requests with semaphore to limit concurrency
        semaphore = asyncio.Semaphore(10)  # Max 10 concurrent requests
        
        async def fetch_kline(symbol: str) -> Tuple[str, Optional[Dict]]:
            async with semaphore:
                kline_url = f"{config.BINANCE_BASE_URL}/api/v3/klines?symbol={symbol}&interval=1m&limit=1"
                try:
                    async with session.get(kline_url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data:
                                k = data[0]
                                return symbol, {
                                    "price": float(k[4]),   # Close
                                    "high": float(k[2]),    # High
                                    "low": float(k[3]),     # Low
                                    "open": float(k[1]),    # Open
                                }
                except Exception as e:
                    logger.debug(f"Kline fetch error for {symbol}: {e}")
                return symbol, None
        
        # Fetch klines in parallel for better accuracy
        if symbols:
            tasks = [fetch_kline(s) for s in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, tuple) and result[1] is not None:
                    symbol, data = result
                    prices[symbol] = data  # Override with more accurate kline data
                    
    except Exception as e:
        logger.error(f"Error fetching prices: {e}")
    
    return prices


async def monitor_dip_trades(session: aiohttp.ClientSession):
    """
    Background task to monitor active DIP trades for TP/SL conditions.
    Runs every DIP_PRICE_CHECK_INTERVAL_SECONDS.
    
    Uses OHLC data for accurate TP/SL detection with SL priority
    when both are hit in the same candle.
    """
    check_interval = getattr(config, "DIP_PRICE_CHECK_INTERVAL_SECONDS", 60)
    notify_enabled = getattr(config, "ENABLE_DIP_TP_SL_NOTIFICATIONS", True)
    
    logger.info(
        f"🎯 DIP trade monitor started (interval: {check_interval}s, "
        f"TP1: +{config.DIP_TP1_PCT}%, TP2: +{config.DIP_TP2_PCT}%, SL: {config.DIP_SL_PCT}%)"
    )
    
    while True:
        try:
            await asyncio.sleep(check_interval)
            
            tracker = get_tracker()
            active_symbols = tracker.get_active_symbols()
            
            if not active_symbols:
                continue
            
            # Fetch current prices with OHLC data for active trades (parallel)
            prices = await fetch_current_prices(session, active_symbols)
            
            if not prices:
                logger.debug("No prices fetched for DIP trades")
                continue
            
            # Check for TP/SL hits (SL has priority when both hit)
            status_changes = tracker.check_prices(prices)
            
            # Log active trade count periodically
            if len(active_symbols) > 0 and len(status_changes) > 0:
                logger.debug(f"DIP Monitor: {len(active_symbols)} active, {len(status_changes)} changes")
            
            for trade, old_status, new_status in status_changes:
                # Determine emoji based on status
                emoji = "✅" if new_status in (TradeStatus.TP1, TradeStatus.TP2) else "❌" if new_status == TradeStatus.SL else "⏱️"
                
                # Log the status change
                logger.info(
                    f"{emoji} DIP {new_status.value}: {trade.symbol} | "
                    f"Entry: ${trade.entry_price:.6f} | "
                    f"Exit: ${trade.exit_price:.6f} | "
                    f"Return: {trade.return_pct:+.2f}%"
                )
                
                # Send Telegram notification if enabled
                if config.ENABLE_TELEGRAM and notify_enabled:
                    try:
                        message = format_status_change_notification(trade, new_status)
                        await telegram_bot.send_telegram_message(message)
                        logger.info(f"DIP {new_status.value} notification sent for {trade.symbol}")
                    except Exception as e:
                        logger.error(f"Failed to send DIP notification: {e}")
            
            # Log periodic stats summary
            if active_symbols and len(status_changes) > 0:
                stats = tracker.get_stats_summary()
                logger.info(f"DIP Monitor: {stats}")
                
        except asyncio.CancelledError:
            logger.info("DIP trade monitor stopped")
            break
        except Exception as e:
            logger.error(f"DIP monitor error: {e}")
            await asyncio.sleep(10)  # Brief pause before retry
                
        except asyncio.CancelledError:
            logger.info("DIP trade monitor stopped")
            break
        except Exception as e:
            logger.error(f"DIP monitor error: {e}")
            await asyncio.sleep(10)  # Brief pause before retry


async def main_loop():
    """Main bot loop with proper error handling"""
    logger.info("=" * 60)
    logger.info("🚀 Starting Binance USDT Signal Bot")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"  • Min 24h Volume: ${config.MIN_24H_QUOTE_VOLUME:,.0f}")
    logger.info(f"  • 24h Change Range: {config.MIN_24H_CHANGE}% to {config.MAX_24H_CHANGE}%")
    logger.info(f"  • Min Price: ${config.MIN_PRICE_USDT}")
    logger.info("  • Core Score Thresholds:")
    logger.info(
        "    - WATCH when core ≥ %d (below this = NO_SIGNAL)",
        config.CORE_SCORE_WATCH_MIN,
    )
    logger.info(
        "    - STRONG_BUY when core ≥ %d with Trend≥%d & Vol≥%d",
        config.CORE_SCORE_STRONG_MIN,
        config.TREND_MIN_FOR_STRONG,
        config.VOL_MIN_FOR_STRONG,
    )
    logger.info(
        "    - ULTRA_BUY when core ≥ %d plus Trend≥%d, Osc≥%d, Vol≥%d, HTF≥%d",
        config.CORE_SCORE_ULTRA_MIN,
        config.TREND_MIN_FOR_ULTRA,
        config.OSC_MIN_FOR_ULTRA,
        config.VOL_MIN_FOR_ULTRA,
        config.HTF_MIN_FOR_ULTRA,
    )
    logger.info(f"  • Cooldown: {config.COOLDOWN_MINUTES} minutes")
    logger.info(f"  • Telegram: {'✅ Enabled' if config.ENABLE_TELEGRAM else '❌ Disabled'}")
    logger.info(f"  • Main Timeframe: {config.MAIN_TIMEFRAME}")
    logger.info(f"  • Multi-timeframe: {', '.join(config.TIMEFRAMES)}")
    
    # DIP Tracking status
    dip_tracking = getattr(config, "ENABLE_DIP_TRACKING", False)
    if dip_tracking:
        logger.info("  • DIP Tracking: ✅ Enabled")
        logger.info(f"    - TP1: +{config.DIP_TP1_PCT}% | TP2: +{config.DIP_TP2_PCT}% | SL: {config.DIP_SL_PCT}%")
        logger.info(f"    - Timeout: {config.DIP_TIMEOUT_HOURS}h | Check interval: {config.DIP_PRICE_CHECK_INTERVAL_SECONDS}s")
    else:
        logger.info("  • DIP Tracking: ❌ Disabled")
    
    logger.info("=" * 60)

    start_time = datetime.now()
    warmup_duration = timedelta(minutes=3)
    warmup_notice_logged = False
    warmup_complete_logged = False

    if config.ENABLE_TELEGRAM:
        try:
            await telegram_bot.send_simple_message(
                "🤖 Bot started! Ready to catch dreams... 🚀"
            )
            for preview_label in ("STRONG_BUY", "ULTRA_BUY"):
                demo_signal = _build_demo_signal(preview_label)
                preview_message = telegram_bot.format_signal_message(demo_signal)
                await telegram_bot.send_telegram_message(preview_message)
                await asyncio.sleep(0.3)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to send Telegram start message: {exc}")
    
    # Create persistent session with connection pooling
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        scan_count = 0
        total_signals = 0
        watch_premium_sent = 0
        
        # Start DIP trade monitor if enabled
        dip_monitor_task = None
        if getattr(config, "ENABLE_DIP_TRACKING", False):
            dip_monitor_task = asyncio.create_task(monitor_dip_trades(session))
            logger.info("🎯 DIP trade tracking enabled")
        
        while True:
            try:
                scan_count += 1
                scan_start = datetime.now()
                elapsed = datetime.now() - start_time
                in_warmup = elapsed < warmup_duration

                if config.ENABLE_TELEGRAM:
                    if in_warmup and not warmup_notice_logged:
                        logger.info("Skipping Telegram sends during warmup window (first 3 minutes)")
                        warmup_notice_logged = True
                    elif not in_warmup and not warmup_complete_logged:
                        logger.info("Warmup complete; Telegram notifications enabled")
                        warmup_complete_logged = True
                
                logger.info(f"\n🔍 Starting scan #{scan_count} at {scan_start.strftime('%H:%M:%S')}")
                
                await signal_monitor.evaluate_active_signals(session)
                failure_events = signal_monitor.drain_failure_events()
                if failure_events:
                    await _handle_post_signal_failures(failure_events, in_warmup)

                # Perform market scan
                signals = await scan_market(session)
                
                # Process signals
                if signals:
                    logger.info(f"✨ Generated {len(signals)} signals!")
                    
                    for signal in signals:
                        try:
                            symbol = signal.get('symbol')
                            if not symbol:
                                logger.debug("Signal missing symbol, skipping")
                                continue
                            if symbol in config.STABLE_SYMBOLS:
                                logger.debug("Ignoring stablecoin signal %s", symbol)
                                continue

                            label = signal.get('label')
                            total_signals += 1
                            score_total_value = signal.get('score_total', signal.get('total_score'))
                            try:
                                score_total_float = float(score_total_value) if score_total_value is not None else None
                            except (TypeError, ValueError):
                                score_total_float = None
                            original_label = label

                            # === DETERMINE SIGNAL TYPE FOR COOLDOWN ===
                            signal_type = None
                            should_notify = False
                            
                            if original_label == "DIP_ALERT":
                                signal_type = "DIP_ALERT"
                                should_notify = getattr(config, "DIP_NOTIFY_TELEGRAM", True)
                            elif original_label == "MOMENTUM_ALERT":
                                signal_type = "MOMENTUM_ALERT"
                                should_notify = getattr(config, "MOMENTUM_NOTIFY_TELEGRAM", True)
                            elif original_label == "PUMP_ALERT":
                                signal_type = "PUMP_ALERT"
                                should_notify = getattr(config, "PUMP_NOTIFY_TELEGRAM", True)
                            elif original_label == "ULTRA_BUY":
                                signal_type = "ULTRA_BUY"
                                should_notify = True
                            elif original_label == "STRONG_BUY":
                                signal_type = "STRONG_BUY"
                                should_notify = True
                            elif original_label == "WATCH" and score_total_float is not None and score_total_float >= config.WATCH_PREMIUM_MIN_SCORE:
                                signal_type = "WATCH_PREMIUM"
                                should_notify = getattr(config, "ENABLE_WATCH_PREMIUM", True)

                            # === CHECK COOLDOWN FOR THIS SIGNAL TYPE ===
                            if signal_type and is_in_cooldown(symbol, signal_type):
                                logger.debug(f"{signal_type} for {symbol} skipped (cooldown)")
                                # Still log to CSV for analysis, but don't notify
                                log_module.log_signal_to_csv(
                                    path=config.LOG_CSV_PATH,
                                    signal=signal,
                                    extra_fields={
                                        'price': signal.get('price'),
                                        'change_24h': signal.get('price_change_pct'),
                                        'quote_vol_24h': signal.get('quote_volume'),
                                        'cooldown_skipped': True,
                                    }
                                )
                                continue  # Skip to next signal

                            # === SIGNAL GUARD BLOCKING (for STRONG/ULTRA only) ===
                            blocked_reason = None
                            if original_label in {"STRONG_BUY", "ULTRA_BUY"} and signal_monitor.is_symbol_blocked(symbol):
                                blocked_reason = signal_monitor.get_block_reason(symbol) or "Post-signal block active"
                                label = "WATCH"
                                signal['label'] = label
                                logger.info("Blocking STRONG/ULTRA for %s due to %s", symbol, blocked_reason)
                                # Re-evaluate as WATCH_PREMIUM
                                if score_total_float is not None and score_total_float >= config.WATCH_PREMIUM_MIN_SCORE:
                                    signal_type = "WATCH_PREMIUM"
                                    should_notify = getattr(config, "ENABLE_WATCH_PREMIUM", True)
                                else:
                                    signal_type = None
                                    should_notify = False

                            should_notify_strong = label in {"STRONG_BUY", "ULTRA_BUY"}
                            should_register = should_notify_strong
                            
                            # Log to CSV
                            log_module.log_signal_to_csv(
                                path=config.LOG_CSV_PATH,
                                signal=signal,
                                extra_fields={
                                    'price': signal.get('price'),
                                    'change_24h': signal.get('price_change_pct'),
                                    'quote_vol_24h': signal.get('quote_volume'),
                                }
                            )
                            
                            # === SEND TO TELEGRAM WITH SIGNAL-TYPE SPECIFIC COOLDOWN ===
                            if config.ENABLE_TELEGRAM and not in_warmup and should_notify and signal_type:
                                if signal_type == "DIP_ALERT":
                                    message = telegram_bot.format_signal_message(
                                        signal,
                                        display_label="🎯 DIP_ALERT",
                                        info_note="Dip bounce opportunity — momentum reversal expected.",
                                    )
                                    await telegram_bot.send_telegram_message(message)
                                    set_cooldown(symbol, signal_type)
                                    logger.info("DIP_ALERT sent for %s", symbol)
                                    
                                    # Track the DIP trade for TP/SL monitoring
                                    if getattr(config, "ENABLE_DIP_TRACKING", False):
                                        entry_price = signal.get('price', 0)
                                        if entry_price > 0:
                                            trade = add_dip_trade(symbol, entry_price)
                                            logger.info(f"DIP trade tracked: {symbol} @ ${entry_price:.6f} (TP1=${trade.tp1_price:.6f}, SL=${trade.sl_price:.6f})")
                                elif signal_type == "MOMENTUM_ALERT":
                                    message = telegram_bot.format_signal_message(
                                        signal,
                                        display_label="🚀 MOMENTUM_ALERT",
                                        info_note="Strong trend continuation — 65-72% win rate.",
                                    )
                                    await telegram_bot.send_telegram_message(message)
                                    set_cooldown(symbol, signal_type)
                                    logger.info("MOMENTUM_ALERT sent for %s", symbol)
                                elif signal_type == "PUMP_ALERT":
                                    message = telegram_bot.format_signal_message(
                                        signal,
                                        display_label="📈 PUMP_ALERT",
                                        info_note="Recovery + pump momentum — 60-67% win rate.",
                                    )
                                    await telegram_bot.send_telegram_message(message)
                                    set_cooldown(symbol, signal_type)
                                    logger.info("PUMP_ALERT sent for %s", symbol)
                                elif signal_type in ("STRONG_BUY", "ULTRA_BUY"):
                                    message = telegram_bot.format_signal_message(signal)
                                    await telegram_bot.send_telegram_message(message)
                                    set_cooldown(symbol, signal_type)
                                elif signal_type == "WATCH_PREMIUM":
                                    info_note = "Early alert only — not actionable. STRONG/ULTRA rules unchanged."
                                    message = telegram_bot.format_signal_message(
                                        signal,
                                        display_label=config.WATCH_PREMIUM_TG_LABEL,
                                        info_note=info_note,
                                    )
                                    await telegram_bot.send_telegram_message(message)
                                    set_cooldown(symbol, signal_type)
                                    watch_premium_sent += 1
                                    log_score = score_total_float if score_total_float is not None else score_total_value
                                    logger.info("WATCH_PREMIUM sent for %s score=%s", symbol, log_score)
                            
                            # Console output
                            display_label = signal_type if signal_type else label
                            label_emoji = {
                                "ULTRA_BUY": "🚀",
                                "STRONG_BUY": "📈",
                                "WATCH_PREMIUM": "💎",
                                "DIP_ALERT": "🎯",
                                "MOMENTUM_ALERT": "🚀",
                                "PUMP_ALERT": "📈",
                                "WATCH": "👀",
                            }.get(display_label, "ℹ️")
                            logger.info(f"{label_emoji} {display_label}: {symbol}")
                            core_score = signal.get('score_core', signal.get('total_score', 0))
                            htf_bonus = signal.get('htf_bonus', 0)
                            total_score = signal.get('score_total', signal.get('total_score', 0))
                            logger.info(f"   Price: ${signal.get('price', 0):.6f}")
                            logger.info(
                                "   Scores: T=%s O=%s V=%s PA=%s | Core=%s HTF+%s Total=%s",
                                signal.get('trend_score'),
                                signal.get('osc_score'),
                                signal.get('vol_score'),
                                signal.get('pa_score'),
                                core_score,
                                htf_bonus,
                                total_score,
                            )
                            risk_tag = signal.get('risk_tag')
                            if risk_tag and risk_tag != "NORMAL":
                                logger.info("   Risk Tag: %s", risk_tag)
                            if blocked_reason:
                                logger.info("   Block Reason: %s", blocked_reason)
                            filter_notes = signal.get('filter_notes')
                            if filter_notes:
                                for note in filter_notes:
                                    logger.info("   %s", note)

                            if should_register:
                                signal_monitor.register_signal(signal)
                            
                        except Exception as e:
                            logger.error(f"Error processing signal: {e}")
                else:
                    logger.info("No signals generated in this scan")
                
                # Scan statistics
                scan_duration = (datetime.now() - scan_start).seconds
                logger.info(f"Scan completed in {scan_duration} seconds")
                logger.info(f"Total signals generated: {total_signals}")
                logger.info(f"Watch premium alerts sent: {watch_premium_sent}")
                
                # Clean up expired cooldowns from all three dictionaries
                cleanup_expired_cooldowns()
                
                # Wait before next scan
                logger.info(f"⏰ Waiting 60 seconds before next scan...")
                await asyncio.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Received shutdown signal...")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}", exc_info=True)
                logger.info("Recovering in 30 seconds...")
                await asyncio.sleep(30)
        
        # Cleanup DIP monitor task
        if dip_monitor_task:
            dip_monitor_task.cancel()
            try:
                await dip_monitor_task
            except asyncio.CancelledError:
                pass
            logger.info("DIP trade monitor stopped")
    
    logger.info("Bot shutdown complete")


def main():
    """Entry point with proper error handling"""
    try:
        # Run the async main loop
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("\n👋 Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()