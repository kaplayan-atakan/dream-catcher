"""
Binance USDT Signal Bot - Main Module
Complete implementation with prefilters, cooldown, and error handling
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set

# Import modules
import config
import data_fetcher
import analyzer
import telegram_bot
import logger as log_module
from signal_guard import SignalMonitor, SignalFailureEvent

# Setup logging
logger = log_module.setup_logger()

# Global state
last_signal_times: Dict[str, datetime] = {}
active_symbols: Set[str] = set()
signal_monitor = SignalMonitor()


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
            "cooldown": 0,
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
                
                # PREFILTER 4: Skip if in cooldown
                if symbol in last_signal_times:
                    time_since_signal = datetime.now() - last_signal_times[symbol]
                    if time_since_signal < timedelta(minutes=config.COOLDOWN_MINUTES):
                        logger.debug(f"Skipping {symbol} - in cooldown ({time_since_signal.seconds//60} min)")
                        prefilter_stats["cooldown"] += 1
                        continue
                
                filtered_symbols.append(parsed)
                
            except Exception as e:
                logger.error(f"Error parsing ticker: {e}")
                continue
        
        logger.info(
            "Prefilter summary - kept: %d | volume: %d | price: %d | change: %d | cooldown: %d | non-USDT: %d",
            len(filtered_symbols),
            prefilter_stats["volume"],
            prefilter_stats["price"],
            prefilter_stats["change"],
            prefilter_stats["cooldown"],
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

                            # Cooldown is applied only to actionable signals.
                            # WATCH should not block the symbol from generating future STRONG/ULTRA.
                            if original_label in {"STRONG_BUY", "ULTRA_BUY"}:
                                last_signal_times[symbol] = datetime.now()

                            blocked_reason = None
                            if original_label in {"STRONG_BUY", "ULTRA_BUY"} and signal_monitor.is_symbol_blocked(symbol):
                                blocked_reason = signal_monitor.get_block_reason(symbol) or "Post-signal block active"
                                label = "WATCH"
                                signal['label'] = label
                                logger.info("Blocking STRONG/ULTRA for %s due to %s", symbol, blocked_reason)

                            should_notify_strong = label in {"STRONG_BUY", "ULTRA_BUY"}
                            should_register = should_notify_strong
                            should_notify_watch_premium = (
                                original_label == "WATCH"
                                and score_total_float is not None
                                and config.ENABLE_WATCH_PREMIUM
                                and score_total_float >= config.WATCH_PREMIUM_MIN_SCORE
                            )
                            
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
                            
                            # Send to Telegram
                            if config.ENABLE_TELEGRAM and not in_warmup:
                                if should_notify_strong:
                                    message = telegram_bot.format_signal_message(signal)
                                    await telegram_bot.send_telegram_message(message)
                                elif should_notify_watch_premium:
                                    info_note = "Early alert only — not actionable. STRONG/ULTRA rules unchanged."
                                    message = telegram_bot.format_signal_message(
                                        signal,
                                        display_label=config.WATCH_PREMIUM_TG_LABEL,
                                        info_note=info_note,
                                    )
                                    await telegram_bot.send_telegram_message(message)
                                    watch_premium_sent += 1
                                    log_score = score_total_float if score_total_float is not None else score_total_value
                                    logger.info("WATCH_PREMIUM sent for %s score=%s", symbol, log_score)
                            
                            # Console output
                            label_emoji = {
                                "ULTRA_BUY": "🚀",
                                "STRONG_BUY": "📈",
                                "WATCH": "👀",
                            }.get(label, "ℹ️")
                            logger.info(f"{label_emoji} {label}: {symbol}")
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
                
                # Clean up old cooldowns
                current_time = datetime.now()
                expired_symbols = [
                    symbol for symbol, signal_time in last_signal_times.items()
                    if current_time - signal_time > timedelta(minutes=config.COOLDOWN_MINUTES * 2)
                ]
                for symbol in expired_symbols:
                    del last_signal_times[symbol]
                
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