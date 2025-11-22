"""
Binance USDT Signal Bot - Main Module
Complete implementation with prefilters, cooldown, and error handling
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set
import json

# Import modules
import config
import data_fetcher
import analyzer
import telegram_bot
import logger as log_module

# Setup logging
logger = log_module.setup_logger()

# Global state
last_signal_times: Dict[str, datetime] = {}
active_symbols: Set[str] = set()


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
        
        for ticker in tickers:
            try:
                # Parse ticker data
                parsed = data_fetcher.parse_ticker_data(ticker)
                if parsed is None:
                    logger.debug("Skipping non-USDT ticker")
                    continue
                
                # PREFILTER 1: Minimum 24h volume
                if parsed['quote_volume'] < config.MIN_24H_QUOTE_VOLUME:
                    continue
                
                # PREFILTER 2: Minimum price
                if parsed['price'] < config.MIN_PRICE_USDT:
                    continue
                
                # PREFILTER 3: 24h price change range
                if not (config.MIN_24H_CHANGE <= parsed['price_change_pct'] <= config.MAX_24H_CHANGE):
                    continue
                
                # PREFILTER 4: Skip if in cooldown
                symbol = parsed['symbol']
                if symbol in last_signal_times:
                    time_since_signal = datetime.now() - last_signal_times[symbol]
                    if time_since_signal < timedelta(minutes=config.COOLDOWN_MINUTES):
                        logger.debug(f"Skipping {symbol} - in cooldown ({time_since_signal.seconds//60} min)")
                        continue
                
                filtered_symbols.append(parsed)
                
            except Exception as e:
                logger.error(f"Error parsing ticker: {e}")
                continue
        
        logger.info(f"After prefilters: {len(filtered_symbols)} symbols to analyze")
        
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
        
        # Step 5: Filter for actual signals (not NONE)
        valid_signals = [s for s in all_signals if s and s.get('label') != 'NONE']
        
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
    logger.info(f"  • Price Range: {config.MIN_24H_CHANGE}% to {config.MAX_24H_CHANGE}%")
    logger.info(f"  • Min Price: ${config.MIN_PRICE_USDT}")
    logger.info(f"  • Score Thresholds:")
    logger.info(f"    - STRONG_BUY: {config.STRONG_BUY_SCORE}")
    logger.info(f"    - ULTRA_BUY: {config.ULTRA_BUY_SCORE}")
    logger.info(f"  • Cooldown: {config.COOLDOWN_MINUTES} minutes")
    logger.info(f"  • Telegram: {'✅ Enabled' if config.ENABLE_TELEGRAM else '❌ Disabled'}")
    logger.info(f"  • Main Timeframe: {config.MAIN_TIMEFRAME}")
    logger.info(f"  • Multi-timeframe: {', '.join(config.TIMEFRAMES)}")
    logger.info("=" * 60)

    start_time = datetime.now()
    warmup_duration = timedelta(minutes=3)
    warmup_notice_logged = False
    warmup_complete_logged = False

    # if config.ENABLE_TELEGRAM:
    #     try:
    #         await telegram_bot.send_telegram_message(
    #             "Hadi, başlıyorum! Hazır mıyız? Time to fly 🚀"
    #         )
    #     except Exception as exc:  # noqa: BLE001
    #         logger.warning(f"Failed to send Telegram start message: {exc}")
    
    # Create persistent session with connection pooling
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        scan_count = 0
        total_signals = 0
        
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
                
                # Perform market scan
                signals = await scan_market(session)
                
                # Process signals
                if signals:
                    logger.info(f"✨ Generated {len(signals)} signals!")
                    
                    for signal in signals:
                        try:
                            # Update cooldown
                            symbol = signal['symbol']
                            last_signal_times[symbol] = datetime.now()
                            total_signals += 1
                            
                            # Log to CSV
                            log_module.log_signal_to_csv(
                                path=config.LOG_CSV_PATH,
                                signal=signal,
                                extra_fields={
                                    'price': signal.get('price'),
                                    'change_24h': signal.get('price_change_pct'),
                                    'quote_vol_24h': signal.get('quote_volume')
                                }
                            )
                            
                            # Send to Telegram
                            if config.ENABLE_TELEGRAM and not in_warmup:
                                message = telegram_bot.format_signal_message(signal)
                                await telegram_bot.send_telegram_message(message)
                            
                            # Console output
                            label_emoji = "🚀" if signal['label'] == "ULTRA_BUY" else "📈"
                            logger.info(f"{label_emoji} {signal['label']}: {symbol}")
                            logger.info(f"   Price: ${signal.get('price', 0):.6f}")
                            logger.info(f"   Scores: T={signal['trend_score']} O={signal['osc_score']} " +
                                      f"V={signal['vol_score']} PA={signal['pa_score']} " +
                                      f"TOTAL={signal['total_score']}")
                            
                        except Exception as e:
                            logger.error(f"Error processing signal: {e}")
                else:
                    logger.info("No signals generated in this scan")
                
                # Scan statistics
                scan_duration = (datetime.now() - scan_start).seconds
                logger.info(f"Scan completed in {scan_duration} seconds")
                logger.info(f"Total signals generated: {total_signals}")
                
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