"""
Configuration - Complete parameters for production
"""

# Binance API
BINANCE_BASE_URL = "https://api.binance.com"
SYMBOL_FILTER_SUFFIX = "USDT"

# Symbols to exclude entirely (stablecoins, etc.)
STABLE_SYMBOLS = {
    "USDCUSDT",
    "BUSDUSDT",
    "TUSDUSDT",
    "DAIUSDT",
    "USDPUSDT",
    "GUSDUSDT",
    "LUSDUSDT",
    "USDEUSDT",
    "USD1USDT",
    "XUSDUSDT",
    "FDUSDUSDT",
    "USDSUSDT",      # bazı borsalarda stable
    "EURSUSDT",      # euro stable (isteğe bağlı)
    "BFUSDUSDT",
}

# Timeframes
TIMEFRAMES = ["15m", "1h", "4h"]
MAIN_TIMEFRAME = "15m"
MAIN_TIMEFRAME_MINUTES = 15

# === PREFILTERS ===
MIN_24H_QUOTE_VOLUME = 10_000_000  # $10M minimum volume per revised spec
MIN_PRICE_USDT = 0.04  # Filter illiquid penny assets
MIN_24H_CHANGE = -10.0  # Reject symbols dumping more than 10%
MAX_24H_CHANGE = 20.0  # Reject symbols pumping beyond 20%

# Signal Settings
COOLDOWN_MINUTES = 60
MAX_SYMBOLS_PER_SCAN = 50  # Limit to top 50 by volume

# WATCH_PREMIUM controls
# - ENABLE_WATCH_PREMIUM=False disables these informational alerts immediately
# - Adjust WATCH_PREMIUM_MIN_SCORE to tune how many WATCH results get promoted
# - The label used in Telegram is WATCH_PREMIUM_TG_LABEL and the log line "WATCH_PREMIUM sent…".
# - Verify by sending a WATCH with score>=WATCH_PREMIUM_MIN_SCORE and watching for the INFO line + Telegram payload.
ENABLE_WATCH_PREMIUM = True  # Toggle informational WATCH alerts (no post-signal effects)
WATCH_PREMIUM_MIN_SCORE = 18
WATCH_PREMIUM_TG_LABEL = "WATCH_PREMIUM"

# Pre-signal gating filters
MA60_PERIOD = 60
RSI_PRE_FILTER_THRESHOLD = 60.0
RSI_MOMENTUM_LOOKBACK = 10
RSI_MOMENTUM_MIN_MULTIPLIER = 1.02
MACD_1H_MIN_VALUE = 0.0
MACD_1H_HIST_MIN_VALUE = 0.0
ENABLE_SSR_EMA20_GATE = True

# === LATE SPIKE / OVEREXTENSION GUARD (Revizyon 1) ===
ENABLE_LATE_SPIKE_FILTER = True
ENABLE_OVEREXTENSION_FILTER = True
ENABLE_PARABOLIC_SPIKE_FILTER = False
EXHAUSTION_LOOKBACK = 8
LATE_PUMP_EMA_DIST_PCT = 3.0
LATE_PUMP_RUNUP_PCT = 8.0

# === TREND REVERSAL SAFEGUARDS (Revizyon 2) ===
ENABLE_CANDLE_DIRECTION_FILTER = True
ENABLE_MOMENTUM_TURNING_FILTER = True
ENABLE_LOCAL_BOTTOM_FILTER = True
LOCAL_BOTTOM_LOOKBACK = 6

# Post-signal validation
FOLLOW_THROUGH_TARGET_MULTIPLIER = 1.008  # Require +0.8% within validation window
FOLLOW_THROUGH_BARS = 8
BLOCK_DURATION_MINUTES = 180

# Legacy aliases (Phase 2 compatibility)
POST_SIGNAL_TARGET_PCT = (FOLLOW_THROUGH_TARGET_MULTIPLIER - 1.0) * 100
POST_SIGNAL_MONITOR_BARS = FOLLOW_THROUGH_BARS
POST_SIGNAL_BLOCK_MINUTES = BLOCK_DURATION_MINUTES

# Telegram
ENABLE_TELEGRAM = True  # Set True and add credentials
TELEGRAM_BOT_TOKEN = "7611453017:AAFAz9jBsUQ-N6RUdQ8pnct0gIzV2UeEmIM"
TELEGRAM_CHAT_ID = "5883922751"

# API Settings
REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_BACKOFF = 2

# Logging
LOG_CSV_PATH = "signals_log.csv"
LOG_LEVEL = "INFO"

# === SIGNAL THRESHOLDS (legacy, retained for telemetry) ===
ULTRA_BUY_SCORE = 12
STRONG_BUY_SCORE = 8
ULTRA_BUY_MIN_TREND = 3
ULTRA_BUY_MIN_OSC = 2
ULTRA_BUY_MIN_VOL_PA = 3
ULTRA_BUY_MAX_RSI = 65
STRONG_BUY_MIN_TREND = 2
STRONG_BUY_MIN_OSC = 1
STRONG_BUY_MIN_VOL_PA = 2

# Block toggles (Phase 3 future-ready hooks)
ENABLE_TREND_BLOCK = True
ENABLE_OSC_BLOCK = True
ENABLE_VOLUME_BLOCK = True
ENABLE_PRICE_ACTION_BLOCK = True

# === INDICATOR PARAMETERS ===
# Moving Averages
EMA_FAST = 20
EMA_SLOW = 50

# Trend
ADX_PERIOD = 14
ADX_STRONG_TREND = 20.0
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_HIST_RISING_BARS = 3
MOMENTUM_PERIOD = 10

# Oscillators
RSI_PERIOD = 14
RSI_HEALTHY_MIN = 45
RSI_HEALTHY_MAX = 65
STOCH_K_PERIOD = 14
STOCH_D_PERIOD = 3
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80
CCI_PERIOD = 20
WILLIAMS_PERIOD = 14
WILLIAMS_BULLISH = -60
UO_PERIODS = (7, 14, 28)
UO_BULLISH = 50

# Volume
OBV_TREND_LOOKBACK = 10
VOLUME_SPIKE_MULTIPLIER = 1.5
VOLUME_LOOKBACK = 20

# Price Action
MIN_BODY_PCT = 1.0  # 1% minimum candle body
COLLAPSE_MAX_DROP_PCT = 20.0
COLLAPSE_LOOKBACK_BARS = 96
MIN_BAR_VOLUME_USDT = 10000
EMA_BREAK_LOOKBACK = 10
EMA_RETEST_LOOKBACK = 10
EMA_NEAR_TOLERANCE = 0.01  # 1% proximity check for EMA20 retest detection
EMA_SIMILARITY_TOLERANCE = 0.01  # EMA20 considered ~ EMA50 when within 1%
STRONG_GREEN_BODY_MULTIPLIER = 1.5  # strong vs average body size
STRONG_GREEN_LOOKBACK = 20
LONG_WICK_MIN_RATIO = 0.40

# Volume scoring helpers
VOLUME_SPIKE_STRONG = 1.5
VOLUME_SPIKE_MEDIUM = 1.2
OBV_UPTREND_MIN_PCT = 2.0
OBV_SIDEWAYS_MIN_PCT = 0.5
BULL_POWER_DOMINANCE = 0.0  # Bull power must be > 0 and bear < 0 for bonus

# Higher timeframe confirmation
HTF_EMA_SLOPE_LOOKBACK = 5
HTF_SLOPE_MIN_PCT = 0.0  # Slope must be non-negative for bonus

# Risk tagging thresholds
RISK_LATE_PUMP_CHANGE = 15.0
RISK_VOL_STRONG = 3
RISK_TREND_WEAK = 2

# Core score thresholds (Phase 3 revised spec)
CORE_SCORE_WATCH_MIN = 8
CORE_SCORE_STRONG_MIN = 11
CORE_SCORE_ULTRA_MIN = 14
TREND_MIN_FOR_STRONG = 3
VOL_MIN_FOR_STRONG = 2
TREND_MIN_FOR_ULTRA = 4
OSC_MIN_FOR_ULTRA = 3
VOL_MIN_FOR_ULTRA = 3
HTF_MIN_FOR_ULTRA = 2

# Oscillator / momentum guardrails
RSI_STRONG_MIN = 50
RSI_STRONG_MAX = 65
RSI_BUFFER_MIN = 45
RSI_BUFFER_MAX = 70
STOCH_K_MIDLINE = 50
CCI_STRONG_THRESHOLD = 100
STOCH_RSI_BULL_LEVEL = 50
UO_RISING_MIN_DELTA = 0.5

# === DİP YAKALAMA FİLTRELERİ / BOTTOM-FISHING FILTERS ===
ENABLE_BOTTOM_FISHING = True  # Master toggle for dip-focused logic

# RSI dip zone
RSI_OVERSOLD_ZONE = 35        # r < 35 considered oversold
RSI_RECOVERY_MIN = 38         # Early recovery level from oversold
RSI_OVERSOLD_EXIT = 40        # Full exit from oversold

# Stochastic recovery
STOCH_OVERSOLD_EXIT = 25      # Exiting oversold region
STOCH_RECOVERY = 30           # Recovery confirmation level

# Price action dip patterns
REQUIRE_HAMMER_OR_ENGULFING = True
MIN_BOUNCE_FROM_SUPPORT = 0.5  # 0.5% bounce from EMA/support

# Volume confirmation on bounce
REQUIRE_VOLUME_INCREASE_ON_BOUNCE = True
MIN_VOLUME_INCREASE_RATIO = 1.3  # 30% increase vs prior bar

# Support level detection
SUPPORT_EMA_PROXIMITY_PCT = 1.5  # 1.5% proximity to EMA20 for support
SUPPORT_LOOKBACK_BARS = 20       # Recent lows lookback

# Momentum shift detection
REQUIRE_MOMENTUM_SHIFT = True    # Require clear upward shift from dip
MIN_RSI_RISE_LAST_3 = 3          # RSI gained at least 3 points in last 3 bars
MOMENTUM_SHIFT_LOOKBACK = 5      # Bars to look back for oversold condition

# Dip bonus scoring
DIP_REVERSAL_BONUS = 2           # Bonus for reversal pattern detection
SUPPORT_BOUNCE_BONUS = 1         # Bonus for bouncing from support
EMA_SUPPORT_BONUS = 1            # Bonus for near EMA20 support
RSI_RECOVERY_BONUS = 2           # Bonus for RSI exiting oversold

# Top filter (avoid late entries)
ENABLE_TOP_FILTER = True         # Downgrade signals at overbought tops
RSI_TOP_THRESHOLD = 70           # RSI above this = momentum exhausted

# MACD histogram turning detection
MACD_HIST_NEG_TO_POS_BARS = 3    # Lookback bars for histogram neg→pos turn

# Momentum shift detection master toggle
ENABLE_MOMENTUM_SHIFT_DETECTION = True

# Backtest
BACKTEST_TP_PERCENTS = [2.0, 3.0, 5.0, 10.0]
BACKTEST_SL_PERCENTS = [1.0, 2.0, 3.0]
BACKTEST_LOOKAHEAD_BARS = 96

# === PREFILTER RELAXATION (when bottom-fishing is enabled) ===
# These overrides allow catching dip reversals that would otherwise be filtered out
if ENABLE_BOTTOM_FISHING:
    MIN_24H_QUOTE_VOLUME = 5_000_000   # $5M (relaxed from $10M)
    MAX_SYMBOLS_PER_SCAN = 100          # Increased from 50
    COOLDOWN_MINUTES = 30               # Reduced from 60
    MAX_24H_CHANGE = 8.0                # Reduced from 20%
    MIN_24H_CHANGE = -5.0               # Relaxed from -10%
    RSI_STRONG_MIN = 38                 # Lowered from 50 for earlier entries
    RSI_STRONG_MAX = 55                 # Lowered from 65 to avoid late entries