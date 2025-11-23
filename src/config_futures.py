"""
Futures-mode configuration profile.

Import this module as `config` when running the bot against USDT-margined
futures markets. All values default to the spot profile from config.py and are
then tightened/extended for leveraged trading.
"""
from .config import *  # noqa: F401,F403

# === PREFILTER OVERRIDES ===
MIN_24H_QUOTE_VOLUME = 10_000_000  # Futures-mode: require deeper liquidity
MIN_PRICE_USDT = 0.05  # Futures-mode: avoid sub-5¢ instruments
MIN_24H_CHANGE = -10.0  # Narrow the acceptable daily drawdown range
MAX_24H_CHANGE = 15.0   # Cap daily pumps to reduce chasing breakouts

# === SIGNAL THRESHOLDS ===
STRONG_BUY_SCORE = 10  # Futures-mode: demand higher overall confluence
ULTRA_BUY_SCORE = 14

# ULTRA_BUY block minimums tightened to favor stronger setups
ULTRA_BUY_MIN_TREND = 4
ULTRA_BUY_MIN_VOL_PA = 4
ULTRA_BUY_MAX_RSI = 60

# === PRICE ACTION / RISK FILTERS ===
COLLAPSE_MAX_DROP_PCT = 10.0  # Reject charts with sharper dumps in lookback
MIN_BAR_VOLUME_USDT = 20_000  # Ensure each candle carries enough liquidity

# === FUTURES-SPECIFIC FLAGS ===
ASSUMED_MAX_LEVERAGE = 5  # Reference only; execution layer can use later
REQUIRE_4H_CONFIRM_FOR_ULTRA = True  # Optional hook for stricter HTF checks
FOUR_H_MIN_SLOPE_PCT = 0.5  # Minimum EMA20 slope (%) to treat 4h as supportive
