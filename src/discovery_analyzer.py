"""
Discovery Mode Analyzer
Temporarily relaxes filters to find optimal entry zones.
Does NOT modify config.py - uses internal overrides only.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config

# === DISCOVERY MODE OVERRIDES ===
# These override config values for analysis only
DISCOVERY_OVERRIDES = {
    "RSI_STRONG_MIN": 25,        # See all RSI levels
    "RSI_STRONG_MAX": 80,        # See all RSI levels
    "CORE_SCORE_STRONG_MIN": 7,  # Lower threshold = more signals
    "CORE_SCORE_WATCH_MIN": 5,   # Lower threshold = more signals
    "MIN_24H_CHANGE": -15.0,     # Allow bigger dips
    "MAX_24H_CHANGE": 25.0,      # Allow bigger pumps
    "RSI_TOP_THRESHOLD": 85,     # Relaxed top filter
}


def get_discovery_value(key: str):
    """Get value from overrides or fall back to config."""
    return DISCOVERY_OVERRIDES.get(key, getattr(config, key, None))


# === ZONE CLASSIFIERS ===
def classify_rsi_zone(rsi: float) -> str:
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


def classify_ema_zone(dist_pct: float) -> str:
    if dist_pct < -3.0:
        return "far_below"
    elif dist_pct < -1.0:
        return "below"
    elif dist_pct <= 1.0:
        return "near"
    elif dist_pct <= 3.0:
        return "above"
    else:
        return "far_above"


def classify_24h_zone(change: float) -> str:
    if change < -5.0:
        return "dump"
    elif change < -2.0:
        return "down"
    elif change <= 2.0:
        return "flat"
    elif change <= 5.0:
        return "up"
    else:
        return "pump"


@dataclass
class DiscoverySignal:
    symbol: str
    timestamp: datetime
    entry_price: float
    
    # Scores
    score_total: int
    trend_score: int
    osc_score: int
    vol_score: int
    pa_score: int
    
    # Context
    rsi_value: float
    rsi_zone: str
    ema20_dist_pct: float
    ema20_zone: str
    change_24h_pct: float
    change_24h_zone: str
    macd_hist: float
    stoch_k: float
    
    # Performance (filled after forward analysis)
    max_rise_pct: float = 0.0
    bars_to_peak: int = 0
    final_return_pct: float = 0.0
    win_2pct: bool = False
    win_3pct: bool = False


def load_data(symbol: str) -> tuple:
    """Load 15m and 1h data for symbol."""
    data_dir = Path("data")
    
    f15m = data_dir / "precomputed_15m" / f"{symbol}_15m_features.parquet"
    f1h = data_dir / "precomputed_1h" / f"{symbol}_1h_features.parquet"
    
    if not f15m.exists() or not f1h.exists():
        return None, None
    
    df_15m = pd.read_parquet(f15m)
    df_1h = pd.read_parquet(f1h)
    
    return df_15m, df_1h


def analyze_forward_performance(
    closes: np.ndarray,
    highs: np.ndarray,
    entry_idx: int,
    entry_price: float,
    lookahead: int = 48
) -> dict:
    """Analyze price action after signal."""
    
    end_idx = min(entry_idx + lookahead, len(closes))
    if end_idx <= entry_idx + 1:
        return {"max_rise_pct": 0, "bars_to_peak": 0, "final_return_pct": 0}
    
    future_highs = highs[entry_idx + 1:end_idx]
    future_closes = closes[entry_idx + 1:end_idx]
    
    if len(future_highs) == 0:
        return {"max_rise_pct": 0, "bars_to_peak": 0, "final_return_pct": 0}
    
    max_high = np.max(future_highs)
    max_rise = (max_high - entry_price) / entry_price * 100
    bars_to_peak = int(np.argmax(future_highs)) + 1
    final_return = (future_closes[-1] - entry_price) / entry_price * 100
    
    return {
        "max_rise_pct": round(max_rise, 4),
        "bars_to_peak": bars_to_peak,
        "final_return_pct": round(final_return, 4)
    }


def generate_discovery_signals(symbol: str, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> List[DiscoverySignal]:
    """Generate signals with relaxed filters."""
    
    signals = []
    closes = df_15m["close"].values
    highs = df_15m["high"].values
    opens = df_15m["open"].values
    
    cooldown_until = 0
    
    for idx in range(100, len(df_15m) - 50):  # Skip edges
        
        if idx < cooldown_until:
            continue
        
        row = df_15m.iloc[idx]
        close = row["close"]
        
        # Basic data extraction
        rsi = row.get("rsi", 50)
        if pd.isna(rsi):
            rsi = 50
        ema20 = row.get("ema_fast", close)
        if pd.isna(ema20):
            ema20 = close
        ema50 = row.get("ema_slow", close)
        if pd.isna(ema50):
            ema50 = close
        macd_hist = row.get("macd_hist", 0)
        if pd.isna(macd_hist):
            macd_hist = 0
        stoch_k = row.get("stoch_k", 50)
        if pd.isna(stoch_k):
            stoch_k = 50
        adx = row.get("adx", 20)
        if pd.isna(adx):
            adx = 20
        plus_di = row.get("plus_di", 25)
        if pd.isna(plus_di):
            plus_di = 25
        minus_di = row.get("minus_di", 25)
        if pd.isna(minus_di):
            minus_di = 25
        
        # Calculate context
        ema20_dist = ((close - ema20) / ema20 * 100) if ema20 > 0 else 0
        
        # Simulate 24h change
        if idx >= 96:
            price_96_ago = closes[idx - 96]
            change_24h = (close - price_96_ago) / price_96_ago * 100
        else:
            change_24h = 0
        
        # === RELAXED PREFILTER ===
        if rsi < get_discovery_value("RSI_STRONG_MIN"):
            continue
        if rsi > get_discovery_value("RSI_STRONG_MAX"):
            continue
        if change_24h < get_discovery_value("MIN_24H_CHANGE"):
            continue
        if change_24h > get_discovery_value("MAX_24H_CHANGE"):
            continue
        
        # === BUILD SCORING BLOCKS ===
        # Simplified scoring for discovery
        
        # Trend score (0-5)
        trend_score = 0
        if close > ema20:
            trend_score += 1
        if close > ema50:
            trend_score += 1
        if ema20 > ema50:
            trend_score += 1
        if adx > 20:
            trend_score += 1
        if plus_di > minus_di:
            trend_score += 1
        
        # Oscillator score (0-4)
        osc_score = 0
        if 40 < rsi < 70:
            osc_score += 1
        if macd_hist > 0:
            osc_score += 1
        if idx > 0:
            prev_macd = df_15m.iloc[idx-1].get("macd_hist", 0)
            if pd.isna(prev_macd):
                prev_macd = 0
            if macd_hist > prev_macd:
                osc_score += 1
        if 30 < stoch_k < 80:
            osc_score += 1
        
        # Volume score (0-3)
        vol = row.get("volume", 0)
        if pd.isna(vol):
            vol = 0
        avg_vol = df_15m["volume"].iloc[max(0, idx-20):idx].mean() if idx > 20 else vol
        if pd.isna(avg_vol) or avg_vol == 0:
            avg_vol = vol if vol > 0 else 1
        vol_ratio = vol / avg_vol if avg_vol > 0 else 1
        
        vol_score = 0
        if vol_ratio > 1.0:
            vol_score += 1
        if vol_ratio > 1.5:
            vol_score += 1
        if vol_ratio > 2.0:
            vol_score += 1
        
        # PA score (0-3)
        pa_score = 0
        if close > opens[idx]:
            pa_score += 1  # Green candle
        body = abs(close - opens[idx])
        low_val = row.get("low", close)
        wick_lower = min(close, opens[idx]) - low_val
        if body > 0 and wick_lower > body * 0.5:
            pa_score += 1  # Lower wick
        if idx > 0 and close > closes[idx-1]:
            pa_score += 1  # Higher close
        
        # Total score
        score_total = trend_score + osc_score + vol_score + pa_score
        
        # === RELAXED SCORE THRESHOLD ===
        if score_total < get_discovery_value("CORE_SCORE_WATCH_MIN"):
            continue
        
        # === CREATE SIGNAL ===
        perf = analyze_forward_performance(closes, highs, idx, close, lookahead=48)
        
        ts_val = row.get("timestamp", None)
        if ts_val is not None and not pd.isna(ts_val):
            sig_time = datetime.fromtimestamp(ts_val / 1000)
        else:
            sig_time = datetime.now()
        
        sig = DiscoverySignal(
            symbol=symbol,
            timestamp=sig_time,
            entry_price=close,
            score_total=score_total,
            trend_score=trend_score,
            osc_score=osc_score,
            vol_score=vol_score,
            pa_score=pa_score,
            rsi_value=round(rsi, 1),
            rsi_zone=classify_rsi_zone(rsi),
            ema20_dist_pct=round(ema20_dist, 2),
            ema20_zone=classify_ema_zone(ema20_dist),
            change_24h_pct=round(change_24h, 2),
            change_24h_zone=classify_24h_zone(change_24h),
            macd_hist=round(macd_hist, 4),
            stoch_k=round(stoch_k, 1),
            max_rise_pct=perf["max_rise_pct"],
            bars_to_peak=perf["bars_to_peak"],
            final_return_pct=perf["final_return_pct"],
            win_2pct=perf["max_rise_pct"] >= 2.0,
            win_3pct=perf["max_rise_pct"] >= 3.0,
        )
        
        signals.append(sig)
        cooldown_until = idx + 4  # 1 hour cooldown
    
    return signals


def run_discovery(symbols: List[str], output_dir: str = "results/discovery"):
    """Main discovery analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DISCOVERY MODE ANALYSIS")
    print("=" * 60)
    print(f"\nOverrides active:")
    for k, v in DISCOVERY_OVERRIDES.items():
        orig = getattr(config, k, "N/A")
        print(f"  {k}: {orig} → {v}")
    print()
    
    all_signals = []
    
    for symbol in symbols:
        print(f"Processing {symbol}...", end=" ")
        df_15m, df_1h = load_data(symbol)
        
        if df_15m is None:
            print("SKIP (no data)")
            continue
        
        sigs = generate_discovery_signals(symbol, df_15m, df_1h)
        all_signals.extend(sigs)
        print(f"{len(sigs)} signals")
    
    if not all_signals:
        print("No signals generated!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame([asdict(s) for s in all_signals])
    df.to_csv(output_path / "discovery_signals.csv", index=False)
    print(f"\nTotal signals: {len(df)}")
    
    # === ZONE ANALYSIS ===
    print("\n" + "=" * 60)
    print("ZONE PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # RSI Zone Analysis
    print("\n📊 RSI ZONE PERFORMANCE:")
    print("-" * 50)
    rsi_stats = df.groupby("rsi_zone").agg({
        "win_2pct": ["count", "sum", "mean"],
        "max_rise_pct": "mean",
        "final_return_pct": "mean"
    }).round(3)
    rsi_stats.columns = ["count", "wins", "win_rate", "avg_rise", "avg_return"]
    rsi_stats["win_rate"] = (rsi_stats["win_rate"] * 100).round(1)
    rsi_stats = rsi_stats.sort_values("win_rate", ascending=False)
    print(rsi_stats.to_string())
    
    # EMA Zone Analysis
    print("\n📊 EMA20 DISTANCE ZONE PERFORMANCE:")
    print("-" * 50)
    ema_stats = df.groupby("ema20_zone").agg({
        "win_2pct": ["count", "sum", "mean"],
        "max_rise_pct": "mean"
    }).round(3)
    ema_stats.columns = ["count", "wins", "win_rate", "avg_rise"]
    ema_stats["win_rate"] = (ema_stats["win_rate"] * 100).round(1)
    ema_stats = ema_stats.sort_values("win_rate", ascending=False)
    print(ema_stats.to_string())
    
    # 24h Change Zone Analysis
    print("\n📊 24H CHANGE ZONE PERFORMANCE:")
    print("-" * 50)
    change_stats = df.groupby("change_24h_zone").agg({
        "win_2pct": ["count", "sum", "mean"],
        "max_rise_pct": "mean"
    }).round(3)
    change_stats.columns = ["count", "wins", "win_rate", "avg_rise"]
    change_stats["win_rate"] = (change_stats["win_rate"] * 100).round(1)
    change_stats = change_stats.sort_values("win_rate", ascending=False)
    print(change_stats.to_string())
    
    # Score Analysis
    print("\n📊 SCORE PERFORMANCE:")
    print("-" * 50)
    score_stats = df.groupby("score_total").agg({
        "win_2pct": ["count", "sum", "mean"],
        "max_rise_pct": "mean"
    }).round(3)
    score_stats.columns = ["count", "wins", "win_rate", "avg_rise"]
    score_stats["win_rate"] = (score_stats["win_rate"] * 100).round(1)
    print(score_stats.to_string())
    
    # === OPTIMAL CONFIG RECOMMENDATIONS ===
    print("\n" + "=" * 60)
    print("OPTIMAL CONFIG RECOMMENDATIONS")
    print("=" * 60)
    
    # Find best RSI zone
    best_rsi_zone = rsi_stats["win_rate"].idxmax()
    best_rsi_rate = rsi_stats.loc[best_rsi_zone, "win_rate"]
    best_rsi_count = rsi_stats.loc[best_rsi_zone, "count"]
    
    print(f"\n🎯 Best RSI Zone: {best_rsi_zone} ({best_rsi_rate}% win rate, {best_rsi_count} signals)")
    
    # RSI range of successful signals
    successful = df[df["win_2pct"] == True]
    rsi_q25 = 0
    rsi_q75 = 100
    ema_q95 = 5.0
    change_q95 = 20.0
    
    if len(successful) > 20:
        rsi_q25 = successful["rsi_value"].quantile(0.25)
        rsi_q75 = successful["rsi_value"].quantile(0.75)
        rsi_median = successful["rsi_value"].median()
        
        print(f"\n📈 Successful Signals RSI Profile:")
        print(f"   Median: {rsi_median:.0f}")
        print(f"   IQR: {rsi_q25:.0f} - {rsi_q75:.0f}")
        print(f"   Suggested RSI_STRONG_MIN: {int(rsi_q25)}")
        print(f"   Suggested RSI_STRONG_MAX: {int(rsi_q75)}")
        
        ema_q95 = successful["ema20_dist_pct"].quantile(0.95)
        print(f"\n   Suggested PARABOLIC_EMA_DIST_PCT: {ema_q95:.1f}")
        
        change_q95 = successful["change_24h_pct"].quantile(0.95)
        print(f"   Suggested MAX_24H_CHANGE: {change_q95:.0f}")
    
    # Score threshold analysis
    print("\n📊 Score Threshold Analysis:")
    for threshold in [5, 6, 7, 8, 9, 10, 11, 12]:
        subset = df[df["score_total"] >= threshold]
        if len(subset) > 10:
            wr = subset["win_2pct"].mean() * 100
            print(f"   Score >= {threshold}: {len(subset):4d} signals, {wr:.1f}% win rate")
    
    # Save markdown report
    with open(output_path / "discovery_report.md", "w", encoding="utf-8") as f:
        f.write("# Discovery Mode Analysis Report\n\n")
        f.write("## Overrides Used\n\n")
        f.write("| Parameter | Original | Discovery |\n")
        f.write("|-----------|----------|----------|\n")
        for k, v in DISCOVERY_OVERRIDES.items():
            orig = getattr(config, k, "N/A")
            f.write(f"| {k} | {orig} | {v} |\n")
        
        f.write(f"\n## Summary\n\n")
        f.write(f"- Total Signals: {len(df)}\n")
        f.write(f"- Win Rate (2%+): {df['win_2pct'].mean()*100:.1f}%\n")
        f.write(f"- Avg Max Rise: {df['max_rise_pct'].mean():.2f}%\n\n")
        
        f.write("## RSI Zone Performance\n\n")
        f.write("| Zone | Count | Wins | Win Rate | Avg Rise | Avg Return |\n")
        f.write("|------|-------|------|----------|----------|------------|\n")
        for zone in rsi_stats.index:
            row = rsi_stats.loc[zone]
            f.write(f"| {zone} | {int(row['count'])} | {int(row['wins'])} | {row['win_rate']}% | {row['avg_rise']:.2f}% | {row['avg_return']:.2f}% |\n")
        f.write("\n")
        
        f.write("## EMA20 Distance Performance\n\n")
        f.write("| Zone | Count | Wins | Win Rate | Avg Rise |\n")
        f.write("|------|-------|------|----------|----------|\n")
        for zone in ema_stats.index:
            row = ema_stats.loc[zone]
            f.write(f"| {zone} | {int(row['count'])} | {int(row['wins'])} | {row['win_rate']}% | {row['avg_rise']:.2f}% |\n")
        f.write("\n")
        
        f.write("## 24h Change Performance\n\n")
        f.write("| Zone | Count | Wins | Win Rate | Avg Rise |\n")
        f.write("|------|-------|------|----------|----------|\n")
        for zone in change_stats.index:
            row = change_stats.loc[zone]
            f.write(f"| {zone} | {int(row['count'])} | {int(row['wins'])} | {row['win_rate']}% | {row['avg_rise']:.2f}% |\n")
        f.write("\n")
        
        f.write("## Score Threshold Analysis\n\n")
        f.write("| Min Score | Signals | Win Rate |\n")
        f.write("|-----------|---------|----------|\n")
        for threshold in [5, 6, 7, 8, 9, 10, 11, 12]:
            subset = df[df["score_total"] >= threshold]
            if len(subset) > 10:
                wr = subset["win_2pct"].mean() * 100
                f.write(f"| >= {threshold} | {len(subset)} | {wr:.1f}% |\n")
        f.write("\n")
        
        if len(successful) > 20:
            f.write("## Recommended Config\n\n")
            f.write(f"Based on {len(successful)} successful signals (2%+ rise):\n\n")
            f.write(f"```python\n")
            f.write(f"RSI_STRONG_MIN = {int(rsi_q25)}\n")
            f.write(f"RSI_STRONG_MAX = {int(rsi_q75)}\n")
            f.write(f"PARABOLIC_EMA_DIST_PCT = {ema_q95:.1f}\n")
            f.write(f"MAX_24H_CHANGE = {change_q95:.0f}\n")
            f.write(f"```\n")
    
    print(f"\n✅ Report saved to: {output_path}/discovery_report.md")
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Discovery Mode Analyzer")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
    parser.add_argument("--output-dir", default="results/discovery")
    
    args = parser.parse_args()
    
    run_discovery(args.symbols, args.output_dir)
