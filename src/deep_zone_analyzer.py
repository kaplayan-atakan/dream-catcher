"""
Deep Cross-Zone Analyzer
Finds optimal 3-way zone combinations for new alert types.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Any


def load_discovery_data(path: str = "results/discovery/discovery_signals.csv") -> pd.DataFrame:
    """Load discovery signals with zone classifications."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} signals")
    return df


def analyze_triple_combinations(
    df: pd.DataFrame,
    min_samples: int = 25,
    min_win_rate: float = 45.0,
) -> pd.DataFrame:
    """
    Analyze ALL 3-way zone combinations.
    
    Zones analyzed:
    - rsi_zone: deep_oversold, oversold, recovery, neutral, strong, overbought
    - ema20_zone: far_below, below, near, above, far_above
    - change_24h_zone: dump, down, flat, up, pump
    """
    
    results = []
    
    # Get unique values for each zone
    rsi_zones = df["rsi_zone"].unique()
    ema_zones = df["ema20_zone"].unique()
    change_zones = df["change_24h_zone"].unique()
    
    print(f"RSI zones: {list(rsi_zones)}")
    print(f"EMA zones: {list(ema_zones)}")
    print(f"24h zones: {list(change_zones)}")
    
    total_combos = len(rsi_zones) * len(ema_zones) * len(change_zones)
    print(f"\nAnalyzing {total_combos} combinations...")
    
    for rsi_z in rsi_zones:
        for ema_z in ema_zones:
            for chg_z in change_zones:
                subset = df[
                    (df["rsi_zone"] == rsi_z) & 
                    (df["ema20_zone"] == ema_z) & 
                    (df["change_24h_zone"] == chg_z)
                ]
                
                count = len(subset)
                if count < min_samples:
                    continue
                
                wins = subset["win_2pct"].sum()
                win_rate = (wins / count * 100) if count > 0 else 0
                
                if win_rate < min_win_rate:
                    continue
                
                # Calculate additional metrics
                avg_rise = subset["max_rise_pct"].mean()
                avg_return = subset["final_return_pct"].mean()
                win_3pct = subset["win_3pct"].sum() / count * 100 if count > 0 else 0
                
                # Score distribution
                avg_score = subset["score_total"].mean()
                
                # RSI stats
                avg_rsi = subset["rsi_value"].mean()
                
                results.append({
                    "rsi_zone": rsi_z,
                    "ema_zone": ema_z,
                    "change_zone": chg_z,
                    "combination": f"RSI:{rsi_z} | EMA:{ema_z} | 24h:{chg_z}",
                    "count": count,
                    "wins": int(wins),
                    "win_rate_2pct": round(win_rate, 1),
                    "win_rate_3pct": round(win_3pct, 1),
                    "avg_rise": round(avg_rise, 2),
                    "avg_return": round(avg_return, 2),
                    "avg_score": round(avg_score, 1),
                    "avg_rsi": round(avg_rsi, 1),
                    "confidence": "HIGH" if count >= 50 else "MEDIUM" if count >= 30 else "LOW",
                })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("win_rate_2pct", ascending=False)
    
    print(f"\nFound {len(df_results)} combinations with {min_samples}+ samples and {min_win_rate}%+ win rate")
    
    return df_results


def analyze_score_threshold_impact(
    df: pd.DataFrame,
    top_combinations: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each top combination, analyze how score thresholds affect win rate.
    """
    
    results = []
    
    for _, combo in top_combinations.head(10).iterrows():
        subset = df[
            (df["rsi_zone"] == combo["rsi_zone"]) & 
            (df["ema20_zone"] == combo["ema_zone"]) & 
            (df["change_24h_zone"] == combo["change_zone"])
        ]
        
        for min_score in [5, 6, 7, 8, 9, 10]:
            filtered = subset[subset["score_total"] >= min_score]
            if len(filtered) < 15:
                continue
            
            win_rate = filtered["win_2pct"].mean() * 100
            
            results.append({
                "combination": combo["combination"],
                "min_score": min_score,
                "count": len(filtered),
                "win_rate": round(win_rate, 1),
                "avg_rise": round(filtered["max_rise_pct"].mean(), 2),
            })
    
    return pd.DataFrame(results)


def identify_alert_candidates(df_results: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Identify candidate alert types from top combinations.
    
    Criteria:
    - Win rate >= 50%
    - Sample size >= 30 (MEDIUM+ confidence)
    - Not already covered by DIP_ALERT
    """
    
    candidates = []
    
    # DIP_ALERT covers: oversold + below/far_below + dump
    dip_alert_covered = [
        ("oversold", "below", "dump"),
        ("oversold", "far_below", "dump"),
        ("deep_oversold", "below", "dump"),
        ("deep_oversold", "far_below", "dump"),
    ]
    
    for _, row in df_results.iterrows():
        # Skip if already covered by DIP_ALERT
        combo_tuple = (row["rsi_zone"], row["ema_zone"], row["change_zone"])
        if combo_tuple in dip_alert_covered:
            continue
        
        # Skip low confidence
        if row["confidence"] == "LOW":
            continue
        
        # Determine potential alert type
        alert_type = None
        emoji = ""
        
        # MOMENTUM_ALERT: strong RSI + above EMA + positive 24h
        if row["rsi_zone"] in ("strong", "overbought") and row["ema_zone"] in ("above", "far_above") and row["change_zone"] in ("up", "pump"):
            alert_type = "MOMENTUM_ALERT"
            emoji = "🚀"
        
        # PUMP_ALERT: pump day with decent RSI
        elif row["change_zone"] == "pump" and row["rsi_zone"] in ("recovery", "neutral", "strong"):
            alert_type = "PUMP_ALERT"
            emoji = "📈"
        
        # RECOVERY_ALERT: below EMA but 24h turning positive
        elif row["ema_zone"] in ("below", "far_below") and row["change_zone"] in ("up", "pump"):
            alert_type = "RECOVERY_ALERT"
            emoji = "🔄"
        
        # BREAKOUT_ALERT: neutral/strong RSI + near EMA + up movement
        elif row["rsi_zone"] in ("neutral", "strong") and row["ema_zone"] == "near" and row["change_zone"] in ("up", "pump"):
            alert_type = "BREAKOUT_ALERT"
            emoji = "💥"
        
        # STRENGTH_ALERT: strong trend continuation
        elif row["rsi_zone"] == "strong" and row["ema_zone"] == "above":
            alert_type = "STRENGTH_ALERT"
            emoji = "💪"
        
        if alert_type:
            candidates.append({
                "alert_type": alert_type,
                "emoji": emoji,
                "rsi_zone": row["rsi_zone"],
                "ema_zone": row["ema_zone"],
                "change_zone": row["change_zone"],
                "win_rate": row["win_rate_2pct"],
                "win_rate_3pct": row["win_rate_3pct"],
                "count": row["count"],
                "avg_rise": row["avg_rise"],
                "confidence": row["confidence"],
                "priority": "HIGH" if row["win_rate_2pct"] >= 55 and row["count"] >= 40 else "MEDIUM",
            })
    
    # Sort by win rate and deduplicate alert types (keep best)
    seen_types = set()
    unique_candidates = []
    for c in sorted(candidates, key=lambda x: x["win_rate"], reverse=True):
        if c["alert_type"] not in seen_types:
            seen_types.add(c["alert_type"])
            unique_candidates.append(c)
    
    return unique_candidates


def generate_report(
    df: pd.DataFrame,
    df_results: pd.DataFrame,
    score_impact: pd.DataFrame,
    candidates: List[Dict],
    output_dir: Path,
):
    """Generate comprehensive markdown report."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSVs
    df_results.to_csv(output_dir / "triple_combinations.csv", index=False)
    score_impact.to_csv(output_dir / "score_threshold_impact.csv", index=False)
    
    # Generate markdown report
    with open(output_dir / "deep_zone_report.md", "w", encoding="utf-8") as f:
        f.write("# Deep Cross-Zone Analysis Report\n\n")
        f.write(f"**Total Signals Analyzed:** {len(df)}\n")
        f.write(f"**High-Performing Combinations Found:** {len(df_results)}\n\n")
        
        f.write("---\n\n")
        
        # Top 20 combinations
        f.write("## 🏆 Top 20 Triple Combinations (by Win Rate)\n\n")
        f.write("| Rank | RSI Zone | EMA Zone | 24h Zone | Count | Win Rate | Avg Rise | Confidence |\n")
        f.write("|------|----------|----------|----------|-------|----------|----------|------------|\n")
        for i, row in df_results.head(20).iterrows():
            rank = df_results.index.get_loc(i) + 1
            f.write(f"| {rank} | {row['rsi_zone']} | {row['ema_zone']} | {row['change_zone']} | {row['count']} | {row['win_rate_2pct']}% | {row['avg_rise']}% | {row['confidence']} |\n")
        
        f.write("\n---\n\n")
        
        # High confidence only (50+ samples)
        f.write("## 📊 High Confidence Combinations (50+ samples)\n\n")
        high_conf = df_results[df_results["confidence"] == "HIGH"].head(10)
        f.write("| RSI Zone | EMA Zone | 24h Zone | Count | Win Rate | Win 3%+ | Avg Rise |\n")
        f.write("|----------|----------|----------|-------|----------|---------|----------|\n")
        for _, row in high_conf.iterrows():
            f.write(f"| {row['rsi_zone']} | {row['ema_zone']} | {row['change_zone']} | {row['count']} | {row['win_rate_2pct']}% | {row['win_rate_3pct']}% | {row['avg_rise']}% |\n")
        
        f.write("\n---\n\n")
        
        # Alert candidates
        f.write("## 🚨 Recommended New Alert Types\n\n")
        for c in candidates:
            f.write(f"### {c['emoji']} {c['alert_type']}\n\n")
            f.write(f"**Conditions:**\n")
            f.write(f"- RSI Zone: `{c['rsi_zone']}`\n")
            f.write(f"- EMA Zone: `{c['ema_zone']}`\n")
            f.write(f"- 24h Change Zone: `{c['change_zone']}`\n\n")
            f.write(f"**Performance:**\n")
            f.write(f"- Win Rate (2%+): **{c['win_rate']}%**\n")
            f.write(f"- Win Rate (3%+): {c['win_rate_3pct']}%\n")
            f.write(f"- Average Rise: {c['avg_rise']}%\n")
            f.write(f"- Sample Size: {c['count']}\n")
            f.write(f"- Confidence: {c['confidence']}\n")
            f.write(f"- Implementation Priority: **{c['priority']}**\n\n")
        
        f.write("---\n\n")
        
        # Score threshold analysis
        f.write("## 📈 Score Threshold Impact\n\n")
        f.write("How minimum score affects win rate for top combinations:\n\n")
        f.write("| Combination | Min Score | Count | Win Rate |\n")
        f.write("|-------------|-----------|-------|----------|\n")
        for _, row in score_impact.head(30).iterrows():
            combo_short = row['combination'][:40] + "..." if len(row['combination']) > 40 else row['combination']
            f.write(f"| {combo_short} | {row['min_score']} | {row['count']} | {row['win_rate']}% |\n")
        
        f.write("\n---\n\n")
        
        # Implementation priority matrix
        f.write("## 🎯 Implementation Priority Matrix\n\n")
        f.write("| Alert Type | Win Rate | Confidence | Priority | vs DIP_ALERT |\n")
        f.write("|------------|----------|------------|----------|---------------|\n")
        f.write("| 🎯 DIP_ALERT (current) | ~55% | HIGH | ✅ LIVE | baseline |\n")
        for c in candidates:
            vs_dip = "better" if c["win_rate"] > 55 else "similar" if c["win_rate"] >= 50 else "lower"
            f.write(f"| {c['emoji']} {c['alert_type']} | {c['win_rate']}% | {c['confidence']} | {c['priority']} | {vs_dip} |\n")
    
    print(f"\n✅ Report saved to: {output_dir}/deep_zone_report.md")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep Cross-Zone Analyzer")
    parser.add_argument("--input", default="results/discovery/discovery_signals.csv")
    parser.add_argument("--output-dir", default="results/deep_zone_analysis")
    parser.add_argument("--min-samples", type=int, default=25)
    parser.add_argument("--min-win-rate", type=float, default=45.0)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DEEP CROSS-ZONE ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = load_discovery_data(args.input)
    
    # Analyze triple combinations
    print("\n" + "=" * 60)
    print("ANALYZING 3-WAY COMBINATIONS")
    print("=" * 60)
    df_results = analyze_triple_combinations(
        df, 
        min_samples=args.min_samples,
        min_win_rate=args.min_win_rate,
    )
    
    # Print top 15
    print("\n🏆 TOP 15 COMBINATIONS:")
    print("-" * 80)
    for i, row in df_results.head(15).iterrows():
        print(f"{row['combination']}")
        print(f"   Count: {row['count']} | Win Rate: {row['win_rate_2pct']}% | Avg Rise: {row['avg_rise']}% | {row['confidence']}")
        print()
    
    # Score threshold analysis
    print("\n" + "=" * 60)
    print("SCORE THRESHOLD IMPACT")
    print("=" * 60)
    score_impact = analyze_score_threshold_impact(df, df_results)
    
    # Identify alert candidates
    print("\n" + "=" * 60)
    print("ALERT CANDIDATES")
    print("=" * 60)
    candidates = identify_alert_candidates(df_results)
    
    for c in candidates:
        print(f"\n{c['emoji']} {c['alert_type']}")
        print(f"   Zones: RSI={c['rsi_zone']} | EMA={c['ema_zone']} | 24h={c['change_zone']}")
        print(f"   Win Rate: {c['win_rate']}% | Count: {c['count']} | Priority: {c['priority']}")
    
    # Generate report
    output_path = Path(args.output_dir)
    generate_report(df, df_results, score_impact, candidates, output_path)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total combinations analyzed: {len(df_results)}")
    print(f"Alert candidates found: {len(candidates)}")
    print(f"Reports saved to: {output_path}")
    
    return df_results, candidates


if __name__ == "__main__":
    main()
