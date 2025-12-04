"""
Cross-Zone Analyzer
Finds optimal zone combinations for highest win rates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


def load_signals(path: str = "results/discovery/discovery_signals.csv") -> pd.DataFrame:
    """Load signals with zone classifications."""
    return pd.read_csv(path)


def analyze_cross_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze all zone combinations."""
    
    results = []
    
    # Define zone columns to analyze
    zone_cols = ["rsi_zone", "ema20_zone", "change_24h_zone"]
    
    # Single zone analysis
    for col in zone_cols:
        for zone in df[col].unique():
            subset = df[df[col] == zone]
            if len(subset) >= 20:  # Minimum sample size
                results.append({
                    "combination": f"{col}={zone}",
                    "zones": 1,
                    "count": len(subset),
                    "wins": int(subset["win_2pct"].sum()),
                    "win_rate": round(subset["win_2pct"].mean() * 100, 1),
                    "avg_rise": round(subset["max_rise_pct"].mean(), 2),
                    "avg_return": round(subset["final_return_pct"].mean(), 2),
                })
    
    # Two-zone combinations
    for col1, col2 in [("rsi_zone", "ema20_zone"), 
                        ("rsi_zone", "change_24h_zone"),
                        ("ema20_zone", "change_24h_zone")]:
        for z1 in df[col1].unique():
            for z2 in df[col2].unique():
                subset = df[(df[col1] == z1) & (df[col2] == z2)]
                if len(subset) >= 15:
                    results.append({
                        "combination": f"{col1}={z1} + {col2}={z2}",
                        "zones": 2,
                        "count": len(subset),
                        "wins": int(subset["win_2pct"].sum()),
                        "win_rate": round(subset["win_2pct"].mean() * 100, 1),
                        "avg_rise": round(subset["max_rise_pct"].mean(), 2),
                        "avg_return": round(subset["final_return_pct"].mean(), 2),
                    })
    
    # Three-zone combinations
    for rsi_z in df["rsi_zone"].unique():
        for ema_z in df["ema20_zone"].unique():
            for chg_z in df["change_24h_zone"].unique():
                subset = df[
                    (df["rsi_zone"] == rsi_z) & 
                    (df["ema20_zone"] == ema_z) & 
                    (df["change_24h_zone"] == chg_z)
                ]
                if len(subset) >= 10:
                    results.append({
                        "combination": f"RSI={rsi_z} + EMA={ema_z} + 24h={chg_z}",
                        "zones": 3,
                        "count": len(subset),
                        "wins": int(subset["win_2pct"].sum()),
                        "win_rate": round(subset["win_2pct"].mean() * 100, 1),
                        "avg_rise": round(subset["max_rise_pct"].mean(), 2),
                        "avg_return": round(subset["final_return_pct"].mean(), 2),
                    })
    
    return pd.DataFrame(results)


def find_best_combinations(df_results: pd.DataFrame, min_count: int = 20) -> pd.DataFrame:
    """Find combinations with highest win rates and sufficient samples."""
    
    # Filter by minimum count
    df = df_results[df_results["count"] >= min_count].copy()
    
    # Sort by win rate descending
    df = df.sort_values("win_rate", ascending=False)
    
    return df.head(20)


def generate_filter_rules(best_combos: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate actionable filter rules from best combinations."""
    
    rules = []
    
    for _, row in best_combos.head(10).iterrows():
        combo = row["combination"]
        wr = row["win_rate"]
        count = row["count"]
        
        rules.append({
            "rule": combo,
            "win_rate": f"{wr}%",
            "sample_size": int(count),
            "confidence": "HIGH" if count >= 50 else "MEDIUM" if count >= 30 else "LOW"
        })
    
    return rules


def save_report(df_results: pd.DataFrame, best_combos: pd.DataFrame, rules: List[Dict], output_dir: Path):
    """Save analysis report."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    df_results.to_csv(output_dir / "cross_zone_full.csv", index=False)
    
    # Save best combinations
    best_combos.to_csv(output_dir / "cross_zone_best.csv", index=False)
    
    # Save markdown report
    with open(output_dir / "cross_zone_report.md", "w", encoding="utf-8") as f:
        f.write("# Cross-Zone Analysis Report\n\n")
        
        f.write("## Top 20 Combinations by Win Rate\n\n")
        f.write("| Combination | Count | Wins | Win Rate | Avg Rise |\n")
        f.write("|-------------|-------|------|----------|----------|\n")
        for _, row in best_combos.iterrows():
            f.write(f"| {row['combination']} | {row['count']} | {row['wins']} | {row['win_rate']}% | {row['avg_rise']}% |\n")
        
        f.write("\n## Recommended Filter Rules\n\n")
        for i, rule in enumerate(rules, 1):
            f.write(f"### Rule {i}: {rule['rule']}\n")
            f.write(f"- Win Rate: **{rule['win_rate']}**\n")
            f.write(f"- Sample Size: {rule['sample_size']}\n")
            f.write(f"- Confidence: {rule['confidence']}\n\n")
        
        f.write("\n## Implementation Suggestions\n\n")
        f.write("```python\n")
        f.write("# Add to rules.py or config.py:\n\n")
        f.write("# High-value zone combinations\n")
        f.write("PREFERRED_ZONES = {\n")
        for rule in rules[:5]:
            f.write(f"    # {rule['rule']} -> {rule['win_rate']} win rate\n")
        f.write("}\n")
        f.write("```\n")
    
    print(f"Report saved to: {output_dir}/cross_zone_report.md")


def analyze_score_zones(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by score thresholds combined with zones."""
    
    results = []
    
    for score_min in [5, 6, 7, 8, 9, 10, 11, 12]:
        for zone_col in ["rsi_zone", "ema20_zone", "change_24h_zone"]:
            for zone_val in df[zone_col].unique():
                subset = df[(df["score_total"] >= score_min) & (df[zone_col] == zone_val)]
                if len(subset) >= 15:
                    results.append({
                        "combination": f"score>={score_min} + {zone_col}={zone_val}",
                        "count": len(subset),
                        "wins": int(subset["win_2pct"].sum()),
                        "win_rate": round(subset["win_2pct"].mean() * 100, 1),
                        "avg_rise": round(subset["max_rise_pct"].mean(), 2),
                    })
    
    return pd.DataFrame(results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Zone Analyzer")
    parser.add_argument("--input", default="results/discovery/discovery_signals.csv")
    parser.add_argument("--output-dir", default="results/cross_zone")
    parser.add_argument("--min-count", type=int, default=20)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CROSS-ZONE ANALYSIS")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading signals from: {args.input}")
    df = load_signals(args.input)
    print(f"Total signals: {len(df)}")
    print(f"Overall win rate: {df['win_2pct'].mean() * 100:.1f}%")
    
    # Analyze
    print("\nAnalyzing zone combinations...")
    results = analyze_cross_zones(df)
    print(f"Total combinations analyzed: {len(results)}")
    
    # Find best
    print(f"\nFinding best combinations (min {args.min_count} samples)...")
    best = find_best_combinations(results, min_count=args.min_count)
    
    print("\n" + "=" * 60)
    print("TOP 10 COMBINATIONS")
    print("=" * 60)
    print(best.head(10).to_string(index=False))
    
    # Analyze score + zone combinations
    print("\n" + "=" * 60)
    print("SCORE + ZONE COMBINATIONS")
    print("=" * 60)
    score_zones = analyze_score_zones(df)
    if not score_zones.empty:
        top_score_zones = score_zones.sort_values("win_rate", ascending=False).head(10)
        print(top_score_zones.to_string(index=False))
    
    # Generate rules
    rules = generate_filter_rules(best)
    
    # Save
    output_path = Path(args.output_dir)
    save_report(results, best, rules, output_path)
    
    # Also save score+zone analysis
    if not score_zones.empty:
        score_zones.to_csv(output_path / "score_zone_combos.csv", index=False)
    
    print("\n" + "=" * 60)
    print("RECOMMENDED FILTER RULES")
    print("=" * 60)
    for rule in rules[:5]:
        print(f"  → {rule['rule']}: {rule['win_rate']} ({rule['confidence']})")
    
    # Summary insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    
    # Best single zone
    single_zones = results[results["zones"] == 1].sort_values("win_rate", ascending=False)
    if not single_zones.empty:
        best_single = single_zones.iloc[0]
        print(f"Best single zone: {best_single['combination']} ({best_single['win_rate']}%)")
    
    # Best two-zone combo
    two_zones = results[results["zones"] == 2].sort_values("win_rate", ascending=False)
    if not two_zones.empty:
        best_two = two_zones.iloc[0]
        print(f"Best 2-zone combo: {best_two['combination']} ({best_two['win_rate']}%)")
    
    # Best three-zone combo
    three_zones = results[results["zones"] == 3].sort_values("win_rate", ascending=False)
    if not three_zones.empty:
        best_three = three_zones.iloc[0]
        print(f"Best 3-zone combo: {best_three['combination']} ({best_three['win_rate']}%)")
    
    return best


if __name__ == "__main__":
    main()
