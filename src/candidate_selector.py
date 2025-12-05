"""
Candidate Selector for Alert Parameter Optimization
Selects top 10 diverse candidates from each alert type for comprehensive backtesting.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_optimization_results(csv_path: str, alert_name: str) -> pd.DataFrame:
    """Analyze optimization results and select top candidates."""
    
    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"{alert_name} ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nTotal configurations tested: {len(df)}")
    
    # Basic stats
    print(f"\nWin Rate Distribution:")
    print(f"  Min: {df['win_rate_2pct'].min():.1f}%")
    print(f"  Max: {df['win_rate_2pct'].max():.1f}%")
    print(f"  Mean: {df['win_rate_2pct'].mean():.1f}%")
    print(f"  Median: {df['win_rate_2pct'].median():.1f}%")
    
    print(f"\nSample Size Distribution:")
    print(f"  Min: {df['count'].min()}")
    print(f"  Max: {df['count'].max()}")
    print(f"  Mean: {df['count'].mean():.1f}")
    
    # Score candidates based on multiple criteria
    df["quality_score"] = (
        df["win_rate_2pct"] * 0.4 +           # 40% weight on win rate
        df["count"].clip(upper=100) * 0.3 +   # 30% weight on sample size (capped)
        df["avg_rise"] * 10 * 0.2 +           # 20% weight on avg rise
        df["win_rate_3pct"] * 0.1             # 10% weight on 3% win rate
    )
    
    # Categorize by sample size
    df["size_category"] = pd.cut(
        df["count"], 
        bins=[0, 15, 25, 40, 100, 1000],
        labels=["tiny", "small", "medium", "large", "xlarge"]
    )
    
    return df


def select_top_candidates(
    df: pd.DataFrame, 
    alert_name: str,
    n_select: int = 10,
) -> pd.DataFrame:
    """Select top N candidates with diversity."""
    
    print(f"\n{'='*60}")
    print(f"SELECTING TOP {n_select} {alert_name} CANDIDATES")
    print(f"{'='*60}")
    
    # Filter: minimum quality thresholds
    filtered = df[
        (df["win_rate_2pct"] >= 60) &
        (df["count"] >= 15) &
        (df["avg_rise"] >= 1.5)
    ].copy()
    
    print(f"\nAfter filtering (WR≥60%, n≥15, rise≥1.5%): {len(filtered)} configs")
    
    if len(filtered) == 0:
        print("⚠️ No configs meet strict criteria, relaxing...")
        filtered = df[
            (df["win_rate_2pct"] >= 50) &
            (df["count"] >= 10)
        ].copy()
        print(f"After relaxed filtering: {len(filtered)} configs")
    
    if len(filtered) == 0:
        print("⚠️ Still no configs, using all available")
        filtered = df.copy()
    
    # Sort by quality score
    filtered = filtered.sort_values("quality_score", ascending=False)
    
    # Select with diversity: try to get different parameter values
    selected = []
    seen_combos = set()
    
    for _, row in filtered.iterrows():
        if len(selected) >= n_select:
            break
        
        # Create a "signature" to avoid too similar configs
        if alert_name == "DIP_ALERT":
            signature = (
                round(row.get("rsi_max", 0) / 5) * 5,
                round(row.get("ema_dist_max", 0)),
                round(row.get("change_24h_max", 0) / 3) * 3,
            )
        elif alert_name == "MOMENTUM_ALERT":
            signature = (
                round(row.get("rsi_min", 0) / 5) * 5,
                round(row.get("min_score", 0)),
                round(row.get("change_24h_min", 0) / 2) * 2,
            )
        else:  # PUMP_ALERT
            signature = (
                round(row.get("rsi_min", 0) / 5) * 5,
                round(row.get("rsi_max", 0) / 5) * 5,
                round(row.get("change_24h_min", 0)),
            )
        
        # Allow some duplicates but prefer diversity
        if signature not in seen_combos or len(selected) < 5:
            selected.append(row)
            seen_combos.add(signature)
    
    result = pd.DataFrame(selected)
    
    print(f"\nSelected {len(result)} diverse candidates")
    
    return result


def create_selection_report(
    dip_selected: pd.DataFrame,
    momentum_selected: pd.DataFrame,
    pump_selected: pd.DataFrame,
    output_dir: Path,
):
    """Create markdown report with selected candidates."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "selected_candidates_report.md", "w", encoding="utf-8") as f:
        f.write("# Selected Candidates for Comprehensive Backtest\n\n")
        f.write("These candidates were selected based on:\n")
        f.write("- Win Rate ≥ 60% (or ≥50% if insufficient)\n")
        f.write("- Sample Size ≥ 15 (or ≥10 if insufficient)\n")
        f.write("- Avg Rise ≥ 1.5%\n")
        f.write("- Parameter diversity\n\n")
        
        f.write("---\n\n")
        
        # DIP_ALERT
        f.write("## 🎯 DIP_ALERT Candidates\n\n")
        if len(dip_selected) > 0:
            f.write("| # | RSI Max | EMA Dist | 24h Max | Score | Count | Win Rate | Avg Rise | Quality |\n")
            f.write("|---|---------|----------|---------|-------|-------|----------|----------|--------|\n")
            for idx, (_, row) in enumerate(dip_selected.iterrows(), 1):
                f.write(f"| {idx} | ")
                f.write(f"{row.get('rsi_max', 'N/A')} | ")
                f.write(f"{row.get('ema_dist_max', 'N/A')} | ")
                f.write(f"{row.get('change_24h_max', 'N/A')} | ")
                f.write(f"{row.get('min_score', 'N/A')} | ")
                f.write(f"{row['count']:.0f} | ")
                f.write(f"**{row['win_rate_2pct']:.1f}%** | ")
                f.write(f"{row['avg_rise']:.2f}% | ")
                f.write(f"{row['quality_score']:.1f} |\n")
        else:
            f.write("No candidates selected.\n")
        
        f.write("\n---\n\n")
        
        # MOMENTUM_ALERT
        f.write("## 🚀 MOMENTUM_ALERT Candidates\n\n")
        if len(momentum_selected) > 0:
            f.write("| # | RSI Range | EMA Dist | 24h Range | Score | Count | Win Rate | Avg Rise | Quality |\n")
            f.write("|---|-----------|----------|-----------|-------|-------|----------|----------|--------|\n")
            for idx, (_, row) in enumerate(momentum_selected.iterrows(), 1):
                f.write(f"| {idx} | ")
                f.write(f"{row.get('rsi_min', 'N/A')}-{row.get('rsi_max', 'N/A')} | ")
                f.write(f"{row.get('ema_dist_min', 'N/A')}-{row.get('ema_dist_max', 'N/A')} | ")
                f.write(f"{row.get('change_24h_min', 'N/A')}-{row.get('change_24h_max', 'N/A')} | ")
                f.write(f"{row.get('min_score', 'N/A')} | ")
                f.write(f"{row['count']:.0f} | ")
                f.write(f"**{row['win_rate_2pct']:.1f}%** | ")
                f.write(f"{row['avg_rise']:.2f}% | ")
                f.write(f"{row['quality_score']:.1f} |\n")
        else:
            f.write("No candidates selected.\n")
        
        f.write("\n---\n\n")
        
        # PUMP_ALERT
        f.write("## 📈 PUMP_ALERT Candidates\n\n")
        if len(pump_selected) > 0:
            f.write("| # | RSI Range | EMA Dist | 24h Min | Score | Count | Win Rate | Avg Rise | Quality |\n")
            f.write("|---|-----------|----------|---------|-------|-------|----------|----------|--------|\n")
            for idx, (_, row) in enumerate(pump_selected.iterrows(), 1):
                f.write(f"| {idx} | ")
                f.write(f"{row.get('rsi_min', 'N/A')}-{row.get('rsi_max', 'N/A')} | ")
                f.write(f"{row.get('ema_dist_min', 'N/A')}-{row.get('ema_dist_max', 'N/A')} | ")
                f.write(f"{row.get('change_24h_min', 'N/A')} | ")
                f.write(f"{row.get('min_score', 'N/A')} | ")
                f.write(f"{row['count']:.0f} | ")
                f.write(f"**{row['win_rate_2pct']:.1f}%** | ")
                f.write(f"{row['avg_rise']:.2f}% | ")
                f.write(f"{row['quality_score']:.1f} |\n")
        else:
            f.write("No candidates selected.\n")
        
        f.write("\n---\n\n")
        
        # Summary stats
        f.write("## 📊 Selection Summary\n\n")
        f.write("| Alert Type | Candidates | Avg Win Rate | Avg Sample | Best WR | Best Sample |\n")
        f.write("|------------|------------|--------------|------------|---------|-------------|\n")
        
        for name, selected_df in [("🎯 DIP_ALERT", dip_selected), ("🚀 MOMENTUM_ALERT", momentum_selected), ("📈 PUMP_ALERT", pump_selected)]:
            if len(selected_df) > 0:
                f.write(f"| {name} | {len(selected_df)} | {selected_df['win_rate_2pct'].mean():.1f}% | {selected_df['count'].mean():.0f} | {selected_df['win_rate_2pct'].max():.1f}% | {selected_df['count'].max():.0f} |\n")
            else:
                f.write(f"| {name} | 0 | - | - | - | - |\n")
        
        f.write("\n---\n\n")
        
        # Recommended configs
        f.write("## 🏆 Recommended Final Configs\n\n")
        f.write("Based on balance of win rate and sample size:\n\n")
        
        if len(dip_selected) > 0:
            # Find best balance (high WR with decent sample)
            dip_balanced = dip_selected[dip_selected['count'] >= 20].head(1)
            if len(dip_balanced) == 0:
                dip_balanced = dip_selected.head(1)
            best = dip_balanced.iloc[0]
            f.write(f"### 🎯 DIP_ALERT\n")
            f.write(f"- RSI_MAX: **{best.get('rsi_max', 'N/A')}**\n")
            f.write(f"- EMA_DIST_MAX: **{best.get('ema_dist_max', 'N/A')}**\n")
            f.write(f"- 24H_CHANGE_MAX: **{best.get('change_24h_max', 'N/A')}**\n")
            f.write(f"- MIN_SCORE: **{best.get('min_score', 'N/A')}**\n")
            f.write(f"- Expected Win Rate: **{best['win_rate_2pct']:.1f}%** (n={best['count']:.0f})\n\n")
        
        if len(momentum_selected) > 0:
            mom_balanced = momentum_selected[momentum_selected['count'] >= 15].head(1)
            if len(mom_balanced) == 0:
                mom_balanced = momentum_selected.head(1)
            best = mom_balanced.iloc[0]
            f.write(f"### 🚀 MOMENTUM_ALERT\n")
            f.write(f"- RSI_MIN: **{best.get('rsi_min', 'N/A')}**\n")
            f.write(f"- RSI_MAX: **{best.get('rsi_max', 'N/A')}**\n")
            f.write(f"- EMA_DIST_MIN: **{best.get('ema_dist_min', 'N/A')}**\n")
            f.write(f"- EMA_DIST_MAX: **{best.get('ema_dist_max', 'N/A')}**\n")
            f.write(f"- 24H_CHANGE_MIN: **{best.get('change_24h_min', 'N/A')}**\n")
            f.write(f"- 24H_CHANGE_MAX: **{best.get('change_24h_max', 'N/A')}**\n")
            f.write(f"- MIN_SCORE: **{best.get('min_score', 'N/A')}**\n")
            f.write(f"- Expected Win Rate: **{best['win_rate_2pct']:.1f}%** (n={best['count']:.0f})\n\n")
        
        if len(pump_selected) > 0:
            pump_balanced = pump_selected[pump_selected['count'] >= 15].head(1)
            if len(pump_balanced) == 0:
                pump_balanced = pump_selected.head(1)
            best = pump_balanced.iloc[0]
            f.write(f"### 📈 PUMP_ALERT\n")
            f.write(f"- RSI_MIN: **{best.get('rsi_min', 'N/A')}**\n")
            f.write(f"- RSI_MAX: **{best.get('rsi_max', 'N/A')}**\n")
            f.write(f"- EMA_DIST_MIN: **{best.get('ema_dist_min', 'N/A')}**\n")
            f.write(f"- EMA_DIST_MAX: **{best.get('ema_dist_max', 'N/A')}**\n")
            f.write(f"- 24H_CHANGE_MIN: **{best.get('change_24h_min', 'N/A')}**\n")
            f.write(f"- MIN_SCORE: **{best.get('min_score', 'N/A')}**\n")
            f.write(f"- Expected Win Rate: **{best['win_rate_2pct']:.1f}%** (n={best['count']:.0f})\n\n")
    
    # Save selected candidates as CSVs
    if len(dip_selected) > 0:
        dip_selected.to_csv(output_dir / "dip_selected_candidates.csv", index=False)
    if len(momentum_selected) > 0:
        momentum_selected.to_csv(output_dir / "momentum_selected_candidates.csv", index=False)
    if len(pump_selected) > 0:
        pump_selected.to_csv(output_dir / "pump_selected_candidates.csv", index=False)
    
    print(f"\n✅ Report saved to: {output_dir}/selected_candidates_report.md")


def main():
    base_path = Path("results/alert_optimization")
    
    # Analyze each alert type
    dip_df = analyze_optimization_results(
        base_path / "dip_alert_optimization.csv",
        "DIP_ALERT"
    )
    
    momentum_df = analyze_optimization_results(
        base_path / "momentum_alert_optimization.csv",
        "MOMENTUM_ALERT"
    )
    
    pump_df = analyze_optimization_results(
        base_path / "pump_alert_optimization.csv",
        "PUMP_ALERT"
    )
    
    # Select top candidates
    dip_selected = select_top_candidates(dip_df, "DIP_ALERT", n_select=10)
    momentum_selected = select_top_candidates(momentum_df, "MOMENTUM_ALERT", n_select=10)
    pump_selected = select_top_candidates(pump_df, "PUMP_ALERT", n_select=10)
    
    # Display selected candidates
    print("\n" + "="*80)
    print("SELECTED DIP_ALERT CANDIDATES")
    print("="*80)
    display_cols = ["rsi_max", "ema_dist_max", "change_24h_max", "min_score", "count", "win_rate_2pct", "avg_rise", "quality_score"]
    available_cols = [c for c in display_cols if c in dip_selected.columns]
    if len(available_cols) > 0 and len(dip_selected) > 0:
        print(dip_selected[available_cols].to_string(index=False))
    else:
        print("No candidates")
    
    print("\n" + "="*80)
    print("SELECTED MOMENTUM_ALERT CANDIDATES")
    print("="*80)
    display_cols = ["rsi_min", "rsi_max", "ema_dist_min", "ema_dist_max", "change_24h_min", "min_score", "count", "win_rate_2pct", "avg_rise", "quality_score"]
    available_cols = [c for c in display_cols if c in momentum_selected.columns]
    if len(available_cols) > 0 and len(momentum_selected) > 0:
        print(momentum_selected[available_cols].to_string(index=False))
    else:
        print("No candidates")
    
    print("\n" + "="*80)
    print("SELECTED PUMP_ALERT CANDIDATES")
    print("="*80)
    display_cols = ["rsi_min", "rsi_max", "ema_dist_min", "ema_dist_max", "change_24h_min", "min_score", "count", "win_rate_2pct", "avg_rise", "quality_score"]
    available_cols = [c for c in display_cols if c in pump_selected.columns]
    if len(available_cols) > 0 and len(pump_selected) > 0:
        print(pump_selected[available_cols].to_string(index=False))
    else:
        print("No candidates")
    
    # Create report
    create_selection_report(
        dip_selected,
        momentum_selected,
        pump_selected,
        base_path,
    )
    
    return dip_selected, momentum_selected, pump_selected


if __name__ == "__main__":
    main()
