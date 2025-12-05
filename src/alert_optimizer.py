"""
Alert Parameter Optimizer
Finds optimal configuration for DIP_ALERT, MOMENTUM_ALERT, and PUMP_ALERT.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class AlertConfig:
    """Configuration for an alert type."""
    name: str
    rsi_min: float
    rsi_max: float
    ema_dist_min: float
    ema_dist_max: float
    change_24h_min: float
    change_24h_max: float
    min_score: int


def load_discovery_data(path: str = "results/discovery/discovery_signals.csv") -> pd.DataFrame:
    """Load discovery signals."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} signals")
    return df


def simulate_alert(
    df: pd.DataFrame,
    config: AlertConfig,
) -> Dict[str, Any]:
    """
    Simulate an alert configuration against historical data.
    
    Returns metrics: count, wins, win_rate, avg_rise, etc.
    """
    # Filter by RSI
    mask = (df["rsi_value"] >= config.rsi_min) & (df["rsi_value"] <= config.rsi_max)
    
    # Filter by EMA distance
    mask &= (df["ema20_dist_pct"] >= config.ema_dist_min) & (df["ema20_dist_pct"] <= config.ema_dist_max)
    
    # Filter by 24h change
    mask &= (df["change_24h_pct"] >= config.change_24h_min) & (df["change_24h_pct"] <= config.change_24h_max)
    
    # Filter by score
    mask &= df["score_total"] >= config.min_score
    
    subset = df[mask]
    count = len(subset)
    
    if count == 0:
        return {
            "config": config,
            "count": 0,
            "wins": 0,
            "win_rate_2pct": 0,
            "win_rate_3pct": 0,
            "avg_rise": 0,
            "avg_return": 0,
        }
    
    wins_2pct = subset["win_2pct"].sum()
    wins_3pct = subset["win_3pct"].sum()
    
    return {
        "config": config,
        "count": count,
        "wins": int(wins_2pct),
        "win_rate_2pct": round(wins_2pct / count * 100, 1),
        "win_rate_3pct": round(wins_3pct / count * 100, 1),
        "avg_rise": round(subset["max_rise_pct"].mean(), 2),
        "avg_return": round(subset["final_return_pct"].mean(), 2),
    }


def optimize_dip_alert(df: pd.DataFrame) -> pd.DataFrame:
    """Find optimal DIP_ALERT parameters."""
    
    results = []
    
    # Parameter grid for DIP (oversold + below EMA + dump)
    rsi_max_values = [30, 33, 35, 38, 40]
    ema_dist_max_values = [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]  # Must be BELOW EMA
    change_24h_max_values = [-3, -5, -7, -8, -10]
    min_score_values = [5, 6, 7, 8]
    
    total = len(rsi_max_values) * len(ema_dist_max_values) * len(change_24h_max_values) * len(min_score_values)
    print(f"Testing {total} DIP_ALERT configurations...")
    
    for rsi_max in rsi_max_values:
        for ema_dist_max in ema_dist_max_values:
            for change_24h_max in change_24h_max_values:
                for min_score in min_score_values:
                    config = AlertConfig(
                        name="DIP_ALERT",
                        rsi_min=0,
                        rsi_max=rsi_max,
                        ema_dist_min=-999,  # No lower limit on how far below
                        ema_dist_max=ema_dist_max,  # Must be at least this far below
                        change_24h_min=-999,
                        change_24h_max=change_24h_max,
                        min_score=min_score,
                    )
                    
                    result = simulate_alert(df, config)
                    
                    if result["count"] >= 15:  # Minimum sample
                        results.append({
                            "rsi_max": rsi_max,
                            "ema_dist_max": ema_dist_max,
                            "change_24h_max": change_24h_max,
                            "min_score": min_score,
                            **{k: v for k, v in result.items() if k != "config"}
                        })
    
    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        df_results = df_results.sort_values("win_rate_2pct", ascending=False)
    
    return df_results


def optimize_momentum_alert(df: pd.DataFrame) -> pd.DataFrame:
    """Find optimal MOMENTUM_ALERT parameters."""
    
    results = []
    
    # Parameter grid for MOMENTUM (strong RSI + above EMA + up 24h)
    rsi_min_values = [50, 53, 55, 58, 60]
    rsi_max_values = [62, 65, 68, 70, 72]
    ema_dist_min_values = [0.5, 1.0, 1.5, 2.0]
    ema_dist_max_values = [3.0, 4.0, 5.0, 6.0]
    change_24h_min_values = [1, 2, 3, 4]
    change_24h_max_values = [6, 8, 10, 15]
    min_score_values = [8, 9, 10, 11, 12]
    
    tested = 0
    for rsi_min in rsi_min_values:
        for rsi_max in rsi_max_values:
            if rsi_max <= rsi_min:
                continue
            for ema_dist_min in ema_dist_min_values:
                for ema_dist_max in ema_dist_max_values:
                    if ema_dist_max <= ema_dist_min:
                        continue
                    for change_24h_min in change_24h_min_values:
                        for change_24h_max in change_24h_max_values:
                            if change_24h_max <= change_24h_min:
                                continue
                            for min_score in min_score_values:
                                config = AlertConfig(
                                    name="MOMENTUM_ALERT",
                                    rsi_min=rsi_min,
                                    rsi_max=rsi_max,
                                    ema_dist_min=ema_dist_min,
                                    ema_dist_max=ema_dist_max,
                                    change_24h_min=change_24h_min,
                                    change_24h_max=change_24h_max,
                                    min_score=min_score,
                                )
                                
                                result = simulate_alert(df, config)
                                tested += 1
                                
                                if result["count"] >= 10:
                                    results.append({
                                        "rsi_min": rsi_min,
                                        "rsi_max": rsi_max,
                                        "ema_dist_min": ema_dist_min,
                                        "ema_dist_max": ema_dist_max,
                                        "change_24h_min": change_24h_min,
                                        "change_24h_max": change_24h_max,
                                        "min_score": min_score,
                                        **{k: v for k, v in result.items() if k != "config"}
                                    })
    
    print(f"Tested {tested} MOMENTUM_ALERT configurations...")
    
    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        df_results = df_results.sort_values("win_rate_2pct", ascending=False)
    
    return df_results


def optimize_pump_alert(df: pd.DataFrame) -> pd.DataFrame:
    """Find optimal PUMP_ALERT parameters."""
    
    results = []
    
    # Parameter grid for PUMP (recovery RSI + near EMA + pump 24h)
    rsi_min_values = [30, 33, 35, 38]
    rsi_max_values = [42, 45, 48, 50, 52]
    ema_dist_min_values = [-2.0, -1.5, -1.0, -0.5]
    ema_dist_max_values = [0.5, 1.0, 1.5, 2.0]
    change_24h_min_values = [4, 5, 6, 7, 8]
    min_score_values = [5, 6, 7, 8]
    
    tested = 0
    for rsi_min in rsi_min_values:
        for rsi_max in rsi_max_values:
            if rsi_max <= rsi_min:
                continue
            for ema_dist_min in ema_dist_min_values:
                for ema_dist_max in ema_dist_max_values:
                    for change_24h_min in change_24h_min_values:
                        for min_score in min_score_values:
                            config = AlertConfig(
                                name="PUMP_ALERT",
                                rsi_min=rsi_min,
                                rsi_max=rsi_max,
                                ema_dist_min=ema_dist_min,
                                ema_dist_max=ema_dist_max,
                                change_24h_min=change_24h_min,
                                change_24h_max=999,  # No upper limit
                                min_score=min_score,
                            )
                            
                            result = simulate_alert(df, config)
                            tested += 1
                            
                            if result["count"] >= 10:
                                results.append({
                                    "rsi_min": rsi_min,
                                    "rsi_max": rsi_max,
                                    "ema_dist_min": ema_dist_min,
                                    "ema_dist_max": ema_dist_max,
                                    "change_24h_min": change_24h_min,
                                    "min_score": min_score,
                                    **{k: v for k, v in result.items() if k != "config"}
                                })
    
    print(f"Tested {tested} PUMP_ALERT configurations...")
    
    df_results = pd.DataFrame(results)
    if len(df_results) > 0:
        df_results = df_results.sort_values("win_rate_2pct", ascending=False)
    
    return df_results


def generate_optimization_report(
    dip_results: pd.DataFrame,
    momentum_results: pd.DataFrame,
    pump_results: pd.DataFrame,
    output_dir: Path,
):
    """Generate optimization report."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSVs
    if len(dip_results) > 0:
        dip_results.to_csv(output_dir / "dip_alert_optimization.csv", index=False)
    if len(momentum_results) > 0:
        momentum_results.to_csv(output_dir / "momentum_alert_optimization.csv", index=False)
    if len(pump_results) > 0:
        pump_results.to_csv(output_dir / "pump_alert_optimization.csv", index=False)
    
    # Generate markdown report
    with open(output_dir / "optimization_report.md", "w", encoding="utf-8") as f:
        f.write("# Alert Parameter Optimization Report\n\n")
        
        # DIP_ALERT
        f.write("## 🎯 DIP_ALERT Optimization\n\n")
        
        if len(dip_results) > 0:
            best_dip = dip_results.iloc[0]
            f.write("### Current Config vs Optimal\n\n")
            f.write("| Parameter | Current | Optimal | Change |\n")
            f.write("|-----------|---------|---------|--------|\n")
            f.write(f"| RSI_MAX | 35 | **{best_dip['rsi_max']}** | {'✅' if best_dip['rsi_max'] != 35 else '='} |\n")
            f.write(f"| EMA_DIST_MAX | -1.0 | **{best_dip['ema_dist_max']}** | {'✅' if best_dip['ema_dist_max'] != -1.0 else '='} |\n")
            f.write(f"| 24H_CHANGE_MAX | -5.0 | **{best_dip['change_24h_max']}** | {'✅' if best_dip['change_24h_max'] != -5.0 else '='} |\n")
            f.write(f"| MIN_SCORE | 6 | **{best_dip['min_score']}** | {'✅' if best_dip['min_score'] != 6 else '='} |\n")
            f.write(f"\n**Optimal Win Rate: {best_dip['win_rate_2pct']}%** (n={best_dip['count']})\n\n")
            
            f.write("### Top 10 Configurations\n\n")
            f.write("| RSI Max | EMA Dist Max | 24h Max | Score | Count | Win Rate | Avg Rise |\n")
            f.write("|---------|--------------|---------|-------|-------|----------|----------|\n")
            for _, row in dip_results.head(10).iterrows():
                f.write(f"| {row['rsi_max']} | {row['ema_dist_max']} | {row['change_24h_max']} | {row['min_score']} | {row['count']} | **{row['win_rate_2pct']}%** | {row['avg_rise']}% |\n")
        else:
            f.write("No valid configurations found.\n")
        
        f.write("\n---\n\n")
        
        # MOMENTUM_ALERT
        f.write("## 🚀 MOMENTUM_ALERT Optimization\n\n")
        
        if len(momentum_results) > 0:
            best_mom = momentum_results.iloc[0]
            f.write("### Current Config vs Optimal\n\n")
            f.write("| Parameter | Current | Optimal | Change |\n")
            f.write("|-----------|---------|---------|--------|\n")
            f.write(f"| RSI_MIN | 55 | **{best_mom['rsi_min']}** | {'✅' if best_mom['rsi_min'] != 55 else '='} |\n")
            f.write(f"| RSI_MAX | 65 | **{best_mom['rsi_max']}** | {'✅' if best_mom['rsi_max'] != 65 else '='} |\n")
            f.write(f"| EMA_DIST_MIN | 1.0 | **{best_mom['ema_dist_min']}** | {'✅' if best_mom['ema_dist_min'] != 1.0 else '='} |\n")
            f.write(f"| EMA_DIST_MAX | 3.0 | **{best_mom['ema_dist_max']}** | {'✅' if best_mom['ema_dist_max'] != 3.0 else '='} |\n")
            f.write(f"| 24H_CHANGE_MIN | 2.0 | **{best_mom['change_24h_min']}** | {'✅' if best_mom['change_24h_min'] != 2.0 else '='} |\n")
            f.write(f"| 24H_CHANGE_MAX | 8.0 | **{best_mom['change_24h_max']}** | {'✅' if best_mom['change_24h_max'] != 8.0 else '='} |\n")
            f.write(f"| MIN_SCORE | 10 | **{best_mom['min_score']}** | {'✅' if best_mom['min_score'] != 10 else '='} |\n")
            f.write(f"\n**Optimal Win Rate: {best_mom['win_rate_2pct']}%** (n={best_mom['count']})\n\n")
            
            f.write("### Top 10 Configurations\n\n")
            f.write("| RSI Range | EMA Dist | 24h Range | Score | Count | Win Rate | Avg Rise |\n")
            f.write("|-----------|----------|-----------|-------|-------|----------|----------|\n")
            for _, row in momentum_results.head(10).iterrows():
                f.write(f"| {row['rsi_min']}-{row['rsi_max']} | {row['ema_dist_min']}-{row['ema_dist_max']} | {row['change_24h_min']}-{row['change_24h_max']} | {row['min_score']} | {row['count']} | **{row['win_rate_2pct']}%** | {row['avg_rise']}% |\n")
        else:
            f.write("No valid configurations found.\n")
        
        f.write("\n---\n\n")
        
        # PUMP_ALERT
        f.write("## 📈 PUMP_ALERT Optimization\n\n")
        
        if len(pump_results) > 0:
            best_pump = pump_results.iloc[0]
            f.write("### Current Config vs Optimal\n\n")
            f.write("| Parameter | Current | Optimal | Change |\n")
            f.write("|-----------|---------|---------|--------|\n")
            f.write(f"| RSI_MIN | 35 | **{best_pump['rsi_min']}** | {'✅' if best_pump['rsi_min'] != 35 else '='} |\n")
            f.write(f"| RSI_MAX | 45 | **{best_pump['rsi_max']}** | {'✅' if best_pump['rsi_max'] != 45 else '='} |\n")
            f.write(f"| EMA_DIST_MIN | -1.0 | **{best_pump['ema_dist_min']}** | {'✅' if best_pump['ema_dist_min'] != -1.0 else '='} |\n")
            f.write(f"| EMA_DIST_MAX | 1.0 | **{best_pump['ema_dist_max']}** | {'✅' if best_pump['ema_dist_max'] != 1.0 else '='} |\n")
            f.write(f"| 24H_CHANGE_MIN | 5.0 | **{best_pump['change_24h_min']}** | {'✅' if best_pump['change_24h_min'] != 5.0 else '='} |\n")
            f.write(f"| MIN_SCORE | 7 | **{best_pump['min_score']}** | {'✅' if best_pump['min_score'] != 7 else '='} |\n")
            f.write(f"\n**Optimal Win Rate: {best_pump['win_rate_2pct']}%** (n={best_pump['count']})\n\n")
            
            f.write("### Top 10 Configurations\n\n")
            f.write("| RSI Range | EMA Dist | 24h Min | Score | Count | Win Rate | Avg Rise |\n")
            f.write("|-----------|----------|---------|-------|-------|----------|----------|\n")
            for _, row in pump_results.head(10).iterrows():
                f.write(f"| {row['rsi_min']}-{row['rsi_max']} | {row['ema_dist_min']} to {row['ema_dist_max']} | {row['change_24h_min']} | {row['min_score']} | {row['count']} | **{row['win_rate_2pct']}%** | {row['avg_rise']}% |\n")
        else:
            f.write("No valid configurations found.\n")
        
        f.write("\n---\n\n")
        
        # Summary
        f.write("## 📊 Summary & Recommendations\n\n")
        f.write("| Alert | Current WR | Optimal WR | Improvement | Sample Size |\n")
        f.write("|-------|------------|------------|-------------|-------------|\n")
        
        if len(dip_results) > 0:
            best_dip = dip_results.iloc[0]
            improvement = best_dip['win_rate_2pct'] - 68.0
            f.write(f"| 🎯 DIP_ALERT | ~68% | **{best_dip['win_rate_2pct']}%** | {improvement:+.1f}% | n={best_dip['count']} |\n")
        
        if len(momentum_results) > 0:
            best_mom = momentum_results.iloc[0]
            improvement = best_mom['win_rate_2pct'] - 63.0
            f.write(f"| 🚀 MOMENTUM_ALERT | ~63% | **{best_mom['win_rate_2pct']}%** | {improvement:+.1f}% | n={best_mom['count']} |\n")
        
        if len(pump_results) > 0:
            best_pump = pump_results.iloc[0]
            improvement = best_pump['win_rate_2pct'] - 60.0
            f.write(f"| 📈 PUMP_ALERT | ~60% | **{best_pump['win_rate_2pct']}%** | {improvement:+.1f}% | n={best_pump['count']} |\n")
        
        f.write("\n---\n\n")
        
        # Config code generation
        f.write("## 💻 Optimal Config Code\n\n")
        f.write("Copy-paste ready for `config.py`:\n\n")
        f.write("```python\n")
        
        if len(dip_results) > 0:
            best_dip = dip_results.iloc[0]
            f.write("# === DIP_ALERT OPTIMIZED (V7.1) ===\n")
            f.write(f"DIP_RSI_MAX = {best_dip['rsi_max']}\n")
            f.write(f"DIP_EMA_DIST_MIN_PCT = {best_dip['ema_dist_max']}  # Must be below this\n")
            f.write(f"DIP_24H_CHANGE_MAX = {best_dip['change_24h_max']}\n")
            f.write(f"DIP_MIN_SCORE = {best_dip['min_score']}\n")
            f.write("\n")
        
        if len(momentum_results) > 0:
            best_mom = momentum_results.iloc[0]
            f.write("# === MOMENTUM_ALERT OPTIMIZED (V7.1) ===\n")
            f.write(f"MOMENTUM_RSI_MIN = {best_mom['rsi_min']}\n")
            f.write(f"MOMENTUM_RSI_MAX = {best_mom['rsi_max']}\n")
            f.write(f"MOMENTUM_EMA_DIST_MIN_PCT = {best_mom['ema_dist_min']}\n")
            f.write(f"MOMENTUM_EMA_DIST_MAX_PCT = {best_mom['ema_dist_max']}\n")
            f.write(f"MOMENTUM_24H_CHANGE_MIN = {best_mom['change_24h_min']}\n")
            f.write(f"MOMENTUM_24H_CHANGE_MAX = {best_mom['change_24h_max']}\n")
            f.write(f"MOMENTUM_MIN_SCORE = {best_mom['min_score']}\n")
            f.write("\n")
        
        if len(pump_results) > 0:
            best_pump = pump_results.iloc[0]
            f.write("# === PUMP_ALERT OPTIMIZED (V7.1) ===\n")
            f.write(f"PUMP_RSI_MIN = {best_pump['rsi_min']}\n")
            f.write(f"PUMP_RSI_MAX = {best_pump['rsi_max']}\n")
            f.write(f"PUMP_EMA_DIST_MIN_PCT = {best_pump['ema_dist_min']}\n")
            f.write(f"PUMP_EMA_DIST_MAX_PCT = {best_pump['ema_dist_max']}\n")
            f.write(f"PUMP_24H_CHANGE_MIN = {best_pump['change_24h_min']}\n")
            f.write(f"PUMP_MIN_SCORE = {best_pump['min_score']}\n")
        
        f.write("```\n")
    
    print(f"\n✅ Report saved to: {output_dir}/optimization_report.md")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Alert Parameter Optimizer")
    parser.add_argument("--input", default="results/discovery/discovery_signals.csv")
    parser.add_argument("--output-dir", default="results/alert_optimization")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ALERT PARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Load data
    df = load_discovery_data(args.input)
    
    # Check required columns
    required_cols = ["rsi_value", "ema20_dist_pct", "change_24h_pct", "score_total", "win_2pct", "win_3pct", "max_rise_pct", "final_return_pct"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"⚠️ Missing columns: {missing}")
        print("Available columns:", df.columns.tolist())
        return
    
    # Optimize each alert type
    print("\n" + "=" * 60)
    print("OPTIMIZING DIP_ALERT")
    print("=" * 60)
    dip_results = optimize_dip_alert(df)
    if len(dip_results) > 0:
        print(f"\nTop 5 DIP_ALERT configs:")
        print(dip_results.head(5).to_string(index=False))
    else:
        print("No valid DIP_ALERT configurations found.")
    
    print("\n" + "=" * 60)
    print("OPTIMIZING MOMENTUM_ALERT")
    print("=" * 60)
    momentum_results = optimize_momentum_alert(df)
    if len(momentum_results) > 0:
        print(f"\nTop 5 MOMENTUM_ALERT configs:")
        print(momentum_results.head(5).to_string(index=False))
    else:
        print("No valid MOMENTUM_ALERT configurations found.")
    
    print("\n" + "=" * 60)
    print("OPTIMIZING PUMP_ALERT")
    print("=" * 60)
    pump_results = optimize_pump_alert(df)
    if len(pump_results) > 0:
        print(f"\nTop 5 PUMP_ALERT configs:")
        print(pump_results.head(5).to_string(index=False))
    else:
        print("No valid PUMP_ALERT configurations found.")
    
    # Generate report
    output_path = Path(args.output_dir)
    generate_optimization_report(dip_results, momentum_results, pump_results, output_path)
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
