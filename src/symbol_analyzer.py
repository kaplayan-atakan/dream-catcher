"""
Symbol Analyzer - Find all symbols with complete data for backtesting.

Analyzes precomputed 15m feature files to identify symbols with:
- At least 16,000 rows (sufficient history)
- At least 30 columns (complete feature set)
"""

import os
import pandas as pd
from pathlib import Path
from datetime import datetime


def analyze_symbols(data_dir: str = "data/precomputed_15m") -> pd.DataFrame:
    """
    Analyze all symbols and return those with complete data.
    
    Args:
        data_dir: Directory containing precomputed parquet files
        
    Returns:
        DataFrame with symbol analysis results
    """
    results = []
    parquet_files = list(Path(data_dir).glob("*_15m_features.parquet"))
    
    print(f"Found {len(parquet_files)} symbol files in {data_dir}")
    
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            symbol = f.stem.replace("_15m_features", "")
            
            # Determine date range
            if hasattr(df.index, 'min') and df.index.dtype != 'int64':
                start_date = df.index.min()
                end_date = df.index.max()
            elif 'timestamp' in df.columns:
                start_date = df['timestamp'].min()
                end_date = df['timestamp'].max()
            elif 'open_time' in df.columns:
                start_date = df['open_time'].min()
                end_date = df['open_time'].max()
            else:
                start_date = None
                end_date = None
            
            # Check for key indicators
            has_rsi = any('rsi' in c.lower() for c in df.columns)
            has_ema = any('ema' in c.lower() for c in df.columns)
            has_macd = any('macd' in c.lower() for c in df.columns)
            has_volume = any('volume' in c.lower() for c in df.columns)
            
            results.append({
                "symbol": symbol,
                "rows": len(df),
                "cols": len(df.columns),
                "start_date": start_date,
                "end_date": end_date,
                "has_rsi": has_rsi,
                "has_ema": has_ema,
                "has_macd": has_macd,
                "has_volume": has_volume,
                "complete": len(df) >= 16000 and len(df.columns) >= 30,
            })
        except Exception as e:
            print(f"  ⚠️ Error reading {f.name}: {e}")
    
    df_results = pd.DataFrame(results)
    
    if len(df_results) == 0:
        print("\n❌ No symbol files found!")
        return df_results
    
    # Sort by rows descending
    df_results = df_results.sort_values("rows", ascending=False).reset_index(drop=True)
    
    # Filter complete symbols
    complete = df_results[df_results["complete"]]
    incomplete = df_results[~df_results["complete"]]
    
    print(f"\n{'='*60}")
    print("SYMBOL ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total symbols analyzed: {len(df_results)}")
    print(f"Complete symbols (≥16k rows, ≥30 cols): {len(complete)}")
    print(f"Incomplete symbols: {len(incomplete)}")
    
    print(f"\n📊 Row Distribution:")
    print(df_results["rows"].describe().to_string())
    
    print(f"\n📊 Column Distribution:")
    print(df_results["cols"].describe().to_string())
    
    # Feature coverage
    print(f"\n📊 Feature Coverage:")
    print(f"  RSI: {df_results['has_rsi'].sum()}/{len(df_results)} ({100*df_results['has_rsi'].mean():.1f}%)")
    print(f"  EMA: {df_results['has_ema'].sum()}/{len(df_results)} ({100*df_results['has_ema'].mean():.1f}%)")
    print(f"  MACD: {df_results['has_macd'].sum()}/{len(df_results)} ({100*df_results['has_macd'].mean():.1f}%)")
    print(f"  Volume: {df_results['has_volume'].sum()}/{len(df_results)} ({100*df_results['has_volume'].mean():.1f}%)")
    
    if len(complete) > 0:
        print(f"\n✅ COMPLETE SYMBOLS ({len(complete)}):")
        print("-" * 60)
        for i, row in complete.head(30).iterrows():
            date_range = ""
            if row['start_date'] is not None and row['end_date'] is not None:
                try:
                    start_str = pd.Timestamp(row['start_date']).strftime('%Y-%m-%d')
                    end_str = pd.Timestamp(row['end_date']).strftime('%Y-%m-%d')
                    date_range = f" | {start_str} → {end_str}"
                except:
                    pass
            print(f"  {row['symbol']:15s} | {row['rows']:,} rows | {row['cols']} cols{date_range}")
        
        if len(complete) > 30:
            print(f"  ... and {len(complete) - 30} more")
    
    if len(incomplete) > 0 and len(incomplete) <= 20:
        print(f"\n⚠️ INCOMPLETE SYMBOLS ({len(incomplete)}):")
        for i, row in incomplete.iterrows():
            reason = []
            if row['rows'] < 16000:
                reason.append(f"rows={row['rows']}")
            if row['cols'] < 30:
                reason.append(f"cols={row['cols']}")
            print(f"  {row['symbol']:15s} | {', '.join(reason)}")
    
    return df_results


def analyze_1h_symbols(data_dir: str = "data/precomputed_1h") -> pd.DataFrame:
    """Analyze 1h feature files for HTF data availability."""
    
    results = []
    parquet_files = list(Path(data_dir).glob("*_1h_features.parquet"))
    
    print(f"\nFound {len(parquet_files)} 1h symbol files in {data_dir}")
    
    for f in parquet_files:
        try:
            df = pd.read_parquet(f)
            symbol = f.stem.replace("_1h_features", "")
            results.append({
                "symbol": symbol,
                "rows_1h": len(df),
                "cols_1h": len(df.columns),
            })
        except Exception as e:
            print(f"  ⚠️ Error reading {f.name}: {e}")
    
    return pd.DataFrame(results)


def get_backtest_ready_symbols(
    data_dir_15m: str = "data/precomputed_15m",
    data_dir_1h: str = "data/precomputed_1h",
    min_rows_15m: int = 16000,
    min_cols: int = 30
) -> list:
    """
    Get list of symbols ready for backtesting (have both 15m and 1h data).
    
    Returns:
        List of symbol names ready for backtest
    """
    # Analyze 15m
    df_15m = analyze_symbols(data_dir_15m)
    complete_15m = set(df_15m[df_15m["complete"]]["symbol"].tolist())
    
    # Analyze 1h
    df_1h = analyze_1h_symbols(data_dir_1h)
    symbols_1h = set(df_1h["symbol"].tolist()) if len(df_1h) > 0 else set()
    
    # Intersection
    if len(symbols_1h) > 0:
        ready_symbols = sorted(complete_15m & symbols_1h)
        print(f"\n{'='*60}")
        print(f"BACKTEST-READY SYMBOLS")
        print(f"{'='*60}")
        print(f"Complete 15m data: {len(complete_15m)}")
        print(f"Have 1h data: {len(symbols_1h)}")
        print(f"Both (ready for backtest): {len(ready_symbols)}")
    else:
        ready_symbols = sorted(complete_15m)
        print(f"\n⚠️ No 1h data found, using 15m-only symbols: {len(ready_symbols)}")
    
    return ready_symbols


def save_symbol_list(symbols: list, output_path: str = "data/backtest_symbols.txt"):
    """Save symbol list to file for use by backtester."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in symbols:
            f.write(f"{s}\n")
    print(f"\n💾 Saved {len(symbols)} symbols to {output_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("SYMBOL ANALYZER FOR VECTORBT BACKTESTING")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Get backtest-ready symbols
    ready_symbols = get_backtest_ready_symbols()
    
    if len(ready_symbols) > 0:
        # Save to file
        save_symbol_list(ready_symbols)
        
        print(f"\n{'='*60}")
        print("READY FOR BACKTEST")
        print(f"{'='*60}")
        print(f"Total symbols: {len(ready_symbols)}")
        print(f"\nFirst 20: {ready_symbols[:20]}")
    else:
        print("\n❌ No symbols ready for backtesting!")
        print("Check that data/precomputed_15m/ contains parquet files.")
