import pandas as pd
import os

files = [
    ('BTCUSDT', 'data/BTCUSDT_1m.parquet', 'data/precomputed_15m/BTCUSDT_15m_features.parquet', 'data/precomputed_1h/BTCUSDT_1h_features.parquet'),
    ('ETHUSDT', 'data/ETHUSDT_1m.parquet', 'data/precomputed_15m/ETHUSDT_15m_features.parquet', 'data/precomputed_1h/ETHUSDT_1h_features.parquet'),
    ('SOLUSDT', 'data/SOLUSDT_1m.parquet', 'data/precomputed_15m/SOLUSDT_15m_features.parquet', 'data/precomputed_1h/SOLUSDT_1h_features.parquet'),
    ('XRPUSDT', 'data/XRPUSDT_1m.parquet', 'data/precomputed_15m/XRPUSDT_15m_features.parquet', 'data/precomputed_1h/XRPUSDT_1h_features.parquet'),
]

print("=" * 70)
print("PRECOMPUTED DATA ANALYSIS - BTC, ETH, SOL, XRP")
print("=" * 70)

for symbol, raw_1m, feat_15m, feat_1h in files:
    print(f'\n{"="*50}')
    print(f'{symbol}')
    print(f'{"="*50}')
    
    # 1m raw
    if os.path.exists(raw_1m):
        df = pd.read_parquet(raw_1m)
        print(f'\n1m Raw Data:')
        print(f'  Rows: {df.shape[0]:,} | Cols: {df.shape[1]}')
        if 'timestamp' in df.columns:
            print(f'  Date Range: {df["timestamp"].min()} -> {df["timestamp"].max()}')
        elif df.index.name == 'timestamp':
            print(f'  Date Range: {df.index.min()} -> {df.index.max()}')
        print(f'  Columns: {list(df.columns[:8])}')
    
    # 15m features
    if os.path.exists(feat_15m):
        df15 = pd.read_parquet(feat_15m)
        print(f'\n15m Precomputed Features:')
        print(f'  Rows: {df15.shape[0]:,} | Cols: {df15.shape[1]}')
        if 'timestamp' in df15.columns:
            print(f'  Date Range: {df15["timestamp"].min()} -> {df15["timestamp"].max()}')
        elif df15.index.name == 'timestamp':
            print(f'  Date Range: {df15.index.min()} -> {df15.index.max()}')
        print(f'  Sample Columns: {list(df15.columns[:15])}')
    
    # 1h features
    if os.path.exists(feat_1h):
        df1h = pd.read_parquet(feat_1h)
        print(f'\n1h Precomputed Features:')
        print(f'  Rows: {df1h.shape[0]:,} | Cols: {df1h.shape[1]}')
        if 'timestamp' in df1h.columns:
            print(f'  Date Range: {df1h["timestamp"].min()} -> {df1h["timestamp"].max()}')
        elif df1h.index.name == 'timestamp':
            print(f'  Date Range: {df1h.index.min()} -> {df1h.index.max()}')
        print(f'  Sample Columns: {list(df1h.columns[:15])}')

# Show full column list for one feature file
print("\n" + "=" * 70)
print("FULL COLUMN LIST (15m Features Example - BTCUSDT)")
print("=" * 70)
if os.path.exists('data/precomputed_15m/BTCUSDT_15m_features.parquet'):
    df_full = pd.read_parquet('data/precomputed_15m/BTCUSDT_15m_features.parquet')
    cols = list(df_full.columns)
    for i, col in enumerate(cols, 1):
        print(f'{i:3}. {col}')
    
# Count total symbols
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
raw_count = len([f for f in os.listdir('data') if f.endswith('_1m.parquet')])
feat15_count = len([f for f in os.listdir('data/precomputed_15m') if f.endswith('_features.parquet')])
feat1h_count = len([f for f in os.listdir('data/precomputed_1h') if f.endswith('_features.parquet')])
print(f'Total 1m Raw Files: {raw_count}')
print(f'Total 15m Feature Files: {feat15_count}')
print(f'Total 1h Feature Files: {feat1h_count}')
