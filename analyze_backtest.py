import pandas as pd
import numpy as np

# Summary oku
df = pd.read_excel('results/spot_backtest/spot_summary.xlsx')
trades_df = pd.read_excel('results/spot_backtest/spot_trades.xlsx')

print('='*70)
print('SPOT BACKTEST SUMMARY ANALYSIS')
print('='*70)

# Genel istatistikler
print(f'Toplam Sembol-RR kombinasyonu: {len(df)}')
print(f'Unique Symbol sayisi: {df["symbol"].nunique()}')
print(f'RR Profilleri: {df["rr_name"].unique().tolist()}')
print()

# RR profil bazinda toplam sonuclar
print('RR Profil Bazinda Ortalamalar:')
for rr in df['rr_name'].unique():
    subset = df[df['rr_name'] == rr]
    print(f'  {rr}:')
    print(f'    Ortalama Win Rate: {subset["win_rate"].mean():.2f}%')
    print(f'    Ortalama Total Return: {subset["total_return_pct"].mean():.2f}%')
    pf = subset["profit_factor"].replace([np.inf, -np.inf], np.nan).mean()
    print(f'    Ortalama Profit Factor: {pf:.2f}')
    print(f'    Ortalama Trade/Ay: {subset["trades_per_month"].mean():.2f}')
print()

# En iyi performans gosteren semboller (total_return > 0)
profitable = df[df['total_return_pct'] > 0].sort_values('total_return_pct', ascending=False)
print(f'Kar eden kombinasyon sayisi: {len(profitable)} / {len(df)} ({100*len(profitable)/len(df):.1f}%)')
print()

if len(profitable) > 0:
    print('En iyi 10 performans (Total Return):')
    top10 = profitable.head(10)[['symbol', 'rr_name', 'n_trades', 'win_rate', 'total_return_pct', 'profit_factor', 'max_drawdown_pct']]
    print(top10.to_string(index=False))
    print()

# Win rate > 50 olanlar
high_winrate = df[df['win_rate'] > 50].sort_values('win_rate', ascending=False)
print(f'Win Rate > 50% olan kombinasyon sayisi: {len(high_winrate)}')
if len(high_winrate) > 0:
    print('En yuksek Win Rate (Top 10):')
    top_wr = high_winrate.head(10)[['symbol', 'rr_name', 'n_trades', 'win_rate', 'total_return_pct', 'profit_factor']]
    print(top_wr.to_string(index=False))

print()
print('='*70)
print('TRADES ANALYSIS')
print('='*70)

print(f'Toplam Trade sayisi: {len(trades_df)}')
print(f'Unique Symbol: {trades_df["symbol"].nunique()}')

# Signal type distribution
print(f'\nSignal Type Dagilimi:')
print(trades_df['entry_signal_type'].value_counts().to_string())

# Exit reason distribution
print(f'\nExit Reason Dagilimi:')
print(trades_df['exit_reason'].value_counts().to_string())

# Win/Loss analysis
wins = trades_df[trades_df['net_pnl'] > 0]
losses = trades_df[trades_df['net_pnl'] <= 0]
print(f'\nWin/Loss: {len(wins)} / {len(losses)} ({100*len(wins)/len(trades_df):.1f}% win rate)')

# Average metrics
print(f'\nOrtalama Net PNL: ${trades_df["net_pnl"].mean():.4f}')
print(f'Ortalama Return %: {trades_df["return_pct"].mean():.4f}%')
print(f'Toplam Net PNL: ${trades_df["net_pnl"].sum():.2f}')

# Score analysis
print(f'\nScore Distribution:')
print(f'  Core Score (Ort): {trades_df["core_score"].mean():.2f}')
print(f'  Core Score (Min-Max): {trades_df["core_score"].min():.2f} - {trades_df["core_score"].max():.2f}')

# Best trades
print('\nEn Karlı 10 Trade:')
best = trades_df.nlargest(10, 'net_pnl')[['symbol', 'entry_signal_type', 'entry_price', 'exit_price', 'net_pnl', 'return_pct', 'exit_reason', 'core_score']]
print(best.to_string(index=False))

# Worst trades
print('\nEn Zararlı 10 Trade:')
worst = trades_df.nsmallest(10, 'net_pnl')[['symbol', 'entry_signal_type', 'entry_price', 'exit_price', 'net_pnl', 'return_pct', 'exit_reason', 'core_score']]
print(worst.to_string(index=False))
