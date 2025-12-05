"""
Vectorized Backtester for Alert Rule Sets

Features:
- Look-ahead bias prevention (entry at t+1 open)
- Vectorized signal generation using numpy
- Parallel symbol processing
- Queue-based batch database writing
- Support for DIP_ALERT, MOMENTUM_ALERT, PUMP_ALERT candidates

Usage:
    python src/vectorized_backtester.py --workers 8
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue
import threading
import sqlite3
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Try to import vectorbt for enhanced functionality
try:
    import vectorbt as vbt
    HAS_VECTORBT = True
except ImportError:
    HAS_VECTORBT = False


@dataclass
class RuleSet:
    """Configuration for an alert rule set."""
    name: str
    alert_type: str  # DIP_ALERT, MOMENTUM_ALERT, PUMP_ALERT
    
    # RSI conditions
    rsi_min: float = 0
    rsi_max: float = 100
    
    # EMA distance conditions (percentage)
    ema_dist_min: float = -100
    ema_dist_max: float = 100
    
    # 24h change conditions (percentage)
    change_24h_min: float = -100
    change_24h_max: float = 100
    
    # Score threshold
    min_score: float = 0
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "alert_type": self.alert_type,
            "rsi_min": self.rsi_min,
            "rsi_max": self.rsi_max,
            "ema_dist_min": self.ema_dist_min,
            "ema_dist_max": self.ema_dist_max,
            "change_24h_min": self.change_24h_min,
            "change_24h_max": self.change_24h_max,
            "min_score": self.min_score,
        }
    
    def __repr__(self):
        return (f"RuleSet({self.name}: RSI {self.rsi_min}-{self.rsi_max}, "
                f"EMA {self.ema_dist_min} to {self.ema_dist_max}%, "
                f"24h {self.change_24h_min} to {self.change_24h_max}%, "
                f"score≥{self.min_score})")


@dataclass
class BacktestConfig:
    """Backtest configuration parameters."""
    holding_period: int = 24  # 24 bars = 6 hours for 15m timeframe
    take_profit_pct: float = 3.0
    stop_loss_pct: float = -2.0
    min_bars_between_signals: int = 4  # Minimum 1 hour between signals
    max_workers: int = 8


class VectorizedBacktester:
    """
    High-performance vectorized backtester.
    
    Features:
    - Look-ahead bias prevention (entry at t+1 open)
    - Vectorized signal generation
    - Parallel symbol processing
    - Batch result writing
    """
    
    def __init__(
        self,
        data_dir_15m: str = "data/precomputed_15m",
        data_dir_1h: str = "data/precomputed_1h",
        config: BacktestConfig = None,
    ):
        self.data_dir_15m = Path(data_dir_15m)
        self.data_dir_1h = Path(data_dir_1h)
        self.config = config or BacktestConfig()
        
        # Cache for loaded data
        self._data_cache = {}
    
    def load_symbol_data(self, symbol: str, use_cache: bool = True) -> pd.DataFrame:
        """Load 15m data for a symbol."""
        if use_cache and symbol in self._data_cache:
            return self._data_cache[symbol]
        
        path_15m = self.data_dir_15m / f"{symbol}_15m_features.parquet"
        
        if not path_15m.exists():
            raise FileNotFoundError(f"Data not found for {symbol}")
        
        df = pd.read_parquet(path_15m)
        
        # Convert timestamp to datetime index
        if "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("datetime")
        
        if use_cache:
            self._data_cache[symbol] = df
        
        return df
    
    def calculate_ema_distance(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate EMA distance percentage (vectorized)."""
        if "ema_fast" in df.columns:
            ema = df["ema_fast"].values
            close = df["close"].values
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                dist = np.where(ema != 0, (close - ema) / ema * 100, 0)
            return dist
        return np.zeros(len(df))
    
    def calculate_24h_change(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate 24h price change (96 bars for 15m, vectorized)."""
        close = df["close"].values
        # 96 bars = 24 hours for 15m timeframe
        change = np.zeros(len(close))
        change[96:] = (close[96:] - close[:-96]) / close[:-96] * 100
        return change
    
    def calculate_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate a simplified score based on available indicators.
        This approximates the full scoring system used in rules.py
        
        Score ranges from 0-15 typically
        """
        n = len(df)
        score = np.zeros(n)
        
        # --- Trend Block (0-5 points) ---
        if "adx" in df.columns:
            adx = df["adx"].values
            score += np.where(adx >= 25, 2, np.where(adx >= 20, 1, 0))
        
        if "plus_di" in df.columns and "minus_di" in df.columns:
            plus_di = df["plus_di"].values
            minus_di = df["minus_di"].values
            score += np.where(plus_di > minus_di, 1, 0)
        
        if "ema_fast" in df.columns and "ema_slow" in df.columns:
            close = df["close"].values
            ema_fast = df["ema_fast"].values
            ema_slow = df["ema_slow"].values
            bullish_structure = (close > ema_fast) & (ema_fast > ema_slow)
            score += np.where(bullish_structure, 2, 0)
        
        # --- Oscillator Block (0-5 points) ---
        if "rsi" in df.columns:
            rsi = df["rsi"].values
            # Sweet spot for momentum: 50-65
            score += np.where((rsi >= 50) & (rsi <= 65), 2,
                             np.where((rsi >= 45) & (rsi <= 70), 1, 0))
        
        if "stoch_k" in df.columns:
            stoch = df["stoch_k"].values
            score += np.where(stoch > 50, 1, 0)
        
        if "macd_hist" in df.columns:
            macd_hist = df["macd_hist"].values
            score += np.where(macd_hist > 0, 1, 0)
        
        # --- Volume Block (0-3 points) ---
        if "pa_volume_spike" in df.columns:
            score += np.where(df["pa_volume_spike"].values, 1, 0)
        
        if "pa_min_volume" in df.columns:
            score += np.where(df["pa_min_volume"].values, 1, 0)
        
        # --- Price Action Block (0-2 points) ---
        if "pa_strong_green" in df.columns:
            score += np.where(df["pa_strong_green"].values, 1, 0)
        
        if "pa_ema_breakout" in df.columns:
            score += np.where(df["pa_ema_breakout"].values, 1, 0)
        
        return score
    
    def generate_signals_vectorized(
        self,
        df: pd.DataFrame,
        rule: RuleSet,
    ) -> np.ndarray:
        """
        Generate signals using vectorized operations.
        Returns boolean array where True = signal triggered.
        """
        n = len(df)
        
        # Get required data (vectorized)
        rsi = df["rsi"].values if "rsi" in df.columns else np.full(n, 50.0)
        ema_dist = self.calculate_ema_distance(df)
        change_24h = self.calculate_24h_change(df)
        score = self.calculate_score(df)
        
        # Apply rule conditions (fully vectorized)
        signals = (
            (rsi >= rule.rsi_min) & (rsi <= rule.rsi_max) &
            (ema_dist >= rule.ema_dist_min) & (ema_dist <= rule.ema_dist_max) &
            (change_24h >= rule.change_24h_min) & (change_24h <= rule.change_24h_max) &
            (score >= rule.min_score)
        )
        
        # Handle NaN values from indicator warmup period
        signals[:100] = False  # First 100 bars are warmup
        
        # Enforce minimum bars between signals
        if self.config.min_bars_between_signals > 1:
            signals = self._enforce_signal_spacing(signals, self.config.min_bars_between_signals)
        
        return signals
    
    def _enforce_signal_spacing(self, signals: np.ndarray, min_spacing: int) -> np.ndarray:
        """Enforce minimum spacing between signals (vectorized)."""
        result = np.zeros_like(signals, dtype=bool)
        last_signal_idx = -min_spacing
        
        for i in range(len(signals)):
            if signals[i] and (i - last_signal_idx) >= min_spacing:
                result[i] = True
                last_signal_idx = i
        
        return result
    
    def calculate_trade_outcomes_vectorized(
        self,
        df: pd.DataFrame,
        signals: np.ndarray,
        symbol: str,
        rule_name: str,
    ) -> List[Dict]:
        """
        Calculate trade outcomes for all signals using vectorization.
        
        CRITICAL: Entry at t+1 Open (Look-Ahead Bias Prevention)
        - Signal generated at bar t (when we see close of bar t)
        - Entry price is OPEN of bar t+1 (next bar)
        - This prevents look-ahead bias
        """
        signal_indices = np.where(signals)[0]
        
        if len(signal_indices) == 0:
            return []
        
        # Pre-extract arrays for faster access
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        timestamps = df.index if hasattr(df.index, '__iter__') else np.arange(len(df))
        
        holding_period = self.config.holding_period
        results = []
        
        for idx in signal_indices:
            # Skip if not enough data for entry + holding period
            if idx + 1 + holding_period >= len(df):
                continue
            
            # ============================================
            # LOOK-AHEAD BIAS PREVENTION
            # ============================================
            # Signal at bar t → Entry at bar t+1 OPEN
            # We cannot know bar t's close until bar t ends
            # Therefore we enter at the OPEN of the next bar
            # ============================================
            
            entry_idx = idx + 1
            entry_price = opens[entry_idx]
            entry_time = timestamps[entry_idx]
            
            # Skip if entry price is invalid
            if entry_price <= 0 or np.isnan(entry_price):
                continue
            
            # Calculate outcomes over holding period
            exit_idx = entry_idx + holding_period - 1
            
            # Slice arrays for holding period
            holding_highs = highs[entry_idx:exit_idx + 1]
            holding_lows = lows[entry_idx:exit_idx + 1]
            
            # Max rise during holding period (best case)
            max_high = np.max(holding_highs)
            max_rise_pct = (max_high - entry_price) / entry_price * 100
            
            # Max drawdown during holding period (worst case)
            min_low = np.min(holding_lows)
            max_drawdown_pct = (min_low - entry_price) / entry_price * 100
            
            # Exit at end of holding period (close of last bar)
            exit_price = closes[exit_idx]
            exit_time = timestamps[exit_idx]
            
            # Final return
            return_pct = (exit_price - entry_price) / entry_price * 100
            
            results.append({
                "symbol": symbol,
                "rule_name": rule_name,
                "signal_idx": idx,
                "entry_idx": entry_idx,
                "entry_time": str(entry_time),
                "entry_price": float(entry_price),
                "exit_idx": exit_idx,
                "exit_time": str(exit_time),
                "exit_price": float(exit_price),
                "return_pct": float(return_pct),
                "max_rise_pct": float(max_rise_pct),
                "max_drawdown_pct": float(max_drawdown_pct),
                "holding_bars": holding_period,
                "win_2pct": bool(max_rise_pct >= 2.0),
                "win_3pct": bool(max_rise_pct >= 3.0),
            })
        
        return results
    
    def backtest_symbol(
        self,
        symbol: str,
        rule: RuleSet,
    ) -> Dict[str, Any]:
        """Backtest a single symbol with a rule set."""
        try:
            df = self.load_symbol_data(symbol, use_cache=False)
            
            # Generate signals
            signals = self.generate_signals_vectorized(df, rule)
            
            # Calculate outcomes
            trades = self.calculate_trade_outcomes_vectorized(
                df, signals, symbol, rule.name
            )
            
            if len(trades) == 0:
                return {
                    "symbol": symbol,
                    "rule_name": rule.name,
                    "alert_type": rule.alert_type,
                    "total_trades": 0,
                    "wins_2pct": 0,
                    "wins_3pct": 0,
                    "win_rate_2pct": 0.0,
                    "win_rate_3pct": 0.0,
                    "avg_return": 0.0,
                    "avg_rise": 0.0,
                    "avg_drawdown": 0.0,
                    "trades": [],
                }
            
            # Aggregate stats
            trades_df = pd.DataFrame(trades)
            
            return {
                "symbol": symbol,
                "rule_name": rule.name,
                "alert_type": rule.alert_type,
                "total_trades": len(trades_df),
                "wins_2pct": int(trades_df["win_2pct"].sum()),
                "wins_3pct": int(trades_df["win_3pct"].sum()),
                "win_rate_2pct": float(trades_df["win_2pct"].mean() * 100),
                "win_rate_3pct": float(trades_df["win_3pct"].mean() * 100),
                "avg_return": float(trades_df["return_pct"].mean()),
                "avg_rise": float(trades_df["max_rise_pct"].mean()),
                "avg_drawdown": float(trades_df["max_drawdown_pct"].mean()),
                "trades": trades,
            }
        
        except Exception as e:
            return {
                "symbol": symbol,
                "rule_name": rule.name,
                "alert_type": rule.alert_type,
                "error": str(e),
                "total_trades": 0,
                "trades": [],
            }
    
    def clear_cache(self):
        """Clear the data cache."""
        self._data_cache.clear()


def backtest_symbol_wrapper(args: Tuple) -> Dict:
    """Wrapper for parallel execution."""
    symbol, rule_dict, config_dict, data_dir_15m = args
    
    # Reconstruct objects (needed for multiprocessing)
    rule = RuleSet(**rule_dict)
    config = BacktestConfig(**config_dict)
    
    backtester = VectorizedBacktester(
        data_dir_15m=data_dir_15m,
        config=config,
    )
    
    return backtester.backtest_symbol(symbol, rule)


class BatchDBWriter:
    """
    Queue-based batch writer for SQLite database.
    Writes results in batches for efficiency.
    """
    
    def __init__(self, db_path: str = "results/backtest_results.db", batch_size: int = 100):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.queue = queue.Queue()
        self.running = False
        self.thread = None
        self._lock = threading.Lock()
    
    def start(self):
        """Start the background writer thread."""
        self.running = True
        self._init_db()
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the writer and flush remaining items."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        self._flush_remaining()
    
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(str(self.db_path))
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT,
                alert_type TEXT,
                symbol TEXT,
                signal_idx INTEGER,
                entry_idx INTEGER,
                entry_time TEXT,
                entry_price REAL,
                exit_idx INTEGER,
                exit_time TEXT,
                exit_price REAL,
                return_pct REAL,
                max_rise_pct REAL,
                max_drawdown_pct REAL,
                holding_bars INTEGER,
                win_2pct INTEGER,
                win_3pct INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_name TEXT,
                alert_type TEXT,
                total_symbols INTEGER,
                symbols_with_trades INTEGER,
                total_trades INTEGER,
                wins_2pct INTEGER,
                wins_3pct INTEGER,
                win_rate_2pct REAL,
                win_rate_3pct REAL,
                avg_return REAL,
                avg_rise REAL,
                avg_drawdown REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_rule ON backtest_trades(rule_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON backtest_trades(symbol)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_alert ON backtest_trades(alert_type)")
        
        conn.commit()
        conn.close()
    
    def add_trade(self, trade: Dict, alert_type: str = ""):
        """Add a trade to the write queue."""
        trade["alert_type"] = alert_type
        self.queue.put(("trade", trade))
    
    def add_trades(self, trades: List[Dict], alert_type: str = ""):
        """Add multiple trades to the write queue."""
        for trade in trades:
            trade["alert_type"] = alert_type
            self.queue.put(("trade", trade))
    
    def add_summary(self, summary: Dict):
        """Add a rule summary to the write queue."""
        self.queue.put(("summary", summary))
    
    def _writer_loop(self):
        """Background loop that writes batches to DB."""
        batch = []
        
        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=1)
                batch.append(item)
                
                if len(batch) >= self.batch_size:
                    self._write_batch(batch)
                    batch = []
            
            except queue.Empty:
                if batch:
                    self._write_batch(batch)
                    batch = []
    
    def _write_batch(self, batch: List[Tuple]):
        """Write a batch of items to database."""
        if not batch:
            return
        
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            trades_written = 0
            summaries_written = 0
            
            for item_type, item in batch:
                if item_type == "trade":
                    try:
                        cursor.execute("""
                            INSERT INTO backtest_trades 
                            (rule_name, alert_type, symbol, signal_idx, entry_idx,
                             entry_time, entry_price, exit_idx, exit_time, exit_price,
                             return_pct, max_rise_pct, max_drawdown_pct, holding_bars,
                             win_2pct, win_3pct)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            item.get("rule_name"),
                            item.get("alert_type", ""),
                            item.get("symbol"),
                            item.get("signal_idx"),
                            item.get("entry_idx"),
                            item.get("entry_time"),
                            item.get("entry_price"),
                            item.get("exit_idx"),
                            item.get("exit_time"),
                            item.get("exit_price"),
                            item.get("return_pct"),
                            item.get("max_rise_pct"),
                            item.get("max_drawdown_pct"),
                            item.get("holding_bars"),
                            1 if item.get("win_2pct") else 0,
                            1 if item.get("win_3pct") else 0,
                        ))
                        trades_written += 1
                    except Exception as e:
                        print(f"  [DB] Error writing trade: {e}")
                
                elif item_type == "summary":
                    try:
                        cursor.execute("""
                            INSERT INTO backtest_summary
                            (rule_name, alert_type, total_symbols, symbols_with_trades,
                             total_trades, wins_2pct, wins_3pct, win_rate_2pct,
                             win_rate_3pct, avg_return, avg_rise, avg_drawdown)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            item.get("rule_name"),
                            item.get("alert_type"),
                            item.get("total_symbols"),
                            item.get("symbols_with_trades"),
                            item.get("total_trades"),
                            item.get("wins_2pct"),
                            item.get("wins_3pct"),
                            item.get("win_rate_2pct"),
                            item.get("win_rate_3pct"),
                            item.get("avg_return"),
                            item.get("avg_rise"),
                            item.get("avg_drawdown"),
                        ))
                        summaries_written += 1
                    except Exception as e:
                        print(f"  [DB] Error writing summary: {e}")
            
            conn.commit()
            conn.close()
            
            if trades_written > 0 or summaries_written > 0:
                print(f"  [DB] Wrote {trades_written} trades, {summaries_written} summaries")
    
    def _flush_remaining(self):
        """Flush any remaining items in queue."""
        batch = []
        while not self.queue.empty():
            try:
                batch.append(self.queue.get_nowait())
            except queue.Empty:
                break
        if batch:
            self._write_batch(batch)


def load_selected_candidates() -> Tuple[List[RuleSet], List[RuleSet], List[RuleSet]]:
    """Load selected candidates from CSV files."""
    
    base_path = Path("results/alert_optimization")
    
    dip_rules = []
    momentum_rules = []
    pump_rules = []
    
    # Load DIP candidates
    dip_path = base_path / "dip_selected_candidates.csv"
    if dip_path.exists():
        dip_df = pd.read_csv(dip_path)
        print(f"  Loading DIP candidates: {len(dip_df)} rows")
        print(f"  Columns: {list(dip_df.columns)}")
        
        for i, row in dip_df.head(10).iterrows():
            dip_rules.append(RuleSet(
                name=f"DIP_{i+1}",
                alert_type="DIP_ALERT",
                rsi_min=0,
                rsi_max=float(row.get("rsi_max", 35)),
                ema_dist_min=-100,
                ema_dist_max=float(row.get("ema_dist_max", -1.0)),
                change_24h_min=-100,
                change_24h_max=float(row.get("change_24h_max", -5.0)),
                min_score=float(row.get("min_score", 5)),
            ))
    else:
        print(f"  ⚠️ DIP candidates file not found: {dip_path}")
    
    # Load MOMENTUM candidates
    mom_path = base_path / "momentum_selected_candidates.csv"
    if mom_path.exists():
        mom_df = pd.read_csv(mom_path)
        print(f"  Loading MOMENTUM candidates: {len(mom_df)} rows")
        print(f"  Columns: {list(mom_df.columns)}")
        
        for i, row in mom_df.head(10).iterrows():
            momentum_rules.append(RuleSet(
                name=f"MOMENTUM_{i+1}",
                alert_type="MOMENTUM_ALERT",
                rsi_min=float(row.get("rsi_min", 55)),
                rsi_max=float(row.get("rsi_max", 65)),
                ema_dist_min=float(row.get("ema_dist_min", 1.0)),
                ema_dist_max=float(row.get("ema_dist_max", 3.0)),
                change_24h_min=float(row.get("change_24h_min", 2.0)),
                change_24h_max=float(row.get("change_24h_max", 10.0)) if "change_24h_max" in row else 100.0,
                min_score=float(row.get("min_score", 10)),
            ))
    else:
        print(f"  ⚠️ MOMENTUM candidates file not found: {mom_path}")
    
    # Load PUMP candidates
    pump_path = base_path / "pump_selected_candidates.csv"
    if pump_path.exists():
        pump_df = pd.read_csv(pump_path)
        print(f"  Loading PUMP candidates: {len(pump_df)} rows")
        print(f"  Columns: {list(pump_df.columns)}")
        
        for i, row in pump_df.head(10).iterrows():
            pump_rules.append(RuleSet(
                name=f"PUMP_{i+1}",
                alert_type="PUMP_ALERT",
                rsi_min=float(row.get("rsi_min", 35)),
                rsi_max=float(row.get("rsi_max", 45)),
                ema_dist_min=float(row.get("ema_dist_min", -1.0)),
                ema_dist_max=float(row.get("ema_dist_max", 1.0)),
                change_24h_min=float(row.get("change_24h_min", 5.0)),
                change_24h_max=100.0,
                min_score=float(row.get("min_score", 7)),
            ))
    else:
        print(f"  ⚠️ PUMP candidates file not found: {pump_path}")
    
    return dip_rules, momentum_rules, pump_rules


def load_symbols(symbols_file: str = "data/backtest_symbols.txt") -> List[str]:
    """Load symbols from file."""
    path = Path(symbols_file)
    if not path.exists():
        raise FileNotFoundError(f"Symbols file not found: {symbols_file}")
    
    with open(path, "r") as f:
        symbols = [line.strip() for line in f if line.strip()]
    
    return symbols


def backtest_rule_parallel(
    rule: RuleSet,
    symbols: List[str],
    config: BacktestConfig,
    data_dir_15m: str = "data/precomputed_15m",
    max_workers: int = 8,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Backtest a rule set across multiple symbols in parallel.
    
    Returns:
        Tuple of (results DataFrame, all trades list)
    """
    print(f"\n{'='*60}")
    print(f"Backtesting: {rule.name} ({rule.alert_type})")
    print(f"  {rule}")
    print(f"Symbols: {len(symbols)} | Workers: {max_workers}")
    print(f"{'='*60}")
    
    # Prepare arguments for parallel execution
    rule_dict = rule.to_dict()
    config_dict = {
        "holding_period": config.holding_period,
        "take_profit_pct": config.take_profit_pct,
        "stop_loss_pct": config.stop_loss_pct,
        "min_bars_between_signals": config.min_bars_between_signals,
    }
    
    args_list = [
        (symbol, rule_dict, config_dict, data_dir_15m)
        for symbol in symbols
    ]
    
    results = []
    all_trades = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(backtest_symbol_wrapper, args): args[0]
            for args in args_list
        }
        
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                
                # Collect trades
                if result.get("trades"):
                    all_trades.extend(result["trades"])
                
                if completed % 50 == 0 or completed == len(symbols):
                    print(f"  Progress: {completed}/{len(symbols)}")
            
            except Exception as e:
                print(f"  Error for {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "rule_name": rule.name,
                    "alert_type": rule.alert_type,
                    "error": str(e),
                    "total_trades": 0,
                })
    
    df_results = pd.DataFrame(results)
    
    # Aggregate stats
    valid = df_results[df_results["total_trades"] > 0]
    
    if len(valid) > 0:
        total_trades = valid["total_trades"].sum()
        total_wins = valid["wins_2pct"].sum()
        overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        
        print(f"\n📊 Results for {rule.name}:")
        print(f"  Symbols with trades: {len(valid)}/{len(df_results)}")
        print(f"  Total trades: {total_trades:,}")
        print(f"  Wins (≥2%): {total_wins:,}")
        print(f"  Overall Win Rate: {overall_wr:.1f}%")
        print(f"  Avg Rise: {valid['avg_rise'].mean():.2f}%")
        print(f"  Avg Return: {valid['avg_return'].mean():.2f}%")
    else:
        print(f"\n⚠️ No trades generated for {rule.name}")
    
    return df_results, all_trades


def generate_report(
    all_results: List[pd.DataFrame],
    output_dir: Path,
) -> pd.DataFrame:
    """Generate summary report from backtest results."""
    
    summaries = []
    
    for df in all_results:
        if len(df) == 0:
            continue
        
        rule_name = df["rule_name"].iloc[0]
        alert_type = df["alert_type"].iloc[0] if "alert_type" in df.columns else ""
        
        valid = df[df["total_trades"] > 0]
        
        if len(valid) == 0:
            continue
        
        total_trades = valid["total_trades"].sum()
        total_wins_2 = valid["wins_2pct"].sum()
        total_wins_3 = valid["wins_3pct"].sum() if "wins_3pct" in valid.columns else 0
        
        summaries.append({
            "rule_name": rule_name,
            "alert_type": alert_type,
            "total_symbols": len(df),
            "symbols_with_trades": len(valid),
            "total_trades": int(total_trades),
            "wins_2pct": int(total_wins_2),
            "wins_3pct": int(total_wins_3),
            "win_rate_2pct": float(total_wins_2 / total_trades * 100) if total_trades > 0 else 0,
            "win_rate_3pct": float(total_wins_3 / total_trades * 100) if total_trades > 0 else 0,
            "avg_return": float(valid["avg_return"].mean()),
            "avg_rise": float(valid["avg_rise"].mean()),
            "avg_drawdown": float(valid["avg_drawdown"].mean()) if "avg_drawdown" in valid.columns else 0,
        })
    
    summary_df = pd.DataFrame(summaries)
    
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values("win_rate_2pct", ascending=False)
    
    return summary_df


def main():
    """Main backtest orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vectorized Backtester for Alert Rules")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--output-dir", default="results/backtest_v7", help="Output directory")
    parser.add_argument("--holding-period", type=int, default=24, help="Holding period in bars (24 = 6h)")
    parser.add_argument("--symbols-file", default="data/backtest_symbols.txt", help="Symbols file")
    parser.add_argument("--limit-symbols", type=int, default=0, help="Limit symbols (0=all)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("VECTORIZED BACKTEST - 30 RULE SETS (10 DIP + 10 MOMENTUM + 10 PUMP)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output: {args.output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Holding Period: {args.holding_period} bars ({args.holding_period * 15 / 60:.1f} hours)")
    
    # 1. Load symbols
    print("\n[1/5] Loading symbols...")
    symbols = load_symbols(args.symbols_file)
    
    if args.limit_symbols > 0:
        symbols = symbols[:args.limit_symbols]
        print(f"  Limited to {len(symbols)} symbols")
    else:
        print(f"  Loaded {len(symbols)} symbols")
    
    # 2. Load rule sets
    print("\n[2/5] Loading rule sets...")
    dip_rules, momentum_rules, pump_rules = load_selected_candidates()
    all_rules = dip_rules + momentum_rules + pump_rules
    print(f"  Loaded {len(all_rules)} rule sets:")
    print(f"    DIP_ALERT: {len(dip_rules)}")
    print(f"    MOMENTUM_ALERT: {len(momentum_rules)}")
    print(f"    PUMP_ALERT: {len(pump_rules)}")
    
    if len(all_rules) == 0:
        print("\n❌ No rule sets loaded! Check candidate CSV files.")
        return
    
    # 3. Initialize config and output
    print("\n[3/5] Initializing...")
    config = BacktestConfig(
        holding_period=args.holding_period,
        max_workers=args.workers,
    )
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize DB writer
    db_writer = BatchDBWriter(db_path=str(output_path / "backtest_results.db"))
    db_writer.start()
    
    # 4. Run backtests
    print("\n[4/5] Running backtests...")
    all_results = []
    all_trades = []
    
    for rule in all_rules:
        df_results, trades = backtest_rule_parallel(
            rule=rule,
            symbols=symbols,
            config=config,
            max_workers=args.workers,
        )
        all_results.append(df_results)
        
        # Queue trades for DB writing
        if trades:
            db_writer.add_trades(trades, alert_type=rule.alert_type)
    
    # 5. Generate report
    print("\n[5/5] Generating report...")
    
    summary_df = generate_report(all_results, output_path)
    
    # Save summary to CSV and DB
    if len(summary_df) > 0:
        summary_df.to_csv(output_path / "backtest_summary.csv", index=False)
        
        for _, row in summary_df.iterrows():
            db_writer.add_summary(row.to_dict())
    
    # Stop DB writer
    db_writer.stop()
    
    # Combine all per-symbol results
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(output_path / "all_symbol_results.csv", index=False)
    
    # Print final report
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    
    if len(summary_df) > 0:
        print("\n📊 RESULTS BY RULE SET (sorted by Win Rate):")
        print("-" * 100)
        print(f"{'Rule':<15} {'Type':<15} {'Trades':>8} {'Wins':>8} {'WR%':>8} {'WR3%':>8} {'AvgRet':>8} {'AvgRise':>8}")
        print("-" * 100)
        
        for _, row in summary_df.iterrows():
            print(f"{row['rule_name']:<15} {row['alert_type']:<15} "
                  f"{row['total_trades']:>8,} {row['wins_2pct']:>8,} "
                  f"{row['win_rate_2pct']:>7.1f}% {row['win_rate_3pct']:>7.1f}% "
                  f"{row['avg_return']:>7.2f}% {row['avg_rise']:>7.2f}%")
        
        print("-" * 100)
        
        # Group by alert type
        print("\n📈 RESULTS BY ALERT TYPE:")
        for alert_type in ["DIP_ALERT", "MOMENTUM_ALERT", "PUMP_ALERT"]:
            subset = summary_df[summary_df["alert_type"] == alert_type]
            if len(subset) > 0:
                total_trades = subset["total_trades"].sum()
                total_wins = subset["wins_2pct"].sum()
                avg_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
                best = subset.iloc[0]
                print(f"  {alert_type}:")
                print(f"    Total Trades: {total_trades:,}")
                print(f"    Overall Win Rate: {avg_wr:.1f}%")
                print(f"    Best Rule: {best['rule_name']} ({best['win_rate_2pct']:.1f}% WR)")
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"  - backtest_summary.csv")
    print(f"  - all_symbol_results.csv")
    print(f"  - backtest_results.db")


if __name__ == "__main__":
    main()
