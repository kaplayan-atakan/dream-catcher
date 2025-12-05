"""
DIP Trade Tracker Module (V8)
Tracks all DIP_ALERT signals as virtual trades and monitors for TP/SL conditions.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

import config


class TradeStatus(Enum):
    ACTIVE = "ACTIVE"
    TP1 = "TP1"
    TP2 = "TP2"
    SL = "SL"
    TIMEOUT = "TIMEOUT"


@dataclass
class DipTrade:
    """Represents a single DIP trade."""
    
    id: Optional[int]
    symbol: str
    entry_time: datetime
    entry_price: float
    current_price: float = 0.0
    max_price: float = 0.0
    min_price: float = float('inf')
    status: TradeStatus = TradeStatus.ACTIVE
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    return_pct: Optional[float] = None
    
    def __post_init__(self):
        """Initialize max/min prices from entry if not set."""
        if self.max_price == 0.0:
            self.max_price = self.entry_price
        if self.min_price == float('inf'):
            self.min_price = self.entry_price
        if self.current_price == 0.0:
            self.current_price = self.entry_price
    
    @property
    def tp1_price(self) -> float:
        """Price target for TP1 (+2%)."""
        return self.entry_price * (1 + config.DIP_TP1_PCT / 100)
    
    @property
    def tp2_price(self) -> float:
        """Price target for TP2 (+3%)."""
        return self.entry_price * (1 + config.DIP_TP2_PCT / 100)
    
    @property
    def sl_price(self) -> float:
        """Price target for SL (-3%)."""
        return self.entry_price * (1 + config.DIP_SL_PCT / 100)
    
    @property
    def timeout_time(self) -> datetime:
        """Time when trade times out."""
        return self.entry_time + timedelta(hours=config.DIP_TIMEOUT_HOURS)
    
    @property
    def current_return_pct(self) -> float:
        """Current return percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100
    
    @property
    def max_return_pct(self) -> float:
        """Maximum return percentage reached."""
        if self.entry_price == 0:
            return 0.0
        return (self.max_price - self.entry_price) / self.entry_price * 100
    
    @property
    def max_drawdown_pct(self) -> float:
        """Maximum drawdown percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.min_price - self.entry_price) / self.entry_price * 100
    
    def update_price(
        self, 
        price: float, 
        current_time: Optional[datetime] = None,
        high: Optional[float] = None,
        low: Optional[float] = None
    ) -> Optional[Tuple[TradeStatus, TradeStatus]]:
        """
        Update current price and check for TP/SL conditions.
        
        SL PRIORITY: When both TP and SL are hit in same candle (high >= TP and low <= SL),
        ALWAYS return SL (more conservative/realistic assumption).
        
        Args:
            price: Current/close price
            current_time: Current time (defaults to now)
            high: Candle high (optional, for same-candle detection)
            low: Candle low (optional, for same-candle detection)
        
        Returns:
            Tuple of (old_status, new_status) if status changed, None otherwise
        """
        if self.status != TradeStatus.ACTIVE:
            return None
        
        current_time = current_time or datetime.now(timezone.utc)
        old_status = self.status
        
        # Use high/low if provided, otherwise use current price
        check_high = high if high is not None else price
        check_low = low if low is not None else price
        
        # Update price tracking
        self.current_price = price
        self.max_price = max(self.max_price, check_high)
        self.min_price = min(self.min_price, check_low)
        
        # Check timeout first
        if current_time >= self.timeout_time:
            self._close_trade(TradeStatus.TIMEOUT, price, current_time)
            return (old_status, self.status)
        
        # Check if BOTH TP and SL hit in same candle
        tp1_hit = check_high >= self.tp1_price
        sl_hit = check_low <= self.sl_price
        
        # SL PRIORITY: If both hit, ALWAYS return SL (conservative)
        if tp1_hit and sl_hit:
            self._close_trade(TradeStatus.SL, self.sl_price, current_time)
            self.return_pct = config.DIP_SL_PCT  # Use exact SL percentage
            return (old_status, self.status)
        
        # Check SL first (priority over TP)
        if sl_hit:
            self._close_trade(TradeStatus.SL, check_low, current_time)
            return (old_status, self.status)
        
        # Check TP2 (higher priority than TP1)
        if check_high >= self.tp2_price:
            self._close_trade(TradeStatus.TP2, self.tp2_price, current_time)
            self.return_pct = config.DIP_TP2_PCT  # Use exact TP2 percentage
            return (old_status, self.status)
        
        # Check TP1
        if tp1_hit:
            self._close_trade(TradeStatus.TP1, self.tp1_price, current_time)
            self.return_pct = config.DIP_TP1_PCT  # Use exact TP1 percentage
            return (old_status, self.status)
        
        return None
    
    def _close_trade(self, status: TradeStatus, price: float, exit_time: datetime):
        """Close the trade with given status."""
        self.status = status
        self.exit_price = price
        self.exit_time = exit_time
        self.return_pct = (price - self.entry_price) / self.entry_price * 100


class DipTradeTracker:
    """Tracks and manages DIP trades with SQLite persistence."""
    
    def __init__(self, db_path: str = "data/dip_trades.db"):
        """Initialize tracker with database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        self.active_trades: Dict[str, DipTrade] = {}
        self._load_active_trades()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dip_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_time TEXT,
                    exit_price REAL,
                    max_price REAL,
                    min_price REAL,
                    return_pct REAL,
                    status TEXT DEFAULT 'ACTIVE',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dip_trades_status 
                ON dip_trades(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dip_trades_symbol 
                ON dip_trades(symbol)
            """)
            conn.commit()
    
    def _load_active_trades(self):
        """Load active trades from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM dip_trades WHERE status = 'ACTIVE'
            """)
            
            for row in cursor.fetchall():
                entry_time = datetime.fromisoformat(row['entry_time'])
                # Ensure timezone-aware
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                    
                trade = DipTrade(
                    id=row['id'],
                    symbol=row['symbol'],
                    entry_time=entry_time,
                    entry_price=row['entry_price'],
                    max_price=row['max_price'] or row['entry_price'],
                    min_price=row['min_price'] or row['entry_price'],
                    status=TradeStatus.ACTIVE
                )
                self.active_trades[row['symbol']] = trade
    
    def add_trade(self, symbol: str, price: float, entry_time: Optional[datetime] = None) -> DipTrade:
        """
        Add a new DIP trade.
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            price: Entry price
            entry_time: Entry time (defaults to now)
            
        Returns:
            Created DipTrade object
        """
        entry_time = entry_time or datetime.now(timezone.utc)
        
        # Check if already tracking this symbol
        if symbol in self.active_trades:
            # Update existing trade's price tracking
            self.active_trades[symbol].update_price(price, entry_time)
            return self.active_trades[symbol]
        
        # Create new trade
        trade = DipTrade(
            id=None,
            symbol=symbol,
            entry_time=entry_time,
            entry_price=price
        )
        
        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO dip_trades (symbol, entry_time, entry_price, max_price, min_price, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                entry_time.isoformat(),
                price,
                price,
                price,
                TradeStatus.ACTIVE.value
            ))
            trade.id = cursor.lastrowid
            conn.commit()
        
        self.active_trades[symbol] = trade
        return trade
    
    def check_prices(self, prices_dict: Dict[str, any], current_time: Optional[datetime] = None) -> List[Tuple[DipTrade, TradeStatus, TradeStatus]]:
        """
        Check all active trades for TP/SL conditions.
        
        SL PRIORITY: When both TP and SL are hit in same candle, SL wins.
        
        Args:
            prices_dict: Dict mapping symbol -> price data
                        Can be: symbol -> float (just price)
                        Or: symbol -> {"price": float, "high": float, "low": float}
            current_time: Current time (defaults to now)
            
        Returns:
            List of (trade, old_status, new_status) for trades that changed status
        """
        current_time = current_time or datetime.now(timezone.utc)
        status_changes = []
        trades_to_remove = []
        
        for symbol, trade in self.active_trades.items():
            price_data = prices_dict.get(symbol)
            if price_data is None:
                # Check timeout even without price update
                if current_time >= trade.timeout_time:
                    result = trade.update_price(trade.current_price, current_time)
                    if result:
                        status_changes.append((trade, result[0], result[1]))
                        trades_to_remove.append(symbol)
                        self._update_trade_in_db(trade)
                continue
            
            # Handle both dict and float formats
            if isinstance(price_data, dict):
                price = price_data.get("price", price_data.get("close", 0))
                high = price_data.get("high")
                low = price_data.get("low")
            else:
                price = float(price_data)
                high = None
                low = None
            
            # Update price with OHLC data for accurate TP/SL detection
            result = trade.update_price(price, current_time, high=high, low=low)
            if result:
                old_status, new_status = result
                status_changes.append((trade, old_status, new_status))
                trades_to_remove.append(symbol)
                self._update_trade_in_db(trade)
            else:
                # Update max/min in database periodically
                self._update_trade_prices_in_db(trade)
        
        # Remove closed trades from active dict
        for symbol in trades_to_remove:
            del self.active_trades[symbol]
        
        return status_changes
    
    def _update_trade_in_db(self, trade: DipTrade):
        """Update trade in database after closure."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE dip_trades 
                SET exit_time = ?, exit_price = ?, max_price = ?, min_price = ?, 
                    return_pct = ?, status = ?
                WHERE id = ?
            """, (
                trade.exit_time.isoformat() if trade.exit_time else None,
                trade.exit_price,
                trade.max_price,
                trade.min_price,
                trade.return_pct,
                trade.status.value,
                trade.id
            ))
            conn.commit()
    
    def _update_trade_prices_in_db(self, trade: DipTrade):
        """Update max/min prices in database for active trade."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE dip_trades 
                SET max_price = ?, min_price = ?
                WHERE id = ?
            """, (trade.max_price, trade.min_price, trade.id))
            conn.commit()
    
    def get_stats_summary(self) -> str:
        """
        Get formatted stats summary string.
        
        Returns:
            Formatted string like: 📊 DIP Stats: 🟢 Active: 5 | ✅ TP: 12 | ❌ SL: 3 | ⏱️ Timeout: 2 | 📈 WR: 70.6%
        """
        stats = self.get_detailed_stats()
        
        active = stats['active']
        tp_total = stats['tp1'] + stats['tp2']
        sl = stats['sl']
        timeout = stats['timeout']
        win_rate = stats['win_rate']
        
        return (
            f"📊 DIP Stats: 🟢 Active: {active} | "
            f"✅ TP: {tp_total} | ❌ SL: {sl} | ⏱️ Timeout: {timeout} | "
            f"📈 WR: {win_rate:.1f}%"
        )
    
    def get_detailed_stats(self) -> Dict:
        """
        Get detailed statistics.
        
        Returns:
            Dict with all statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Count by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count, 
                       AVG(return_pct) as avg_return,
                       SUM(return_pct) as total_return
                FROM dip_trades
                GROUP BY status
            """)
            
            stats = {
                'active': 0,
                'tp1': 0,
                'tp2': 0,
                'sl': 0,
                'timeout': 0,
                'total': 0,
                'closed': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'total_return': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
            
            for row in cursor.fetchall():
                status = row['status']
                count = row['count']
                avg_ret = row['avg_return'] or 0
                total_ret = row['total_return'] or 0
                
                stats['total'] += count
                
                if status == 'ACTIVE':
                    stats['active'] = count
                elif status == 'TP1':
                    stats['tp1'] = count
                    stats['wins'] += count
                    stats['closed'] += count
                elif status == 'TP2':
                    stats['tp2'] = count
                    stats['wins'] += count
                    stats['closed'] += count
                elif status == 'SL':
                    stats['sl'] = count
                    stats['losses'] += count
                    stats['closed'] += count
                elif status == 'TIMEOUT':
                    stats['timeout'] = count
                    stats['closed'] += count
                    # Timeout can be win or loss based on return
                    if avg_ret and avg_ret > 0:
                        stats['wins'] += count
                    else:
                        stats['losses'] += count
            
            # Calculate win rate
            if stats['closed'] > 0:
                stats['win_rate'] = stats['wins'] / stats['closed'] * 100
            
            # Get average returns for wins and losses
            cursor = conn.execute("""
                SELECT 
                    AVG(CASE WHEN return_pct > 0 THEN return_pct END) as avg_win,
                    AVG(CASE WHEN return_pct <= 0 THEN return_pct END) as avg_loss,
                    AVG(return_pct) as avg_return,
                    SUM(return_pct) as total_return
                FROM dip_trades
                WHERE status != 'ACTIVE'
            """)
            row = cursor.fetchone()
            if row:
                stats['avg_win'] = row['avg_win'] or 0
                stats['avg_loss'] = row['avg_loss'] or 0
                stats['avg_return'] = row['avg_return'] or 0
                stats['total_return'] = row['total_return'] or 0
            
            return stats
    
    def get_active_trades(self) -> List[DipTrade]:
        """Get list of active trades."""
        return list(self.active_trades.values())
    
    def get_active_symbols(self) -> List[str]:
        """Get list of symbols with active trades."""
        return list(self.active_trades.keys())
    
    def format_trade_status_message(self, trade: DipTrade, old_status: TradeStatus, new_status: TradeStatus) -> str:
        """
        Format a status change message for Telegram.
        
        Args:
            trade: The trade that changed status
            old_status: Previous status
            new_status: New status
            
        Returns:
            Formatted message string
        """
        emoji_map = {
            TradeStatus.TP1: "✅",
            TradeStatus.TP2: "🎯",
            TradeStatus.SL: "❌",
            TradeStatus.TIMEOUT: "⏱️"
        }
        
        status_text_map = {
            TradeStatus.TP1: "TP1 HIT (+2%)",
            TradeStatus.TP2: "TP2 HIT (+3%)",
            TradeStatus.SL: "STOP LOSS (-3%)",
            TradeStatus.TIMEOUT: "TIMEOUT (24h)"
        }
        
        emoji = emoji_map.get(new_status, "📊")
        status_text = status_text_map.get(new_status, str(new_status.value))
        
        return_emoji = "🟢" if (trade.return_pct or 0) > 0 else "🔴"
        
        # Calculate hold time
        if trade.exit_time and trade.entry_time:
            hold_time = trade.exit_time - trade.entry_time
            hours = hold_time.total_seconds() / 3600
            hold_str = f"{hours:.1f}h"
        else:
            hold_str = "N/A"
        
        message = (
            f"{emoji} **DIP {status_text}**\n\n"
            f"🪙 Symbol: `{trade.symbol}`\n"
            f"📥 Entry: ${trade.entry_price:.6f}\n"
            f"📤 Exit: ${trade.exit_price:.6f}\n"
            f"{return_emoji} Return: {trade.return_pct:+.2f}%\n"
            f"📈 Max: {trade.max_return_pct:+.2f}%\n"
            f"📉 Min: {trade.max_drawdown_pct:+.2f}%\n"
            f"⏱️ Hold: {hold_str}\n\n"
            f"{self.get_stats_summary()}"
        )
        
        return message
    
    def close_all_active(self, prices_dict: Dict[str, float], reason: str = "MANUAL") -> List[DipTrade]:
        """
        Close all active trades manually.
        
        Args:
            prices_dict: Current prices for exit
            reason: Reason for closure
            
        Returns:
            List of closed trades
        """
        closed = []
        current_time = datetime.now(timezone.utc)
        
        for symbol, trade in list(self.active_trades.items()):
            price = prices_dict.get(symbol, trade.current_price)
            trade._close_trade(TradeStatus.TIMEOUT, price, current_time)
            self._update_trade_in_db(trade)
            closed.append(trade)
        
        self.active_trades.clear()
        return closed


# Singleton instance
_tracker_instance: Optional[DipTradeTracker] = None


def get_tracker() -> DipTradeTracker:
    """Get or create the singleton tracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = DipTradeTracker()
    return _tracker_instance


# Convenience functions
def add_dip_trade(symbol: str, price: float) -> DipTrade:
    """Add a new DIP trade."""
    return get_tracker().add_trade(symbol, price)


def check_dip_prices(prices_dict: Dict[str, float]) -> List[Tuple[DipTrade, TradeStatus, TradeStatus]]:
    """Check all active DIP trades for TP/SL."""
    return get_tracker().check_prices(prices_dict)


def get_dip_stats() -> str:
    """Get DIP stats summary."""
    return get_tracker().get_stats_summary()


if __name__ == "__main__":
    # Test the tracker
    tracker = DipTradeTracker(db_path="data/dip_trades_test.db")
    
    # Add test trades
    trade1 = tracker.add_trade("BTCUSDT", 100000.0)
    trade2 = tracker.add_trade("ETHUSDT", 3500.0)
    
    print(f"Added trades: {trade1.symbol}, {trade2.symbol}")
    print(f"TP1 targets: BTC=${trade1.tp1_price:.2f}, ETH=${trade2.tp1_price:.2f}")
    print(f"SL targets: BTC=${trade1.sl_price:.2f}, ETH=${trade2.sl_price:.2f}")
    
    # Simulate price updates
    changes = tracker.check_prices({
        "BTCUSDT": 102000.0,  # +2% -> TP1
        "ETHUSDT": 3400.0,    # -2.8% -> still active
    })
    
    for trade, old, new in changes:
        print(f"\n{trade.symbol}: {old.value} -> {new.value}")
        print(tracker.format_trade_status_message(trade, old, new))
    
    print(f"\n{tracker.get_stats_summary()}")
    print(f"\nDetailed stats: {tracker.get_detailed_stats()}")
