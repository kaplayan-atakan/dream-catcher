"""
DIP TP/SL Notification Formatter (V8)
Formats Telegram notifications for DIP trade exits.
"""

from datetime import datetime, timezone
from typing import Optional

from dip_tracker import DipTrade, TradeStatus, get_tracker


def _format_price(price: float) -> str:
    """Format price with appropriate decimal places."""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"


def _format_hold_time(trade: DipTrade) -> str:
    """Format hold time as hours."""
    if trade.exit_time and trade.entry_time:
        delta = trade.exit_time - trade.entry_time
        hours = delta.total_seconds() / 3600
        return f"{hours:.1f}h"
    return "N/A"


def _get_stats_line() -> str:
    """Get the running stats summary line."""
    return get_tracker().get_stats_summary()


def format_tp_notification(trade: DipTrade, tp_level: int = 1) -> str:
    """
    Format a Take Profit notification.
    
    Args:
        trade: The DipTrade that hit TP
        tp_level: 1 for TP1 (+2%), 2 for TP2 (+3%)
        
    Returns:
        Formatted Telegram message
    """
    tp_label = f"TP{tp_level}"
    
    return f"""✅ {tp_label} HIT - DIP_ALERT
{trade.symbol}
━━━━━━━━━━━━━━━━━━━━
💰 Entry: {_format_price(trade.entry_price)}
🎯 Exit: {_format_price(trade.exit_price or trade.current_price)}
📈 Profit: +{trade.return_pct:.2f}%
━━━━━━━━━━━━━━━━━━━━
⏱️ Hold Time: {_format_hold_time(trade)}
📊 Max Rise: +{trade.max_return_pct:.2f}%
📉 Max Drop: {trade.max_drawdown_pct:.2f}%
━━━━━━━━━━━━━━━━━━━━
{_get_stats_line()}"""


def format_sl_notification(trade: DipTrade) -> str:
    """
    Format a Stop Loss notification.
    
    Args:
        trade: The DipTrade that hit SL
        
    Returns:
        Formatted Telegram message
    """
    return f"""❌ SL HIT - DIP_ALERT
{trade.symbol}
━━━━━━━━━━━━━━━━━━━━
💰 Entry: {_format_price(trade.entry_price)}
🛑 Exit: {_format_price(trade.exit_price or trade.current_price)}
📉 Loss: {trade.return_pct:.2f}%
━━━━━━━━━━━━━━━━━━━━
⏱️ Hold Time: {_format_hold_time(trade)}
📊 Max Rise: +{trade.max_return_pct:.2f}%
━━━━━━━━━━━━━━━━━━━━
{_get_stats_line()}"""


def format_timeout_notification(trade: DipTrade) -> str:
    """
    Format a Timeout notification.
    
    Args:
        trade: The DipTrade that timed out
        
    Returns:
        Formatted Telegram message
    """
    # Determine return emoji based on profit/loss
    return_emoji = "🟢" if (trade.return_pct or 0) > 0 else "🔴"
    return_sign = "+" if (trade.return_pct or 0) > 0 else ""
    
    return f"""⏱️ TIMEOUT - DIP_ALERT
{trade.symbol}
━━━━━━━━━━━━━━━━━━━━
💰 Entry: {_format_price(trade.entry_price)}
📤 Exit: {_format_price(trade.exit_price or trade.current_price)}
{return_emoji} Return: {return_sign}{trade.return_pct:.2f}%
━━━━━━━━━━━━━━━━━━━━
⏱️ Held for 24h (max)
📊 Max Rise: +{trade.max_return_pct:.2f}%
📉 Max Drop: {trade.max_drawdown_pct:.2f}%
━━━━━━━━━━━━━━━━━━━━
{_get_stats_line()}"""


def format_status_change_notification(trade: DipTrade, new_status: TradeStatus) -> str:
    """
    Format notification based on new status.
    
    Args:
        trade: The DipTrade that changed status
        new_status: The new status
        
    Returns:
        Formatted Telegram message
    """
    if new_status == TradeStatus.TP1:
        return format_tp_notification(trade, tp_level=1)
    elif new_status == TradeStatus.TP2:
        return format_tp_notification(trade, tp_level=2)
    elif new_status == TradeStatus.SL:
        return format_sl_notification(trade)
    elif new_status == TradeStatus.TIMEOUT:
        return format_timeout_notification(trade)
    else:
        return f"Unknown status change: {new_status}"


if __name__ == "__main__":
    # Test with mock trade
    from src.dip_tracker import DipTradeTracker
    
    tracker = DipTradeTracker(db_path="data/dip_trades_test.db")
    
    # Add and close a test trade
    trade = tracker.add_trade("BTCUSDT", 95234.50)
    
    # Simulate TP1 hit
    changes = tracker.check_prices({"BTCUSDT": 97139.19})
    
    if changes:
        t, old, new = changes[0]
        print("=== TP1 NOTIFICATION ===")
        print(format_status_change_notification(t, new))
    
    # Add another trade for SL test
    trade2 = tracker.add_trade("SOLUSDT", 142.50)
    changes = tracker.check_prices({"SOLUSDT": 138.23})
    
    if changes:
        t, old, new = changes[0]
        print("\n=== SL NOTIFICATION ===")
        print(format_status_change_notification(t, new))
    
    # Add another trade for timeout test  
    trade3 = tracker.add_trade("ETHUSDT", 3150.00)
    # Manually set exit for demo
    trade3.exit_price = 3180.50
    trade3.return_pct = 0.97
    trade3.max_price = 3208.28  # +1.85%
    trade3.min_price = 3112.20  # -1.20%
    trade3.status = TradeStatus.TIMEOUT
    
    print("\n=== TIMEOUT NOTIFICATION ===")
    print(format_timeout_notification(trade3))
