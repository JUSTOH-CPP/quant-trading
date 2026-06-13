import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_atr(high, low, close, period=14):
    """
    Average True Range — volatility measure that
    accounts for gaps between sessions.
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low  - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(period).mean()
    return atr


def simulate_with_stop(trades_df, stop_type="none", stop_value=0.02):
    """
    Recalculate trade returns assuming a stop loss.
    stop_type: 'none', 'fixed', 'atr'
    """
    new_returns = []
    for _, trade in trades_df.iterrows():
        if stop_type == "none":
            new_returns.append(trade['return'])
        elif stop_type == "fixed":
            if trade['mae'] < -stop_value:
                new_returns.append(-stop_value)  # stopped out
            else:
                new_returns.append(trade['return'])
        elif stop_type == "atr":
            atr_pct = trade['atr_at_entry'] / trade['entry_price']
            stop_threshold = -stop_value * atr_pct
            if trade['mae'] < stop_threshold:
                new_returns.append(stop_threshold)
            else:
                new_returns.append(trade['return'])
    return pd.Series(new_returns)
