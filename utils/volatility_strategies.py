import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def vix_regime(v):
    if   v < 15: return "LOW"
    elif v < 25: return "NORMAL"
    elif v < 35: return "HIGH"
    else:        return "EXTREME"

def vol_target_returns(returns, target_vol=0.10, window=20):
    """
    Apply volatility targeting to a return stream.
    Scales position to maintain target annual volatility.
    """
    realized_vol = returns.rolling(window).std() * np.sqrt(252)
    vol_scalar   = (target_vol / realized_vol).clip(0, 1)
    return returns * vol_scalar.shift(1)

def vix_contrarian_backtest(spy_returns, vix,
                             entry_vix=30, exit_vix=20,
                             cost_pct=0.0006):
    """
    Buy SPY when VIX spikes above entry_vix.
    Exit when VIX calms below exit_vix.
    """
    vix_aligned = vix.reindex(spy_returns.index).ffill()
    pos_list, cp = [], 0
    for i in range(len(vix_aligned)):
        if cp == 0 and vix_aligned.iloc[i] > entry_vix:
            cp = 1
        elif cp == 1 and vix_aligned.iloc[i] < exit_vix:
            cp = 0
        pos_list.append(cp)
    position = pd.Series(pos_list,
                         index=spy_returns.index).shift(1).fillna(0)
    trades   = position.diff().abs().fillna(0)
    net_ret  = position * spy_returns - trades * cost_pct
    return net_ret
