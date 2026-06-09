import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys
sys.path.append("../utils")
from indicators import compute_rsi

def multi_signal_backtest(spy, vix, log_returns,
                           target_vol=0.10,
                           cost_pct=0.0006):
    """
    Combined SMA trend + RSI(2) mean reversion +
    volatility targeting strategy.
    """
    sma50  = spy.rolling(50).mean()
    sma200 = spy.rolling(200).mean()
    sma_signal = (sma50 > sma200).astype(int).shift(1).fillna(0)

    rsi2 = compute_rsi(spy, period=2)
    pos_list, cp = [], 0
    for i in range(len(rsi2)):
        in_uptrend = spy.iloc[i] > sma200.iloc[i]
        if cp == 0 and rsi2.iloc[i] < 10 and in_uptrend:
            cp = 1
        elif cp == 1 and (rsi2.iloc[i] > 90 or not in_uptrend):
            cp = 0
        pos_list.append(cp)
    rsi2_signal = pd.Series(pos_list,
                             index=spy.index).shift(1).fillna(0)

    realized_vol = log_returns.rolling(20).std() * np.sqrt(252)
    vol_scalar   = (target_vol / realized_vol).clip(0, 1)

    sma_base = sma_signal.reindex(log_returns.index).fillna(0)
    rsi2_enh = rsi2_signal.reindex(log_returns.index).fillna(0)
    position = sma_base.clip(0, 1)
    position = position.where(rsi2_enh == 0, 1.0)

    trades  = position.diff().abs().fillna(0)
    net_ret = position * log_returns * vol_scalar.shift(1)
    net_ret = net_ret - trades * cost_pct

    ar = net_ret.mean() * 252
    av = net_ret.std()  * np.sqrt(252)
    sh = (ar - 0.02) / av
    w  = (1 + net_ret).cumprod()
    dd = (w - w.cummax()) / w.cummax()

    print(f"Multi-Signal Strategy:")
    print(f"  Sharpe : {sh:.2f}")
    print(f"  Ann Ret: {ar:.2%}")
    print(f"  Max DD : {dd.min():.2%}")

    return net_ret
