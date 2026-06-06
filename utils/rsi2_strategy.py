import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys
sys.path.append("../utils")
from indicators import compute_rsi

def rsi2_backtest(price, use_filter=True, cost_pct=0.0006):
    """
    RSI(2) mean reversion backtest.
    use_filter=True adds 200 SMA trend filter.
    """
    log_ret = np.log(price / price.shift(1)).dropna()
    rsi2    = compute_rsi(price, period=2)
    sma200  = price.rolling(200).mean()

    # State machine
    pos_list, cp = [], 0
    for i in range(len(rsi2)):
        in_uptrend = price.iloc[i] > sma200.iloc[i]
        if cp == 0:
            if use_filter:
                if rsi2.iloc[i] < 10 and in_uptrend:
                    cp = 1
            else:
                if rsi2.iloc[i] < 10:
                    cp = 1
        elif cp == 1:
            if rsi2.iloc[i] > 90 or (use_filter and not in_uptrend):
                cp = 0
        pos_list.append(cp)

    position = pd.Series(pos_list,
                         index=rsi2.index).shift(1).fillna(0)
    trades   = position.diff().abs().fillna(0)
    lr       = log_ret.reindex(position.index).fillna(0)
    net_ret  = position * lr - trades * cost_pct
    equity   = (1 + net_ret).cumprod()

    ar = net_ret.mean() * 252
    av = net_ret.std()  * np.sqrt(252)
    sh = (ar - 0.02)   / av
    dd = (equity - equity.cummax()) / equity.cummax()

    return {
        "ann_ret" : ar,
        "ann_vol" : av,
        "sharpe"  : sh,
        "max_dd"  : dd.min(),
        "n_trades": int(trades.sum()),
        "time_mkt": position.mean(),
        "equity"  : equity,
        "net_ret" : net_ret
    }
