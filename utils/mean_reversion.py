import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def mean_reversion_backtest(ticker, window=20,
                             entry_z=-2, exit_z=0,
                             start="2010-01-01",
                             end="2024-01-01",
                             cost_pct=0.0006):
    """
    Z-score mean reversion backtest.
    Buy when z < entry_z, exit when z > exit_z.
    """
    # Data
    df = yf.download(ticker, start=start,
                     end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    price   = df["Close"].squeeze()
    log_ret = np.log(price / price.shift(1)).dropna()

    # Z-score
    rm     = price.rolling(window).mean()
    rs     = price.rolling(window).std()
    zscore = (price - rm) / rs

    # State machine
    pos_list, cp = [], 0
    for i in range(len(zscore)):
        if cp == 0 and zscore.iloc[i] < entry_z:
            cp = 1
        elif cp == 1 and zscore.iloc[i] > exit_z:
            cp = 0
        pos_list.append(cp)

    position = pd.Series(pos_list,
                         index=zscore.index).shift(1).fillna(0)
    trades   = position.diff().abs().fillna(0)

    # Returns
    lr      = log_ret.reindex(position.index).fillna(0)
    net_ret = position * lr - trades * cost_pct

    # Metrics
    ar = net_ret.mean() * 252
    av = net_ret.std()  * np.sqrt(252)
    sh = (ar - 0.02) / av
    w  = (1 + net_ret).cumprod()
    dd = (w - w.cummax()) / w.cummax()

    print(f"\n{'='*50}")
    print(f"  {ticker} MEAN REVERSION BACKTEST")
    print(f"{'='*50}")
    print(f"  Window     : {window} days")
    print(f"  Entry z    : {entry_z}")
    print(f"  Exit z     : {exit_z}")
    print(f"  Ann Return : {ar:.2%}")
    print(f"  Ann Vol    : {av:.2%}")
    print(f"  Sharpe     : {sh:.2f}")
    print(f"  Max DD     : {dd.min():.2%}")
    print(f"  N Trades   : {int(trades.sum())}")
    print(f"  Time in mkt: {position.mean():.1%}")
    print(f"{'='*50}")

    return {"sharpe": sh, "ann_ret": ar,
            "max_dd": dd.min(),
            "n_trades": int(trades.sum())}
