import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def sma_backtest(ticker, fast=50, slow=200,
                 start="2010-01-01", end="2024-01-01",
                 cost_pct=0.0006):
    """
    SMA crossover backtest with costs and benchmark comparison.
    Returns performance dict.
    """
    # Data
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    price = df["Close"].squeeze()
    log_ret = np.log(price / price.shift(1)).dropna()

    # Signals
    sma_fast = price.rolling(fast).mean()
    sma_slow = price.rolling(slow).mean()
    position = (sma_fast > sma_slow).astype(int).shift(1).fillna(0)

    # Returns
    log_ret_aligned = log_ret.reindex(position.index).fillna(0)
    strat_ret       = position * log_ret_aligned
    trades          = position.diff().abs().fillna(0)
    net_ret         = strat_ret - (trades * cost_pct)

    # Metrics
    def metrics(r):
        ar = r.mean() * 252
        av = r.std()  * np.sqrt(252)
        sh = (ar - 0.02) / av
        w  = (1 + r).cumprod()
        dd = (w - w.cummax()) / w.cummax()
        return ar, av, sh, dd.min()

    s_ar, s_av, s_sh, s_dd = metrics(net_ret)
    b_ar, b_av, b_sh, b_dd = metrics(log_ret_aligned)

    print(f"\n{'='*58}")
    print(f"  {ticker} SMA {fast}/{slow} BACKTEST")
    print(f"{'='*58}")
    print(f"  {'Strategy':<20} {'Ann Ret':>8} {'Ann Vol':>8} {'Sharpe':>7} {'Max DD':>9}")
    print(f"  {'-'*54}")
    print(f"  {'SMA Strategy':<20} {s_ar:>8.2%} {s_av:>8.2%} {s_sh:>7.2f} {s_dd:>9.2%}")
    print(f"  {'Buy & Hold':<20} {b_ar:>8.2%} {b_av:>8.2%} {b_sh:>7.2f} {b_dd:>9.2%}")
    print(f"{'='*58}")
    print(f"  Total trades: {int(trades.sum())} over {len(log_ret)/252:.0f} years")

    return {"sharpe": s_sh, "ann_ret": s_ar,
            "max_dd": s_dd, "n_trades": int(trades.sum())}
