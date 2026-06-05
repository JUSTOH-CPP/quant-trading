import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from itertools import combinations

def pairs_backtest(y, x, hedge_ratio, 
                   entry_z=2.0, exit_z=0.5,
                   window=20, cost_pct=0.0006):
    """
    Pairs trading backtest with configurable parameters.
    """
    # Spread z-score
    spread      = y - hedge_ratio * x
    spread_mean = spread.rolling(window).mean()
    spread_std  = spread.rolling(window).std()
    zscore      = (spread - spread_mean) / spread_std

    # State machine
    pos_list, cp = [], 0
    for i in range(len(zscore)):
        if cp == 0 and zscore.iloc[i] < -entry_z:
            cp = 1
        elif cp == 0 and zscore.iloc[i] > entry_z:
            cp = -1
        elif cp != 0 and abs(zscore.iloc[i]) < exit_z:
            cp = 0
        pos_list.append(cp)

    position = pd.Series(pos_list,
                         index=zscore.index).shift(1).fillna(0)

    # Returns
    y_ret = np.log(y / y.shift(1)).reindex(position.index).fillna(0)
    x_ret = np.log(x / x.shift(1)).reindex(position.index).fillna(0)
    spread_ret = position * (y_ret - hedge_ratio * x_ret)

    # Costs
    trades  = position.diff().abs().fillna(0)
    net_ret = spread_ret - trades * cost_pct
    equity  = (1 + net_ret).cumprod()

    # Metrics
    ar = net_ret.mean() * 252
    av = net_ret.std()  * np.sqrt(252)
    sh = (ar - 0.02)   / av
    dd = (equity - equity.cummax()) / equity.cummax()

    return {
        "ann_ret"  : ar,
        "ann_vol"  : av,
        "sharpe"   : sh,
        "max_dd"   : dd.min(),
        "n_trades" : int(trades.sum()),
        "equity"   : equity
    }


def find_cointegrated_pairs(tickers, start="2015-01-01",
                             end="2024-01-01"):
    """
    Test all pairs in a universe for cointegration.
    Returns ranked list of cointegrated pairs.
    """
    raw = yf.download(tickers, start=start,
                      end=end, auto_adjust=True)
    prices = raw["Close"]
    prices.columns = prices.columns.get_level_values(0)

    results = []
    pairs   = list(combinations(tickers, 2))

    print(f"Testing {len(pairs)} pairs...")

    for t1, t2 in pairs:
        try:
            s1 = prices[t1].dropna()
            s2 = prices[t2].dropna()
            # Align
            s1, s2 = s1.align(s2, join='inner')
            if len(s1) < 252:
                continue
            _, pval, _ = coint(s1, s2)
            results.append({
                "pair"  : f"{t1}/{t2}",
                "pvalue": pval,
                "cointegrated": pval < 0.05
            })
        except:
            continue

    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("pvalue")

    print(f"\nCointegrated pairs (p < 0.05):")
    print("=" * 40)
    coint_pairs = df_res[df_res["cointegrated"]]
    if len(coint_pairs) == 0:
        print("  None found in this universe")
    else:
        for _, row in coint_pairs.iterrows():
            print(f"  {row['pair']:<15} p={row['pvalue']:.4f}")
    print("=" * 40)
    print(f"\nTop 5 by p-value (all pairs):")
    print(df_res.head().to_string(index=False))

    return df_res
