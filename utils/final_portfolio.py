import numpy as np
import pandas as pd
import yfinance as yf

def build_final_portfolio(s1, s2, s3,
                           w1=0.60, w2=0.15, w3=0.25):
    """
    Combine three strategy return streams into
    final multi-strategy portfolio.

    s1: Core SPY SMA+RSI2+VolTgt (60%)
    s2: BTC SMA+VolTgt satellite (15%)
    s3: Equal-weight multi-asset (25%)
    """
    common = s1.index.intersection(
             s2.index.intersection(s3.index))
    s1 = s1.reindex(common).fillna(0)
    s2 = s2.reindex(common).fillna(0)
    s3 = s3.reindex(common).fillna(0)
    return w1*s1 + w2*s2 + w3*s3


def portfolio_metrics(returns, name="Portfolio", days=252):
    """Full performance metrics for any return stream."""
    ar = returns.mean() * days
    av = returns.std()  * np.sqrt(days)
    sh = (ar - 0.02) / av if av > 0 else 0
    w  = (1 + returns).cumprod()
    dd = ((w - w.cummax()) / w.cummax()).min()
    print(f"  {name:<30} AR:{ar:>7.2%} Vol:{av:>7.2%}"
          f" Sh:{sh:>5.2f} DD:{dd:>8.2%}")
    return {"ann_ret": ar, "ann_vol": av,
            "sharpe": sh, "max_dd": dd, "equity": w}


def strategy_correlation(return_streams):
    """
    Check correlation between strategy return streams.
    Low correlation = genuine diversification benefit.
    """
    df = pd.DataFrame(return_streams)
    return df.corr()
