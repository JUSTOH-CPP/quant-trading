import numpy as np
import pandas as pd
import yfinance as yf

def load_crypto(ticker="BTC-USD",
                start="2018-01-01",
                end="2024-01-01"):
    """Load crypto OHLCV data from Yahoo Finance."""
    raw = yf.download(ticker, start=start,
                      end=end, auto_adjust=True)
    raw.columns = raw.columns.get_level_values(0)
    return raw["Close"].squeeze()


def crypto_stats(returns, name, days=365):
    """Print annualised stats for crypto strategy."""
    ar = returns.mean() * days
    av = returns.std()  * np.sqrt(days)
    sh = (ar - 0.02) / av if av > 0 else 0
    w  = (1 + returns).cumprod()
    dd = ((w - w.cummax()) / w.cummax()).min()
    print(f"  {name:<25} AR:{ar:>7.2%} Vol:{av:>7.2%} "
          f"Sh:{sh:>5.2f} DD:{dd:>8.2%}")
    return {"ann_ret": ar, "ann_vol": av,
            "sharpe": sh, "max_dd": dd}


def crypto_vol_target(returns, target_vol=0.20, window=20):
    """
    Volatility targeting for crypto.
    Higher target (20%) than equities (10%) due to
    extreme crypto volatility.
    """
    realized_vol = returns.rolling(window).std() * np.sqrt(365)
    vol_scalar   = (target_vol / realized_vol).clip(0, 1)
    return vol_scalar


def sma_crypto_backtest(price, fast=50, slow=200,
                         target_vol=0.20, cost_pct=0.001):
    """
    SMA crossover strategy adapted for crypto.
    Includes vol targeting and higher transaction costs.
    """
    returns  = np.log(price / price.shift(1)).dropna()
    sma_fast = price.rolling(fast).mean()
    sma_slow = price.rolling(slow).mean()
    position = (sma_fast > sma_slow).astype(int).shift(1).fillna(0)
    position = position.reindex(returns.index).fillna(0)
    trades   = position.diff().abs().fillna(0)

    vol_scalar = crypto_vol_target(returns, target_vol)
    net_ret    = (position * returns *
                  vol_scalar.shift(1).fillna(1) -
                  trades * cost_pct)
    return net_ret
