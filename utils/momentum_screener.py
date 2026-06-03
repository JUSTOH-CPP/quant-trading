import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def momentum_screener(prices, top_n=5):
    """
    Full momentum screener using pre-downloaded prices.
    Ranks stocks by 12-1 month momentum.
    """
    lookback = 252  # 12 months
    skip     = 21   # 1 month

    # Check we have enough data
    if len(prices) < lookback + skip:
        print(f"Need at least {lookback+skip} rows, have {len(prices)}")
        return None

    # Momentum = price 1 month ago / price 12 months ago - 1
    price_1m  = prices.iloc[-(skip+1)]
    price_12m = prices.iloc[-(lookback+skip+1)]

    momentum = (price_1m / price_12m - 1).dropna()
    momentum = momentum.sort_values(ascending=False)

    top    = momentum.head(top_n)
    bottom = momentum.tail(top_n)

    print(f"\n{'='*45}")
    print(f"  MOMENTUM SCREENER")
    print(f"  Date: {prices.index[-1].date()}")
    print(f"{'='*45}")
    print(f"\n  LONG (Top {top_n}):")
    for t, s in top.items():
        print(f"    {t:<6} {s:>+8.2%}")
    print(f"\n  SHORT (Bottom {top_n}):")
    for t, s in bottom.items():
        print(f"    {t:<6} {s:>+8.2%}")
    print(f"{'='*45}")

    return {"long": top, "short": bottom, "all": momentum}
