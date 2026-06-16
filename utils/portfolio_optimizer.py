import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns

def build_portfolios(prices):
    """
    Build equal weight, max Sharpe and min vol portfolios.
    Returns dict of weights for each approach.
    """
    mu = expected_returns.mean_historical_return(prices)
    S  = risk_models.sample_cov(prices)

    n = len(prices.columns)
    ew = {t: 1/n for t in prices.columns}

    ef1 = EfficientFrontier(mu, S)
    ef1.max_sharpe()
    ms = dict(ef1.clean_weights())

    ef2 = EfficientFrontier(mu, S)
    ef2.min_volatility()
    mv = dict(ef2.clean_weights())

    return {"equal_weight": ew,
            "max_sharpe"  : ms,
            "min_vol"     : mv}

def portfolio_returns(weights_dict, returns):
    """Calculate portfolio return stream from weights dict."""
    w = pd.Series(weights_dict).reindex(returns.columns).fillna(0)
    w = w / w.sum()
    return (returns * w).sum(axis=1)

def efficient_frontier(returns, n_sims=3000):
    """Plot efficient frontier from random portfolio simulations."""
    n = returns.shape[1]
    rand_ret, rand_vol, rand_sh = [], [], []

    for _ in range(n_sims):
        w = np.random.random(n)
        w = w / w.sum()
        r = np.sum(returns.mean() * w) * 252
        v = np.sqrt(np.dot(w.T, np.dot(returns.cov()*252, w)))
        rand_ret.append(r)
        rand_vol.append(v)
        rand_sh.append((r-0.02)/v)

    return np.array(rand_ret), np.array(rand_vol), np.array(rand_sh)
