import numpy as np
import pandas as pd
from scipy import stats

def var_historical(returns, confidence=0.95, capital=100000):
    """Historical VaR — no distribution assumptions."""
    var_pct = np.percentile(returns, (1-confidence)*100)
    return var_pct, var_pct * capital

def var_parametric(returns, confidence=0.95, capital=100000):
    """Parametric VaR — assumes normal distribution."""
    mu      = returns.mean()
    sigma   = returns.std()
    var_pct = mu + stats.norm.ppf(1-confidence) * sigma
    return var_pct, var_pct * capital

def var_montecarlo(returns, confidence=0.95,
                    capital=100000, n_sims=10000):
    """Monte Carlo VaR — simulate from fitted distribution."""
    mu, sigma = returns.mean(), returns.std()
    simulated = np.random.normal(mu, sigma, n_sims)
    var_pct   = np.percentile(simulated, (1-confidence)*100)
    return var_pct, var_pct * capital

def cvar(returns, confidence=0.95, capital=100000):
    """CVaR/Expected Shortfall."""
    var_pct, _ = var_historical(returns, confidence)
    tail       = returns[returns <= var_pct]
    cvar_pct   = tail.mean() if len(tail) > 0 else var_pct
    return cvar_pct, cvar_pct * capital

def stress_test(strategy_returns, benchmark_returns,
                scenarios):
    """
    Apply historical crisis scenarios to strategy.
    scenarios: dict of {name: (start_date, end_date)}
    Returns DataFrame of results.
    """
    results = []
    for name, (start, end) in scenarios.items():
        s = strategy_returns.loc[start:end]
        b = benchmark_returns.loc[start:end]
        if len(s) == 0:
            continue
        s_cum = (1 + s).prod() - 1
        b_cum = (1 + b).prod() - 1
        results.append({
            "scenario"  : name,
            "start"     : start,
            "end"       : end,
            "strategy"  : s_cum,
            "benchmark" : b_cum,
            "protection": s_cum - b_cum
        })
    return pd.DataFrame(results)

def rolling_var(returns, window=252, confidence=0.95):
    """Rolling historical VaR."""
    return returns.rolling(window).apply(
        lambda x: np.percentile(x, (1-confidence)*100),
        raw=True)
