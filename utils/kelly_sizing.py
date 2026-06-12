import numpy as np
import pandas as pd

def kelly_fraction(returns):
    """
    Kelly fraction for continuous return strategy.
    Kelly = mean(returns) / variance(returns)
    """
    mu     = returns.mean()
    sigma2 = returns.var()
    return mu / sigma2


def position_sizer(capital, kelly_fraction, price,
                   kelly_multiplier=0.5, max_leverage=2.0):
    """
    Position sizer with leverage cap.
    kelly_multiplier=0.5 for half-Kelly (recommended)
    """
    effective_fraction = kelly_fraction * kelly_multiplier
    effective_fraction = min(effective_fraction, max_leverage)
    effective_fraction = max(effective_fraction, 0.0)

    position_value = capital * effective_fraction
    shares = position_value / price

    return {
        "effective_fraction": effective_fraction,
        "position_value": position_value,
        "shares": shares
    }


def position_sizer_no_leverage(capital, kelly_fraction, price,
                                kelly_multiplier=0.5):
    """
    Position sizer capped at 1.0x — for retail accounts
    without margin access.
    """
    return position_sizer(capital, kelly_fraction, price,
                          kelly_multiplier, max_leverage=1.0)


def kelly_simulation(returns, fractions, rf=0.02):
    """
    Simulate equity curves at different Kelly fractions.
    Returns DataFrame of results.
    """
    results = []
    for name, frac in fractions.items():
        scaled = returns * frac
        equity = (1 + scaled).cumprod()
        ar = scaled.mean() * 252
        av = scaled.std()  * np.sqrt(252)
        sh = (ar - rf) / av
        dd = ((equity - equity.cummax()) / equity.cummax()).min()
        results.append({"name": name, "fraction": frac,
                        "ann_ret": ar, "sharpe": sh,
                        "max_dd": dd, "equity": equity})
    return results
