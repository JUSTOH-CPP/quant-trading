import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sma_backtest_quick(price, returns, fast, slow, cost=0.0006):
    """Quick SMA backtest returning Sharpe."""
    if fast >= slow:
        return -999
    sma_f  = price.rolling(fast).mean()
    sma_s  = price.rolling(slow).mean()
    pos    = (sma_f > sma_s).astype(int).shift(1).fillna(0)
    pos    = pos.reindex(returns.index).fillna(0)
    trades = pos.diff().abs().fillna(0)
    ret    = pos * returns - trades * cost
    ar     = ret.mean() * 252
    av     = ret.std()  * np.sqrt(252)
    return (ar - 0.02) / av if av > 0 else -999

def bonferroni_threshold(n_tests, alpha=0.05):
    """Return Bonferroni-corrected significance threshold."""
    corrected = alpha / n_tests
    print(f"Tests performed  : {n_tests}")
    print(f"Standard alpha   : {alpha:.3f}")
    print(f"Bonferroni alpha : {corrected:.4f}")
    return corrected

def overfitting_audit(strategy_name, rules_results):
    """
    Print anti-overfitting audit.
    rules_results: list of (rule_name, passed, note)
    """
    print(f"ANTI-OVERFITTING AUDIT — {strategy_name}")
    print("=" * 55)
    passes = 0
    for rule, passed, note in rules_results:
        symbol = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"  {symbol} {rule}")
        print(f"    [{status}] {note}")
        if passed: passes += 1
    print("=" * 55)
    print(f"  Score  : {passes}/{len(rules_results)}")
    verdict = "TRADEABLE" if passes >= len(rules_results)*0.85 else "NEEDS WORK"
    print(f"  Verdict: {verdict}")
    return passes
